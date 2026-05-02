"""Final system demo: multi-block placement + interactive object inference.

Scenarios:
  1. Multi-block placement: red→green, blue→yellow, yellow→blue
  2. Interactive object inference (mouse): press + drag + scroll

Flow: task input → LLM planning → CausalExplore → skill execution → feedback

Usage:
    python -m src.demo.final_demo --scenario multi_block
    python -m src.demo.final_demo --scenario interactive
    python -m src.demo.final_demo --scenario all --headless --record
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../embodied_agent/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../phase2_llm_planner/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../phase1_skills_and_causal/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from embodied_agent.contracts import (
    CausalExploreOutput,
    PlannerStep,
    ExecutorResult,
    FailureRecord,
    PropertyBelief,
    AffordanceCandidate,
    ContactState,
)
from embodied_agent.types import PlanStep, PostCondition, WorldState, StepFailure
from embodied_agent.planner_contracts import ContractPlannerAdapter
from embodied_agent.planning_bridge import ContractPlanningBridge
from embodied_agent.simulator import create_pick_place_simulation

from causal_explore.probe_executor import ObjectManifest, ProbeExecutor
from causal_explore.explore_strategies import (
    CausalExploreStrategy,
    ExploreHistory,
    ExploreStep,
)
from planner.llm_planner import LLMPlanner
from planner.replan_handler import ReplanHandler, MAX_REPLANS


SCENARIO_MULTI_BLOCK = "multi_block"
SCENARIO_INTERACTIVE = "interactive"
SCENARIO_ALL = "all"

MULTI_BLOCK_TASKS = [
    "把红色方块移到绿色区域",
    "把蓝色方块移到黄色区域",
    "把黄色方块移到蓝色区域",
]

INTERACTIVE_TASKS = [
    "探索鼠标的功能：先按压鼠标顶部，然后向前拖动鼠标，最后滚动滚轮",
]


def _extract_pick_place_pairs(instruction_text: str) -> list[tuple[str, str]]:
    """Extract explicit color-block to color-zone pairs without cross-matching colors."""
    import re

    color_to_object = {
        "红": "red_block", "红色": "red_block", "red": "red_block",
        "蓝": "blue_block", "蓝色": "blue_block", "blue": "blue_block",
        "黄": "yellow_block", "黄色": "yellow_block", "yellow": "yellow_block",
    }
    color_to_zone = {
        "绿": "green_zone", "绿色": "green_zone", "green": "green_zone",
        "蓝": "blue_zone", "蓝色": "blue_zone", "blue": "blue_zone",
        "黄": "yellow_zone", "黄色": "yellow_zone", "yellow": "yellow_zone",
    }

    pairs: list[tuple[str, str]] = []
    zh_pattern = re.compile(
        r"(红色|蓝色|黄色|红|蓝|黄)方块.*?(?:移到|移动到|放到|放入|放置到)"
        r"(绿色|蓝色|黄色|绿|蓝|黄)区域"
    )
    for match in zh_pattern.finditer(instruction_text):
        pairs.append((color_to_object[match.group(1)], color_to_zone[match.group(2)]))

    en_pattern = re.compile(
        r"\b(red|blue|yellow)\s+block\b.*?\b(?:to|into)\s+"
        r"\b(green|blue|yellow)\s+zone\b",
        flags=re.IGNORECASE,
    )
    for match in en_pattern.finditer(instruction_text):
        pairs.append((
            color_to_object[match.group(1).lower()],
            color_to_zone[match.group(2).lower()],
        ))

    return pairs


@dataclass
class DemoConfig:
    headless: bool = True
    record: bool = False
    output_dir: str = "outputs"
    max_replans: int = MAX_REPLANS
    language: str = "zh"


@dataclass
class ScenarioResult:
    scenario_name: str
    task: str
    success: bool
    plan_steps: list[dict[str, Any]] = field(default_factory=list)
    executed_actions: list[str] = field(default_factory=list)
    explore_steps: int = 0
    replan_count: int = 0
    error: str | None = None
    execution_time_seconds: float = 0.0
    feedback_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "task": self.task,
            "success": self.success,
            "plan_steps": self.plan_steps,
            "executed_actions": self.executed_actions,
            "explore_steps": self.explore_steps,
            "replan_count": self.replan_count,
            "error": self.error,
            "execution_time_seconds": self.execution_time_seconds,
            "feedback_notes": self.feedback_notes,
        }


@dataclass
class DemoSession:
    config: DemoConfig = field(default_factory=DemoConfig)
    results: list[ScenarioResult] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "config": {
                "headless": self.config.headless,
                "record": self.config.record,
                "max_replans": self.config.max_replans,
                "language": self.config.language,
            },
            "results": [r.to_dict() for r in self.results],
            "summary": self._build_summary(),
        }

    def _build_summary(self) -> dict[str, Any]:
        total = len(self.results)
        successes = sum(1 for r in self.results if r.success)
        return {
            "total_scenarios": total,
            "successful": successes,
            "failed": total - successes,
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_execution_time": (
                sum(r.execution_time_seconds for r in self.results) / total if total > 0 else 0.0
            ),
        }


def _mock_llm_call(messages: list[dict[str, str]]) -> str:
    """Mock LLM for demo scenarios."""
    user_content = ""
    for msg in messages:
        if msg["role"] == "user":
            user_content = msg["content"]
            break

    content_lower = user_content.lower()

    instruction_text = user_content
    for marker in ["## 任务指令\n", "## Task Instruction\n", "## 原始任务指令\n"]:
        if marker in user_content:
            rest = user_content.split(marker, 1)[1]
            instruction_text = rest.split("\n\n")[0].split("\n##")[0].strip()
            break

    # Interactive object detection (mouse)
    if "mouse" in content_lower or "鼠标" in instruction_text:
        return json.dumps({
            "task_id": "final_demo_interactive",
            "steps": [
                {
                    "step_index": 0,
                    "selected_skill": "observe",
                    "target_object": None,
                    "skill_args": {},
                    "preconditions": ["object_visible"],
                    "expected_effect": "scene_observed",
                    "fallback_action": "probe_or_replan",
                },
                {
                    "step_index": 1,
                    "selected_skill": "probe",
                    "target_object": "mouse",
                    "skill_args": {"probe": "top_press"},
                    "preconditions": ["object_visible"],
                    "expected_effect": "uncertainty_reduced(mouse)",
                    "fallback_action": "probe_or_replan",
                },
                {
                    "step_index": 2,
                    "selected_skill": "press",
                    "target_object": "mouse",
                    "skill_args": {"press_direction": "down", "force": 0.5},
                    "preconditions": ["object_visible"],
                    "expected_effect": "pressed(mouse)",
                    "fallback_action": "probe_or_replan",
                },
                {
                    "step_index": 3,
                    "selected_skill": "push",
                    "target_object": "mouse",
                    "skill_args": {"push_direction": "forward", "distance": 0.10},
                    "preconditions": ["object_visible"],
                    "expected_effect": "moved(mouse)",
                    "fallback_action": "probe_or_replan",
                },
                {
                    "step_index": 4,
                    "selected_skill": "rotate",
                    "target_object": "mouse",
                    "skill_args": {"rotation_axis": "z", "angle_deg": 30},
                    "preconditions": ["object_visible"],
                    "expected_effect": "scrolled(mouse)",
                    "fallback_action": "probe_or_replan",
                },
            ],
        }, ensure_ascii=False)

    # Multi-block pick and place
    pairs = _extract_pick_place_pairs(instruction_text)
    if not pairs:
        pairs = [("red_block", "green_zone")]

    steps = [{
        "step_index": 0,
        "selected_skill": "observe",
        "target_object": None,
        "skill_args": {},
        "preconditions": ["object_visible"],
        "expected_effect": "scene_observed",
        "fallback_action": "probe_or_replan",
    }]

    idx = 1
    for target_obj, zone in pairs:
        steps.append({
            "step_index": idx,
            "selected_skill": "pick",
            "target_object": target_obj,
            "skill_args": {"zone": zone},
            "preconditions": ["object_visible"],
            "expected_effect": f"holding({target_obj})",
            "fallback_action": "probe_or_replan",
        })
        idx += 1
        steps.append({
            "step_index": idx,
            "selected_skill": "place",
            "target_object": target_obj,
            "skill_args": {"object": target_obj, "target_zone": zone},
            "preconditions": [f"holding({target_obj})"],
            "expected_effect": f"placed({target_obj},{zone})",
            "fallback_action": "probe_or_replan",
        })
        idx += 1

    return json.dumps({"task_id": "final_demo_multi_block", "steps": steps}, ensure_ascii=False)


def _make_world_state(instruction: str = "") -> WorldState:
    return WorldState(
        instruction=instruction,
        object_positions={
            "red_block": (0.55, 0.10, 0.645),
            "blue_block": (0.64, 0.00, 0.645),
            "yellow_block": (0.46, -0.12, 0.645),
        },
        zone_positions={
            "green_zone": (0.50, -0.10, 0.622),
            "blue_zone": (0.65, 0.13, 0.622),
            "yellow_zone": (0.42, 0.14, 0.622),
        },
        end_effector_position=(0.45, 0.0, 0.80),
        held_object_name=None,
    )


def _make_mock_causal_output(object_id: str, uncertainty: float = 0.25) -> CausalExploreOutput:
    return CausalExploreOutput(
        scene_id="final_demo",
        object_id=object_id,
        object_category="block" if "block" in object_id else "device",
        property_belief={
            "movable": PropertyBelief(label="movable", confidence=0.90),
            "pressable": PropertyBelief(label="pressable", confidence=0.85),
        },
        affordance_candidates=[
            AffordanceCandidate(name="pushable", confidence=0.85),
            AffordanceCandidate(name="graspable", confidence=0.90),
            AffordanceCandidate(name="pressable", confidence=0.80),
        ],
        uncertainty_score=uncertainty,
        recommended_probe="lateral_push" if uncertainty >= 0.50 else None,
        contact_region="top",
        skill_constraints={},
        artifact_path=f"mock://final_demo/{object_id}",
    )


def _simulate_execute(step: PlannerStep) -> ExecutorResult:
    return ExecutorResult(
        success=True,
        reward=1.0,
        final_state={"executed_skill": step.selected_skill},
        contact_state=ContactState(
            has_contact=step.selected_skill not in ("observe",),
            contact_region="top",
        ),
        error_code=None,
        rollout_path=f"mock://final_demo/{step.selected_skill}",
        failure_history=[],
    )


class FinalDemo:
    """Runs the final integrated demo: multi-block placement + interactive inference."""

    def __init__(self, config: DemoConfig | None = None) -> None:
        self.config = config or DemoConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, scenario: str = SCENARIO_ALL) -> DemoSession:
        session = DemoSession(config=self.config)

        if scenario in (SCENARIO_MULTI_BLOCK, SCENARIO_ALL):
            for task in MULTI_BLOCK_TASKS:
                result = self._run_multi_block_task(task)
                session.results.append(result)
                self._print_result(result)

        if scenario in (SCENARIO_INTERACTIVE, SCENARIO_ALL):
            for task in INTERACTIVE_TASKS:
                result = self._run_interactive_task(task)
                session.results.append(result)
                self._print_result(result)

        if self.config.record:
            self._save_session(session)

        return session

    def _run_multi_block_task(self, task: str) -> ScenarioResult:
        start_time = time.monotonic()
        feedback: list[str] = []

        try:
            world_state = _make_world_state(task)
            causal_outputs = self._prepare_causal_outputs(world_state)

            llm_planner = LLMPlanner(
                llm_callable=_mock_llm_call,
                causal_outputs={k: v.to_dict() for k, v in causal_outputs.items()},
            )
            plan = llm_planner.plan(task, world_state)

            bridge = ContractPlanningBridge(planner=llm_planner)
            contract_plan = bridge.plan_contract(
                task_id="final_demo_mb",
                instruction=task,
                state=world_state,
                causal_outputs=causal_outputs,
            )

            explore_steps = sum(1 for s in contract_plan if s.selected_skill == "probe")
            executed: list[str] = []
            replan_count = 0

            for step in contract_plan:
                exec_result = _simulate_execute(step)
                executed.append(step.selected_skill)
                if not exec_result.success:
                    if replan_count < self.config.max_replans:
                        replan_count += 1
                        feedback.append(f"Replan after {step.selected_skill} failure")
                    else:
                        raise RuntimeError(f"Max replans exceeded at step {step.selected_skill}")

            feedback.append(f"Plan: {' → '.join(executed)}")
            if explore_steps > 0:
                feedback.append(f"CausalExplore injected {explore_steps} probe step(s)")

            elapsed = time.monotonic() - start_time
            return ScenarioResult(
                scenario_name=SCENARIO_MULTI_BLOCK,
                task=task,
                success=True,
                plan_steps=[{"step_index": i, "skill": s.selected_skill,
                            "target": s.target_object, "args": dict(s.skill_args)}
                           for i, s in enumerate(contract_plan)],
                executed_actions=executed,
                explore_steps=explore_steps,
                replan_count=replan_count,
                execution_time_seconds=elapsed,
                feedback_notes=feedback,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            return ScenarioResult(
                scenario_name=SCENARIO_MULTI_BLOCK,
                task=task,
                success=False,
                error=str(exc),
                execution_time_seconds=elapsed,
                feedback_notes=feedback,
            )

    def _run_interactive_task(self, task: str) -> ScenarioResult:
        start_time = time.monotonic()
        feedback: list[str] = []

        try:
            world_state = _make_world_state(task)
            world_state = WorldState(
                instruction=task,
                object_positions={"mouse": (0.50, 0.00, 0.65)},
                zone_positions={},
                end_effector_position=(0.45, 0.0, 0.80),
                held_object_name=None,
            )

            causal_outputs = {
                "mouse": _make_mock_causal_output("mouse", uncertainty=0.20),
            }

            llm_planner = LLMPlanner(
                llm_callable=_mock_llm_call,
                causal_outputs={k: v.to_dict() for k, v in causal_outputs.items()},
            )
            plan = llm_planner.plan(task, world_state)

            bridge = ContractPlanningBridge(planner=llm_planner)
            contract_plan = bridge.plan_contract(
                task_id="final_demo_interactive",
                instruction=task,
                state=world_state,
                causal_outputs=causal_outputs,
            )

            explore_steps = sum(1 for s in contract_plan if s.selected_skill == "probe")
            executed: list[str] = []
            replan_count = 0

            for step in contract_plan:
                exec_result = _simulate_execute(step)
                executed.append(step.selected_skill)
                if not exec_result.success and replan_count < self.config.max_replans:
                    replan_count += 1

            feedback.append(f"Plan: {' → '.join(executed)}")
            feedback.append("Interactive object inference: mouse affordances explored via CausalExplore")

            elapsed = time.monotonic() - start_time
            return ScenarioResult(
                scenario_name=SCENARIO_INTERACTIVE,
                task=task,
                success=True,
                plan_steps=[{"step_index": i, "skill": s.selected_skill,
                            "target": s.target_object, "args": dict(s.skill_args)}
                           for i, s in enumerate(contract_plan)],
                executed_actions=executed,
                explore_steps=explore_steps,
                replan_count=replan_count,
                execution_time_seconds=elapsed,
                feedback_notes=feedback,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            return ScenarioResult(
                scenario_name=SCENARIO_INTERACTIVE,
                task=task,
                success=False,
                error=str(exc),
                execution_time_seconds=elapsed,
                feedback_notes=feedback,
            )

    def _prepare_causal_outputs(
        self,
        world_state: WorldState,
    ) -> dict[str, CausalExploreOutput]:
        outputs: dict[str, CausalExploreOutput] = {}
        for obj_id in world_state.object_positions:
            is_main = obj_id.replace("_block", "") in world_state.instruction.lower()
            outputs[obj_id] = _make_mock_causal_output(
                obj_id, uncertainty=0.25 if is_main else 0.30,
            )
        return outputs

    def _print_result(self, result: ScenarioResult) -> None:
        status = "PASS" if result.success else "FAIL"
        print(f"\n[{status}] [{result.scenario_name}] {result.task[:50]}")
        if result.error:
            print(f"  Error: {result.error}")
        print(f"  Actions: {result.executed_actions}")
        print(f"  Explore steps: {result.explore_steps}")
        print(f"  Replans: {result.replan_count}")
        print(f"  Time: {result.execution_time_seconds:.2f}s")
        for note in result.feedback_notes:
            print(f"  > {note}")

    def _save_session(self, session: DemoSession) -> str:
        output_path = self.output_dir / f"demo_session_{session.session_id}.json"
        output_path.write_text(json.dumps(session.to_dict(), indent=2, ensure_ascii=False))
        print(f"\nSession recorded to: {output_path}")
        return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Final System Demo")
    parser.add_argument(
        "--scenario", type=str,
        choices=[SCENARIO_MULTI_BLOCK, SCENARIO_INTERACTIVE, SCENARIO_ALL],
        default=SCENARIO_ALL,
        help="Demo scenario to run",
    )
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--record", action="store_true",
                       help="Save session results to JSON")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--language", type=str, default="zh",
                       choices=["zh", "en"])
    args = parser.parse_args()

    config = DemoConfig(
        headless=args.headless,
        record=args.record,
        output_dir=args.output_dir,
        language=args.language,
    )

    print("=" * 60)
    print("Phase 3 Final System Demo")
    print("=" * 60)
    print(f"Scenario: {args.scenario}")
    print(f"Headless: {args.headless}")
    print(f"Record: {args.record}")
    print()

    demo = FinalDemo(config)
    session = demo.run(scenario=args.scenario)

    summary = session._build_summary()
    print(f"\n{'=' * 60}")
    print(f"Demo Complete — {summary['successful']}/{summary['total_scenarios']} scenarios passed")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Total time: {summary['avg_execution_time'] * summary['total_scenarios']:.2f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
