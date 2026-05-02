"""Closed-loop demo: natural language → LLM plan → explore → execute → feedback → replan.

Usage:
    python -m src.demo.closed_loop_demo --task "把红色方块移到绿色区域"
    python -m src.demo.closed_loop_demo --scenario multi_block --headless
    python -m src.demo.closed_loop_demo --scenario interactive --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../phase1_skills_and_causal/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../embodied_agent/src'))

from embodied_agent.contracts import (
    PlannerStep,
    CausalExploreOutput,
    ExecutorResult,
    FailureRecord,
    PropertyBelief,
    AffordanceCandidate,
    build_planner_step_from_causal_output,
    build_planner_step_from_plan_step,
)
from embodied_agent.types import PlanStep, PostCondition, WorldState, StepFailure
from embodied_agent.planner import Planner
from embodied_agent.planner_contracts import ContractPlannerAdapter
from embodied_agent.planning_bridge import ContractPlanningBridge

from src.planner.llm_planner import LLMPlanner, LlmCallable
from src.planner.uncertainty_handler import UncertaintyHandler
from src.planner.replan_handler import ReplanHandler, MAX_REPLANS, MAX_SKILL_RETRIES


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


def _mock_instruction_llm(messages: list[dict[str, str]]) -> str:
    """Mock LLM that parses natural language instructions into structured plans.

    Supports Chinese and English instructions for pick-place tasks.
    """
    user_content = ""
    for msg in messages:
        if msg["role"] == "user":
            user_content = msg["content"]
            break

    content_lower = user_content.lower()

    task_id = "demo_task"

    # Check if this is a replan request
    is_replan = "失败" in user_content or "failed" in content_lower or "replan" in content_lower

    # Extract just the instruction text (first line after "任务指令"/"Task Instruction")
    instruction_text = user_content
    for marker in ["## 任务指令\n", "## Task Instruction\n", "## 原始任务指令\n", "## Original Task Instruction\n"]:
        if marker in user_content:
            rest = user_content.split(marker, 1)[1]
            instruction_text = rest.split("\n\n")[0].split("\n##")[0].strip()
            break

    # Interactive object detection (mouse, etc.)
    if "mouse" in content_lower or "鼠标" in instruction_text:
        return json.dumps({
            "task_id": task_id,
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
            ],
        }, ensure_ascii=False)

    # Multi-block pick and place
    pairs = _extract_pick_place_pairs(instruction_text)
    if not pairs:
        pairs = [("red_block", "green_zone")]

    steps = [
        {
            "step_index": 0,
            "selected_skill": "observe",
            "target_object": None,
            "skill_args": {},
            "preconditions": ["object_visible"],
            "expected_effect": "scene_observed",
            "fallback_action": "probe_or_replan",
        },
    ]

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

    if is_replan:
        for step in steps:
            if step["selected_skill"] in ("pick", "place"):
                step["skill_args"]["retry"] = True

    return json.dumps({"task_id": task_id, "steps": steps}, ensure_ascii=False)


@dataclass
class DemoConfig:
    headless: bool = True
    dry_run: bool = False
    max_replans: int = MAX_REPLANS
    max_skill_retries: int = MAX_SKILL_RETRIES
    language: str = "zh"


@dataclass
class DemoResult:
    instruction: str
    success: bool = False
    plan_steps: list[dict[str, Any]] = field(default_factory=list)
    executed_steps: list[str] = field(default_factory=list)
    failure_history: list[dict[str, Any]] = field(default_factory=list)
    replan_count: int = 0
    error: str | None = None


def make_mock_causal_output(
    object_id: str,
    uncertainty: float = 0.3,
    affordances: list[tuple[str, float]] | None = None,
) -> CausalExploreOutput:
    """Build a mock CausalExploreOutput for testing/demo."""
    if affordances is None:
        affordances = [
            ("pushable", 0.85),
            ("graspable", 0.90),
            ("pressable", 0.80),
        ]
    return CausalExploreOutput(
        scene_id="demo_scene",
        object_id=object_id,
        object_category="block",
        property_belief={
            "movable": PropertyBelief(label="movable", confidence=0.90),
            "rigid": PropertyBelief(label="rigid", confidence=0.85),
        },
        affordance_candidates=[
            AffordanceCandidate(name=name, confidence=conf)
            for name, conf in affordances
        ],
        uncertainty_score=uncertainty,
        recommended_probe="lateral_push" if uncertainty >= 0.50 else None,
        contact_region="top",
        skill_constraints={},
        artifact_path=f"mock://demo/{object_id}",
    )


def make_mock_world_state(
    instruction: str = "",
    objects: dict[str, tuple[float, float, float]] | None = None,
    zones: dict[str, tuple[float, float, float]] | None = None,
) -> WorldState:
    """Build a mock WorldState for testing/demo."""
    return WorldState(
        instruction=instruction,
        object_positions=objects or {
            "red_block": (0.55, 0.10, 0.645),
            "blue_block": (0.64, 0.00, 0.645),
            "yellow_block": (0.46, -0.12, 0.645),
        },
        zone_positions=zones or {
            "green_zone": (0.50, -0.10, 0.622),
            "blue_zone": (0.65, 0.13, 0.622),
            "yellow_zone": (0.42, 0.14, 0.622),
        },
        end_effector_position=(0.45, 0.0, 0.80),
        held_object_name=None,
    )


class ClosedLoopDemo:
    """Runs the full closed-loop demo: plan → explore → execute → feedback → replan."""

    def __init__(self, config: DemoConfig | None = None) -> None:
        self.config = config or DemoConfig()
        self.replan_handler = ReplanHandler()
        self.uncertainty_handler = UncertaintyHandler()

    def run(self, instruction: str) -> DemoResult:
        """Run closed-loop demo with the given natural language instruction."""
        result = DemoResult(instruction=instruction)

        world_state = make_mock_world_state(instruction=instruction)
        causal_outputs = self._prepare_causal_outputs(instruction, world_state)

        llm_callable: LlmCallable = _mock_instruction_llm
        llm_planner = LLMPlanner(
            llm_callable=llm_callable,
            language=self.config.language,
            causal_outputs={obj_id: co.to_dict() for obj_id, co in causal_outputs.items()},
        )

        plan = llm_planner.plan(instruction, world_state)
        result.plan_steps = [self._plan_step_to_dict(s) for s in plan]

        if self.config.dry_run:
            result.success = True
            result.executed_steps = [s.action for s in plan]
            return result

        contract_plan = ContractPlannerAdapter(llm_planner).plan_contract(
            task_id="demo_task", instruction=instruction, state=world_state,
        )

        bridge = ContractPlanningBridge(planner=llm_planner)
        contract_plan = bridge._inject_probe_if_needed(
            task_id="demo_task",
            contract_plan=contract_plan,
            causal_outputs=causal_outputs,
        )

        replan_count = 0
        failure_history: list[dict[str, Any]] = []

        for step_idx, step in enumerate(contract_plan):
            try:
                exec_result = self._simulate_execute(step)
                result.executed_steps.append(step.selected_skill)

                if not exec_result.success:
                    failure_record = {
                        "step_index": step_idx,
                        "selected_skill": step.selected_skill,
                        "failure_source": "execution_error",
                        "reason": exec_result.error_code or "execution_failed",
                        "replan_attempt": replan_count,
                    }
                    failure_history.append(failure_record)

                    if replan_count >= self.config.max_replans:
                        result.error = "Max replans exhausted"
                        break

                    replan_count += 1
                    remaining = contract_plan[step_idx + 1:]
                    failed_ps = PlanStep(
                        action=step.selected_skill,
                        target=step.target_object,
                        parameters=step.skill_args,
                    )
                    failure = StepFailure(
                        failed_step=failed_ps,
                        source="execution_error",
                        reason=exec_result.error_code or "execution_failed",
                        replan_attempt=replan_count,
                    )
                    new_plan = llm_planner.replan(
                        instruction, world_state, failed_ps, remaining, failure,
                    )
                    llm_planner.update_causal_outputs({
                        obj_id: co.to_dict() for obj_id, co in causal_outputs.items()
                    })
                    contract_plan = ContractPlannerAdapter(llm_planner).replan_contract(
                        task_id="demo_task",
                        instruction=instruction,
                        state=world_state,
                        failed_step=failed_ps,
                        remaining_plan=remaining,
                        failure=failure,
                    )
                    contract_plan = bridge._inject_probe_if_needed(
                        task_id="demo_task",
                        contract_plan=contract_plan,
                        causal_outputs=causal_outputs,
                    )
                    result.plan_steps = [
                        {"step_index": i, "selected_skill": s.selected_skill,
                         "target_object": s.target_object, "skill_args": dict(s.skill_args)}
                        for i, s in enumerate(contract_plan)
                    ]

            except Exception as exc:
                result.error = str(exc)
                break

        result.success = result.error is None
        result.failure_history = failure_history
        result.replan_count = replan_count
        return result

    def _prepare_causal_outputs(
        self,
        instruction: str,
        world_state: WorldState,
    ) -> dict[str, CausalExploreOutput]:
        """Prepare CausalExplore outputs for objects in the scene."""
        outputs: dict[str, CausalExploreOutput] = {}
        for obj_id in world_state.object_positions:
            is_main_target = obj_id.replace("_block", "") in instruction.lower()
            uncertainty = 0.25 if is_main_target else 0.30
            outputs[obj_id] = make_mock_causal_output(
                object_id=obj_id,
                uncertainty=uncertainty,
            )
        return outputs

    @staticmethod
    def _simulate_execute(step: PlannerStep) -> ExecutorResult:
        """Simulate skill execution (mock for dry-run/demo)."""
        return ExecutorResult(
            success=True,
            reward=1.0,
            final_state={"executed_skill": step.selected_skill},
            contact_state=type("ContactState", (), {
                "has_contact": step.selected_skill != "observe",
                "contact_region": "top",
                "to_dict": lambda: {"has_contact": True, "contact_region": "top"},
            })(),
            error_code=None,
            rollout_path=f"mock://demo/{step.selected_skill}",
            failure_history=[],
        )

    @staticmethod
    def _plan_step_to_dict(step: PlanStep) -> dict[str, Any]:
        return {
            "action": step.action,
            "target": step.target,
            "parameters": dict(step.parameters),
            "post_condition": step.post_condition.to_dict() if step.post_condition else None,
        }


def run_multi_block_demo(config: DemoConfig) -> list[DemoResult]:
    """Run multi-block placement demo: red→green, blue→yellow."""
    instructions = [
        "把红色方块移到绿色区域",
        "把蓝色方块移到黄色区域",
    ]
    demo = ClosedLoopDemo(config)
    results = []
    for instruction in instructions:
        result = demo.run(instruction)
        results.append(result)
        status = "PASS" if result.success else "FAIL"
        print(f"[{status}] {instruction}")
        if result.error:
            print(f"  Error: {result.error}")
        print(f"  Steps: {result.executed_steps}")
        print(f"  Replans: {result.replan_count}")
    return results


def run_interactive_demo(config: DemoConfig) -> DemoResult:
    """Run interactive object inference demo: mouse press + drag."""
    instruction = "探索鼠标的功能：先按压，然后向前拖动"
    demo = ClosedLoopDemo(config)
    result = demo.run(instruction)
    status = "PASS" if result.success else "FAIL"
    print(f"[{status}] {instruction}")
    if result.error:
        print(f"  Error: {result.error}")
    print(f"  Steps: {result.executed_steps}")
    print(f"  Replans: {result.replan_count}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-Loop LLM Planner Demo")
    parser.add_argument(
        "--task", type=str, default=None,
        help="Natural language task instruction",
    )
    parser.add_argument(
        "--scenario", type=str, choices=["multi_block", "interactive"],
        default=None,
        help="Built-in demo scenario",
    )
    parser.add_argument("--headless", action="store_true", default=True,
                       help="Run headless (no GUI)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run without simulation execution")
    parser.add_argument("--language", type=str, default="zh",
                       choices=["zh", "en"],
                       help="Prompt language")
    args = parser.parse_args()

    config = DemoConfig(
        headless=args.headless,
        dry_run=args.dry_run,
        language=args.language,
    )

    print("=" * 60)
    print("Phase 2 Closed-Loop LLM Planner Demo")
    print("=" * 60)

    if args.task:
        demo = ClosedLoopDemo(config)
        result = demo.run(args.task)
        status = "PASS" if result.success else "FAIL"
        print(f"\n[{status}] Task: {args.task}")
        if result.error:
            print(f"  Error: {result.error}")
        print(f"  Plan steps: {len(result.plan_steps)}")
        print(f"  Executed: {result.executed_steps}")
        print(f"  Replans: {result.replan_count}")
        print(f"  Failures: {len(result.failure_history)}")
    elif args.scenario == "multi_block":
        print("\n--- Multi-Block Placement Demo ---")
        run_multi_block_demo(config)
    elif args.scenario == "interactive":
        print("\n--- Interactive Object Inference Demo ---")
        run_interactive_demo(config)
    else:
        print("\nRunning all scenarios...")
        print("\n--- Multi-Block Placement ---")
        run_multi_block_demo(config)
        print("\n--- Interactive Object Inference ---")
        run_interactive_demo(config)

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
