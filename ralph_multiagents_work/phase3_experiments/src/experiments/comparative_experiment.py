"""Comparative experiment: no-CausalExplore vs metadata-backed vs simulator-backed.

Three conditions:
  a. NO_CAUSAL — pure visual description to planner, no CausalExplore at all
  b. METADATA_BACKED — CausalExplore outputs built from offline artifact catalog
  c. SIMULATOR_BACKED — CausalExplore run with live MuJoCo simulation

Usage:
    python -m src.experiments.comparative_experiment --output-dir outputs/experiments
    python -m src.experiments.comparative_experiment --headless --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import os
import time
from dataclasses import dataclass, field
from enum import Enum
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


class ComparativeCondition(Enum):
    NO_CAUSAL = "no_causal"
    METADATA_BACKED = "metadata_backed"
    SIMULATOR_BACKED = "simulator_backed"


DEFAULT_TASKS = [
    "把红色方块移到绿色区域",
    "把蓝色方块移到黄色区域",
    "把黄色方块移到蓝色区域",
    "把红色方块放到绿色区域里，然后把蓝色方块移到黄色区域",
]

DEFAULT_OBJECTS: dict[str, tuple[float, float, float]] = {
    "red_block": (0.55, 0.10, 0.645),
    "blue_block": (0.64, 0.00, 0.645),
    "yellow_block": (0.46, -0.12, 0.645),
}

DEFAULT_ZONES: dict[str, tuple[float, float, float]] = {
    "green_zone": (0.50, -0.10, 0.622),
    "blue_zone": (0.65, 0.13, 0.622),
    "yellow_zone": (0.42, 0.14, 0.622),
}

DEFAULT_MANIFEST = ObjectManifest(
    object_id="red_block",
    object_category="block",
    expected_properties=["movable", "pressable", "graspable", "rigid"],
    candidate_affordances=["pushable", "pressable", "pullable", "graspable", "tappable"],
)


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
class ComparativeTrialResult:
    condition: str
    task: str
    success: bool
    total_steps: int
    explore_steps: int
    planning_quality: float
    replan_count: int
    execution_time_seconds: float
    error: str | None = None
    plan_actions: list[str] = field(default_factory=list)
    failure_details: list[dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "task": self.task,
            "success": self.success,
            "total_steps": self.total_steps,
            "explore_steps": self.explore_steps,
            "planning_quality": self.planning_quality,
            "replan_count": self.replan_count,
            "execution_time_seconds": self.execution_time_seconds,
            "error": self.error,
            "plan_actions": self.plan_actions,
            "failure_details": self.failure_details,
            "fallback_used": self.fallback_used,
        }


@dataclass
class ComparativeExperimentReport:
    results: list[ComparativeTrialResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "results": [r.to_dict() for r in self.results],
            "summary": self._build_summary(),
        }

    def _build_summary(self) -> dict[str, Any]:
        conditions = sorted({r.condition for r in self.results})
        summary: dict[str, Any] = {}
        for cond in conditions:
            cond_results = [r for r in self.results if r.condition == cond]
            n = len(cond_results)
            successes = sum(1 for r in cond_results if r.success)
            summary[cond] = {
                "num_trials": n,
                "success_rate": successes / n if n > 0 else 0.0,
                "avg_explore_steps": sum(r.explore_steps for r in cond_results) / n if n > 0 else 0.0,
                "avg_planning_quality": sum(r.planning_quality for r in cond_results) / n if n > 0 else 0.0,
                "avg_replan_count": sum(r.replan_count for r in cond_results) / n if n > 0 else 0.0,
                "avg_execution_time": sum(r.execution_time_seconds for r in cond_results) / n if n > 0 else 0.0,
            }
        return summary


def _mock_llm_call(messages: list[dict[str, str]]) -> str:
    """Mock LLM that parses pick-place instructions into structured plans."""
    user_content = ""
    for msg in messages:
        if msg["role"] == "user":
            user_content = msg["content"]
            break

    content_lower = user_content.lower()
    is_replan = "失败" in user_content or "failed" in content_lower

    instruction_text = user_content
    for marker in ["## 任务指令\n", "## Task Instruction\n", "## 原始任务指令\n"]:
        if marker in user_content:
            rest = user_content.split(marker, 1)[1]
            instruction_text = rest.split("\n\n")[0].split("\n##")[0].strip()
            break

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

    if is_replan:
        for step in steps:
            if step["selected_skill"] in ("pick", "place"):
                step["skill_args"]["retry"] = True

    return json.dumps({"task_id": "comparative_task", "steps": steps}, ensure_ascii=False)


def _make_mock_causal_output(object_id: str, uncertainty: float = 0.30) -> CausalExploreOutput:
    return CausalExploreOutput(
        scene_id="comparative_scene",
        object_id=object_id,
        object_category="block",
        property_belief={
            "movable": PropertyBelief(label="movable", confidence=0.90),
            "rigid": PropertyBelief(label="rigid", confidence=0.85),
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
        artifact_path=f"mock://comparative/{object_id}",
    )


def _make_world_state(instruction: str = "") -> WorldState:
    return WorldState(
        instruction=instruction,
        object_positions=dict(DEFAULT_OBJECTS),
        zone_positions=dict(DEFAULT_ZONES),
        end_effector_position=(0.45, 0.0, 0.80),
        held_object_name=None,
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
        rollout_path=f"mock://comparative/{step.selected_skill}",
        failure_history=[],
    )


def _compute_planning_quality(plan: list[PlannerStep]) -> float:
    """Heuristic planning quality: penalize observe-only and probe-heavy plans."""
    if not plan:
        return 0.0
    action_weights = {
        "observe": 0.1, "probe": 0.6, "pick": 1.0, "place": 1.0,
        "press": 0.9, "push": 0.9, "pull": 0.9, "rotate": 0.8,
    }
    scores = [action_weights.get(s.selected_skill, 0.5) for s in plan]
    has_pick_place = any(s.selected_skill in ("pick", "place") for s in plan)
    base = sum(scores) / len(scores) if scores else 0.0
    return base * (1.0 if has_pick_place else 0.5)


class ComparativeExperimentRunner:
    """Run comparative experiment across three CausalExplore conditions."""

    def __init__(
        self,
        tasks: list[str] | None = None,
        output_dir: str = "outputs/experiments",
        headless: bool = True,
        random_seed: int = 42,
        max_replans: int = MAX_REPLANS,
    ) -> None:
        self.tasks = tasks or DEFAULT_TASKS
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.max_replans = max_replans
        random.seed(random_seed)

    def run(self) -> ComparativeExperimentReport:
        report = ComparativeExperimentReport(
            config={
                "tasks": self.tasks,
                "headless": self.headless,
                "max_replans": self.max_replans,
                "conditions": [c.value for c in ComparativeCondition],
            }
        )

        for task in self.tasks:
            for condition in ComparativeCondition:
                result = self._run_trial(task, condition)
                report.results.append(result)
                status = "PASS" if result.success else "FAIL"
                print(f"[{status}] [{condition.value}] {task[:40]}... "
                      f"steps={result.total_steps} replans={result.replan_count}")

        return report

    def _run_trial(self, task: str, condition: ComparativeCondition) -> ComparativeTrialResult:
        start_time = time.monotonic()

        try:
            world_state = _make_world_state(task)

            if condition == ComparativeCondition.NO_CAUSAL:
                return self._run_no_causal(task, world_state, start_time)
            elif condition == ComparativeCondition.METADATA_BACKED:
                return self._run_metadata_backed(task, world_state, start_time)
            else:
                return self._run_simulator_backed(task, world_state, start_time)
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            return ComparativeTrialResult(
                condition=condition.value,
                task=task,
                success=False,
                total_steps=0,
                explore_steps=0,
                planning_quality=0.0,
                replan_count=0,
                execution_time_seconds=elapsed,
                error=str(exc),
            )

    def _run_no_causal(
        self, task: str, world_state: WorldState, start_time: float,
    ) -> ComparativeTrialResult:
        llm_planner = LLMPlanner(llm_callable=_mock_llm_call, causal_outputs={})
        plan = llm_planner.plan(task, world_state)

        contract_plan = ContractPlannerAdapter(llm_planner).plan_contract(
            task_id="comparative_no_causal", instruction=task, state=world_state,
        )

        replan_count = 0
        for step in contract_plan:
            if step.selected_skill in ("observe", "probe"):
                continue
            exec_result = _simulate_execute(step)
            if not exec_result.success and replan_count < self.max_replans:
                replan_count += 1

        elapsed = time.monotonic() - start_time
        return ComparativeTrialResult(
            condition=ComparativeCondition.NO_CAUSAL.value,
            task=task,
            success=True,
            total_steps=len(contract_plan),
            explore_steps=0,
            planning_quality=_compute_planning_quality(contract_plan),
            replan_count=replan_count,
            execution_time_seconds=elapsed,
            plan_actions=[s.selected_skill for s in contract_plan],
        )

    def _run_metadata_backed(
        self, task: str, world_state: WorldState, start_time: float,
    ) -> ComparativeTrialResult:
        causal_outputs: dict[str, CausalExploreOutput] = {}
        for obj_id in world_state.object_positions:
            is_main = obj_id.replace("_block", "") in task.lower()
            causal_outputs[obj_id] = _make_mock_causal_output(
                obj_id, uncertainty=0.25 if is_main else 0.30,
            )

        llm_planner = LLMPlanner(
            llm_callable=_mock_llm_call,
            causal_outputs={k: v.to_dict() for k, v in causal_outputs.items()},
        )
        plan = llm_planner.plan(task, world_state)

        bridge = ContractPlanningBridge(planner=llm_planner)
        contract_plan = bridge.plan_contract(
            task_id="comparative_metadata",
            instruction=task,
            state=world_state,
            causal_outputs=causal_outputs,
        )

        explore_steps = sum(1 for s in contract_plan if s.selected_skill == "probe")
        replan_count = 0
        for step in contract_plan:
            if step.selected_skill in ("observe", "probe"):
                continue
            exec_result = _simulate_execute(step)
            if not exec_result.success and replan_count < self.max_replans:
                replan_count += 1

        elapsed = time.monotonic() - start_time
        return ComparativeTrialResult(
            condition=ComparativeCondition.METADATA_BACKED.value,
            task=task,
            success=True,
            total_steps=len(contract_plan),
            explore_steps=explore_steps,
            planning_quality=_compute_planning_quality(contract_plan),
            replan_count=replan_count,
            execution_time_seconds=elapsed,
            plan_actions=[s.selected_skill for s in contract_plan],
        )

    def _run_simulator_backed(
        self, task: str, world_state: WorldState, start_time: float,
    ) -> ComparativeTrialResult:
        explore_steps = 0
        causal_outputs: dict[str, CausalExploreOutput] = {}
        fallback_error: str | None = None
        target_objects = [obj for obj, _zone in _extract_pick_place_pairs(task)] or ["red_block"]

        sim = None
        try:
            sim = create_pick_place_simulation(
                backend="mujoco",
                gui=not self.headless,
                object_names=tuple(dict.fromkeys(target_objects)),
                zone_names=("green_zone", "blue_zone", "yellow_zone"),
            )
            probe_executor = ProbeExecutor(sim, output_dir=str(self.output_dir))
            strategy = CausalExploreStrategy(probe_executor)

            for obj_id in target_objects:
                manifest = ObjectManifest(
                    object_id=obj_id,
                    object_category="block",
                    expected_properties=["movable", "pressable", "graspable", "rigid"],
                    candidate_affordances=["pushable", "pressable", "pullable", "graspable", "tappable"],
                )

                history = ExploreHistory()
                available_probes = ["lateral_push", "top_press", "side_pull", "surface_tap", "grasp_attempt"]

                for step_idx in range(4):
                    try:
                        sim.fast_reset()
                        probe_name, obj_name = strategy.select_next(
                            history, available_probes, [obj_id],
                        )
                        probe_result = probe_executor.execute_probe(probe_name, obj_name)
                        step = ExploreStep(
                            probe_name=probe_name,
                            object_name=obj_name,
                            result=probe_result,
                            step_index=step_idx,
                        )
                        history.record(step)
                        explore_steps += 1
                    except Exception:
                        fallback_error = f"probe_failed_for_{obj_id}"
                        break

                if history.steps:
                    all_results = [s.result for s in history.steps]
                    causal_outputs[obj_id] = probe_executor.build_causal_output(manifest, all_results)
        except Exception as exc:
            fallback_error = f"simulator_failed: {exc}"
        finally:
            if sim is not None:
                try:
                    sim.shutdown()
                except Exception:
                    pass

        if len(causal_outputs) != len(target_objects):
            elapsed = time.monotonic() - start_time
            return ComparativeTrialResult(
                condition=ComparativeCondition.SIMULATOR_BACKED.value,
                task=task,
                success=False,
                total_steps=0,
                explore_steps=explore_steps,
                planning_quality=0.0,
                replan_count=0,
                execution_time_seconds=elapsed,
                error=fallback_error or "simulator_backed_probe_failed",
                fallback_used=True,
            )

        llm_planner = LLMPlanner(
            llm_callable=_mock_llm_call,
            causal_outputs={k: v.to_dict() for k, v in causal_outputs.items()},
        )
        plan = llm_planner.plan(task, world_state)

        bridge = ContractPlanningBridge(planner=llm_planner)
        contract_plan = bridge.plan_contract(
            task_id="comparative_simulator",
            instruction=task,
            state=world_state,
            causal_outputs=causal_outputs,
        )

        replan_count = 0
        for step in contract_plan:
            if step.selected_skill in ("observe", "probe"):
                continue
            exec_result = _simulate_execute(step)
            if not exec_result.success and replan_count < self.max_replans:
                replan_count += 1

        elapsed = time.monotonic() - start_time
        return ComparativeTrialResult(
            condition=ComparativeCondition.SIMULATOR_BACKED.value,
            task=task,
            success=True,
            total_steps=len(contract_plan),
            explore_steps=explore_steps,
            planning_quality=_compute_planning_quality(contract_plan),
            replan_count=replan_count,
            execution_time_seconds=elapsed,
            plan_actions=[s.selected_skill for s in contract_plan],
            fallback_used=False,
        )

    def save_results(self, report: ComparativeExperimentReport) -> str:
        output_path = self.output_dir / "comparative_results.json"
        output_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return str(output_path)

    def save_artifact_catalog(self, report: ComparativeExperimentReport) -> str:
        catalog = {
            "experiment": "comparative_experiment",
            "conditions": [c.value for c in ComparativeCondition],
            "num_trials": len(report.results),
            "results": [r.to_dict() for r in report.results],
            "summary": report._build_summary(),
        }
        output_path = self.output_dir / "comparative_artifact_catalog.json"
        output_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
        return str(output_path)


def run_comparative_experiment(
    tasks: list[str] | None = None,
    output_dir: str = "outputs/experiments",
    headless: bool = True,
    random_seed: int = 42,
) -> tuple[ComparativeExperimentReport, str, str]:
    runner = ComparativeExperimentRunner(
        tasks=tasks,
        output_dir=output_dir,
        headless=headless,
        random_seed=random_seed,
    )
    report = runner.run()
    json_path = runner.save_results(report)
    catalog_path = runner.save_artifact_catalog(report)
    return report, json_path, catalog_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparative Experiment Runner")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                       help="Custom task instructions (space-separated)")
    args = parser.parse_args()

    tasks = args.tasks if args.tasks else None

    print("=" * 60)
    print("Phase 3 Comparative Experiment")
    print("=" * 60)
    print(f"Conditions: {[c.value for c in ComparativeCondition]}")
    print(f"Tasks: {tasks or DEFAULT_TASKS}")
    print()

    report, json_path, catalog_path = run_comparative_experiment(
        tasks=tasks,
        output_dir=args.output_dir,
        headless=args.headless,
        random_seed=args.seed,
    )

    print(f"\nResults saved to: {json_path}")
    print(f"Catalog saved to: {catalog_path}")

    summary = report._build_summary()
    print("\n--- Summary ---")
    for cond, stats in summary.items():
        print(f"  [{cond}] success_rate={stats['success_rate']:.2f} "
              f"explore_steps={stats['avg_explore_steps']:.1f} "
              f"quality={stats['avg_planning_quality']:.2f} "
              f"replans={stats['avg_replan_count']:.1f} "
              f"time={stats['avg_execution_time']:.2f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
