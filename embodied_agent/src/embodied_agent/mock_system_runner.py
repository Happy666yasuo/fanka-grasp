from __future__ import annotations

import json
from pathlib import Path

from embodied_agent.contracts import ContactState, ExecutorResult, FailureRecord, PlannerStep
from embodied_agent.mock_planning_runner import DEFAULT_REGISTRY_PATH, build_default_world_state
from embodied_agent.planner import RuleBasedPlanner
from embodied_agent.planning_bridge import (
    ArtifactRegistryCausalOutputProvider,
    ContractPlanningBridge,
    SimulatorArtifactCatalogCausalOutputProvider,
)
from embodied_agent.types import PlanStep, PostCondition, StepFailure, WorldState


DEFAULT_ROLLOUT_PATH = "mock://system_demo/episode_0001"

SYSTEM_SCENARIOS: dict[str, dict[str, str | None]] = {
    "blue_success": {
        "instruction": "put the blue block in the yellow zone",
        "fail_on_skill": None,
    },
    "red_place_failure": {
        "instruction": "put the red block in the green zone",
        "fail_on_skill": "place",
    },
}


def run_mock_system_demo(
    instruction: str,
    *,
    registry_path: str | Path | None = None,
    catalog_path: str | Path | None = None,
    task_id: str = "mock_task",
    fail_on_skill: str | None = None,
    scenario: str | None = None,
) -> dict[str, object]:
    provider, resolved_registry_path, resolved_catalog_path = _build_provider(
        registry_path=registry_path,
        catalog_path=catalog_path,
    )
    bridge = ContractPlanningBridge(
        RuleBasedPlanner(),
        causal_output_provider=provider,
    )
    state = build_default_world_state(instruction)
    plan = bridge.plan_contract(
        task_id=task_id,
        instruction=instruction,
        state=state,
    )
    causal_outputs = provider.get_outputs(_collect_target_object_ids(plan))
    executor_result, latest_state, failed_step = _execute_mock_plan(
        instruction=instruction,
        plan=plan,
        state=state,
        fail_on_skill=fail_on_skill,
    )

    payload: dict[str, object] = {
        "task_id": task_id,
        "scenario": scenario,
        "instruction": instruction,
        "registry_path": str(resolved_registry_path) if resolved_registry_path is not None else None,
        "catalog_path": str(resolved_catalog_path) if resolved_catalog_path is not None else None,
        "causal_outputs": {
            object_id: causal_output.to_dict()
            for object_id, causal_output in causal_outputs.items()
        },
        "planner_steps": [step.to_dict() for step in plan],
        "executor_result": executor_result.to_dict(),
    }

    if failed_step is not None:
        failure = StepFailure(
            failed_step=failed_step,
            source="post_condition_failed",
            reason="released_outside_zone",
            replan_attempt=1,
            recovery_policy="retry_place_only",
        )
        replanned_steps = bridge.replan_contract(
            task_id=task_id,
            instruction=instruction,
            state=latest_state,
            failed_step=failed_step,
            remaining_plan=[],
            failure=failure,
        )
        payload["replanned_steps"] = [step.to_dict() for step in replanned_steps]

    return payload


def render_mock_system_demo(
    instruction: str,
    *,
    registry_path: str | Path | None = None,
    catalog_path: str | Path | None = None,
    task_id: str = "mock_task",
    fail_on_skill: str | None = None,
    scenario: str | None = None,
) -> str:
    return json.dumps(
        run_mock_system_demo(
            instruction,
            registry_path=registry_path,
            catalog_path=catalog_path,
            task_id=task_id,
            fail_on_skill=fail_on_skill,
            scenario=scenario,
        ),
        indent=2,
        ensure_ascii=False,
    )


def _build_provider(
    *,
    registry_path: str | Path | None,
    catalog_path: str | Path | None,
) -> tuple[object, Path | None, Path | None]:
    if catalog_path is not None:
        resolved_catalog_path = Path(catalog_path).resolve()
        return (
            SimulatorArtifactCatalogCausalOutputProvider(resolved_catalog_path),
            None,
            resolved_catalog_path,
        )

    resolved_registry_path = Path(registry_path or DEFAULT_REGISTRY_PATH).resolve()
    return (
        ArtifactRegistryCausalOutputProvider(resolved_registry_path),
        resolved_registry_path,
        None,
    )


def _execute_mock_plan(
    *,
    instruction: str,
    plan: list[PlannerStep],
    state: WorldState,
    fail_on_skill: str | None,
) -> tuple[ExecutorResult, WorldState, PlanStep | None]:
    latest_state = state
    held_object_name = state.held_object_name

    for step in plan:
        if step.selected_skill == "pick":
            held_object_name = step.target_object
            latest_state = _copy_world_state(state, held_object_name=held_object_name)
            continue

        if step.selected_skill != "place":
            continue

        target_zone = str(step.skill_args.get("target_zone", ""))
        if step.selected_skill == fail_on_skill:
            latest_state = _copy_world_state(latest_state, held_object_name=step.target_object)
            failure_record = FailureRecord(
                step_index=step.step_index,
                selected_skill=step.selected_skill,
                failure_source="post_condition_failed",
                reason="released_outside_zone",
                replan_attempt=1,
                selected_recovery_policy="retry_place_only",
            )
            return (
                ExecutorResult(
                    success=False,
                    reward=0.0,
                    final_state={
                        "holding": True,
                        "object_pose": list(latest_state.object_positions[str(step.target_object)]),
                    },
                    contact_state=ContactState(
                        has_contact=True,
                        contact_region=str(step.skill_args.get("contact_region", "top_center")),
                    ),
                    error_code="released_outside_zone",
                    rollout_path=DEFAULT_ROLLOUT_PATH,
                    failure_history=[failure_record],
                ),
                latest_state,
                _planner_step_to_runtime_plan_step(step, target_zone=target_zone),
            )

        latest_state = _place_object_in_zone(
            latest_state,
            object_name=str(step.target_object),
            zone_name=target_zone,
        )
        held_object_name = None

    final_target = _find_last_place_target(plan)
    if final_target is None:
        final_state = {"holding": held_object_name is not None}
    else:
        final_state = {
            "holding": False,
            "object_pose": list(latest_state.object_positions[final_target[0]]),
            "target_zone": final_target[1],
        }

    return (
        ExecutorResult(
            success=True,
            reward=1.0,
            final_state=final_state,
            contact_state=ContactState(has_contact=True, contact_region="top_center"),
            error_code=None,
            rollout_path=DEFAULT_ROLLOUT_PATH,
            failure_history=[],
        ),
        latest_state,
        None,
    )


def _collect_target_object_ids(plan: list[PlannerStep]) -> list[str]:
    object_ids: list[str] = []
    seen: set[str] = set()
    for step in plan:
        if step.target_object is None or step.target_object in seen:
            continue
        seen.add(step.target_object)
        object_ids.append(step.target_object)
    return object_ids


def _copy_world_state(state: WorldState, *, held_object_name: str | None) -> WorldState:
    return WorldState(
        instruction=state.instruction,
        object_positions=dict(state.object_positions),
        zone_positions=dict(state.zone_positions),
        end_effector_position=state.end_effector_position,
        held_object_name=held_object_name,
    )


def _place_object_in_zone(state: WorldState, *, object_name: str, zone_name: str) -> WorldState:
    target_pose = state.zone_positions[zone_name]
    updated_positions = dict(state.object_positions)
    updated_positions[object_name] = target_pose
    return WorldState(
        instruction=state.instruction,
        object_positions=updated_positions,
        zone_positions=dict(state.zone_positions),
        end_effector_position=state.end_effector_position,
        held_object_name=None,
    )


def _planner_step_to_runtime_plan_step(step: PlannerStep, *, target_zone: str) -> PlanStep:
    if step.selected_skill == "place":
        return PlanStep(
            action="place",
            target=target_zone,
            parameters={"object": step.target_object},
            post_condition=PostCondition(
                kind="placed",
                object_name=step.target_object,
                zone_name=target_zone,
            ),
        )
    return PlanStep(
        action=step.selected_skill,
        target=step.target_object,
        parameters=dict(step.skill_args),
        post_condition=None,
    )


def _find_last_place_target(plan: list[PlannerStep]) -> tuple[str, str] | None:
    for step in reversed(plan):
        if step.selected_skill != "place" or step.target_object is None:
            continue
        target_zone = step.skill_args.get("target_zone")
        if isinstance(target_zone, str):
            return step.target_object, target_zone
    return None
