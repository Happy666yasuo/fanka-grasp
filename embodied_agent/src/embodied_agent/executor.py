from __future__ import annotations

import math
from dataclasses import replace

from embodied_agent.planner import Planner
from embodied_agent.skills import SkillLibrary
from embodied_agent.types import ExecutionResult, PlanStep, StepFailure, WorldState


class TaskExecutor:
    def __init__(self, planner: Planner, skill_library: SkillLibrary, max_replans: int = 1) -> None:
        self.planner = planner
        self.skill_library = skill_library
        self.max_replans = max(0, max_replans)

    def run(self, instruction: str) -> ExecutionResult:
        plan: list[PlanStep] = []
        executed_actions: list[str] = []
        failure_history: list[StepFailure] = []
        replan_count = 0
        latest_state = self.skill_library.observe_scene(instruction)
        object_name = self._maybe_default_object_name(latest_state)
        zone_name = self._maybe_default_zone_name(latest_state)

        try:
            plan = self.planner.plan(instruction, latest_state)
            object_name = self._extract_object_name(plan, latest_state)
            zone_name = self._extract_zone_name(plan, latest_state)

            while True:
                replanned = False
                for step_index, step in enumerate(plan):
                    try:
                        executed_actions.append(self.skill_library.execute(step))
                    except Exception as exc:
                        latest_state = self.skill_library.observe_scene(instruction)
                        failure = StepFailure(
                            failed_step=step,
                            source="execution_error",
                            reason=str(exc),
                            replan_attempt=replan_count + 1,
                        )
                        next_plan, recovery_policy = self._attempt_replan(
                            instruction=instruction,
                            state=latest_state,
                            failed_step=step,
                            remaining_plan=plan[step_index + 1 :],
                            failure=failure,
                            replan_count=replan_count,
                        )
                        failure_history.append(replace(failure, recovery_policy=recovery_policy))
                        if next_plan is None:
                            raise RuntimeError(str(exc)) from exc
                        plan = next_plan
                        object_name = self._extract_object_name(plan, latest_state)
                        zone_name = self._extract_zone_name(plan, latest_state)
                        replan_count += 1
                        replanned = True
                        break

                    latest_state = self.skill_library.observe_scene(instruction)
                    if not self._check_post_condition(step):
                        reason = self._format_post_condition_failure(step)
                        failure = StepFailure(
                            failed_step=step,
                            source="post_condition_failed",
                            reason=reason,
                            replan_attempt=replan_count + 1,
                        )
                        next_plan, recovery_policy = self._attempt_replan(
                            instruction=instruction,
                            state=latest_state,
                            failed_step=step,
                            remaining_plan=plan[step_index + 1 :],
                            failure=failure,
                            replan_count=replan_count,
                        )
                        failure_history.append(replace(failure, recovery_policy=recovery_policy))
                        if next_plan is None:
                            raise RuntimeError(reason)
                        plan = next_plan
                        object_name = self._extract_object_name(plan, latest_state)
                        zone_name = self._extract_zone_name(plan, latest_state)
                        replan_count += 1
                        replanned = True
                        break

                if replanned:
                    continue

                latest_state = self.skill_library.observe_scene(instruction)
                break

            success = self.skill_library.verify(object_name, zone_name)
            goal_distance_xy = self._distance_xy(
                latest_state.object_positions[object_name],
                latest_state.zone_positions[zone_name],
            )
            metrics = {
                "plan_length": float(len(plan)),
                "goal_distance_xy": goal_distance_xy,
                "replan_count": float(replan_count),
                "failure_count": float(len(failure_history)),
                "used_recovery": 1.0 if replan_count > 0 else 0.0,
                "recovered_success": 1.0 if replan_count > 0 and success else 0.0,
                "recovery_failed": 1.0 if replan_count > 0 and not success else 0.0,
                "success": 1.0 if success else 0.0,
            }
            return ExecutionResult(
                success=success,
                plan=plan,
                executed_actions=executed_actions,
                final_object_positions=latest_state.object_positions,
                metrics=metrics,
                error=None,
                replan_count=replan_count,
                failure_history=failure_history,
            )
        except Exception as exc:
            latest_state = self.skill_library.observe_scene(instruction)
            metrics = {
                "plan_length": float(len(plan)),
                "goal_distance_xy": self._safe_goal_distance(latest_state, object_name, zone_name),
                "replan_count": float(replan_count),
                "failure_count": float(len(failure_history)),
                "used_recovery": 1.0 if replan_count > 0 else 0.0,
                "recovered_success": 0.0,
                "recovery_failed": 1.0 if replan_count > 0 else 0.0,
                "success": 0.0,
            }
            return ExecutionResult(
                success=False,
                plan=plan,
                executed_actions=executed_actions,
                final_object_positions=latest_state.object_positions,
                metrics=metrics,
                error=str(exc),
                replan_count=replan_count,
                failure_history=failure_history,
            )

    def _check_post_condition(self, step: PlanStep) -> bool:
        if step.post_condition is None:
            return True

        check_post_condition = getattr(self.skill_library, "check_post_condition", None)
        if check_post_condition is None:
            raise RuntimeError("The skill library does not support step-level post-condition checks.")
        return bool(check_post_condition(step.post_condition))

    def _format_post_condition_failure(self, step: PlanStep) -> str:
        if step.post_condition is None:
            return f"Step '{step.action}' failed verification."
        return (
            f"Step '{step.action}' failed post-condition "
            f"'{step.post_condition.kind}'."
        )

    def _attempt_replan(
        self,
        instruction: str,
        state: WorldState,
        failed_step: PlanStep,
        remaining_plan: list[PlanStep],
        failure: StepFailure,
        replan_count: int,
    ) -> tuple[list[PlanStep] | None, str | None]:
        if replan_count >= self.max_replans:
            return None, None

        replan = getattr(self.planner, "replan", None)
        if replan is None:
            return None, None

        next_plan = replan(
            instruction,
            state,
            failed_step,
            remaining_plan,
            failure,
        )
        if not next_plan:
            return None, None
        return next_plan, self._classify_recovery_policy(failed_step, next_plan)

    def _classify_recovery_policy(self, failed_step: PlanStep, next_plan: list[PlanStep]) -> str:
        action_sequence = [step.action for step in next_plan]

        if failed_step.action == "pick":
            return "retry_pick_then_continue"

        if failed_step.action == "place":
            if action_sequence == ["observe"]:
                return "observe_only"
            if action_sequence == ["observe", "place"]:
                return "retry_place_only"
            if action_sequence == ["observe", "pick", "place"]:
                return "repick_then_place"
            return "place_recovery_other"

        return "generic_replan"

    def _extract_object_name(self, plan: list[PlanStep], state: WorldState) -> str:
        for step in plan:
            if step.action == "pick" and step.target is not None:
                return step.target
            if step.action == "place" and "object" in step.parameters:
                object_name = step.parameters["object"]
                if isinstance(object_name, str):
                    return object_name
        return self._default_object_name(state)

    def _extract_zone_name(self, plan: list[PlanStep], state: WorldState) -> str:
        for step in plan:
            if step.action == "place" and step.target is not None:
                return step.target
        return self._default_zone_name(state)

    def _maybe_default_object_name(self, state: WorldState) -> str | None:
        if len(state.object_positions) == 1:
            return next(iter(state.object_positions.keys()))
        return None

    def _maybe_default_zone_name(self, state: WorldState) -> str | None:
        if len(state.zone_positions) == 1:
            return next(iter(state.zone_positions.keys()))
        return None

    def _default_object_name(self, state: WorldState) -> str:
        if len(state.object_positions) != 1:
            raise ValueError("The executor could not infer a unique object target from the current state.")
        return next(iter(state.object_positions.keys()))

    def _default_zone_name(self, state: WorldState) -> str:
        if len(state.zone_positions) != 1:
            raise ValueError("The executor could not infer a unique zone target from the current state.")
        return next(iter(state.zone_positions.keys()))

    def _safe_goal_distance(self, state: WorldState, object_name: str | None, zone_name: str | None) -> float:
        if object_name is None or zone_name is None:
            return float("nan")
        if object_name not in state.object_positions or zone_name not in state.zone_positions:
            return float("nan")
        return self._distance_xy(state.object_positions[object_name], state.zone_positions[zone_name])

    def _distance_xy(self, first_position: tuple[float, float, float], second_position: tuple[float, float, float]) -> float:
        return math.dist(first_position[:2], second_position[:2])