from __future__ import annotations

import sys
import os
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../embodied_agent/src')))
from embodied_agent.contracts import PlannerStep, ExecutorResult, ContactState, FailureRecord
from embodied_agent.mujoco_simulator import MujocoPickPlaceSimulation

from .skill_registry import SkillRegistry


class UnifiedSkillExecutor:
    """Unified executor that maps PlannerStep to skill execution."""

    def __init__(
        self,
        simulation: MujocoPickPlaceSimulation,
        registry: SkillRegistry | None = None,
        headless: bool = True,
    ) -> None:
        self.simulation = simulation
        self.registry = registry or SkillRegistry()
        self.headless = headless

    def execute(self, step: PlannerStep) -> ExecutorResult:
        skill_name = step.selected_skill

        if skill_name in ("pick", "place"):
            return self._execute_builtin(step)

        skill = self.registry.get_skill(skill_name)
        kwargs: dict[str, Any] = {"target_object": step.target_object}
        kwargs.update(step.skill_args)
        return skill.execute(self.simulation, **kwargs)

    def _execute_builtin(self, step: PlannerStep) -> ExecutorResult:
        skill_name = step.selected_skill
        target = step.target_object or "red_block"
        zone = step.skill_args.get("target_zone", "green_zone")

        try:
            if skill_name == "pick":
                self.simulation.pick_object(target)
                return ExecutorResult(
                    success=True,
                    reward=1.0,
                    final_state={
                        "executed_skill": "pick",
                        "target_object": target,
                    },
                    contact_state=ContactState(
                        has_contact=True,
                        contact_region="top",
                    ),
                    error_code=None,
                    rollout_path="scripted://pick",
                    failure_history=[],
                )
            elif skill_name == "place":
                self.simulation.place_object(zone)
                return ExecutorResult(
                    success=True,
                    reward=1.0,
                    final_state={
                        "executed_skill": "place",
                        "target_zone": zone,
                        "held_object": target,
                    },
                    contact_state=ContactState(
                        has_contact=False,
                        contact_region=None,
                    ),
                    error_code=None,
                    rollout_path="scripted://place",
                    failure_history=[],
                )
            else:
                return ExecutorResult(
                    success=False,
                    reward=0.0,
                    final_state={},
                    contact_state=ContactState(has_contact=False, contact_region=None),
                    error_code=f"unsupported_builtin: {skill_name}",
                    rollout_path="mock://unsupported",
                    failure_history=[FailureRecord(
                        step_index=step.step_index,
                        selected_skill=skill_name,
                        failure_source="execution_error",
                        reason=f"Unsupported builtin skill: {skill_name}",
                        replan_attempt=0,
                    )],
                )
        except Exception as exc:
            return ExecutorResult(
                success=False,
                reward=0.0,
                final_state={"executed_skill": skill_name},
                contact_state=ContactState(has_contact=False, contact_region=None),
                error_code=f"{skill_name}_failed: {exc}",
                rollout_path=f"scripted://{skill_name}/error",
                failure_history=[FailureRecord(
                    step_index=step.step_index,
                    selected_skill=skill_name,
                    failure_source="execution_error",
                    reason=str(exc),
                    replan_attempt=0,
                )],
            )

    def execute_all(self, steps: list[PlannerStep]) -> list[ExecutorResult]:
        results: list[ExecutorResult] = []
        for step in steps:
            result = self.execute(step)
            results.append(result)
            if not result.success:
                break
        return results

    def shutdown(self) -> None:
        if hasattr(self.simulation, 'shutdown'):
            self.simulation.shutdown()
