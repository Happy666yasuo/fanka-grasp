from __future__ import annotations

from dataclasses import dataclass

from embodied_agent.contracts import PlannerStep, build_planner_step_from_plan_step
from embodied_agent.planner import Planner
from embodied_agent.types import PlanStep, StepFailure, WorldState


@dataclass
class ContractPlannerAdapter:
    planner: Planner

    def plan_contract(
        self,
        *,
        task_id: str,
        instruction: str,
        state: WorldState,
    ) -> list[PlannerStep]:
        runtime_plan = self.planner.plan(instruction, state)
        return self._export_plan(task_id=task_id, runtime_plan=runtime_plan)

    def replan_contract(
        self,
        *,
        task_id: str,
        instruction: str,
        state: WorldState,
        failed_step: PlanStep,
        remaining_plan: list[PlanStep],
        failure: StepFailure,
    ) -> list[PlannerStep]:
        runtime_plan = self.planner.replan(
            instruction=instruction,
            state=state,
            failed_step=failed_step,
            remaining_plan=remaining_plan,
            failure=failure,
        )
        return self._export_plan(task_id=task_id, runtime_plan=runtime_plan)

    def _export_plan(
        self,
        *,
        task_id: str,
        runtime_plan: list[PlanStep],
    ) -> list[PlannerStep]:
        return [
            build_planner_step_from_plan_step(
                task_id=task_id,
                step_index=step_index,
                plan_step=plan_step,
            )
            for step_index, plan_step in enumerate(runtime_plan)
        ]
