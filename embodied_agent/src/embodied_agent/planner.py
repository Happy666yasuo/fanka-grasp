from __future__ import annotations

from typing import Protocol

from embodied_agent.types import PlanStep, PostCondition, StepFailure, WorldState


class Planner(Protocol):
    def plan(self, instruction: str, state: WorldState) -> list[PlanStep]:
        ...

    def replan(
        self,
        instruction: str,
        state: WorldState,
        failed_step: PlanStep,
        remaining_plan: list[PlanStep],
        failure: StepFailure,
    ) -> list[PlanStep]:
        ...


class RuleBasedPlanner:
    PLACE_TOKENS = (
        "put",
        "place",
        "move",
        "drop",
        "放",
        "放到",
        "放进",
        "放入",
        "移动",
    )

    OBJECT_ALIASES = {
        "red_block": (
            "red block",
            "red cube",
            "红色方块",
            "红色积木",
        ),
        "blue_block": (
            "blue block",
            "blue cube",
            "蓝色方块",
            "蓝色积木",
        ),
        "yellow_block": (
            "yellow block",
            "yellow cube",
            "黄色方块",
            "黄色积木",
        )
    }

    ZONE_ALIASES = {
        "green_zone": (
            "green zone",
            "green area",
            "绿色区域",
            "绿色区域里",
            "绿色区域内",
            "绿色目标区",
        ),
        "blue_zone": (
            "blue zone",
            "blue area",
            "蓝色区域",
            "蓝色区域里",
            "蓝色区域内",
            "蓝色目标区",
        ),
        "yellow_zone": (
            "yellow zone",
            "yellow area",
            "黄色区域",
            "黄色区域里",
            "黄色区域内",
            "黄色目标区",
        )
    }

    def plan(self, instruction: str, state: WorldState) -> list[PlanStep]:
        normalized_instruction = instruction.strip().lower()

        if not any(token in normalized_instruction for token in self.PLACE_TOKENS):
            raise ValueError("The baseline planner only supports place-style instructions.")

        object_name = self._resolve_target(
            normalized_instruction,
            self.OBJECT_ALIASES,
            fallback_names=tuple(state.object_positions.keys()),
        )
        zone_name = self._resolve_target(
            normalized_instruction,
            self.ZONE_ALIASES,
            fallback_names=tuple(state.zone_positions.keys()),
        )

        return self._build_pick_and_place_plan(object_name, zone_name)

    def replan(
        self,
        instruction: str,
        state: WorldState,
        failed_step: PlanStep,
        remaining_plan: list[PlanStep],
        failure: StepFailure,
    ) -> list[PlanStep]:
        del instruction
        del failure

        object_name = self._resolve_replan_object_name(failed_step, remaining_plan, state)
        zone_name = self._resolve_replan_zone_name(failed_step, remaining_plan, state)

        if failed_step.action == "pick":
            return self._build_pick_recovery_plan(object_name, zone_name)

        if failed_step.action == "place":
            return self._build_place_recovery_plan(object_name, zone_name, state)

        return self._build_pick_and_place_plan(object_name, zone_name)

    def _resolve_target(
        self,
        normalized_instruction: str,
        alias_map: dict[str, tuple[str, ...]],
        fallback_names: tuple[str, ...],
    ) -> str:
        matched_names = [
            canonical_name
            for canonical_name, aliases in alias_map.items()
            if any(alias in normalized_instruction for alias in aliases)
        ]

        if len(matched_names) == 1:
            return matched_names[0]

        if len(matched_names) > 1:
            raise ValueError("The planner matched multiple possible instruction targets.")

        if len(fallback_names) == 1:
            return fallback_names[0]

        raise ValueError("The planner could not resolve the instruction target.")

    def _build_pick_and_place_plan(self, object_name: str, zone_name: str) -> list[PlanStep]:
        return [
            PlanStep(action="observe"),
            self._build_pick_step(object_name, zone_name),
            self._build_place_step(object_name, zone_name),
        ]

    def _build_pick_recovery_plan(self, object_name: str, zone_name: str) -> list[PlanStep]:
        return [
            PlanStep(action="observe"),
            self._build_pick_step(object_name, zone_name),
            self._build_place_step(object_name, zone_name),
        ]

    def _build_place_recovery_plan(
        self,
        object_name: str,
        zone_name: str,
        state: WorldState,
    ) -> list[PlanStep]:
        if self._is_object_already_in_target_zone(state, object_name, zone_name):
            return [PlanStep(action="observe")]

        if state.held_object_name == object_name:
            return [
                PlanStep(action="observe"),
                self._build_place_step(object_name, zone_name),
            ]

        return [
            PlanStep(action="observe"),
            self._build_pick_step(object_name, zone_name),
            self._build_place_step(object_name, zone_name),
        ]

    def _build_pick_step(self, object_name: str, zone_name: str) -> PlanStep:
        return PlanStep(
            action="pick",
            target=object_name,
            parameters={"zone": zone_name},
            post_condition=PostCondition(kind="holding", object_name=object_name),
        )

    def _build_place_step(self, object_name: str, zone_name: str) -> PlanStep:
        return PlanStep(
            action="place",
            target=zone_name,
            parameters={"object": object_name},
            post_condition=PostCondition(
                kind="placed",
                object_name=object_name,
                zone_name=zone_name,
            ),
        )

    def _is_object_already_in_target_zone(
        self,
        state: WorldState,
        object_name: str,
        zone_name: str,
    ) -> bool:
        if state.held_object_name == object_name:
            return False

        object_position = state.object_positions.get(object_name)
        zone_position = state.zone_positions.get(zone_name)
        if object_position is None or zone_position is None:
            return False

        xy_distance = ((object_position[0] - zone_position[0]) ** 2 + (object_position[1] - zone_position[1]) ** 2) ** 0.5
        z_delta = abs(object_position[2] - zone_position[2])
        return xy_distance <= 0.08 and z_delta <= 0.08

    def _resolve_replan_object_name(
        self,
        failed_step: PlanStep,
        remaining_plan: list[PlanStep],
        state: WorldState,
    ) -> str:
        if failed_step.action == "pick" and failed_step.target is not None:
            return failed_step.target

        if failed_step.action == "place":
            object_name = failed_step.parameters.get("object")
            if isinstance(object_name, str):
                return object_name

        for step in remaining_plan:
            if step.action == "pick" and step.target is not None:
                return step.target
            if step.action == "place":
                object_name = step.parameters.get("object")
                if isinstance(object_name, str):
                    return object_name

        fallback_names = tuple(state.object_positions.keys())
        if len(fallback_names) == 1:
            return fallback_names[0]
        raise ValueError("The planner could not resolve the recovery object target.")

    def _resolve_replan_zone_name(
        self,
        failed_step: PlanStep,
        remaining_plan: list[PlanStep],
        state: WorldState,
    ) -> str:
        if failed_step.action == "place" and failed_step.target is not None:
            return failed_step.target

        for step in remaining_plan:
            if step.action == "place" and step.target is not None:
                return step.target

        fallback_names = tuple(state.zone_positions.keys())
        if len(fallback_names) == 1:
            return fallback_names[0]
        raise ValueError("The planner could not resolve the recovery zone target.")