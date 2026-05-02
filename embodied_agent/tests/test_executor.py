from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.executor import TaskExecutor
from embodied_agent.planner import RuleBasedPlanner
from embodied_agent.types import PlanStep, PostCondition, StepFailure, WorldState


class _FakePlanner:
    def __init__(self) -> None:
        self.replan_calls: list[StepFailure] = []

    def plan(self, instruction: str, state: WorldState) -> list[PlanStep]:
        del instruction
        del state
        return self._build_plan()

    def replan(
        self,
        instruction: str,
        state: WorldState,
        failed_step: PlanStep,
        remaining_plan: list[PlanStep],
        failure: StepFailure,
    ) -> list[PlanStep]:
        del instruction
        del state
        del failed_step
        del remaining_plan
        self.replan_calls.append(failure)
        return self._build_plan()

    def _build_plan(self) -> list[PlanStep]:
        return [
            PlanStep(action="observe"),
            PlanStep(
                action="pick",
                target="red_block",
                post_condition=PostCondition(kind="holding", object_name="red_block"),
            ),
            PlanStep(
                action="place",
                target="green_zone",
                parameters={"object": "red_block"},
                post_condition=PostCondition(
                    kind="placed",
                    object_name="red_block",
                    zone_name="green_zone",
                ),
            ),
        ]


class _FakeSkillLibrary:
    def __init__(self, succeed_on_pick_attempt: int) -> None:
        self.succeed_on_pick_attempt = succeed_on_pick_attempt
        self.pick_attempts = 0
        self.current_instruction = ""
        self.holding = False
        self.placed = False

    def observe_scene(self, instruction: str = "") -> WorldState:
        if instruction:
            self.current_instruction = instruction
        object_z = 0.72 if self.holding else 0.645
        zone_position = (0.50, -0.10, 0.623)
        object_position = zone_position if self.placed else (0.55, 0.10, object_z)
        return WorldState(
            instruction=self.current_instruction,
            object_positions={"red_block": object_position},
            zone_positions={"green_zone": zone_position},
            end_effector_position=(0.35, 0.0, 0.75),
            held_object_name="red_block" if self.holding else None,
        )

    def execute(self, step: PlanStep) -> str:
        if step.action == "observe":
            return "observe"
        if step.action == "pick":
            self.pick_attempts += 1
            self.holding = self.pick_attempts >= self.succeed_on_pick_attempt
            self.placed = False
            return f"pick:{step.target}"
        if step.action == "place":
            if not self.holding:
                raise RuntimeError("No object is currently held.")
            self.holding = False
            self.placed = True
            return f"place:{step.target}"
        raise ValueError(f"Unsupported step: {step.action}")

    def check_post_condition(self, post_condition: PostCondition) -> bool:
        if post_condition.kind == "holding":
            return self.holding
        if post_condition.kind == "placed":
            return self.placed
        raise ValueError(f"Unsupported post-condition: {post_condition.kind}")

    def verify(self, object_name: str, zone_name: str) -> bool:
        del object_name
        del zone_name
        return self.placed


class TaskExecutorReplanningTests(unittest.TestCase):
    def test_replans_after_pick_post_condition_failure(self) -> None:
        planner = _FakePlanner()
        skill_library = _FakeSkillLibrary(succeed_on_pick_attempt=2)
        executor = TaskExecutor(planner=planner, skill_library=skill_library, max_replans=1)

        result = executor.run("put the red block in the green zone")

        self.assertTrue(result.success)
        self.assertEqual(result.replan_count, 1)
        self.assertEqual(len(result.failure_history), 1)
        self.assertEqual(result.failure_history[0].source, "post_condition_failed")
        self.assertEqual(
            result.executed_actions,
            ["observe", "pick:red_block", "observe", "pick:red_block", "place:green_zone"],
        )
        self.assertEqual(len(planner.replan_calls), 1)
        self.assertEqual(result.metrics["replan_count"], 1.0)
        self.assertEqual(result.failure_history[0].recovery_policy, "retry_pick_then_continue")

    def test_returns_failure_when_replan_budget_is_exhausted(self) -> None:
        planner = _FakePlanner()
        skill_library = _FakeSkillLibrary(succeed_on_pick_attempt=99)
        executor = TaskExecutor(planner=planner, skill_library=skill_library, max_replans=0)

        result = executor.run("put the red block in the green zone")

        self.assertFalse(result.success)
        self.assertEqual(result.replan_count, 0)
        self.assertEqual(len(result.failure_history), 1)
        self.assertIn("failed post-condition", result.error or "")
        self.assertEqual(result.metrics["failure_count"], 1.0)
        self.assertEqual(len(planner.replan_calls), 0)


class _PlaceRecoverySkillLibrary:
    def __init__(self, keep_holding_after_first_place_failure: bool) -> None:
        self.keep_holding_after_first_place_failure = keep_holding_after_first_place_failure
        self.current_instruction = ""
        self.pick_attempts = 0
        self.place_attempts = 0
        self.holding = False
        self.placed = False
        self.object_position = (0.55, 0.10, 0.645)
        self.zone_position = (0.50, -0.10, 0.623)

    def observe_scene(self, instruction: str = "") -> WorldState:
        if instruction:
            self.current_instruction = instruction
        object_position = self.zone_position if self.placed else self.object_position
        return WorldState(
            instruction=self.current_instruction,
            object_positions={"red_block": object_position},
            zone_positions={"green_zone": self.zone_position},
            end_effector_position=(0.35, 0.0, 0.75),
            held_object_name="red_block" if self.holding else None,
        )

    def execute(self, step: PlanStep) -> str:
        if step.action == "observe":
            return "observe"
        if step.action == "pick":
            self.pick_attempts += 1
            self.holding = True
            self.placed = False
            self.object_position = (0.55, 0.10, 0.72)
            return f"pick:{step.target}"
        if step.action == "place":
            self.place_attempts += 1
            if self.place_attempts == 1:
                if self.keep_holding_after_first_place_failure:
                    self.holding = True
                    self.placed = False
                    self.object_position = (0.52, -0.09, 0.72)
                else:
                    self.holding = False
                    self.placed = False
                    self.object_position = (0.62, 0.12, 0.645)
                return f"place:{step.target}"

            self.holding = False
            self.placed = True
            self.object_position = self.zone_position
            return f"place:{step.target}"
        raise ValueError(f"Unsupported step: {step.action}")

    def check_post_condition(self, post_condition: PostCondition) -> bool:
        if post_condition.kind == "holding":
            return self.holding
        if post_condition.kind == "placed":
            return self.placed
        raise ValueError(f"Unsupported post-condition: {post_condition.kind}")

    def verify(self, object_name: str, zone_name: str) -> bool:
        del object_name
        del zone_name
        return self.placed


class TaskExecutorPlaceRecoveryPolicyTests(unittest.TestCase):
    def test_place_failure_retries_place_only_when_object_is_still_held(self) -> None:
        executor = TaskExecutor(
            planner=RuleBasedPlanner(),
            skill_library=_PlaceRecoverySkillLibrary(keep_holding_after_first_place_failure=True),
            max_replans=1,
        )

        result = executor.run("put the red block in the green zone")

        self.assertTrue(result.success)
        self.assertEqual(
            result.executed_actions,
            ["observe", "pick:red_block", "place:green_zone", "observe", "place:green_zone"],
        )
        self.assertEqual(result.failure_history[0].recovery_policy, "retry_place_only")

    def test_place_failure_repicks_when_object_is_released(self) -> None:
        executor = TaskExecutor(
            planner=RuleBasedPlanner(),
            skill_library=_PlaceRecoverySkillLibrary(keep_holding_after_first_place_failure=False),
            max_replans=1,
        )

        result = executor.run("put the red block in the green zone")

        self.assertTrue(result.success)
        self.assertEqual(
            result.executed_actions,
            [
                "observe",
                "pick:red_block",
                "place:green_zone",
                "observe",
                "pick:red_block",
                "place:green_zone",
            ],
        )
        self.assertEqual(result.failure_history[0].recovery_policy, "repick_then_place")


if __name__ == "__main__":
    unittest.main()