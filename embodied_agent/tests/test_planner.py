from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.planner import RuleBasedPlanner
from embodied_agent.types import PlanStep, StepFailure, WorldState


class RuleBasedPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = RuleBasedPlanner()
        self.state = WorldState(
            instruction="",
            object_positions={
                "red_block": (0.55, 0.10, 0.645),
                "blue_block": (0.63, 0.02, 0.645),
                "yellow_block": (0.47, -0.12, 0.645),
            },
            zone_positions={
                "green_zone": (0.70, -0.18, 0.623),
                "blue_zone": (0.62, 0.14, 0.623),
                "yellow_zone": (0.40, 0.13, 0.623),
            },
            end_effector_position=(0.35, 0.00, 0.75),
        )

    def test_plans_a_chinese_pick_and_place_instruction(self) -> None:
        plan = self.planner.plan("把红色方块放到绿色区域", self.state)
        self.assertEqual([step.action for step in plan], ["observe", "pick", "place"])
        self.assertEqual(plan[1].target, "red_block")
        self.assertEqual(plan[2].target, "green_zone")
        self.assertEqual(plan[1].parameters["zone"], "green_zone")
        self.assertEqual(plan[1].post_condition.kind, "holding")
        self.assertEqual(plan[2].post_condition.kind, "placed")

    def test_plans_an_english_pick_and_place_instruction(self) -> None:
        plan = self.planner.plan("put the red block in the green zone", self.state)
        self.assertEqual(plan[1].target, "red_block")
        self.assertEqual(plan[2].target, "green_zone")

    def test_plans_a_multi_object_instruction(self) -> None:
        plan = self.planner.plan("put the blue block in the yellow zone", self.state)

        self.assertEqual(plan[1].target, "blue_block")
        self.assertEqual(plan[1].parameters["zone"], "yellow_zone")
        self.assertEqual(plan[2].target, "yellow_zone")

    def test_rejects_ambiguous_instruction_when_multiple_targets_exist(self) -> None:
        with self.assertRaises(ValueError):
            self.planner.plan("put the block in the zone", self.state)

    def test_replans_to_recover_full_pick_and_place_sequence(self) -> None:
        failed_step = PlanStep(action="pick", target="red_block")
        failure = StepFailure(
            failed_step=failed_step,
            source="post_condition_failed",
            reason="Step 'pick' failed post-condition 'holding'.",
            replan_attempt=1,
        )

        plan = self.planner.replan(
            instruction="put the red block in the green zone",
            state=self.state,
            failed_step=failed_step,
            remaining_plan=[
                PlanStep(action="place", target="green_zone", parameters={"object": "red_block"})
            ],
            failure=failure,
        )

        self.assertEqual([step.action for step in plan], ["observe", "pick", "place"])
        self.assertEqual(plan[1].target, "red_block")
        self.assertEqual(plan[1].parameters["zone"], "green_zone")
        self.assertEqual(plan[2].target, "green_zone")

    def test_place_replan_retries_place_only_when_object_is_still_held(self) -> None:
        state = WorldState(
            instruction="",
            object_positions={"red_block": (0.50, -0.10, 0.72)},
            zone_positions={"green_zone": (0.50, -0.10, 0.623)},
            end_effector_position=(0.35, 0.00, 0.75),
            held_object_name="red_block",
        )
        failed_step = PlanStep(
            action="place",
            target="green_zone",
            parameters={"object": "red_block"},
        )
        failure = StepFailure(
            failed_step=failed_step,
            source="post_condition_failed",
            reason="Step 'place' failed post-condition 'placed'.",
            replan_attempt=1,
        )

        plan = self.planner.replan(
            instruction="put the red block in the green zone",
            state=state,
            failed_step=failed_step,
            remaining_plan=[],
            failure=failure,
        )

        self.assertEqual([step.action for step in plan], ["observe", "place"])

    def test_place_replan_repicks_when_object_is_not_held(self) -> None:
        state = WorldState(
            instruction="",
            object_positions={"red_block": (0.62, 0.11, 0.645)},
            zone_positions={"green_zone": (0.50, -0.10, 0.623)},
            end_effector_position=(0.35, 0.00, 0.75),
            held_object_name=None,
        )
        failed_step = PlanStep(
            action="place",
            target="green_zone",
            parameters={"object": "red_block"},
        )
        failure = StepFailure(
            failed_step=failed_step,
            source="post_condition_failed",
            reason="Step 'place' failed post-condition 'placed'.",
            replan_attempt=1,
        )

        plan = self.planner.replan(
            instruction="put the red block in the green zone",
            state=state,
            failed_step=failed_step,
            remaining_plan=[],
            failure=failure,
        )

        self.assertEqual([step.action for step in plan], ["observe", "pick", "place"])


if __name__ == "__main__":
    unittest.main()