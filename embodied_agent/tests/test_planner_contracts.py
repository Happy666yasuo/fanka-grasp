from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.planner import RuleBasedPlanner  # noqa: E402
from embodied_agent.planner_contracts import ContractPlannerAdapter  # noqa: E402
from embodied_agent.types import PlanStep, StepFailure, WorldState  # noqa: E402


class ContractPlannerAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = ContractPlannerAdapter(RuleBasedPlanner())
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

    def test_exports_rule_based_plan_as_contract_steps(self) -> None:
        contract_plan = self.adapter.plan_contract(
            task_id="move_red_block_to_green_zone",
            instruction="put the red block in the green zone",
            state=self.state,
        )

        self.assertEqual([step.selected_skill for step in contract_plan], ["observe", "pick", "place"])
        self.assertEqual(contract_plan[1].target_object, "red_block")
        self.assertEqual(contract_plan[2].skill_args["target_zone"], "green_zone")
        self.assertEqual(contract_plan[2].expected_effect, "placed(red_block,green_zone)")

    def test_exports_place_replan_as_contract_steps(self) -> None:
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
        held_state = WorldState(
            instruction="",
            object_positions={"red_block": (0.50, -0.10, 0.72)},
            zone_positions={"green_zone": (0.50, -0.10, 0.623)},
            end_effector_position=(0.35, 0.00, 0.75),
            held_object_name="red_block",
        )

        contract_plan = self.adapter.replan_contract(
            task_id="move_red_block_to_green_zone",
            instruction="put the red block in the green zone",
            state=held_state,
            failed_step=failed_step,
            remaining_plan=[],
            failure=failure,
        )

        self.assertEqual([step.selected_skill for step in contract_plan], ["observe", "place"])
        self.assertEqual(contract_plan[1].target_object, "red_block")
        self.assertEqual(contract_plan[1].skill_args["target_zone"], "green_zone")


if __name__ == "__main__":
    unittest.main()
