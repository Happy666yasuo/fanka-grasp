from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.contracts import (  # noqa: E402
    AffordanceCandidate,
    CausalExploreOutput,
    ContactState,
    ExecutorResult,
    FailureRecord,
    PlannerStep,
    PropertyBelief,
    build_planner_step_from_causal_output,
    build_planner_step_from_plan_step,
    simulate_contract_step,
)
from embodied_agent.types import PlanStep as RuntimePlanStep, PostCondition  # noqa: E402


class SystemContractTests(unittest.TestCase):
    def _build_causal_output(self, uncertainty_score: float) -> CausalExploreOutput:
        return CausalExploreOutput(
            scene_id="scene_0001",
            object_id="red_block",
            object_category="rigid_block",
            property_belief={
                "mass": PropertyBelief(label="light", confidence=0.82),
            },
            affordance_candidates=[
                AffordanceCandidate(name="graspable", confidence=0.91),
            ],
            uncertainty_score=uncertainty_score,
            recommended_probe="lateral_push",
            contact_region="side_center",
            skill_constraints={"preferred_skill": "pick"},
            artifact_path="outputs/causal_explore/run_0001/object_red_block.json",
        )

    def test_causal_explore_output_requests_probe_when_uncertainty_is_high(self) -> None:
        output = self._build_causal_output(uncertainty_score=0.63)

        self.assertTrue(output.requires_probe())
        self.assertEqual(output.to_dict()["object_id"], "red_block")
        self.assertEqual(output.to_dict()["property_belief"]["mass"]["confidence"], 0.82)

    def test_adapter_emits_probe_step_when_causal_output_is_uncertain(self) -> None:
        step = build_planner_step_from_causal_output(
            task_id="inspect_red_block",
            step_index=0,
            causal_output=self._build_causal_output(uncertainty_score=0.63),
            requested_skill="pick",
        )

        self.assertEqual(step.selected_skill, "probe")
        self.assertEqual(step.target_object, "red_block")
        self.assertEqual(step.skill_args["probe"], "lateral_push")
        self.assertEqual(step.skill_args["contact_region"], "side_center")
        self.assertEqual(step.fallback_action, "probe_or_replan")

    def test_adapter_emits_requested_skill_when_causal_output_is_confident(self) -> None:
        step = build_planner_step_from_causal_output(
            task_id="move_red_block_to_green_zone",
            step_index=1,
            causal_output=self._build_causal_output(uncertainty_score=0.24),
            requested_skill="pick",
            skill_args={"target_zone": "green_zone"},
        )

        self.assertEqual(step.selected_skill, "pick")
        self.assertEqual(step.target_object, "red_block")
        self.assertEqual(step.skill_args["target_zone"], "green_zone")
        self.assertIn("affordance.graspable.confidence >= 0.70", step.preconditions)

    def test_planner_step_rejects_continuous_control_arguments(self) -> None:
        with self.assertRaisesRegex(ValueError, "continuous control"):
            PlannerStep(
                task_id="move_red_block_to_green_zone",
                step_index=1,
                selected_skill="pick",
                target_object="red_block",
                skill_args={"joint_positions": [0.1, 0.2]},
                preconditions=["object_visible"],
                expected_effect="holding(red_block)",
                fallback_action="probe_or_replan",
            )

    def test_executor_result_serializes_failure_history(self) -> None:
        result = ExecutorResult(
            success=False,
            reward=0.0,
            final_state={"holding": False, "object_pose": [0.22, -0.18, 0.74]},
            contact_state=ContactState(has_contact=False, contact_region=None),
            error_code="released_outside_zone",
            rollout_path="outputs/evaluations/run_0001/episode_0007.jsonl",
            failure_history=[
                FailureRecord(
                    step_index=2,
                    selected_skill="place",
                    failure_source="execution_error",
                    reason="released_outside_zone",
                    replan_attempt=1,
                    selected_recovery_policy="repick_then_place",
                )
            ],
        )

        payload = result.to_dict()

        self.assertFalse(payload["success"])
        self.assertEqual(payload["contact_state"]["has_contact"], False)
        self.assertEqual(payload["failure_history"][0]["selected_recovery_policy"], "repick_then_place")

    def test_mock_contract_flow_executes_probe_for_uncertain_object(self) -> None:
        result = simulate_contract_step(
            task_id="inspect_red_block",
            step_index=0,
            causal_output=self._build_causal_output(uncertainty_score=0.63),
            requested_skill="pick",
        )

        payload = result.to_dict()

        self.assertTrue(payload["success"])
        self.assertEqual(payload["final_state"]["executed_skill"], "probe")
        self.assertEqual(payload["contact_state"]["contact_region"], "side_center")
        self.assertEqual(payload["error_code"], None)

    def test_mock_contract_flow_executes_requested_skill_for_confident_object(self) -> None:
        result = simulate_contract_step(
            task_id="move_red_block_to_green_zone",
            step_index=1,
            causal_output=self._build_causal_output(uncertainty_score=0.24),
            requested_skill="pick",
            skill_args={"target_zone": "green_zone"},
        )

        payload = result.to_dict()

        self.assertTrue(payload["success"])
        self.assertEqual(payload["final_state"]["executed_skill"], "pick")
        self.assertEqual(payload["final_state"]["target_object"], "red_block")
        self.assertEqual(payload["failure_history"], [])

    def test_exports_runtime_pick_plan_step_to_contract_step(self) -> None:
        runtime_step = RuntimePlanStep(
            action="pick",
            target="red_block",
            parameters={"zone": "green_zone"},
            post_condition=PostCondition(kind="holding", object_name="red_block"),
        )

        contract_step = build_planner_step_from_plan_step(
            task_id="move_red_block_to_green_zone",
            step_index=1,
            plan_step=runtime_step,
        )

        self.assertEqual(contract_step.selected_skill, "pick")
        self.assertEqual(contract_step.target_object, "red_block")
        self.assertEqual(contract_step.skill_args["zone"], "green_zone")
        self.assertEqual(contract_step.expected_effect, "holding(red_block)")

    def test_exports_runtime_place_plan_step_to_contract_step(self) -> None:
        runtime_step = RuntimePlanStep(
            action="place",
            target="green_zone",
            parameters={"object": "red_block"},
            post_condition=PostCondition(
                kind="placed",
                object_name="red_block",
                zone_name="green_zone",
            ),
        )

        contract_step = build_planner_step_from_plan_step(
            task_id="move_red_block_to_green_zone",
            step_index=2,
            plan_step=runtime_step,
        )

        self.assertEqual(contract_step.selected_skill, "place")
        self.assertEqual(contract_step.target_object, "red_block")
        self.assertEqual(contract_step.skill_args["target_zone"], "green_zone")
        self.assertEqual(contract_step.expected_effect, "placed(red_block,green_zone)")


if __name__ == "__main__":
    unittest.main()
