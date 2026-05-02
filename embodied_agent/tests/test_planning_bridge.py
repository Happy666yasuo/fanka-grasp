from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.contracts import (  # noqa: E402
    AffordanceCandidate,
    CausalExploreOutput,
    PropertyBelief,
)
from embodied_agent.planner import RuleBasedPlanner  # noqa: E402
from embodied_agent.planning_bridge import (  # noqa: E402
    ArtifactRegistryCausalOutputProvider,
    ContractPlanningBridge,
)
from embodied_agent.types import PlanStep, StepFailure, WorldState  # noqa: E402


class StubCausalOutputProvider:
    def __init__(self, outputs: dict[str, CausalExploreOutput]) -> None:
        self.outputs = outputs
        self.requested_object_ids: list[str] = []

    def get_outputs(self, object_ids: list[str]) -> dict[str, CausalExploreOutput]:
        self.requested_object_ids.extend(object_ids)
        return {object_id: self.outputs[object_id] for object_id in object_ids if object_id in self.outputs}


class ContractPlanningBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = ContractPlanningBridge(RuleBasedPlanner())
        self.state = WorldState(
            instruction="",
            object_positions={
                "red_block": (0.55, 0.10, 0.645),
                "blue_block": (0.63, 0.02, 0.645),
            },
            zone_positions={
                "green_zone": (0.70, -0.18, 0.623),
                "yellow_zone": (0.40, 0.13, 0.623),
            },
            end_effector_position=(0.35, 0.00, 0.75),
        )

    def _build_causal_output(self, *, object_id: str, uncertainty_score: float) -> CausalExploreOutput:
        return CausalExploreOutput(
            scene_id="scene_0001",
            object_id=object_id,
            object_category="rigid_block",
            property_belief={"mass": PropertyBelief(label="light", confidence=0.82)},
            affordance_candidates=[AffordanceCandidate(name="graspable", confidence=0.91)],
            uncertainty_score=uncertainty_score,
            recommended_probe="lateral_push",
            contact_region="side_center",
            skill_constraints={"preferred_skill": "pick"},
            artifact_path=f"outputs/causal_explore/{object_id}.json",
        )

    def test_inserts_probe_before_pick_when_target_object_is_uncertain(self) -> None:
        plan = self.bridge.plan_contract(
            task_id="move_red_block_to_green_zone",
            instruction="put the red block in the green zone",
            state=self.state,
            causal_outputs={"red_block": self._build_causal_output(object_id="red_block", uncertainty_score=0.63)},
        )

        self.assertEqual([step.selected_skill for step in plan], ["observe", "probe", "pick", "place"])
        self.assertEqual(plan[1].target_object, "red_block")
        self.assertEqual(plan[1].skill_args["probe"], "lateral_push")
        self.assertEqual(plan[2].target_object, "red_block")

    def test_keeps_rule_based_contract_plan_when_target_object_is_confident(self) -> None:
        plan = self.bridge.plan_contract(
            task_id="move_red_block_to_green_zone",
            instruction="put the red block in the green zone",
            state=self.state,
            causal_outputs={"red_block": self._build_causal_output(object_id="red_block", uncertainty_score=0.24)},
        )

        self.assertEqual([step.selected_skill for step in plan], ["observe", "pick", "place"])
        self.assertEqual(plan[1].target_object, "red_block")
        self.assertEqual(plan[2].skill_args["target_zone"], "green_zone")

    def test_inserts_probe_before_place_on_uncertain_replan(self) -> None:
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

        plan = self.bridge.replan_contract(
            task_id="move_red_block_to_green_zone",
            instruction="put the red block in the green zone",
            state=held_state,
            failed_step=failed_step,
            remaining_plan=[],
            failure=failure,
            causal_outputs={"red_block": self._build_causal_output(object_id="red_block", uncertainty_score=0.63)},
        )

        self.assertEqual([step.selected_skill for step in plan], ["observe", "probe", "place"])
        self.assertEqual(plan[1].target_object, "red_block")
        self.assertEqual(plan[2].skill_args["target_zone"], "green_zone")

    def test_replan_keeps_contract_plan_when_no_causal_output_exists(self) -> None:
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

        plan = self.bridge.replan_contract(
            task_id="move_red_block_to_green_zone",
            instruction="put the red block in the green zone",
            state=held_state,
            failed_step=failed_step,
            remaining_plan=[],
            failure=failure,
            causal_outputs={},
        )

        self.assertEqual([step.selected_skill for step in plan], ["observe", "place"])

    def test_uses_provider_when_causal_outputs_are_not_passed(self) -> None:
        provider = StubCausalOutputProvider(
            outputs={"red_block": self._build_causal_output(object_id="red_block", uncertainty_score=0.63)}
        )
        bridge = ContractPlanningBridge(RuleBasedPlanner(), causal_output_provider=provider)

        plan = bridge.plan_contract(
            task_id="move_red_block_to_green_zone",
            instruction="put the red block in the green zone",
            state=self.state,
        )

        self.assertEqual(provider.requested_object_ids, ["red_block"])
        self.assertEqual([step.selected_skill for step in plan], ["observe", "probe", "pick", "place"])

    def test_explicit_causal_outputs_override_provider_lookup(self) -> None:
        provider = StubCausalOutputProvider(
            outputs={"red_block": self._build_causal_output(object_id="red_block", uncertainty_score=0.63)}
        )
        bridge = ContractPlanningBridge(RuleBasedPlanner(), causal_output_provider=provider)

        plan = bridge.plan_contract(
            task_id="move_red_block_to_green_zone",
            instruction="put the red block in the green zone",
            state=self.state,
            causal_outputs={"red_block": self._build_causal_output(object_id="red_block", uncertainty_score=0.24)},
        )

        self.assertEqual(provider.requested_object_ids, [])
        self.assertEqual([step.selected_skill for step in plan], ["observe", "pick", "place"])

    def test_registry_provider_loads_artifact_and_injects_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry_path = root / "causal_registry.json"
            artifact_dir = root / "outputs" / "causal_explore"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / "red_block.json"
            artifact_payload = {
                "scene_id": "scene_0001",
                "object_id": "red_block",
                "object_category": "rigid_block",
                "property_belief": {
                    "mass": {"label": "light", "confidence": 0.82},
                },
                "affordance_candidates": [{"name": "graspable", "confidence": 0.91}],
                "uncertainty_score": 0.63,
                "recommended_probe": "lateral_push",
                "contact_region": "side_center",
                "skill_constraints": {"preferred_skill": "pick"},
            }
            artifact_path.write_text(json.dumps(artifact_payload), encoding="utf-8")
            registry_path.write_text(
                json.dumps(
                    {
                        "objects": {
                            "red_block": "outputs/causal_explore/red_block.json",
                        }
                    }
                ),
                encoding="utf-8",
            )

            bridge = ContractPlanningBridge(
                RuleBasedPlanner(),
                causal_output_provider=ArtifactRegistryCausalOutputProvider(registry_path),
            )
            plan = bridge.plan_contract(
                task_id="move_red_block_to_green_zone",
                instruction="put the red block in the green zone",
                state=self.state,
            )

        self.assertEqual([step.selected_skill for step in plan], ["observe", "probe", "pick", "place"])
        self.assertEqual(plan[1].skill_args["artifact_path"], str(artifact_path.resolve()))

    def test_repo_fixture_registry_loads_probe_artifact(self) -> None:
        registry_path = (
            PROJECT_ROOT
            / "outputs"
            / "causal_explore"
            / "mock_registry_v1"
            / "registry.json"
        )
        bridge = ContractPlanningBridge(
            RuleBasedPlanner(),
            causal_output_provider=ArtifactRegistryCausalOutputProvider(registry_path),
        )

        plan = bridge.plan_contract(
            task_id="move_red_block_to_green_zone",
            instruction="put the red block in the green zone",
            state=self.state,
        )

        self.assertEqual([step.selected_skill for step in plan], ["observe", "probe", "pick", "place"])
        self.assertEqual(plan[1].skill_args["probe"], "lateral_push")
        self.assertTrue(plan[1].skill_args["artifact_path"].endswith("red_block.json"))


if __name__ == "__main__":
    unittest.main()
