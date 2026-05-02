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

from embodied_agent.mock_planning_runner import run_mock_planning_bridge  # noqa: E402
from embodied_agent.mock_system_runner import run_mock_system_demo  # noqa: E402
from embodied_agent.planning_bridge import (  # noqa: E402
    ContractPlanningBridge,
    SimulatorArtifactCatalogCausalOutputProvider,
)
from embodied_agent.planner import RuleBasedPlanner  # noqa: E402
from embodied_agent.types import WorldState  # noqa: E402


SIM_CATALOG_V1_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "causal_explore"
    / "sim_catalog_v1"
    / "catalog.json"
)


class SimulatorCatalogProviderTests(unittest.TestCase):
    def test_catalog_provider_loads_uncertain_object_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = self._write_catalog_fixture(Path(tmpdir), include_red_in_manifest=True)

            provider = SimulatorArtifactCatalogCausalOutputProvider(catalog_path)
            outputs = provider.get_outputs(["red_block"])

        self.assertEqual(outputs["red_block"].scene_id, "scene_sim_0001")
        self.assertEqual(outputs["red_block"].object_id, "red_block")
        self.assertTrue(outputs["red_block"].artifact_path.endswith("red_block.json"))

    def test_catalog_provider_rejects_object_missing_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = self._write_catalog_fixture(Path(tmpdir), include_red_in_manifest=False)
            provider = SimulatorArtifactCatalogCausalOutputProvider(catalog_path)

            with self.assertRaisesRegex(ValueError, "red_block"):
                provider.get_outputs(["red_block"])

    def test_bridge_uses_catalog_provider_without_manual_output_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = self._write_catalog_fixture(Path(tmpdir), include_red_in_manifest=True)
            bridge = ContractPlanningBridge(
                RuleBasedPlanner(),
                causal_output_provider=SimulatorArtifactCatalogCausalOutputProvider(catalog_path),
            )
            state = WorldState(
                instruction="",
                object_positions={
                    "red_block": (0.55, 0.10, 0.645),
                    "blue_block": (0.63, 0.02, 0.645),
                },
                zone_positions={"green_zone": (0.70, -0.18, 0.623)},
                end_effector_position=(0.35, 0.00, 0.75),
            )

            plan = bridge.plan_contract(
                task_id="move_red_block_to_green_zone",
                instruction="put the red block in the green zone",
                state=state,
            )

        self.assertEqual([step.selected_skill for step in plan], ["observe", "probe", "pick", "place"])

    def test_run_mock_planning_bridge_accepts_catalog_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = self._write_catalog_fixture(Path(tmpdir), include_red_in_manifest=True)
            result = run_mock_planning_bridge(
                "put the red block in the green zone",
                catalog_path=catalog_path,
            )

        self.assertEqual(result["catalog_path"], str(catalog_path.resolve()))
        self.assertEqual(
            [step["selected_skill"] for step in result["planner_steps"]],
            ["observe", "probe", "pick", "place"],
        )

    def test_run_mock_system_demo_accepts_catalog_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = self._write_catalog_fixture(Path(tmpdir), include_red_in_manifest=True)
            result = run_mock_system_demo(
                "put the red block in the green zone",
                catalog_path=catalog_path,
                fail_on_skill="place",
            )

        self.assertEqual(result["catalog_path"], str(catalog_path.resolve()))
        self.assertEqual(result["executor_result"]["error_code"], "released_outside_zone")
        self.assertIn("replanned_steps", result)

    def test_repo_sim_catalog_v1_drives_uncertain_planning_path(self) -> None:
        result = run_mock_planning_bridge(
            "put the red block in the green zone",
            catalog_path=SIM_CATALOG_V1_PATH,
            scenario="uncertain_red_to_green",
        )

        self.assertEqual(result["catalog_path"], str(SIM_CATALOG_V1_PATH.resolve()))
        self.assertEqual(
            [step["selected_skill"] for step in result["planner_steps"]],
            ["observe", "probe", "pick", "place"],
        )
        self.assertEqual(result["planner_steps"][1]["skill_args"]["probe"], "lateral_push")

    def test_repo_sim_catalog_v1_drives_confident_system_path(self) -> None:
        result = run_mock_system_demo(
            "put the blue block in the yellow zone",
            catalog_path=SIM_CATALOG_V1_PATH,
            scenario="blue_success",
        )

        self.assertEqual(result["catalog_path"], str(SIM_CATALOG_V1_PATH.resolve()))
        self.assertEqual(
            [step["selected_skill"] for step in result["planner_steps"]],
            ["observe", "pick", "place"],
        )
        self.assertIn("blue_block", result["causal_outputs"])
        blue_output = result["causal_outputs"]["blue_block"]
        self.assertEqual(blue_output["artifact_path"], str((SIM_CATALOG_V1_PATH.parent / "artifacts" / "blue_block.json").resolve()))
        self.assertTrue(result["executor_result"]["success"])

    def test_repo_sim_catalog_v1_drives_failure_replan_path(self) -> None:
        result = run_mock_system_demo(
            "put the red block in the green zone",
            catalog_path=SIM_CATALOG_V1_PATH,
            fail_on_skill="place",
            scenario="red_place_failure",
        )

        self.assertEqual(result["executor_result"]["error_code"], "released_outside_zone")
        self.assertEqual(result["executor_result"]["failure_history"][0]["selected_skill"], "place")
        self.assertEqual(
            [step["selected_skill"] for step in result["replanned_steps"]],
            ["observe", "probe", "place"],
        )

    def _write_catalog_fixture(self, root: Path, *, include_red_in_manifest: bool) -> Path:
        manifest_dir = root / "manifests"
        artifact_dir = root / "artifacts"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = manifest_dir / "scene_sim_0001.json"
        artifact_path = artifact_dir / "red_block.json"
        catalog_path = root / "catalog.json"

        manifest_payload = {
            "scene_id": "scene_sim_0001",
            "objects": [
                {"object_id": "blue_block"},
            ],
        }
        if include_red_in_manifest:
            manifest_payload["objects"].append({"object_id": "red_block"})

        artifact_payload = {
            "scene_id": "scene_sim_0001",
            "object_id": "red_block",
            "object_category": "rigid_block",
            "property_belief": {"mass": {"label": "light", "confidence": 0.82}},
            "affordance_candidates": [{"name": "graspable", "confidence": 0.91}],
            "uncertainty_score": 0.63,
            "recommended_probe": "lateral_push",
            "contact_region": "side_center",
            "skill_constraints": {"preferred_skill": "pick"},
        }
        catalog_payload = {
            "run_id": "sim_run_0001",
            "scene_id": "scene_sim_0001",
            "object_manifest_path": "manifests/scene_sim_0001.json",
            "objects": {
                "red_block": {
                    "artifact_path": "artifacts/red_block.json",
                }
            },
        }

        manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
        artifact_path.write_text(json.dumps(artifact_payload), encoding="utf-8")
        catalog_path.write_text(json.dumps(catalog_payload), encoding="utf-8")
        return catalog_path


if __name__ == "__main__":
    unittest.main()
