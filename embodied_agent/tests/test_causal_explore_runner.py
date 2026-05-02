from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embodied_agent.causal_explore_runner import run_causal_explore_probe  # noqa: E402
from embodied_agent.mock_planning_runner import run_mock_planning_bridge  # noqa: E402
from embodied_agent.planning_bridge import SimulatorArtifactCatalogCausalOutputProvider  # noqa: E402
from run_causal_explore_probe import main  # noqa: E402


class CausalExploreRunnerTests(unittest.TestCase):
    def test_probe_runner_writes_catalog_manifest_artifact_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_causal_explore_probe(
                run_id="test_probe_v1",
                object_ids=("red_block",),
                output_dir=Path(tmpdir) / "test_probe_v1",
                seed=7,
            )

            catalog_path = Path(result["catalog_path"])
            manifest_path = catalog_path.parent / "manifests" / "scene_test_probe_v1.json"
            artifact_path = catalog_path.parent / "artifacts" / "red_block.json"
            evidence_path = catalog_path.parent / "evidence" / "red_block_lateral_push.json"

            self.assertTrue(catalog_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(artifact_path.exists())
            self.assertTrue(evidence_path.exists())

            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            self.assertEqual(catalog["scene_id"], "scene_test_probe_v1")
            self.assertEqual(catalog["objects"]["red_block"]["artifact_path"], "artifacts/red_block.json")

            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["probe_action"], "lateral_push")
            self.assertEqual(evidence["object_id"], "red_block")
            self.assertIn("before_position", evidence)
            self.assertIn("after_position", evidence)
            self.assertGreaterEqual(evidence["displacement_xy"], 0.0)

    def test_probe_catalog_loads_through_simulator_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_causal_explore_probe(
                run_id="provider_probe_v1",
                object_ids=("red_block", "blue_block"),
                output_dir=Path(tmpdir) / "provider_probe_v1",
                seed=7,
            )
            catalog_path = Path(result["catalog_path"])
            provider = SimulatorArtifactCatalogCausalOutputProvider(catalog_path)

            outputs = provider.get_outputs(["red_block", "blue_block"])

            self.assertEqual(set(outputs), {"red_block", "blue_block"})
            self.assertEqual(outputs["red_block"].scene_id, "scene_provider_probe_v1")
            self.assertTrue(outputs["red_block"].artifact_path.endswith("red_block.json"))

    def test_probe_catalog_can_drive_existing_planning_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_causal_explore_probe(
                run_id="planning_probe_v1",
                object_ids=("red_block",),
                output_dir=Path(tmpdir) / "planning_probe_v1",
                seed=7,
            )

            plan_result = run_mock_planning_bridge(
                "put the red block in the green zone",
                catalog_path=result["catalog_path"],
            )

            selected_skills = [step["selected_skill"] for step in plan_result["planner_steps"]]
            self.assertIn("pick", selected_skills)
            self.assertIn("place", selected_skills)
            self.assertEqual(plan_result["catalog_path"], str(Path(result["catalog_path"]).resolve()))

    def test_cli_writes_probe_run_and_prints_json_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--run-id",
                        "cli_probe_v1",
                        "--objects",
                        "red_block",
                        "--output-dir",
                        str(Path(tmpdir) / "cli_probe_v1"),
                    ]
                )

            payload = json.loads(stdout.getvalue())

            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["run_id"], "cli_probe_v1")
            self.assertEqual(payload["objects"], ["red_block"])
            self.assertTrue(Path(payload["catalog_path"]).exists())


if __name__ == "__main__":
    unittest.main()
