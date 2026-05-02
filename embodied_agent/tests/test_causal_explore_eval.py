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

from embodied_agent.causal_explore_runner import run_causal_explore_eval  # noqa: E402
from embodied_agent.planning_bridge import SimulatorArtifactCatalogCausalOutputProvider  # noqa: E402
from run_causal_explore_eval import main  # noqa: E402


class CausalExploreEvalTests(unittest.TestCase):
    def test_eval_runner_writes_strategy_catalogs_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_causal_explore_eval(
                run_id="eval_test_v1",
                strategies=("random", "curiosity", "causal"),
                object_ids=("red_block", "blue_block"),
                episodes=2,
                seed=7,
                output_dir=Path(tmpdir) / "eval_test_v1",
            )

            summary_path = Path(result["summary_path"])
            self.assertTrue(summary_path.exists())
            self.assertEqual(result["run_id"], "eval_test_v1")
            self.assertEqual(
                [item["strategy"] for item in result["strategies"]],
                ["random", "curiosity", "causal"],
            )

            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(summary_payload["strategies"]), 3)
            planner_eval_path = Path(summary_payload["planner_eval_summary_path"])
            system_eval_path = Path(summary_payload["system_eval_summary_path"])
            self.assertTrue(planner_eval_path.exists())
            self.assertTrue(system_eval_path.exists())
            planner_eval_payload = json.loads(planner_eval_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["strategy"] for item in planner_eval_payload["strategies"]],
                ["random", "curiosity", "causal"],
            )
            planner_by_strategy = {
                item["strategy"]: item for item in planner_eval_payload["strategies"]
            }
            for item in summary_payload["strategies"]:
                strategy_dir = Path(item["output_dir"])
                catalog_path = strategy_dir / "catalog.json"
                metrics_path = strategy_dir / "episode_metrics.jsonl"
                self.assertTrue(catalog_path.exists())
                self.assertTrue((strategy_dir / "artifacts").is_dir())
                self.assertTrue((strategy_dir / "evidence").is_dir())
                self.assertTrue(metrics_path.exists())

                self.assertEqual(item["episodes"], 2)
                self.assertEqual(item["object_count"], 2)
                self.assertIn("mean_uncertainty", item)
                self.assertIn("mean_displacement_xy", item)
                self.assertIn("artifact_count", item)
                self.assertIn("planner_probe_count", item)
                self.assertIn("actual_planner_probe_count", item)
                self.assertIn("planner_eval_case_count", item)
                self.assertIn("planner_probe_rate", item)
                self.assertIn("mean_requires_probe_rate", item)
                self.assertIn("system_success_rate", item)
                self.assertIn("system_eval_case_count", item)
                self.assertIn("system_replan_count", item)
                self.assertGreaterEqual(item["mean_uncertainty"], 0.0)
                self.assertGreaterEqual(item["mean_displacement_xy"], 0.0)
                self.assertGreaterEqual(item["planner_probe_rate"], 0.0)
                self.assertEqual(item["planner_eval_case_count"], 4)
                self.assertEqual(
                    planner_by_strategy[item["strategy"]]["planner_eval_case_count"],
                    4,
                )
                self.assertEqual(
                    len(planner_by_strategy[item["strategy"]]["catalog_evaluations"]),
                    2,
                )

                metrics = [
                    json.loads(line)
                    for line in metrics_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(metrics), 4)
                self.assertEqual({metric["strategy"] for metric in metrics}, {item["strategy"]})
                self.assertEqual({metric["object_id"] for metric in metrics}, {"red_block", "blue_block"})
                self.assertIn("requires_probe", metrics[0])
                self.assertIn("artifact_path", metrics[0])
                self.assertIn("evidence_path", metrics[0])

            curiosity_summary = next(
                item
                for item in summary_payload["strategies"]
                if item["strategy"] == "curiosity"
            )
            self.assertGreaterEqual(curiosity_summary["planner_probe_count"], 1)
            self.assertGreaterEqual(curiosity_summary["actual_planner_probe_count"], 1)

    def test_eval_strategy_catalogs_load_through_simulator_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_causal_explore_eval(
                run_id="provider_eval_v1",
                strategies=("random", "curiosity", "causal"),
                object_ids=("red_block", "blue_block"),
                episodes=1,
                seed=7,
                output_dir=Path(tmpdir) / "provider_eval_v1",
            )

            for item in result["strategies"]:
                provider = SimulatorArtifactCatalogCausalOutputProvider(Path(item["catalog_path"]))
                outputs = provider.get_outputs(["red_block", "blue_block"])
                self.assertEqual(set(outputs), {"red_block", "blue_block"})

    def test_cli_runs_eval_and_prints_json_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--run-id",
                        "cli_eval_v1",
                        "--strategies",
                        "random",
                        "curiosity",
                        "causal",
                        "--objects",
                        "red_block",
                        "blue_block",
                        "--episodes",
                        "1",
                        "--seed",
                        "7",
                        "--output-dir",
                        str(Path(tmpdir) / "cli_eval_v1"),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["run_id"], "cli_eval_v1")
            self.assertTrue(Path(payload["summary_path"]).exists())
            self.assertEqual(
                [item["strategy"] for item in payload["strategies"]],
                ["random", "curiosity", "causal"],
            )


if __name__ == "__main__":
    unittest.main()
