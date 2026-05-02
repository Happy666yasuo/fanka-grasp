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

from embodied_agent.causal_explore_runner import run_causal_explore_eval  # noqa: E402


class CausalExploreSystemEvalTests(unittest.TestCase):
    def test_eval_runner_writes_system_eval_summary_and_strategy_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_causal_explore_eval(
                run_id="system_eval_test_v1",
                strategies=("random", "curiosity", "causal"),
                object_ids=("red_block", "blue_block"),
                episodes=2,
                seed=7,
                output_dir=Path(tmpdir) / "system_eval_test_v1",
            )

            system_summary_path = Path(result["system_eval_summary_path"])
            self.assertTrue(system_summary_path.exists())
            system_summary = json.loads(system_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(system_summary["version"], "causal_explore_system_eval_v1")

            for strategy_summary in system_summary["strategies"]:
                strategy = strategy_summary["strategy"]
                self.assertEqual(strategy_summary["system_eval_case_count"], 4)
                self.assertEqual(strategy_summary["success_count"], 2)
                self.assertEqual(strategy_summary["failure_count"], 2)
                self.assertEqual(strategy_summary["replan_count"], 2)
                self.assertGreaterEqual(strategy_summary["probe_step_count"], 0)

                metrics_path = Path(strategy_summary["system_metrics_path"])
                self.assertTrue(metrics_path.exists())
                metrics = [
                    json.loads(line)
                    for line in metrics_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(metrics), 4)
                self.assertEqual({metric["strategy"] for metric in metrics}, {strategy})
                self.assertEqual(
                    {metric["scenario"] for metric in metrics},
                    {"blue_success", "red_place_failure"},
                )

                blue_cases = [
                    metric for metric in metrics if metric["scenario"] == "blue_success"
                ]
                red_failure_cases = [
                    metric
                    for metric in metrics
                    if metric["scenario"] == "red_place_failure"
                ]
                self.assertTrue(all(metric["executor_success"] for metric in blue_cases))
                self.assertTrue(
                    all(
                        metric["executor_error"] == "released_outside_zone"
                        for metric in red_failure_cases
                    )
                )
                self.assertTrue(
                    all(metric["replanned_steps"] for metric in red_failure_cases)
                )

            summary_payload = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            self.assertEqual(
                Path(summary_payload["system_eval_summary_path"]),
                system_summary_path,
            )
            for item in summary_payload["strategies"]:
                self.assertIn("system_success_rate", item)
                self.assertIn("system_eval_case_count", item)
                self.assertIn("system_replan_count", item)


if __name__ == "__main__":
    unittest.main()
