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

from embodied_agent.causal_explore_report import generate_causal_explore_report  # noqa: E402
from run_causal_explore_report import main  # noqa: E402


class CausalExploreReportTests(unittest.TestCase):
    def test_report_summarizes_existing_eval_outputs_without_rerunning_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            planner_eval_path = root / "planner_eval_summary.json"
            summary_path = root / "evaluation_summary.json"
            system_eval_path = root / "system_eval_summary.json"
            planner_eval_path.write_text(
                json.dumps(
                    {
                        "version": "causal_explore_planner_eval_v1",
                        "strategies": [
                            {"strategy": "random", "actual_planner_probe_count": 0},
                            {
                                "strategy": "curiosity",
                                "actual_planner_probe_count": 1,
                                "planner_eval_case_count": 4,
                                "planner_probe_rate": 0.25,
                                "catalog_evaluations": [
                                    {
                                        "episode": 0,
                                        "scenarios": [
                                            {
                                                "scenario": "red_to_green",
                                                "object_id": "red_block",
                                                "zone_id": "green_zone",
                                                "selected_skills": ["observe", "pick", "place"],
                                                "inserted_probe": False,
                                            }
                                        ],
                                    },
                                    {
                                        "episode": 1,
                                        "scenarios": [
                                            {
                                                "scenario": "red_to_green",
                                                "object_id": "red_block",
                                                "zone_id": "green_zone",
                                                "selected_skills": ["observe", "probe", "pick", "place"],
                                                "inserted_probe": True,
                                            }
                                        ],
                                    },
                                ],
                            },
                            {"strategy": "causal", "actual_planner_probe_count": 0},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary_path.write_text(
                json.dumps(
                    {
                        "version": "causal_explore_eval_v1",
                        "run_id": "report_fixture",
                        "planner_eval_summary_path": str(planner_eval_path),
                        "system_eval_summary_path": str(system_eval_path),
                        "strategies": [
                            {
                                "strategy": "random",
                                "mean_displacement_xy": 0.20,
                                "mean_uncertainty": 0.06,
                                "artifact_count": 6,
                                "mean_requires_probe_rate": 0.0,
                                "actual_planner_probe_count": 0,
                                "planner_eval_case_count": 4,
                                "planner_probe_rate": 0.0,
                                "system_eval_case_count": 4,
                                "system_success_rate": 0.5,
                                "system_replan_count": 2,
                            },
                            {
                                "strategy": "curiosity",
                                "mean_displacement_xy": 0.16,
                                "mean_uncertainty": 0.13,
                                "artifact_count": 6,
                                "mean_requires_probe_rate": 0.17,
                                "actual_planner_probe_count": 1,
                                "planner_eval_case_count": 4,
                                "planner_probe_rate": 0.25,
                                "system_eval_case_count": 4,
                                "system_success_rate": 0.5,
                                "system_replan_count": 2,
                            },
                            {
                                "strategy": "causal",
                                "mean_displacement_xy": 0.17,
                                "mean_uncertainty": 0.05,
                                "artifact_count": 6,
                                "mean_requires_probe_rate": 0.0,
                                "actual_planner_probe_count": 0,
                                "planner_eval_case_count": 4,
                                "planner_probe_rate": 0.0,
                                "system_eval_case_count": 4,
                                "system_success_rate": 0.5,
                                "system_replan_count": 2,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            system_eval_path.write_text(
                json.dumps(
                    {
                        "version": "causal_explore_system_eval_v1",
                        "strategies": [
                            {
                                "strategy": "curiosity",
                                "system_eval_case_count": 4,
                                "success_rate": 0.5,
                                "failure_count": 2,
                                "replan_count": 2,
                                "probe_step_count": 3,
                                "cases": [
                                    {
                                        "episode": 0,
                                        "scenario": "blue_success",
                                        "selected_skills": ["observe", "pick", "place"],
                                        "executor_success": True,
                                        "executor_error": None,
                                        "replanned_skills": [],
                                    },
                                    {
                                        "episode": 0,
                                        "scenario": "red_place_failure",
                                        "selected_skills": ["observe", "probe", "pick", "place"],
                                        "executor_success": False,
                                        "executor_error": "released_outside_zone",
                                        "replanned_skills": ["observe", "probe", "place"],
                                    },
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = generate_causal_explore_report(summary_path)
            report_path = Path(result["report_path"])
            report = report_path.read_text(encoding="utf-8")

            self.assertTrue(report_path.exists())
            self.assertIn("| Strategy |", report)
            self.assertIn("| random |", report)
            self.assertIn("| curiosity |", report)
            self.assertIn("| causal |", report)
            self.assertIn("Planner eval cases", report)
            self.assertIn("Planner probe rate", report)
            self.assertIn("System Summary", report)
            self.assertIn("system cases", report)
            self.assertIn("System Scenarios", report)
            self.assertIn("blue_success", report)
            self.assertIn("red_place_failure", report)
            self.assertIn("episode_001", report)
            self.assertIn("report_fixture", report)

    def test_cli_writes_report_and_prints_json_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "evaluation_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "version": "causal_explore_eval_v1",
                        "run_id": "cli_report_fixture",
                        "strategies": [
                            {
                                "strategy": "random",
                                "mean_displacement_xy": 0.20,
                                "mean_uncertainty": 0.06,
                                "artifact_count": 6,
                                "mean_requires_probe_rate": 0.0,
                                "actual_planner_probe_count": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["--summary-path", str(summary_path)])

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertTrue(Path(payload["report_path"]).exists())


if __name__ == "__main__":
    unittest.main()
