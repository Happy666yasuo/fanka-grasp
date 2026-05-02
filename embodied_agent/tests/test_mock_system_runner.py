from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embodied_agent.mock_system_runner import DEFAULT_ROLLOUT_PATH, run_mock_system_demo  # noqa: E402
from run_mock_system_demo import main  # noqa: E402


class MockSystemRunnerTests(unittest.TestCase):
    def test_run_mock_system_demo_returns_successful_executor_result(self) -> None:
        result = run_mock_system_demo("put the blue block in the yellow zone")

        self.assertEqual(
            [step["selected_skill"] for step in result["planner_steps"]],
            ["observe", "pick", "place"],
        )
        self.assertTrue(result["executor_result"]["success"])
        self.assertEqual(result["executor_result"]["failure_history"], [])
        self.assertEqual(result["executor_result"]["rollout_path"], DEFAULT_ROLLOUT_PATH)
        self.assertNotIn("replanned_steps", result)

    def test_run_mock_system_demo_returns_replan_payload_after_place_failure(self) -> None:
        result = run_mock_system_demo(
            "put the red block in the green zone",
            fail_on_skill="place",
        )

        self.assertEqual(
            [step["selected_skill"] for step in result["planner_steps"]],
            ["observe", "probe", "pick", "place"],
        )
        self.assertFalse(result["executor_result"]["success"])
        self.assertEqual(result["executor_result"]["error_code"], "released_outside_zone")
        self.assertEqual(len(result["executor_result"]["failure_history"]), 1)
        self.assertEqual(result["executor_result"]["failure_history"][0]["selected_skill"], "place")
        self.assertEqual(
            [step["selected_skill"] for step in result["replanned_steps"]],
            ["observe", "probe", "place"],
        )

    def test_cli_main_prints_failure_payload_with_replanned_steps(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "--instruction",
                    "put the red block in the green zone",
                    "--fail-on-skill",
                    "place",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertIn("replanned_steps", payload)
        self.assertEqual(payload["executor_result"]["failure_history"][0]["reason"], "released_outside_zone")

    def test_cli_main_accepts_named_failure_scenario(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["--scenario", "red_place_failure"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["scenario"], "red_place_failure")
        self.assertEqual(payload["executor_result"]["error_code"], "released_outside_zone")
        self.assertIn("replanned_steps", payload)


if __name__ == "__main__":
    unittest.main()
