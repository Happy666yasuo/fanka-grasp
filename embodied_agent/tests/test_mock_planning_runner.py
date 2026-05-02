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

from embodied_agent.mock_planning_runner import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    run_mock_planning_bridge,
)
from run_mock_planning_bridge import main  # noqa: E402


class MockPlanningRunnerTests(unittest.TestCase):
    def test_run_mock_planning_bridge_uses_repo_fixture_by_default(self) -> None:
        result = run_mock_planning_bridge("put the red block in the green zone")

        self.assertEqual(result["registry_path"], str(DEFAULT_REGISTRY_PATH.resolve()))
        self.assertEqual(
            [step["selected_skill"] for step in result["planner_steps"]],
            ["observe", "probe", "pick", "place"],
        )
        self.assertEqual(result["planner_steps"][1]["skill_args"]["probe"], "lateral_push")

    def test_run_mock_planning_bridge_accepts_explicit_registry_path(self) -> None:
        result = run_mock_planning_bridge(
            "put the blue block in the yellow zone",
            registry_path=DEFAULT_REGISTRY_PATH,
        )

        self.assertEqual(result["registry_path"], str(DEFAULT_REGISTRY_PATH.resolve()))
        self.assertEqual(
            [step["selected_skill"] for step in result["planner_steps"]],
            ["observe", "pick", "place"],
        )
        self.assertEqual(result["planner_steps"][1]["target_object"], "blue_block")

    def test_cli_main_prints_json_payload(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "--instruction",
                    "put the red block in the green zone",
                    "--registry-path",
                    str(DEFAULT_REGISTRY_PATH),
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["registry_path"], str(DEFAULT_REGISTRY_PATH.resolve()))
        self.assertEqual(payload["planner_steps"][1]["selected_skill"], "probe")

    def test_cli_main_accepts_named_scenario(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["--scenario", "confident_blue_to_yellow"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["scenario"], "confident_blue_to_yellow")
        self.assertEqual(
            [step["selected_skill"] for step in payload["planner_steps"]],
            ["observe", "pick", "place"],
        )


if __name__ == "__main__":
    unittest.main()
