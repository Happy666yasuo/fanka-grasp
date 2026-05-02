from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.mock_planning_runner import DEFAULT_REGISTRY_PATH
from embodied_agent.mock_system_runner import SYSTEM_SCENARIOS, render_mock_system_demo


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the mock planning + executor system demo with repository-backed fixtures."
    )
    parser.add_argument(
        "--instruction",
        default="把红色方块放到绿色区域",
        help="Natural-language instruction for the planner.",
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(SYSTEM_SCENARIOS.keys()),
        help="Optional named demo scenario. Overrides instruction and fail injection when provided.",
    )
    parser.add_argument(
        "--registry-path",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Path to the CausalExplore artifact registry JSON.",
    )
    parser.add_argument(
        "--catalog-path",
        help="Optional simulator-backed artifact catalog JSON. Overrides registry loading when provided.",
    )
    parser.add_argument(
        "--task-id",
        default="mock_task",
        help="Task identifier to attach to the exported planner steps.",
    )
    parser.add_argument(
        "--fail-on-skill",
        choices=("observe", "probe", "pick", "place"),
        help="Optional mock failure injection point.",
    )
    args = parser.parse_args(argv)
    scenario_config = SYSTEM_SCENARIOS.get(args.scenario, {})
    instruction = str(scenario_config.get("instruction", args.instruction))
    fail_on_skill = scenario_config.get("fail_on_skill", args.fail_on_skill)

    print(
        render_mock_system_demo(
            instruction,
            registry_path=args.registry_path,
            catalog_path=args.catalog_path,
            task_id=args.task_id,
            fail_on_skill=str(fail_on_skill) if fail_on_skill is not None else None,
            scenario=args.scenario,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
