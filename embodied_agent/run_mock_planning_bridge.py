from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.mock_planning_runner import (
    DEFAULT_REGISTRY_PATH,
    PLANNING_SCENARIOS,
    render_mock_planning_bridge,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the mock planning bridge with repository-backed CausalExplore fixtures."
    )
    parser.add_argument(
        "--instruction",
        default="把红色方块放到绿色区域",
        help="Natural-language instruction for the planner.",
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(PLANNING_SCENARIOS.keys()),
        help="Optional named demo scenario. Overrides the default instruction when provided.",
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
    args = parser.parse_args(argv)
    instruction = PLANNING_SCENARIOS.get(args.scenario, args.instruction)

    print(
        render_mock_planning_bridge(
            instruction,
            registry_path=args.registry_path,
            catalog_path=args.catalog_path,
            task_id=args.task_id,
            scenario=args.scenario,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
