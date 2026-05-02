from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.causal_explore_runner import (  # noqa: E402
    SUPPORTED_EVAL_STRATEGIES,
    run_causal_explore_eval,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a multi-strategy CausalExplore evaluation and export JSON summary."
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier. Defaults to a timestamped causal_eval id.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["random", "curiosity", "causal"],
        choices=tuple(sorted(SUPPORTED_EVAL_STRATEGIES)),
        help="Exploration strategies to evaluate.",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["red_block", "blue_block"],
        help="Object ids to evaluate.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run per strategy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base random seed for deterministic strategy ordering and layout jitter.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output evaluation directory. Defaults to outputs/causal_explore/<run-id>.",
    )
    args = parser.parse_args(argv)

    result = run_causal_explore_eval(
        run_id=args.run_id,
        strategies=tuple(args.strategies),
        object_ids=tuple(args.objects),
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
