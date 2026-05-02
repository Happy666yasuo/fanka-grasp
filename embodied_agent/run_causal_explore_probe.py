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

from embodied_agent.causal_explore_runner import build_run_id, run_causal_explore_probe


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a PyBullet-backed CausalExplore probe and export simulator-style artifacts."
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier. Defaults to a timestamped pybullet_probe id.",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["red_block", "blue_block"],
        help="Object ids to probe.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output run directory. Defaults to outputs/causal_explore/<run-id>.",
    )
    parser.add_argument(
        "--probe",
        default="lateral_push",
        choices=("lateral_push",),
        help="Probe action to execute. v1 supports lateral_push only.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for small deterministic layout jitter.",
    )
    args = parser.parse_args(argv)

    run_id = args.run_id or build_run_id()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "outputs" / "causal_explore" / run_id)

    result = run_causal_explore_probe(
        run_id=run_id,
        object_ids=tuple(args.objects),
        output_dir=output_dir,
        probe=args.probe,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
