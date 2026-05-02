from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.reporting import plot_success_rate_curve


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot a success-rate curve from an SB3 evaluation log.")
    parser.add_argument("--run-dir", required=True, help="Training run directory containing eval/evaluations.npz")
    parser.add_argument("--title", default="Success rate curve", help="Plot title")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    evaluations_path = run_dir / "eval" / "evaluations.npz"
    output_path = run_dir / "success_rate_curve.png"
    plot_success_rate_curve(evaluations_path, output_path, title=args.title)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())