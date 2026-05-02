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

from embodied_agent.causal_explore_report import generate_causal_explore_report  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a CausalExplore evaluation summary as a Markdown report."
    )
    parser.add_argument(
        "--summary-path",
        required=True,
        help="Path to evaluation_summary.json.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional Markdown output path. Defaults to evaluation_report.md beside the summary.",
    )
    args = parser.parse_args(argv)

    result = generate_causal_explore_report(
        summary_path=args.summary_path,
        output_path=args.output_path,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
