from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.failure_analysis import load_jsonl_records, summarize_pick_failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze learned pick failures from evaluation episode logs.")
    parser.add_argument("--episodes", required=True, help="Path to a *_episodes.jsonl file.")
    parser.add_argument(
        "--compare-episodes",
        help="Optional second *_episodes.jsonl file to compare failed episode overlap.",
    )
    args = parser.parse_args()

    primary_path = Path(args.episodes).resolve()
    primary_records = load_jsonl_records(primary_path)
    summary = summarize_pick_failures(primary_records)

    if args.compare_episodes:
        compare_path = Path(args.compare_episodes).resolve()
        compare_records = load_jsonl_records(compare_path)
        compare_summary = summarize_pick_failures(compare_records)
        primary_failed = {int(item["episode"]) for item in summary["failed_episodes"]}
        compare_failed = {int(item["episode"]) for item in compare_summary["failed_episodes"]}
        summary["compare"] = {
            "path": str(compare_path),
            "failed_episode_overlap": sorted(primary_failed & compare_failed),
            "compare_failure_count": compare_summary["failure_count"],
            "compare_category_counts": compare_summary["category_counts"],
        }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())