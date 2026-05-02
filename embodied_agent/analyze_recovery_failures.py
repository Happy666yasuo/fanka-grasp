from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.failure_analysis import format_recovery_tables, load_jsonl_records, summarize_recovery_failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze executor failure_history records and summarize recovery behavior as tables."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--run-dir",
        help="Path to an evaluation run directory containing one or more *_episodes.jsonl files.",
    )
    input_group.add_argument(
        "--episodes",
        nargs="+",
        help="One or more *_episodes.jsonl files to analyze.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the rendered report. Defaults to stdout only.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON instead of Markdown tables.",
    )
    args = parser.parse_args()

    episode_paths = _resolve_episode_paths(run_dir=args.run_dir, episode_paths=args.episodes)
    records = []
    for episode_path in episode_paths:
        records.extend(load_jsonl_records(episode_path))

    summary = summarize_recovery_failures(records)
    summary["input_files"] = [str(path) for path in episode_paths]
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) if args.json else _render_report(summary)

    if args.output:
        Path(args.output).resolve().write_text(rendered, encoding="utf-8")

    print(rendered)
    return 0


def _resolve_episode_paths(run_dir: str | None, episode_paths: list[str] | None) -> list[Path]:
    if run_dir:
        resolved_run_dir = Path(run_dir).resolve()
        paths = sorted(path for path in resolved_run_dir.glob("*_episodes.jsonl") if path.is_file())
        if not paths:
            raise SystemExit(f"No *_episodes.jsonl files found in {resolved_run_dir}")
        return paths

    if not episode_paths:
        raise SystemExit("No episode files were provided.")
    return [Path(raw_path).resolve() for raw_path in episode_paths]


def _render_report(summary: dict[str, object]) -> str:
    input_files = summary.get("input_files", [])
    inputs = [str(path) for path in input_files] if isinstance(input_files, list) else []
    header_lines = ["Analyzed Files:"]
    header_lines.extend(f"- {path}" for path in inputs)
    header_lines.append("")
    return "\n".join([*header_lines, format_recovery_tables(summary)])


if __name__ == "__main__":
    raise SystemExit(main())