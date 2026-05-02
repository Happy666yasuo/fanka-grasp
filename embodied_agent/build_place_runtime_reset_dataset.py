"""Build strict place runtime-reset datasets for task-pool training.

Target conda environment: beyondmimic (Python 3.10)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluations"
    / "20260420_131259_batch_eval_50_place_runtime_alignment"
)
DEFAULT_SOURCE_PATHS = (
    DEFAULT_SOURCE_DIR / "learned_place_runtime_alignment_raw_entry_50_post_pick_states.jsonl",
    DEFAULT_SOURCE_DIR / "learned_place_runtime_alignment_staged_50_post_pick_states.jsonl",
)
DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / "outputs" / "datasets" / "task_pool_post_pick_curated_v2.jsonl"
DEFAULT_OUTPUT_AUDIT = PROJECT_ROOT / "outputs" / "datasets" / "task_pool_post_pick_curated_v2_audit.json"
DEFAULT_TASK_POOL = (
    {
        "task_id": "red_to_green",
        "instruction": "put the red block in the green zone",
        "object_name": "red_block",
        "zone_name": "green_zone",
    },
    {
        "task_id": "blue_to_yellow",
        "instruction": "put the blue block in the yellow zone",
        "object_name": "blue_block",
        "zone_name": "yellow_zone",
    },
    {
        "task_id": "yellow_to_blue",
        "instruction": "把黄色方块放到蓝色区域",
        "object_name": "yellow_block",
        "zone_name": "blue_zone",
    },
)


@dataclass(frozen=True)
class StrictCuratedThresholds:
    preset: str = "strict"
    min_object_height: float = 0.705
    min_lift_progress: float = 0.065
    max_lift_progress: float = 0.09
    min_object_zone_distance_xy: float = 0.35
    max_object_zone_distance_xy: float = 0.60
    max_ee_object_distance: float = 0.16
    min_held_local_z: float = 0.08

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "capture_stage": "post_pick_success",
            "task_pool": list(DEFAULT_TASK_POOL),
            "min_object_height": self.min_object_height,
            "lift_progress_range": [self.min_lift_progress, self.max_lift_progress],
            "object_zone_distance_xy_range": [
                self.min_object_zone_distance_xy,
                self.max_object_zone_distance_xy,
            ],
            "max_ee_object_distance": self.max_ee_object_distance,
            "min_held_local_z": self.min_held_local_z,
        }


def _resolve_thresholds(preset: str) -> StrictCuratedThresholds:
    if preset == "strict":
        return StrictCuratedThresholds()
    if preset == "broadened_v1":
        return StrictCuratedThresholds(
            preset="broadened_v1",
            min_object_height=0.64,
            min_lift_progress=0.0,
            max_lift_progress=0.20,
            min_object_zone_distance_xy=0.05,
            max_object_zone_distance_xy=0.60,
            max_ee_object_distance=0.50,
            min_held_local_z=-0.20,
        )
    raise ValueError(f"Unknown threshold preset: {preset}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a strict curated post-pick runtime-reset dataset for task-pool place training."
    )
    parser.add_argument(
        "--preset",
        choices=("strict", "broadened_v1"),
        default="strict",
        help="Threshold preset to use. Default preserves the existing strict curated dataset behavior.",
    )
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        help="Input post-pick runtime state JSONL path. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(DEFAULT_OUTPUT_JSONL),
        help="Output JSONL path for curated runtime-reset samples.",
    )
    parser.add_argument(
        "--output-audit",
        default=str(DEFAULT_OUTPUT_AUDIT),
        help="Output audit JSON path.",
    )
    return parser.parse_args()


def _iter_records(paths: Iterable[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                yield path, json.loads(line)


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics = dict(payload.get("metrics", {}))
    object_position = payload.get("object_position") or [0.0, 0.0, 0.0]
    held_local_position = payload.get("held_local_position") or [0.0, 0.0, 0.0]
    metrics.setdefault("object_height", float(object_position[2]))
    metrics.setdefault("lift_progress", 0.0)
    metrics.setdefault("object_zone_distance_xy", float("inf"))
    metrics.setdefault("ee_object_distance", float("inf"))
    metrics["held_local_z"] = float(held_local_position[2])
    return {key: float(value) for key, value in metrics.items()}


def _record_passes_strict_filter(
    payload: dict[str, Any],
    thresholds: StrictCuratedThresholds,
    allowed_task_pairs: set[tuple[str, str]],
) -> tuple[bool, dict[str, float], str | None]:
    if str(payload.get("capture_stage")) != "post_pick_success":
        return False, {}, "capture_stage"
    if not bool(payload.get("holding_target_object", False)):
        return False, {}, "holding_target_object"

    object_name = str(payload.get("object_name", "")).strip()
    zone_name = str(payload.get("zone_name", "")).strip()
    if (object_name, zone_name) not in allowed_task_pairs:
        return False, {}, "task_pool"

    metrics = _extract_metrics(payload)
    if metrics["object_height"] < thresholds.min_object_height:
        return False, metrics, "object_height"
    if not thresholds.min_lift_progress <= metrics["lift_progress"] <= thresholds.max_lift_progress:
        return False, metrics, "lift_progress"
    if not (
        thresholds.min_object_zone_distance_xy
        <= metrics["object_zone_distance_xy"]
        <= thresholds.max_object_zone_distance_xy
    ):
        return False, metrics, "object_zone_distance_xy"
    if metrics["ee_object_distance"] > thresholds.max_ee_object_distance:
        return False, metrics, "ee_object_distance"
    if metrics["held_local_z"] < thresholds.min_held_local_z:
        return False, metrics, "held_local_z"
    return True, metrics, None


def _build_audit_record(payload: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "experiment": payload.get("experiment"),
        "episode": payload.get("episode", payload.get("episode_index")),
        "capture_index": payload.get("capture_index"),
        "task_id": payload.get("task_id"),
        "object_name": payload.get("object_name"),
        "zone_name": payload.get("zone_name"),
        "object_zone_distance_xy": metrics["object_zone_distance_xy"],
        "lift_progress": metrics["lift_progress"],
        "ee_object_distance": metrics["ee_object_distance"],
        "held_local_z": metrics["held_local_z"],
    }


def main() -> int:
    args = _parse_args()
    source_paths = tuple(Path(raw_path).expanduser().resolve() for raw_path in (args.inputs or DEFAULT_SOURCE_PATHS))
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    output_audit = Path(args.output_audit).expanduser().resolve()

    thresholds = _resolve_thresholds(args.preset)
    allowed_task_pairs = {
        (str(task["object_name"]), str(task["zone_name"])) for task in DEFAULT_TASK_POOL
    }
    counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    kept_records: list[dict[str, Any]] = []
    unique_records: list[dict[str, Any]] = []

    for source_path, payload in _iter_records(source_paths):
        counts["total_records"] += 1
        keep_record, metrics, rejection_reason = _record_passes_strict_filter(
            payload,
            thresholds,
            allowed_task_pairs,
        )
        if not keep_record:
            assert rejection_reason is not None
            rejection_counts[rejection_reason] += 1
            continue

        kept_records.append(payload)
        counts["kept_records"] += 1
        counts[f"source::{source_path.name}"] += 1
        counts[f"experiment::{payload.get('experiment', '')}"] += 1
        counts[f"task::{payload.get('task_id', '')}"] += 1
        counts[f"capture_index::{payload.get('capture_index', '')}"] += 1
        unique_records.append(_build_audit_record(payload, metrics))

    if not kept_records:
        raise ValueError("Strict curated filter removed every runtime-reset sample.")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for payload in kept_records:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    audit_payload = {
        "output_path": str(output_jsonl),
        "source_paths": [str(path) for path in source_paths],
        "filters": thresholds.to_dict(),
        "counts": dict(counts),
        "rejection_counts": dict(rejection_counts),
        "unique_records": unique_records,
    }
    output_audit.parent.mkdir(parents=True, exist_ok=True)
    with output_audit.open("w", encoding="utf-8") as handle:
        json.dump(audit_payload, handle, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "output_jsonl": str(output_jsonl),
                "output_audit": str(output_audit),
                "kept_records": counts["kept_records"],
                "task_counts": {
                    key.removeprefix("task::"): value
                    for key, value in counts.items()
                    if key.startswith("task::")
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
