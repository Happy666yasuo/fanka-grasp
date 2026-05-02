from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def classify_pick_failure(debug_info: Mapping[str, Any]) -> str:
    initial_metrics = _as_dict(debug_info.get("initial_metrics"))
    final_metrics = _as_dict(debug_info.get("final_metrics"))

    min_distance = float(debug_info.get("min_ee_object_distance", initial_metrics.get("ee_object_distance", float("inf"))))
    max_lift = float(debug_info.get("max_lift_progress", initial_metrics.get("lift_progress", 0.0)))
    ever_holding = bool(debug_info.get("ever_holding", False))
    ever_release = bool(debug_info.get("ever_release", False))
    final_holding = float(final_metrics.get("holding_flag", 0.0)) >= 0.5
    min_gripper_open_ratio = float(debug_info.get("min_gripper_open_ratio", initial_metrics.get("gripper_open_ratio", 1.0)))

    if not ever_holding:
        if min_distance > 0.08:
            return "approach_failed"
        if min_gripper_open_ratio > 0.25:
            return "gripper_never_closed"
        return "attach_failed"
    if ever_release and not final_holding:
        if max_lift < 0.02:
            return "dropped_before_lift"
        return "dropped_during_lift"
    if max_lift < 0.02:
        return "lift_failed"
    if not final_holding:
        return "dropped_after_grasp"
    return "action_budget_exhausted"


def classify_place_failure(debug_info: Mapping[str, Any]) -> str:
    initial_metrics = _as_dict(debug_info.get("initial_metrics"))
    final_metrics = _as_dict(debug_info.get("final_metrics"))

    min_zone_distance = float(
        debug_info.get("min_object_zone_distance_xy", initial_metrics.get("object_zone_distance_xy", float("inf")))
    )
    ever_release = bool(debug_info.get("ever_release", False))
    final_holding = float(final_metrics.get("holding_flag", 1.0)) >= 0.5

    if not ever_release:
        if min_zone_distance > 0.10:
            return "transport_failed"
        return "release_missing"
    if final_holding:
        return "release_recovered_to_hold"
    return "released_outside_zone"


def extract_skill_debug(record: Mapping[str, Any], skill_name: str) -> dict[str, Any] | None:
    debug_root = record.get("debug")
    if not isinstance(debug_root, Mapping):
        return None
    skills = debug_root.get("skills")
    if not isinstance(skills, Sequence) or isinstance(skills, (str, bytes)):
        return None
    for skill_debug in skills:
        if isinstance(skill_debug, Mapping) and str(skill_debug.get("skill_name")) == skill_name:
            return dict(skill_debug)
    return None


def summarize_pick_failures(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    failed_episodes: list[dict[str, Any]] = []

    for record in records:
        pick_debug = extract_skill_debug(record, "pick")
        if pick_debug is None:
            continue
        if bool(pick_debug.get("success", False)):
            continue

        category = classify_pick_failure(pick_debug)
        category_counts[category] += 1
        failed_episodes.append(
            {
                "episode": int(record.get("episode", -1)),
                "category": category,
                "initial_object_position": _extract_position(record, "initial_object_positions", "red_block"),
                "initial_zone_position": _extract_position(record, "initial_zone_positions", "green_zone"),
                "final_object_position": _extract_position(record, "final_object_positions", "red_block"),
                "error": record.get("error"),
                "debug": pick_debug,
            }
        )

    return {
        "failure_count": len(failed_episodes),
        "category_counts": dict(category_counts),
        "failed_episodes": failed_episodes,
    }


def summarize_recovery_failures(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    experiment_records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    event_counts: Counter[tuple[str, str, str, str]] = Counter()
    reason_counts: Counter[tuple[str, str]] = Counter()
    episode_rows: list[dict[str, Any]] = []

    for record in records:
        experiment = str(record.get("experiment", "unknown"))
        experiment_records[experiment].append(record)

        failures = _normalize_failure_history(record.get("failure_history"))
        if not failures:
            continue

        failure_signatures = [_build_failure_signature(failure) for failure in failures]
        failure_reasons = [_failure_reason(failure) for failure in failures]

        episode_rows.append(
            {
                "experiment": experiment,
                "episode": int(record.get("episode", -1)),
                "success": bool(record.get("success", False)),
                "replan_count": _coerce_int(record.get("replan_count", 0)),
                "failure_events": len(failures),
                "failure_signature": _collapse_repeated_values(failure_signatures),
                "reasons": _collapse_repeated_values(failure_reasons),
                "error": str(record.get("error") or ""),
            }
        )

        for failure in failures:
            source = _failure_source(failure)
            action = _failure_action(failure)
            recovery_policy = _failure_policy(failure)
            reason = _failure_reason(failure)
            event_counts[(experiment, source, action, recovery_policy)] += 1
            reason_counts[(experiment, reason)] += 1

    experiment_summary_rows: list[dict[str, Any]] = []
    for experiment in sorted(experiment_records):
        records_for_experiment = experiment_records[experiment]
        replan_counts = [_coerce_int(record.get("replan_count", 0)) for record in records_for_experiment]
        episodes_with_failures = sum(
            1 for record in records_for_experiment if _normalize_failure_history(record.get("failure_history"))
        )
        episodes_with_replan = sum(1 for count in replan_counts if count > 0)
        successful_recovery_episodes = sum(
            1
            for record, count in zip(records_for_experiment, replan_counts)
            if count > 0 and bool(record.get("success", False))
        )
        total_step_failures = sum(
            len(_normalize_failure_history(record.get("failure_history"))) for record in records_for_experiment
        )
        experiment_summary_rows.append(
            {
                "experiment": experiment,
                "episodes": len(records_for_experiment),
                "episodes_with_failures": episodes_with_failures,
                "episodes_with_replan": episodes_with_replan,
                "successful_recovery_episodes": successful_recovery_episodes,
                "total_step_failures": total_step_failures,
                "recovery_success_rate": (
                    successful_recovery_episodes / episodes_with_replan if episodes_with_replan else 0.0
                ),
            }
        )

    event_rows = [
        {
            "experiment": experiment,
            "source": source,
            "action": action,
            "recovery_policy": recovery_policy,
            "count": count,
        }
        for (experiment, source, action, recovery_policy), count in sorted(
            event_counts.items(), key=lambda item: (item[0][0], -item[1], item[0][1], item[0][2], item[0][3])
        )
    ]
    reason_rows = [
        {
            "experiment": experiment,
            "reason": reason,
            "count": count,
        }
        for (experiment, reason), count in sorted(
            reason_counts.items(), key=lambda item: (item[0][0], -item[1], item[0][1])
        )
    ]
    episode_rows.sort(key=lambda row: (str(row["experiment"]), int(row["episode"])))

    return {
        "experiment_summary_rows": experiment_summary_rows,
        "event_rows": event_rows,
        "reason_rows": reason_rows,
        "episode_rows": episode_rows,
    }


def format_recovery_tables(summary: Mapping[str, Any]) -> str:
    experiment_summary_rows = _as_row_list(summary.get("experiment_summary_rows"))
    event_rows = _as_row_list(summary.get("event_rows"))
    reason_rows = _as_row_list(summary.get("reason_rows"))
    episode_rows = _as_row_list(summary.get("episode_rows"))

    sections = [
        "# Recovery Failure Analysis",
        "",
        "## Experiment Summary",
        _format_markdown_table(
            experiment_summary_rows,
            [
                ("experiment", "Experiment"),
                ("episodes", "Episodes"),
                ("episodes_with_failures", "Fail Episodes"),
                ("episodes_with_replan", "Replan Episodes"),
                ("successful_recovery_episodes", "Recovered Episodes"),
                ("total_step_failures", "Failure Events"),
                ("recovery_success_rate", "Recovery Success Rate"),
            ],
        ),
        "",
        "## Failure Events",
        _format_markdown_table(
            event_rows,
            [
                ("experiment", "Experiment"),
                ("source", "Source"),
                ("action", "Action"),
                ("recovery_policy", "Recovery Policy"),
                ("count", "Count"),
            ],
        ),
        "",
        "## Failure Reasons",
        _format_markdown_table(
            reason_rows,
            [
                ("experiment", "Experiment"),
                ("reason", "Reason"),
                ("count", "Count"),
            ],
        ),
        "",
        "## Failure Episodes",
        _format_markdown_table(
            episode_rows,
            [
                ("experiment", "Experiment"),
                ("episode", "Episode"),
                ("success", "Recovered"),
                ("replan_count", "Replans"),
                ("failure_events", "Failure Events"),
                ("failure_signature", "Signature"),
                ("reasons", "Reasons"),
                ("error", "Final Error"),
            ],
        ),
    ]
    return "\n".join(sections)


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_row_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _extract_position(record: Mapping[str, Any], key: str, name: str) -> list[float] | None:
    values = record.get(key)
    if not isinstance(values, Mapping):
        return None
    position = values.get(name)
    if not isinstance(position, Sequence) or isinstance(position, (str, bytes)):
        return None
    return [float(component) for component in position]


def _normalize_failure_history(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [failure for failure in value if isinstance(failure, Mapping)]


def _failure_source(failure: Mapping[str, Any]) -> str:
    return str(failure.get("source", "unknown"))


def _failure_action(failure: Mapping[str, Any]) -> str:
    failed_step = failure.get("failed_step")
    if isinstance(failed_step, Mapping):
        return str(failed_step.get("action", "unknown"))
    return "unknown"


def _failure_policy(failure: Mapping[str, Any]) -> str:
    recovery_policy = failure.get("recovery_policy")
    if recovery_policy in (None, ""):
        return "-"
    return str(recovery_policy)


def _failure_reason(failure: Mapping[str, Any]) -> str:
    reason = failure.get("reason")
    if reason in (None, ""):
        return "-"
    return str(reason)


def _build_failure_signature(failure: Mapping[str, Any]) -> str:
    return f"{_failure_action(failure)}/{_failure_source(failure)}->{_failure_policy(failure)}"


def _collapse_repeated_values(values: Sequence[str]) -> str:
    counts: Counter[str] = Counter(values)
    collapsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        count = counts[value]
        collapsed.append(f"{value} (x{count})" if count > 1 else value)
    return "; ".join(collapsed)


def _format_markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"

    rendered_rows = [
        [_render_table_value(row.get(key)) for key, _ in columns]
        for row in rows
    ]
    widths = [
        max(len(header), max(len(row[index]) for row in rendered_rows))
        for index, (_, header) in enumerate(columns)
    ]

    header_line = "| " + " | ".join(
        header.ljust(widths[index]) for index, (_, header) in enumerate(columns)
    ) + " |"
    separator_line = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
    data_lines = [
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rendered_rows
    ]
    return "\n".join([header_line, separator_line, *data_lines])


def _render_table_value(value: Any) -> str:
    if value in (None, ""):
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value).replace("|", "\\|")


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0