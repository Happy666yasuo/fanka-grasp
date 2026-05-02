"""Render CausalExplore evaluation summaries as Markdown reports.

Target conda environment: beyondmimic (Python 3.10)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_causal_explore_report(
    summary_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_summary_path = Path(summary_path).resolve()
    summary = json.loads(resolved_summary_path.read_text(encoding="utf-8"))
    planner_summary = _load_planner_summary(summary, resolved_summary_path)
    system_summary = _load_system_summary(summary, resolved_summary_path)
    resolved_output_path = (
        Path(output_path).resolve()
        if output_path is not None
        else resolved_summary_path.parent / "evaluation_report.md"
    )
    report = _render_report(summary, planner_summary, system_summary)
    resolved_output_path.write_text(report, encoding="utf-8")
    return {
        "summary_path": str(resolved_summary_path),
        "planner_eval_summary_path": (
            str(Path(summary["planner_eval_summary_path"]).resolve())
            if summary.get("planner_eval_summary_path")
            else None
        ),
        "system_eval_summary_path": (
            str(Path(summary["system_eval_summary_path"]).resolve())
            if summary.get("system_eval_summary_path")
            else None
        ),
        "report_path": str(resolved_output_path.resolve()),
    }


def _load_planner_summary(
    summary: dict[str, Any],
    summary_path: Path,
) -> dict[str, Any]:
    planner_ref = summary.get("planner_eval_summary_path")
    if not planner_ref:
        return {"strategies": []}
    planner_path = Path(str(planner_ref))
    if not planner_path.is_absolute():
        planner_path = (summary_path.parent / planner_path).resolve()
    return json.loads(planner_path.read_text(encoding="utf-8"))


def _load_system_summary(
    summary: dict[str, Any],
    summary_path: Path,
) -> dict[str, Any]:
    system_ref = summary.get("system_eval_summary_path")
    if not system_ref:
        return {"strategies": []}
    system_path = Path(str(system_ref))
    if not system_path.is_absolute():
        system_path = (summary_path.parent / system_path).resolve()
    return json.loads(system_path.read_text(encoding="utf-8"))


def _render_report(
    summary: dict[str, Any],
    planner_summary: dict[str, Any],
    system_summary: dict[str, Any],
) -> str:
    planner_by_strategy = {
        str(item.get("strategy")): item
        for item in planner_summary.get("strategies", [])
        if isinstance(item, dict)
    }
    system_by_strategy = {
        str(item.get("strategy")): item
        for item in system_summary.get("strategies", [])
        if isinstance(item, dict)
    }
    lines = [
        f"# CausalExplore Evaluation Report: {summary.get('run_id', 'unknown')}",
        "",
        "## Strategy Summary",
        "",
        "| Strategy | Mean displacement XY | Mean uncertainty | Artifact count | Requires probe rate | Planner eval cases | Planner probe count | Planner probe rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary.get("strategies", []):
        if not isinstance(item, dict):
            continue
        strategy = str(item.get("strategy", "unknown"))
        planner_item = planner_by_strategy.get(strategy, {})
        planner_probe_count = item.get(
            "actual_planner_probe_count",
            planner_item.get("actual_planner_probe_count", 0),
        )
        planner_eval_case_count = item.get(
            "planner_eval_case_count",
            planner_item.get("planner_eval_case_count", 0),
        )
        planner_probe_rate = item.get(
            "planner_probe_rate",
            planner_item.get("planner_probe_rate", 0.0),
        )
        lines.append(
            "| {strategy} | {mean_displacement:.4f} | {mean_uncertainty:.4f} | {artifact_count} | {requires_rate:.4f} | {planner_eval_case_count} | {planner_probe_count} | {planner_probe_rate:.4f} |".format(
                strategy=strategy,
                mean_displacement=float(item.get("mean_displacement_xy", 0.0)),
                mean_uncertainty=float(item.get("mean_uncertainty", 0.0)),
                artifact_count=int(item.get("artifact_count", 0)),
                requires_rate=float(item.get("mean_requires_probe_rate", 0.0)),
                planner_eval_case_count=int(planner_eval_case_count),
                planner_probe_count=int(planner_probe_count),
                planner_probe_rate=float(planner_probe_rate),
            )
        )

    if planner_by_strategy:
        lines.extend(["", "## Planner Scenarios", ""])
        for strategy, planner_item in planner_by_strategy.items():
            lines.append(f"### {strategy}")
            lines.append("")
            lines.append("| Catalog episode | Scenario | Object | Zone | Skills | Probe inserted |")
            lines.append("|---|---|---|---|---|---|")
            catalog_evaluations = planner_item.get("catalog_evaluations", [])
            if not catalog_evaluations:
                catalog_evaluations = [
                    {
                        "episode_label": "episode_000",
                        "scenarios": planner_item.get("scenarios", []),
                    }
                ]
            for catalog_eval in catalog_evaluations:
                if not isinstance(catalog_eval, dict):
                    continue
                episode_label = _catalog_episode_label(catalog_eval)
                for scenario in catalog_eval.get("scenarios", []):
                    if not isinstance(scenario, dict):
                        continue
                    skills = " -> ".join(str(skill) for skill in scenario.get("selected_skills", []))
                    lines.append(
                        "| {episode_label} | {scenario} | {object_id} | {zone_id} | {skills} | {inserted_probe} |".format(
                            episode_label=episode_label,
                            scenario=scenario.get("scenario", ""),
                            object_id=scenario.get("object_id", ""),
                            zone_id=scenario.get("zone_id", ""),
                            skills=skills,
                            inserted_probe=str(bool(scenario.get("inserted_probe", False))).lower(),
                        )
                    )
            lines.append("")

    if system_by_strategy:
        lines.extend(
            [
                "",
                "## System Summary",
                "",
                "| Strategy | system cases | Success rate | Failures | Replans | Probe steps |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in summary.get("strategies", []):
            if not isinstance(item, dict):
                continue
            strategy = str(item.get("strategy", "unknown"))
            system_item = system_by_strategy.get(strategy, {})
            if not system_item:
                continue
            lines.append(
                "| {strategy} | {case_count} | {success_rate:.4f} | {failure_count} | {replan_count} | {probe_count} |".format(
                    strategy=strategy,
                    case_count=int(system_item.get("system_eval_case_count", 0)),
                    success_rate=float(system_item.get("success_rate", 0.0)),
                    failure_count=int(system_item.get("failure_count", 0)),
                    replan_count=int(system_item.get("replan_count", 0)),
                    probe_count=int(system_item.get("probe_step_count", 0)),
                )
            )

        lines.extend(["", "## System Scenarios", ""])
        for strategy, system_item in system_by_strategy.items():
            lines.append(f"### {strategy}")
            lines.append("")
            lines.append(
                "| Episode | Scenario | Planner skills | Success | Error | Replanned skills |"
            )
            lines.append("|---|---|---|---|---|---|")
            for case in system_item.get("cases", []):
                if not isinstance(case, dict):
                    continue
                selected_skills = " -> ".join(
                    str(skill) for skill in case.get("selected_skills", [])
                )
                replanned_skills = " -> ".join(
                    str(skill) for skill in case.get("replanned_skills", [])
                )
                lines.append(
                    "| {episode} | {scenario} | {selected_skills} | {success} | {error} | {replanned_skills} |".format(
                        episode=case.get("episode_label", _case_episode_label(case)),
                        scenario=case.get("scenario", ""),
                        selected_skills=selected_skills,
                        success=str(bool(case.get("executor_success", False))).lower(),
                        error=case.get("executor_error") or "",
                        replanned_skills=replanned_skills,
                    )
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _catalog_episode_label(catalog_eval: dict[str, Any]) -> str:
    if catalog_eval.get("episode_label") is not None:
        return str(catalog_eval["episode_label"])
    try:
        return f"episode_{int(catalog_eval.get('episode', 0)):03d}"
    except (TypeError, ValueError):
        return "episode_000"


def _case_episode_label(case: dict[str, Any]) -> str:
    try:
        return f"episode_{int(case.get('episode', 0)):03d}"
    except (TypeError, ValueError):
        return "episode_000"
