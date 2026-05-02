"""Run mock closed-loop system evaluation over CausalExplore catalogs.

Target conda environment: beyondmimic (Python 3.10)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from embodied_agent.mock_system_runner import SYSTEM_SCENARIOS, run_mock_system_demo


SYSTEM_EVAL_SCENARIOS: tuple[dict[str, str], ...] = (
    {
        "scenario": "blue_success",
        "object_id": "blue_block",
    },
    {
        "scenario": "red_place_failure",
        "object_id": "red_block",
    },
)


def run_causal_explore_system_eval(
    *,
    run_id: str,
    strategy_summaries: Sequence[dict[str, Any]],
    object_ids: Sequence[str],
    output_path: str | Path,
) -> dict[str, Any]:
    """Run fixed mock system scenarios against every strategy episode catalog."""

    available_object_ids = {str(object_id) for object_id in object_ids}
    resolved_output_path = Path(output_path).resolve()
    system_strategy_summaries: list[dict[str, Any]] = []

    for strategy_summary in strategy_summaries:
        strategy = str(strategy_summary["strategy"])
        strategy_dir = Path(str(strategy_summary["output_dir"])).resolve()
        metrics_path = strategy_dir / "system_metrics.jsonl"
        if metrics_path.exists():
            metrics_path.write_text("", encoding="utf-8")

        cases: list[dict[str, Any]] = []
        catalog_paths = [
            Path(str(path_ref)).resolve()
            for path_ref in strategy_summary.get("episode_catalog_paths", [])
        ]
        for episode_index, catalog_path in enumerate(catalog_paths):
            for scenario in SYSTEM_EVAL_SCENARIOS:
                if scenario["object_id"] not in available_object_ids:
                    continue
                case = _run_system_case(
                    strategy=strategy,
                    episode_index=episode_index,
                    scenario_name=scenario["scenario"],
                    catalog_path=catalog_path,
                )
                cases.append(case)
                _append_jsonl(metrics_path, case)

        system_strategy_summaries.append(
            _summarize_strategy_system_eval(
                strategy=strategy,
                cases=cases,
                metrics_path=metrics_path,
            )
        )

    payload = {
        "version": "causal_explore_system_eval_v1",
        "run_id": run_id,
        "scenarios": [dict(item) for item in SYSTEM_EVAL_SCENARIOS],
        "strategies": system_strategy_summaries,
        "summary_path": str(resolved_output_path),
    }
    _write_json(resolved_output_path, payload)
    return payload


def _run_system_case(
    *,
    strategy: str,
    episode_index: int,
    scenario_name: str,
    catalog_path: Path,
) -> dict[str, Any]:
    scenario_config = SYSTEM_SCENARIOS[scenario_name]
    instruction = str(scenario_config["instruction"])
    fail_on_skill = scenario_config.get("fail_on_skill")
    result = run_mock_system_demo(
        instruction,
        catalog_path=catalog_path,
        task_id=f"system_eval_{strategy}_ep{episode_index:03d}_{scenario_name}",
        fail_on_skill=str(fail_on_skill) if fail_on_skill is not None else None,
        scenario=scenario_name,
    )
    selected_skills = [
        str(step["selected_skill"])
        for step in result.get("planner_steps", [])  # type: ignore[union-attr]
    ]
    replanned_steps = result.get("replanned_steps", [])
    replanned_skills = [
        str(step["selected_skill"])
        for step in replanned_steps  # type: ignore[union-attr]
    ]
    executor_result = result["executor_result"]  # type: ignore[index]
    return {
        "strategy": strategy,
        "episode": episode_index,
        "episode_label": f"episode_{episode_index:03d}",
        "scenario": scenario_name,
        "catalog_path": str(catalog_path.resolve()),
        "selected_skills": selected_skills,
        "planner_skills": selected_skills,
        "executor_success": bool(executor_result["success"]),  # type: ignore[index]
        "executor_error": executor_result.get("error_code"),  # type: ignore[union-attr]
        "replanned_steps": replanned_steps,
        "replanned_skills": replanned_skills,
    }


def _summarize_strategy_system_eval(
    *,
    strategy: str,
    cases: Sequence[dict[str, Any]],
    metrics_path: Path,
) -> dict[str, Any]:
    case_count = len(cases)
    success_count = sum(1 for case in cases if bool(case["executor_success"]))
    replan_count = sum(1 for case in cases if case.get("replanned_steps"))
    probe_step_count = sum(
        1
        for case in cases
        if "probe" in {str(skill) for skill in case.get("selected_skills", [])}
    )
    replanned_probe_count = sum(
        1
        for case in cases
        if "probe" in {str(skill) for skill in case.get("replanned_skills", [])}
    )
    return {
        "strategy": strategy,
        "system_eval_case_count": case_count,
        "success_count": success_count,
        "success_rate": float(success_count / case_count) if case_count else 0.0,
        "failure_count": case_count - success_count,
        "replan_count": replan_count,
        "probe_step_count": probe_step_count,
        "replanned_probe_count": replanned_probe_count,
        "system_metrics_path": str(metrics_path.resolve()),
        "cases": list(cases),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
