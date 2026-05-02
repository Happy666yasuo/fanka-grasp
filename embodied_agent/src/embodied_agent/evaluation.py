from __future__ import annotations

import argparse
from collections import Counter
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from embodied_agent.experiment import append_jsonl, ensure_evaluation_run_dir, load_yaml_config, save_json, save_yaml
from embodied_agent.executor import TaskExecutor
from embodied_agent.planner import RuleBasedPlanner
from embodied_agent.rl_support import SkillPolicySpec, policy_spec_from_dict
from embodied_agent.simulator import create_pick_place_simulation
from embodied_agent.skills import SkillLibrary, build_learned_skill_policies


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate_from_config(config_path: Path, gui: bool = False, gui_delay: float = 0.0) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    output_root = _resolve_path(config.get("output_root", "outputs"), PROJECT_ROOT)
    run_dir = ensure_evaluation_run_dir(output_root, label=config_path.stem)
    save_yaml(run_dir / "resolved_config.yaml", config)

    experiment_summaries: list[dict[str, Any]] = []
    all_episode_records: list[dict[str, Any]] = []
    for experiment in config.get("experiments", []):
        summary, episode_records = _evaluate_experiment(dict(experiment), run_dir=run_dir, base_dir=config_path.parent, gui=gui, gui_delay=gui_delay)
        experiment_summaries.append(summary)
        all_episode_records.extend(episode_records)

    save_json(run_dir / "evaluation_summary.json", {"experiments": experiment_summaries})
    return {
        "run_dir": str(run_dir.resolve()),
        "experiments": experiment_summaries,
        "episode_count": len(all_episode_records),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch evaluate scripted or learned skills.")
    parser.add_argument("--config", required=True, help="Path to a YAML evaluation config.")
    parser.add_argument("--gui", action="store_true", help="Enable simulator GUI rendering when available.")
    parser.add_argument("--slow", type=float, default=0.0,
                        help="Seconds to pause between steps for GUI viewing (e.g. 0.5)")
    args = parser.parse_args(argv)

    summary = evaluate_from_config(Path(args.config).resolve(), gui=args.gui, gui_delay=args.slow)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _evaluate_experiment(
    experiment: dict[str, Any],
    run_dir: Path,
    base_dir: Path,
    gui: bool = False,
    gui_delay: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    experiment_name = str(experiment.get("name", "experiment"))
    instruction = str(experiment.get("instruction", "put the red block in the green zone"))
    episode_count = int(experiment.get("episodes", 10))
    fallback_to_scripted = bool(experiment.get("fallback_to_scripted", False))
    randomize = bool(experiment.get("randomize", True))
    seed = int(experiment.get("seed", 7))
    rng = random.Random(seed)
    task_pool = _build_task_pool(experiment)
    scene_object_names, scene_zone_names = _resolve_scene_entities(task_pool)

    skill_specs = _build_skill_specs(experiment.get("skills", {}), base_dir)
    learned_skill_policies = build_learned_skill_policies(skill_specs, base_dir=base_dir)

    episode_records: list[dict[str, Any]] = []
    success_values: list[float] = []
    goal_distances: list[float] = []
    runtime_alignment_stage_counts: Counter[str] = Counter()
    episodes_with_post_pick_states = 0
    episodes_with_place_entry_states = 0

    for episode_index in range(episode_count):
        task = _select_task(task_pool, episode_index)
        instruction = str(task["instruction"])
        task_id = str(task["task_id"])
        simulation = create_pick_place_simulation(
            backend="mujoco",
            gui=gui,
            object_names=scene_object_names,
            zone_names=scene_zone_names,
        )
        try:
            if randomize:
                object_layout, zone_layout = simulation.sample_scene_layout(rng=rng)
            else:
                object_layout = simulation.default_object_layout()
                zone_layout = simulation.default_zone_layout()
            simulation.reset_task(object_layout=object_layout, zone_layout=zone_layout, holding_object=False)
            initial_state = simulation.observe_scene(instruction)

            skill_library = SkillLibrary(
                simulation=simulation,
                learned_skill_policies=learned_skill_policies,
                fallback_to_scripted=fallback_to_scripted,
                gui_delay=gui_delay,
            )
            executor = TaskExecutor(planner=RuleBasedPlanner(), skill_library=skill_library)
            result = executor.run(instruction)
            goal_distance = float(result.metrics.get("goal_distance_xy", float("nan")))
            episode_record = {
                "experiment": experiment_name,
                "episode": episode_index,
                "task_id": task_id,
                "instruction": instruction,
                "object_name": task["object_name"],
                "zone_name": task["zone_name"],
                "initial_object_positions": {
                    name: list(position) for name, position in initial_state.object_positions.items()
                },
                "initial_zone_positions": {
                    name: list(position) for name, position in initial_state.zone_positions.items()
                },
                "initial_end_effector_position": list(initial_state.end_effector_position),
                "debug": skill_library.debug_snapshot(),
                **result.to_dict(),
            }
            append_jsonl(run_dir / f"{experiment_name}_episodes.jsonl", episode_record)
            runtime_alignment_info = _append_runtime_alignment_records(
                run_dir=run_dir,
                experiment_name=experiment_name,
                episode_record=episode_record,
            )
            runtime_alignment_stage_counts.update(runtime_alignment_info["stage_counts"])
            episodes_with_post_pick_states += int(runtime_alignment_info["has_post_pick_states"])
            episodes_with_place_entry_states += int(runtime_alignment_info["has_place_entry_states"])
            episode_records.append(episode_record)
            success_values.append(1.0 if result.success else 0.0)
            goal_distances.append(goal_distance)
        finally:
            simulation.shutdown()

    summary = {
        "name": experiment_name,
        "episodes": episode_count,
        "success_rate": mean(success_values) if success_values else 0.0,
        "mean_goal_distance_xy": mean(goal_distances) if goal_distances else float("nan"),
        "fallback_to_scripted": fallback_to_scripted,
        "task_pool_size": len(task_pool),
        "learned_skills": {skill_name: spec.to_dict() for skill_name, spec in skill_specs.items()},
        "recovery": _summarize_recovery_metrics(episode_records),
        "runtime_alignment": {
            "episodes_with_post_pick_states": episodes_with_post_pick_states,
            "episodes_with_place_entry_states": episodes_with_place_entry_states,
            "stage_counts": dict(runtime_alignment_stage_counts),
        },
    }
    save_json(run_dir / f"{experiment_name}_summary.json", summary)
    return summary, episode_records


def _append_runtime_alignment_records(
    run_dir: Path,
    experiment_name: str,
    episode_record: dict[str, Any],
) -> dict[str, Any]:
    debug_info = episode_record.get("debug", {})
    runtime_alignment = debug_info.get("runtime_alignment", {})
    post_pick_states = runtime_alignment.get("post_pick_states", [])
    place_entry_states = runtime_alignment.get("place_entry_states", [])
    stage_counts: Counter[str] = Counter()

    base_record = {
        "experiment": episode_record.get("experiment"),
        "episode": episode_record.get("episode"),
        "task_id": episode_record.get("task_id"),
        "instruction": episode_record.get("instruction"),
        "object_name": episode_record.get("object_name"),
        "zone_name": episode_record.get("zone_name"),
        "success": episode_record.get("success"),
        "replan_count": episode_record.get("replan_count"),
    }

    for file_suffix, records in (
        ("post_pick_states", post_pick_states),
        ("place_entry_states", place_entry_states),
    ):
        for record in records:
            capture_stage = str(record.get("capture_stage", "unknown"))
            stage_counts[capture_stage] += 1
            append_jsonl(
                run_dir / f"{experiment_name}_{file_suffix}.jsonl",
                {
                    **base_record,
                    **record,
                },
            )

    return {
        "has_post_pick_states": bool(post_pick_states),
        "has_place_entry_states": bool(place_entry_states),
        "stage_counts": dict(stage_counts),
    }


def _summarize_recovery_metrics(episode_records: list[dict[str, Any]]) -> dict[str, Any]:
    replan_counts = [int(record.get("replan_count", 0)) for record in episode_records]
    episodes_with_replan = sum(1 for count in replan_counts if count > 0)
    successful_recovery_episodes = sum(
        1
        for record, count in zip(episode_records, replan_counts)
        if count > 0 and bool(record.get("success", False))
    )
    episodes_with_failures = sum(
        1 for record in episode_records if record.get("failure_history")
    )

    failure_source_counts: Counter[str] = Counter()
    failure_action_counts: Counter[str] = Counter()
    recovery_policy_counts: Counter[str] = Counter()
    total_step_failures = 0
    for record in episode_records:
        for failure in record.get("failure_history", []):
            total_step_failures += 1
            source = str(failure.get("source", "unknown"))
            failure_source_counts[source] += 1

            failed_step = failure.get("failed_step", {})
            action = str(failed_step.get("action", "unknown"))
            failure_action_counts[action] += 1

            recovery_policy = failure.get("recovery_policy")
            if recovery_policy:
                recovery_policy_counts[str(recovery_policy)] += 1

    return {
        "mean_replan_count": mean(replan_counts) if replan_counts else 0.0,
        "max_replan_count": max(replan_counts, default=0),
        "episodes_with_replan": episodes_with_replan,
        "successful_recovery_episodes": successful_recovery_episodes,
        "recovery_success_rate": (
            successful_recovery_episodes / episodes_with_replan if episodes_with_replan else 0.0
        ),
        "episodes_with_step_failures": episodes_with_failures,
        "total_step_failures": total_step_failures,
        "failure_source_counts": dict(failure_source_counts),
        "failure_action_counts": dict(failure_action_counts),
        "recovery_policy_counts": dict(recovery_policy_counts),
    }


def _build_skill_specs(skill_config: Any, base_dir: Path) -> dict[str, SkillPolicySpec]:
    if not isinstance(skill_config, dict):
        return {}
    return {
        skill_name: policy_spec_from_dict(skill_name, dict(config), base_dir=base_dir)
        for skill_name, config in skill_config.items()
    }


def _build_task_pool(experiment: dict[str, Any]) -> list[dict[str, str]]:
    raw_task_pool = experiment.get("task_pool")
    if not isinstance(raw_task_pool, Sequence) or isinstance(raw_task_pool, (str, bytes)):
        return [
            {
                "task_id": "task_0",
                "instruction": str(experiment.get("instruction", "put the red block in the green zone")),
                "object_name": str(experiment.get("object_name", "red_block")),
                "zone_name": str(experiment.get("zone_name", "green_zone")),
            }
        ]

    task_pool: list[dict[str, str]] = []
    for task_index, raw_task in enumerate(raw_task_pool):
        if not isinstance(raw_task, dict):
            raise ValueError("Each task_pool entry must be a mapping.")

        instruction = str(raw_task.get("instruction", "")).strip()
        object_name = str(raw_task.get("object_name", "")).strip()
        zone_name = str(raw_task.get("zone_name", "")).strip()
        if not instruction or not object_name or not zone_name:
            raise ValueError("Each task_pool entry must define instruction, object_name, and zone_name.")

        task_pool.append(
            {
                "task_id": str(raw_task.get("task_id", raw_task.get("name", f"task_{task_index}"))),
                "instruction": instruction,
                "object_name": object_name,
                "zone_name": zone_name,
            }
        )
    return task_pool


def _select_task(task_pool: Sequence[dict[str, str]], episode_index: int) -> dict[str, str]:
    return dict(task_pool[episode_index % len(task_pool)])


def _resolve_scene_entities(task_pool: Sequence[dict[str, str]]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    object_names = tuple(dict.fromkeys(task["object_name"] for task in task_pool))
    zone_names = tuple(dict.fromkeys(task["zone_name"] for task in task_pool))
    return object_names, zone_names


def _resolve_path(raw_path: object, base_dir: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
