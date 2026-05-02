"""Run simulator-backed probe actions and export CausalExplore artifacts.

Target conda environment: beyondmimic (Python 3.10)
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from embodied_agent.contracts import causal_output_from_dict
from embodied_agent.causal_explore_system_eval import run_causal_explore_system_eval
from embodied_agent.simulation_protocol import PickPlaceSimulationProtocol
from embodied_agent.simulator import create_pick_place_simulation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "causal_explore"
SUPPORTED_PROBES = {"lateral_push"}
SUPPORTED_EVAL_STRATEGIES = {"random", "curiosity", "causal"}
PLANNER_EVAL_SCENARIOS = (
    {
        "scenario": "red_to_green",
        "instruction": "put the red block in the green zone",
        "object_id": "red_block",
        "zone_id": "green_zone",
    },
    {
        "scenario": "blue_to_yellow",
        "instruction": "put the blue block in the yellow zone",
        "object_id": "blue_block",
        "zone_id": "yellow_zone",
    },
)


@dataclass(frozen=True)
class ProbeRunConfig:
    run_id: str
    object_ids: tuple[str, ...]
    output_dir: Path
    probe: str = "lateral_push"
    seed: int = 7


@dataclass
class EvalStrategyState:
    evidence_counts: dict[str, int]
    uncertainty_by_object: dict[str, float]


def build_run_id() -> str:
    return datetime.now().strftime("mujoco_probe_%Y%m%d_%H%M%S")


def run_causal_explore_probe(
    *,
    run_id: str | None = None,
    object_ids: Sequence[str] = ("red_block", "blue_block"),
    output_dir: str | Path | None = None,
    probe: str = "lateral_push",
    seed: int = 7,
) -> dict[str, Any]:
    resolved_run_id = run_id or build_run_id()
    resolved_object_ids = tuple(str(object_id) for object_id in object_ids)
    if not resolved_object_ids:
        raise ValueError("At least one object id is required.")
    if probe not in SUPPORTED_PROBES:
        raise ValueError(f"Unsupported probe: {probe}")

    resolved_output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_ROOT / resolved_run_id
    config = ProbeRunConfig(
        run_id=resolved_run_id,
        object_ids=resolved_object_ids,
        output_dir=resolved_output_dir.resolve(),
        probe=probe,
        seed=int(seed),
    )

    return _run_probe_config(config)


def run_causal_explore_eval(
    *,
    run_id: str | None = None,
    strategies: Sequence[str] = ("random", "curiosity", "causal"),
    object_ids: Sequence[str] = ("red_block", "blue_block"),
    episodes: int = 5,
    seed: int = 7,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate CausalExplore strategy variants on one object manifest.

    The v1 harness keeps the existing probe artifact schema intact. Each
    strategy writes a simulator-style catalog at its strategy root, while
    additional episodes are stored below ``episodes/`` for metric aggregation.
    """

    resolved_run_id = run_id or f"causal_eval_{build_run_id()}"
    resolved_strategies = tuple(str(strategy) for strategy in strategies)
    if not resolved_strategies:
        raise ValueError("At least one strategy is required.")
    unsupported = sorted(set(resolved_strategies) - SUPPORTED_EVAL_STRATEGIES)
    if unsupported:
        raise ValueError(f"Unsupported eval strategies: {', '.join(unsupported)}")

    resolved_object_ids = tuple(str(object_id) for object_id in object_ids)
    if not resolved_object_ids:
        raise ValueError("At least one object id is required.")
    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    eval_root = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_ROOT / resolved_run_id
    eval_root = eval_root.resolve()
    eval_root.mkdir(parents=True, exist_ok=True)

    strategy_summaries: list[dict[str, Any]] = []
    planner_strategy_summaries: list[dict[str, Any]] = []
    for strategy_index, strategy in enumerate(resolved_strategies):
        strategy_summary = _run_eval_strategy(
            eval_run_id=resolved_run_id,
            strategy=strategy,
            object_ids=resolved_object_ids,
            episodes=int(episodes),
            seed=int(seed) + strategy_index * 1009,
            strategy_dir=eval_root / strategy,
        )
        planner_summary = _evaluate_strategy_planner_paths(
            strategy=strategy,
            catalog_paths=[
                Path(path_ref)
                for path_ref in strategy_summary["episode_catalog_paths"]
            ],
            object_ids=resolved_object_ids,
        )
        strategy_summary.update(
            {
                "actual_planner_probe_count": planner_summary["actual_planner_probe_count"],
                "planner_eval_case_count": planner_summary["planner_eval_case_count"],
                "planner_probe_rate": planner_summary["planner_probe_rate"],
            }
        )
        strategy_summaries.append(strategy_summary)
        planner_strategy_summaries.append(planner_summary)

    summary_path = eval_root / "evaluation_summary.json"
    planner_eval_summary_path = eval_root / "planner_eval_summary.json"
    planner_eval_payload = {
        "version": "causal_explore_planner_eval_v1",
        "run_id": resolved_run_id,
        "strategies": planner_strategy_summaries,
    }
    _write_json(planner_eval_summary_path, planner_eval_payload)
    system_eval_summary_path = eval_root / "system_eval_summary.json"
    system_eval_payload = run_causal_explore_system_eval(
        run_id=resolved_run_id,
        strategy_summaries=strategy_summaries,
        object_ids=resolved_object_ids,
        output_path=system_eval_summary_path,
    )
    system_by_strategy = {
        str(item.get("strategy")): item
        for item in system_eval_payload.get("strategies", [])
        if isinstance(item, dict)
    }
    for strategy_summary in strategy_summaries:
        system_summary = system_by_strategy.get(str(strategy_summary["strategy"]), {})
        strategy_summary.update(
            {
                "system_eval_case_count": int(system_summary.get("system_eval_case_count", 0)),
                "system_success_rate": float(system_summary.get("success_rate", 0.0)),
                "system_failure_count": int(system_summary.get("failure_count", 0)),
                "system_replan_count": int(system_summary.get("replan_count", 0)),
                "system_probe_step_count": int(system_summary.get("probe_step_count", 0)),
                "system_metrics_path": system_summary.get("system_metrics_path"),
            }
        )
    payload = {
        "version": "causal_explore_eval_v1",
        "run_id": resolved_run_id,
        "objects": list(resolved_object_ids),
        "episodes": int(episodes),
        "strategies": strategy_summaries,
        "planner_eval_summary_path": str(planner_eval_summary_path.resolve()),
        "system_eval_summary_path": str(system_eval_summary_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "output_dir": str(eval_root),
    }
    _write_json(summary_path, payload)
    return payload


def _run_eval_strategy(
    *,
    eval_run_id: str,
    strategy: str,
    object_ids: tuple[str, ...],
    episodes: int,
    seed: int,
    strategy_dir: Path,
) -> dict[str, Any]:
    state = EvalStrategyState(
        evidence_counts={object_id: 0 for object_id in object_ids},
        uncertainty_by_object={object_id: 1.0 for object_id in object_ids},
    )
    rng = random.Random(seed)
    displacement_values: list[float] = []
    uncertainty_values: list[float] = []
    planner_probe_count = 0
    catalog_path: Path | None = None
    episode_catalog_paths: list[str] = []
    metrics_path = strategy_dir / "episode_metrics.jsonl"

    for episode_index in range(episodes):
        ordered_object_ids = _order_objects_for_strategy(
            strategy=strategy,
            object_ids=object_ids,
            state=state,
            rng=rng,
        )
        episode_dir = (
            strategy_dir
            if episode_index == 0
            else strategy_dir / "episodes" / f"episode_{episode_index:03d}"
        )
        episode_run_id = f"{eval_run_id}_{strategy}_ep{episode_index:03d}"
        result = run_causal_explore_probe(
            run_id=episode_run_id,
            object_ids=ordered_object_ids,
            output_dir=episode_dir,
            seed=seed + episode_index,
        )
        if episode_index == 0:
            catalog_path = Path(result["catalog_path"])
        episode_catalog_paths.append(str(Path(result["catalog_path"]).resolve()))

        artifact_metrics = {
            metric["object_id"]: metric
            for metric in _read_artifact_metrics(Path(result["catalog_path"]))
        }
        for item in result["summaries"]:
            object_id = str(item["object_id"])
            displacement_xy = float(item["displacement_xy"])
            displacement_values.append(displacement_xy)
            state.evidence_counts[object_id] = state.evidence_counts.get(object_id, 0) + 1
            metric = artifact_metrics[object_id]
            object_id = str(metric["object_id"])
            uncertainty = float(metric["uncertainty_score"])
            uncertainty_values.append(uncertainty)
            state.uncertainty_by_object[object_id] = uncertainty
            if bool(metric["requires_probe"]):
                planner_probe_count += 1
            _append_jsonl(
                metrics_path,
                {
                    "strategy": strategy,
                    "episode": episode_index,
                    "run_id": result["run_id"],
                    "scene_id": result["scene_id"],
                    "object_id": object_id,
                    "displacement_xy": displacement_xy,
                    "uncertainty_score": uncertainty,
                    "requires_probe": bool(metric["requires_probe"]),
                    "artifact_path": metric["artifact_path"],
                    "evidence_path": metric["evidence_path"],
                },
            )

    if catalog_path is None:
        raise RuntimeError(f"No catalog generated for strategy {strategy}.")

    return {
        "strategy": strategy,
        "episodes": episodes,
        "object_count": len(object_ids),
        "mean_uncertainty": _mean(uncertainty_values),
        "mean_displacement_xy": _mean(displacement_values),
        "artifact_count": len(uncertainty_values),
        "planner_probe_count": planner_probe_count,
        "mean_requires_probe_rate": (
            float(planner_probe_count / len(uncertainty_values))
            if uncertainty_values
            else 0.0
        ),
        "actual_planner_probe_count": 0,
        "planner_eval_case_count": 0,
        "planner_probe_rate": 0.0,
        "catalog_path": str(catalog_path.resolve()),
        "episode_catalog_paths": episode_catalog_paths,
        "episode_metrics_path": str(metrics_path.resolve()),
        "output_dir": str(strategy_dir.resolve()),
    }


def _order_objects_for_strategy(
    *,
    strategy: str,
    object_ids: tuple[str, ...],
    state: EvalStrategyState,
    rng: random.Random,
) -> tuple[str, ...]:
    ordered = list(object_ids)
    if strategy == "random":
        rng.shuffle(ordered)
        return tuple(ordered)
    if strategy == "curiosity":
        ordered.sort(
            key=lambda object_id: (
                state.evidence_counts.get(object_id, 0),
                -state.uncertainty_by_object.get(object_id, 1.0),
                object_id,
            )
        )
        return tuple(ordered)
    if strategy == "causal":
        return tuple(ordered)
    raise ValueError(f"Unsupported eval strategy: {strategy}")


def _read_artifact_metrics(catalog_path: Path) -> list[dict[str, Any]]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    object_registry = dict(catalog.get("objects", {}))
    metrics: list[dict[str, Any]] = []
    for object_id, object_entry in object_registry.items():
        artifact_ref = object_entry.get("artifact_path", object_entry)
        artifact_path = Path(str(artifact_ref))
        if not artifact_path.is_absolute():
            artifact_path = (catalog_path.parent / artifact_path).resolve()
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        evidence_path = Path(str(payload.get("evidence_path", "")))
        if str(evidence_path) and not evidence_path.is_absolute():
            evidence_path = (artifact_path.parent.parent / evidence_path).resolve()
        causal_output = causal_output_from_dict(
            payload,
            artifact_path=str(artifact_path),
        )
        metrics.append(
            {
                "object_id": str(object_id),
                "uncertainty_score": causal_output.uncertainty_score,
                "requires_probe": causal_output.requires_probe(),
                "artifact_path": str(artifact_path),
                "evidence_path": str(evidence_path) if str(evidence_path) else "",
            }
        )
    return metrics


def _evaluate_strategy_planner_paths(
    *,
    strategy: str,
    catalog_paths: Sequence[Path],
    object_ids: tuple[str, ...],
) -> dict[str, Any]:
    from embodied_agent.mock_planning_runner import run_mock_planning_bridge

    catalog_evaluations: list[dict[str, Any]] = []
    flattened_scenario_records: list[dict[str, Any]] = []
    for episode_index, catalog_path in enumerate(catalog_paths):
        scenario_records: list[dict[str, Any]] = []
        for scenario in PLANNER_EVAL_SCENARIOS:
            if str(scenario["object_id"]) not in object_ids:
                continue
            result = run_mock_planning_bridge(
                str(scenario["instruction"]),
                catalog_path=catalog_path,
                task_id=f"planner_eval_{strategy}_ep{episode_index:03d}_{scenario['scenario']}",
                scenario=str(scenario["scenario"]),
            )
            selected_skills = [
                str(step["selected_skill"])
                for step in result["planner_steps"]  # type: ignore[index]
            ]
            scenario_record = {
                "scenario": scenario["scenario"],
                "instruction": scenario["instruction"],
                "object_id": scenario["object_id"],
                "zone_id": scenario["zone_id"],
                "selected_skills": selected_skills,
                "inserted_probe": "probe" in selected_skills,
            }
            scenario_records.append(scenario_record)
            flattened_scenario_records.append(
                {
                    "episode": episode_index,
                    "catalog_path": str(catalog_path.resolve()),
                    **scenario_record,
                }
            )
        catalog_evaluations.append(
            {
                "episode": episode_index,
                "episode_label": f"episode_{episode_index:03d}",
                "catalog_path": str(catalog_path.resolve()),
                "scenarios": scenario_records,
            }
        )

    actual_planner_probe_count = sum(
        1 for record in flattened_scenario_records if bool(record["inserted_probe"])
    )
    planner_eval_case_count = len(flattened_scenario_records)
    return {
        "strategy": strategy,
        "catalog_path": str(catalog_paths[0].resolve()) if catalog_paths else "",
        "catalog_count": len(catalog_paths),
        "planner_eval_case_count": planner_eval_case_count,
        "scenario_count": planner_eval_case_count,
        "actual_planner_probe_count": actual_planner_probe_count,
        "planner_probe_rate": (
            float(actual_planner_probe_count / planner_eval_case_count)
            if planner_eval_case_count
            else 0.0
        ),
        "catalog_evaluations": catalog_evaluations,
        "scenarios": flattened_scenario_records,
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _run_probe_config(config: ProbeRunConfig) -> dict[str, Any]:
    scene_id = f"scene_{config.run_id}"
    artifact_dir = config.output_dir / "artifacts"
    evidence_dir = config.output_dir / "evidence"
    manifest_dir = config.output_dir / "manifests"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    simulation = create_pick_place_simulation(
        backend="mujoco",
        gui=False,
        object_names=config.object_ids,
        zone_names=("green_zone", "yellow_zone", "blue_zone"),
    )
    try:
        simulation.reset_task()
        _apply_small_layout_jitter(simulation, config.object_ids, rng)
        object_entries: dict[str, dict[str, str]] = {}
        summaries: list[dict[str, Any]] = []

        for object_id in config.object_ids:
            evidence_payload = _run_lateral_push_probe(
                simulation=simulation,
                scene_id=scene_id,
                object_id=object_id,
                run_id=config.run_id,
            )
            evidence_path = evidence_dir / f"{object_id}_lateral_push.json"
            _write_json(evidence_path, evidence_payload)

            artifact_payload = _build_artifact_payload(
                scene_id=scene_id,
                object_id=object_id,
                evidence_path=f"evidence/{evidence_path.name}",
                evidence_payload=evidence_payload,
            )
            artifact_path = artifact_dir / f"{object_id}.json"
            _write_json(artifact_path, artifact_payload)

            object_entries[object_id] = {"artifact_path": f"artifacts/{artifact_path.name}"}
            summaries.append(
                {
                    "object_id": object_id,
                    "artifact_path": str(artifact_path.resolve()),
                    "evidence_path": str(evidence_path.resolve()),
                    "displacement_xy": evidence_payload["displacement_xy"],
                }
            )
    finally:
        simulation.shutdown()

    manifest_path = manifest_dir / f"{scene_id}.json"
    _write_json(manifest_path, _build_manifest(scene_id=scene_id, object_ids=config.object_ids))

    catalog_path = config.output_dir / "catalog.json"
    catalog_payload = {
        "version": "mujoco_probe_v1",
        "run_id": config.run_id,
        "scene_id": scene_id,
        "object_manifest_path": f"manifests/{manifest_path.name}",
        "objects": object_entries,
    }
    _write_json(catalog_path, catalog_payload)

    return {
        "run_id": config.run_id,
        "scene_id": scene_id,
        "probe": config.probe,
        "objects": list(config.object_ids),
        "catalog_path": str(catalog_path.resolve()),
        "output_dir": str(config.output_dir),
        "summaries": summaries,
    }


def _apply_small_layout_jitter(
    simulation: PickPlaceSimulationProtocol,
    object_ids: Sequence[str],
    rng: random.Random,
) -> None:
    object_layout = simulation.default_object_layout()
    for object_id in object_ids:
        base_xy = object_layout[object_id]
        object_layout[object_id] = (
            base_xy[0] + rng.uniform(-0.005, 0.005),
            base_xy[1] + rng.uniform(-0.005, 0.005),
        )
    simulation.reset_task(object_layout=object_layout)


def _run_lateral_push_probe(
    *,
    simulation: PickPlaceSimulationProtocol,
    scene_id: str,
    object_id: str,
    run_id: str,
) -> dict[str, Any]:
    before_state = simulation.observe_scene(instruction=f"probe {object_id}")
    before_position = before_state.object_positions[object_id]
    approach_position = (
        before_position[0] - 0.055,
        before_position[1],
        simulation.config.table_top_z + 0.045,
    )
    simulation.open_gripper()
    simulation.teleport_end_effector(approach_position)
    simulation.apply_skill_action(
        delta_position=(0.11, 0.0, 0.0),
        gripper_command=1.0,
        action_steps=48,
        object_name=object_id,
    )
    simulation.simulate_steps(48)
    after_state = simulation.observe_scene(instruction=f"probe {object_id}")
    after_position = after_state.object_positions[object_id]
    displacement_xy = math.dist(before_position[:2], after_position[:2])

    return {
        "run_id": run_id,
        "scene_id": scene_id,
        "object_id": object_id,
        "probe_action": "lateral_push",
        "contact_region": "side_center",
        "before_position": _vec_to_list(before_position),
        "after_position": _vec_to_list(after_position),
        "displacement_xy": float(displacement_xy),
        "before_world_state": _world_state_to_dict(before_state),
        "after_world_state": _world_state_to_dict(after_state),
    }


def _build_artifact_payload(
    *,
    scene_id: str,
    object_id: str,
    evidence_path: str,
    evidence_payload: dict[str, Any],
) -> dict[str, Any]:
    displacement_xy = float(evidence_payload["displacement_xy"])
    pushable_confidence = _confidence_from_displacement(displacement_xy)
    graspable_confidence = 0.88
    uncertainty_score = max(0.0, min(1.0, 1.0 - max(pushable_confidence, graspable_confidence)))
    if displacement_xy < 0.005:
        uncertainty_score = max(uncertainty_score, 0.55)

    return {
        "scene_id": scene_id,
        "object_id": object_id,
        "object_category": "rigid_block",
        "property_belief": {
            "mass": {"label": "light", "confidence": 0.78},
            "friction": {"label": "medium", "confidence": 0.72},
            "joint_type": {"label": "none", "confidence": 0.95},
        },
        "affordance_candidates": [
            {"name": "graspable", "confidence": graspable_confidence},
            {"name": "pushable", "confidence": pushable_confidence},
        ],
        "uncertainty_score": uncertainty_score,
        "recommended_probe": "lateral_push",
        "contact_region": "side_center",
        "skill_constraints": {
            "preferred_skill": "pick",
            "max_force": 12.0,
            "approach_axis": "top_down",
        },
        "evidence_path": evidence_path,
    }


def _confidence_from_displacement(displacement_xy: float) -> float:
    return max(0.1, min(0.95, 0.35 + displacement_xy / 0.08))


def _build_manifest(*, scene_id: str, object_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "source": "mujoco_probe_v1",
        "objects": [
            {
                "object_id": object_id,
                "object_category": "rigid_block",
            }
            for object_id in object_ids
        ],
    }


def _world_state_to_dict(state: Any) -> dict[str, Any]:
    return {
        "instruction": state.instruction,
        "object_positions": {
            object_id: _vec_to_list(position)
            for object_id, position in state.object_positions.items()
        },
        "zone_positions": {
            zone_id: _vec_to_list(position)
            for zone_id, position in state.zone_positions.items()
        },
        "end_effector_position": _vec_to_list(state.end_effector_position),
        "held_object_name": state.held_object_name,
    }


def _vec_to_list(vector: Sequence[float]) -> list[float]:
    return [float(value) for value in vector]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
