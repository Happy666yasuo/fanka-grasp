from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from embodied_agent.experiment import append_jsonl, ensure_evaluation_run_dir, load_yaml_config, save_json, save_yaml
from embodied_agent.rl_envs import PickPlaceSkillEnv, SkillEnvSettings
from embodied_agent.rl_support import SkillPolicySpec, load_policy_model, policy_spec_from_dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate_pick_from_config(config_path: Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    output_root = _resolve_path(config.get("output_root", "outputs"), PROJECT_ROOT)
    run_dir = ensure_evaluation_run_dir(output_root, label=config_path.stem)
    save_yaml(run_dir / "resolved_config.yaml", config)

    experiment_summaries: list[dict[str, Any]] = []
    total_episode_count = 0
    for experiment in config.get("experiments", []):
        summary, episode_count = _evaluate_skill_experiment(
            experiment=dict(experiment),
            run_dir=run_dir,
            base_dir=config_path.parent,
        )
        experiment_summaries.append(summary)
        total_episode_count += episode_count

    save_json(run_dir / "evaluation_summary.json", {"experiments": experiment_summaries})
    return {
        "run_dir": str(run_dir.resolve()),
        "experiments": experiment_summaries,
        "episode_count": total_episode_count,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Formally evaluate skill manifests on fixed and randomized layouts.")
    parser.add_argument("--config", required=True, help="Path to a YAML formal evaluation config.")
    args = parser.parse_args(argv)

    summary = evaluate_pick_from_config(Path(args.config).resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _evaluate_skill_experiment(
    experiment: dict[str, Any],
    run_dir: Path,
    base_dir: Path,
) -> tuple[dict[str, Any], int]:
    skill_name = str(experiment.get("skill", "pick"))
    experiment_name = str(experiment.get("name", f"{skill_name}_eval"))
    episode_count = int(experiment.get("episodes", 50))
    seed = int(experiment.get("seed", 7))
    randomize = bool(experiment.get("randomize", True))

    if skill_name in experiment:
        skill_config = dict(experiment[skill_name])
    else:
        skill_config = {"manifest_path": experiment["manifest_path"]}

    spec = policy_spec_from_dict(skill_name, skill_config, base_dir=base_dir)
    model = load_policy_model(spec.algorithm, spec.model_path)

    raw_env_config = dict(experiment.get("env", {}))
    env_config = {
        "object_name": raw_env_config.get("object_name", "red_block"),
        "zone_name": raw_env_config.get("zone_name", "green_zone"),
        "max_steps": int(raw_env_config.get("max_steps", spec.max_steps)),
        "action_scale": float(raw_env_config.get("action_scale", spec.action_scale)),
        "action_repeat": int(raw_env_config.get("action_repeat", spec.action_repeat)),
        "randomize": randomize,
        "seed": seed,
        "use_staging": bool(raw_env_config.get("use_staging", spec.use_staging)),
    }
    for optional_key in (
        "post_release_settle_steps",
        "object_x_range",
        "object_y_range",
        "zone_x_range",
        "zone_y_range",
        "object_candidates",
        "zone_candidates",
    ):
        if optional_key in raw_env_config:
            env_config[optional_key] = raw_env_config[optional_key]
    settings = SkillEnvSettings.from_config(skill_name, env_config)
    env = PickPlaceSkillEnv(settings=settings, gui=False)
    metric_keys = _metric_keys_for_skill(skill_name)

    success_values: list[float] = []
    rewards: list[float] = []
    lengths: list[int] = []
    final_metric_values: dict[str, list[float]] = {metric_key: [] for metric_key in metric_keys}

    try:
        for episode_index in range(episode_count):
            observation, info = env.reset(seed=seed + episode_index)
            done = False
            total_reward = 0.0
            episode_length = 0

            while not done:
                action, _ = model.predict(observation, deterministic=spec.deterministic)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                episode_length += 1
                done = bool(terminated or truncated)

            success_value = 1.0 if info.get("is_success", False) else 0.0
            final_metrics = {
                metric_key: float(info.get(metric_key, float("nan")))
                for metric_key in metric_keys
            }

            success_values.append(success_value)
            rewards.append(total_reward)
            lengths.append(episode_length)
            for metric_key, metric_value in final_metrics.items():
                final_metric_values[metric_key].append(metric_value)

            append_jsonl(
                run_dir / f"{experiment_name}_episodes.jsonl",
                {
                    "experiment": experiment_name,
                    "skill": skill_name,
                    "episode": episode_index,
                    "seed": seed + episode_index,
                    "randomize": randomize,
                    "success": bool(success_value),
                    "episode_reward": total_reward,
                    "episode_length": episode_length,
                    **{f"final_{metric_key}": metric_value for metric_key, metric_value in final_metrics.items()},
                },
            )
    finally:
        env.close()

    success_rate = mean(success_values) if success_values else 0.0
    ci_low, ci_high = _wilson_interval(int(sum(success_values)), episode_count)
    summary = {
        "name": experiment_name,
        "skill": skill_name,
        "episodes": episode_count,
        "randomize": randomize,
        "success_rate": success_rate,
        "success_count": int(sum(success_values)),
        "success_ci95_low": ci_low,
        "success_ci95_high": ci_high,
        "mean_episode_reward": mean(rewards) if rewards else 0.0,
        "mean_episode_length": mean(lengths) if lengths else 0.0,
        "skill_policy": spec.to_dict(),
        f"{skill_name}_policy": spec.to_dict(),
    }
    summary.update(
        {
            f"mean_final_{metric_key}": mean(metric_values) if metric_values else float("nan")
            for metric_key, metric_values in final_metric_values.items()
        }
    )
    save_json(run_dir / f"{experiment_name}_summary.json", summary)
    return summary, episode_count


def _metric_keys_for_skill(skill_name: str) -> tuple[str, ...]:
    if skill_name == "place":
        return (
            "object_zone_distance_xy",
            "object_height",
            "holding_flag",
        )

    return (
        "ee_object_distance",
        "lift_progress",
        "holding_flag",
    )


def _wilson_interval(success_count: int, total_count: int, z_value: float = 1.96) -> tuple[float, float]:
    if total_count <= 0:
        return 0.0, 0.0

    phat = success_count / total_count
    denominator = 1.0 + (z_value**2) / total_count
    center = phat + (z_value**2) / (2.0 * total_count)
    radius = z_value * math.sqrt((phat * (1.0 - phat) + (z_value**2) / (4.0 * total_count)) / total_count)
    lower = max(0.0, (center - radius) / denominator)
    upper = min(1.0, (center + radius) / denominator)
    return lower, upper


def _resolve_path(raw_path: object, base_dir: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()