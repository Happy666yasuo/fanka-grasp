from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from embodied_agent.experiment import ensure_training_run_dir, load_yaml_config, save_json, save_yaml
from embodied_agent.reporting import plot_success_rate_curve
from embodied_agent.rl_envs import PickPlaceSkillEnv, SkillEnvSettings
from embodied_agent.rl_support import SkillPolicySpec, get_algorithm_class, normalize_algorithm_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CurriculumStage:
    until_timestep: int
    randomize_probability: float
    label: str


class RandomizationCurriculumCallback(BaseCallback):
    def __init__(
        self,
        train_env: PickPlaceSkillEnv,
        stages: list[CurriculumStage],
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.train_env = train_env
        self.stages = stages
        self.active_stage_index = -1

    def _on_training_start(self) -> None:
        self._apply_stage()

    def _on_step(self) -> bool:
        self._apply_stage()
        self.logger.record("curriculum/randomize_probability", self.train_env.get_randomize_probability())
        self.logger.record("curriculum/stage_index", float(self.active_stage_index))
        return True

    def _apply_stage(self) -> None:
        if not self.stages:
            return

        stage_index = self._resolve_stage_index(self.num_timesteps)
        if stage_index == self.active_stage_index:
            return

        stage = self.stages[stage_index]
        self.train_env.set_randomize_probability(stage.randomize_probability)
        self.active_stage_index = stage_index
        if self.verbose > 0:
            print(
                "[curriculum] "
                f"step={self.num_timesteps} stage={stage.label} "
                f"randomize_probability={stage.randomize_probability:.2f}",
                flush=True,
            )

    def _resolve_stage_index(self, num_timesteps: int) -> int:
        for index, stage in enumerate(self.stages):
            if num_timesteps <= stage.until_timestep:
                return index
        return len(self.stages) - 1


class SkillSuccessEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env,
        best_model_save_path: str,
        log_path: str,
        eval_freq: int,
        n_eval_episodes: int,
        deterministic: bool = True,
        early_stop_success_rate: float | None = None,
        verbose: int = 1,
    ) -> None:
        self.single_eval_env = eval_env
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=False,
            verbose=verbose,
        )
        self.best_model_dir = Path(best_model_save_path)
        self.eval_log_dir = Path(log_path)
        self.early_stop_success_rate = early_stop_success_rate
        self.best_success_rate = -1.0
        self.best_mean_reward = float("-inf")
        self.step_history: list[int] = []
        self.reward_history: list[list[float]] = []
        self.success_history: list[list[float]] = []
        self.length_history: list[list[int]] = []

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        episode_rewards: list[float] = []
        episode_successes: list[float] = []
        episode_lengths: list[int] = []

        for episode_index in range(self.n_eval_episodes):
            observation, _ = self.single_eval_env.reset(seed=episode_index)
            done = False
            total_reward = 0.0
            episode_length = 0
            info: dict[str, Any] = {}

            while not done:
                action, _ = self.model.predict(observation, deterministic=self.deterministic)
                observation, reward, terminated, truncated, info = self.single_eval_env.step(action)
                total_reward += float(reward)
                episode_length += 1
                done = bool(terminated or truncated)

            episode_rewards.append(total_reward)
            episode_successes.append(1.0 if info.get("is_success", info.get("success", False)) else 0.0)
            episode_lengths.append(episode_length)

        mean_reward = float(np.mean(episode_rewards))
        success_rate = float(np.mean(episode_successes))

        self.last_mean_reward = mean_reward
        self.step_history.append(self.num_timesteps)
        self.reward_history.append(episode_rewards)
        self.success_history.append(episode_successes)
        self.length_history.append(episode_lengths)

        if success_rate > self.best_success_rate or (
            np.isclose(success_rate, self.best_success_rate) and mean_reward > self.best_mean_reward
        ):
            self.best_success_rate = success_rate
            self.best_mean_reward = mean_reward
            self.best_model_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.best_model_dir / "best_model"))

        self.eval_log_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.eval_log_dir / "evaluations.npz",
            timesteps=np.asarray(self.step_history, dtype=np.int64),
            results=np.asarray(self.reward_history, dtype=np.float32),
            successes=np.asarray(self.success_history, dtype=np.float32),
            episode_lengths=np.asarray(self.length_history, dtype=np.int32),
        )

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/mean_ep_length", float(np.mean(episode_lengths)))

        if self.verbose > 0:
            print(
                f"[eval] step={self.num_timesteps} success={success_rate:.2%} mean_reward={mean_reward:.2f}",
                flush=True,
            )

        if self.early_stop_success_rate is not None and success_rate >= self.early_stop_success_rate:
            if self.verbose > 0:
                print(
                    f"[eval] early stop at step={self.num_timesteps} success={success_rate:.2%}",
                    flush=True,
                )
            return False

        return True

    def evaluation_log(self) -> list[dict[str, Any]]:
        return [
            {
                "timestep": timestep,
                "mean_reward": float(np.mean(self.reward_history[index])),
                "success_rate": float(np.mean(self.success_history[index])),
                "mean_episode_length": float(np.mean(self.length_history[index])),
            }
            for index, timestep in enumerate(self.step_history)
        ]


def _maybe_apply_bc_actor_initialization(
    model: Any,
    config: dict[str, Any],
    config_path: Path,
    algorithm_name: str,
) -> dict[str, Any] | None:
    raw_bc_init = config.get("bc_init")
    if raw_bc_init is None:
        return None
    if not isinstance(raw_bc_init, dict):
        raise ValueError("bc_init must be a dictionary when provided.")
    if algorithm_name != "sac":
        raise ValueError("bc_init is only supported for SAC training runs.")

    manifest_raw_path = raw_bc_init.get("manifest_path")
    if manifest_raw_path is None:
        raise ValueError("bc_init.manifest_path is required.")
    manifest_path = _resolve_path(manifest_raw_path, config_path.parent)
    bc_policy, bc_meta, bc_model_path = _load_bc_policy_from_manifest(manifest_path)
    log_std_value = raw_bc_init.get("log_std")
    applied_stats = _initialize_sac_actor_from_bc_policy(
        model.actor,
        bc_policy,
        log_std_value=None if log_std_value is None else float(log_std_value),
    )
    print(
        "[bc init] Initialized SAC actor from "
        f"{bc_model_path} using manifest {manifest_path}",
        flush=True,
    )
    return {
        "manifest_path": str(manifest_path.resolve()),
        "model_path": str(bc_model_path.resolve()),
        "hidden": list(bc_meta.get("hidden", [128, 128])),
        "obs_dim": int(bc_meta["obs_dim"]),
        "act_dim": int(bc_meta["act_dim"]),
        **applied_stats,
    }


def _load_bc_policy_from_manifest(manifest_path: Path):
    from embodied_agent.bc_training import BCPolicy

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("algorithm", "")).strip().lower() != "bc":
        raise ValueError(f"BC initialization requires a BC manifest: {manifest_path}")
    model_path = _resolve_path(manifest.get("model_path", "best_policy.pt"), manifest_path.parent)
    meta_path = model_path.parent / "bc_model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"BC metadata file not found: {meta_path}")
    bc_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bc_policy = BCPolicy(
        obs_dim=int(bc_meta["obs_dim"]),
        act_dim=int(bc_meta["act_dim"]),
        hidden=list(bc_meta.get("hidden", [128, 128])),
    )
    bc_policy.load_state_dict(torch.load(str(model_path), map_location="cpu", weights_only=True))
    bc_policy.eval()
    return bc_policy, bc_meta, model_path


def _initialize_sac_actor_from_bc_policy(
    actor: Any,
    bc_policy: Any,
    log_std_value: float | None = None,
) -> dict[str, Any]:
    from torch import nn

    bc_linears = [module for module in bc_policy.net if isinstance(module, nn.Linear)]
    actor_hidden_linears = [module for module in actor.latent_pi if isinstance(module, nn.Linear)]
    if len(bc_linears) < 1:
        raise ValueError("BC policy does not contain any linear layers.")
    if len(actor_hidden_linears) + 1 != len(bc_linears):
        raise ValueError(
            "BC/SAC architecture mismatch: "
            f"BC has {len(bc_linears)} linear layers, SAC actor has {len(actor_hidden_linears) + 1}."
        )

    with torch.no_grad():
        for bc_layer, actor_layer in zip(bc_linears[:-1], actor_hidden_linears):
            if actor_layer.weight.shape != bc_layer.weight.shape:
                raise ValueError(
                    "BC/SAC hidden layer shape mismatch: "
                    f"{tuple(bc_layer.weight.shape)} vs {tuple(actor_layer.weight.shape)}"
                )
            actor_layer.weight.copy_(bc_layer.weight)
            actor_layer.bias.copy_(bc_layer.bias)

        bc_output_layer = bc_linears[-1]
        if actor.mu.weight.shape != bc_output_layer.weight.shape:
            raise ValueError(
                "BC/SAC output layer shape mismatch: "
                f"{tuple(bc_output_layer.weight.shape)} vs {tuple(actor.mu.weight.shape)}"
            )
        actor.mu.weight.copy_(bc_output_layer.weight)
        actor.mu.bias.copy_(bc_output_layer.bias)

        if log_std_value is not None:
            actor.log_std.weight.zero_()
            actor.log_std.bias.fill_(float(log_std_value))

    return {
        "applied": True,
        "log_std_initialized": log_std_value is not None,
        "log_std_value": float(log_std_value) if log_std_value is not None else None,
    }


def train_from_config(
    config_path: Path,
    total_timesteps_override: int | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    skill_name = str(config.get("skill", "pick"))
    algorithm_name = normalize_algorithm_name(str(config.get("algo", "sac")))
    seed = int(config.get("seed", 7))
    total_timesteps = int(total_timesteps_override or config.get("total_timesteps", 50_000))
    output_root = _resolve_path(config.get("output_root", "outputs"), PROJECT_ROOT)
    run_dir = ensure_training_run_dir(output_root, skill_name, algorithm_name, run_name=run_name)

    env_settings = SkillEnvSettings.from_config(
        skill_name,
        config.get("env"),
        base_dir=config_path.parent,
    )
    eval_settings = SkillEnvSettings.from_config(
        skill_name,
        config.get("eval_env", config.get("env")),
        base_dir=config_path.parent,
    )
    curriculum_stages = _load_curriculum_stages(config.get("curriculum"), total_timesteps)
    set_random_seed(seed)

    train_skill_env = PickPlaceSkillEnv(settings=env_settings, gui=False)
    eval_skill_env = PickPlaceSkillEnv(settings=eval_settings, gui=False)
    train_env = Monitor(train_skill_env)
    eval_env = Monitor(eval_skill_env)

    algorithm_class = get_algorithm_class(algorithm_name)
    algo_kwargs = dict(config.get("algo_kwargs", {}))
    policy_name = str(config.get("policy", "MlpPolicy"))
    model = algorithm_class(
        policy_name,
        train_env,
        seed=seed,
        verbose=1,
        device="cpu",
        **algo_kwargs,
    )
    bc_init_stats = _maybe_apply_bc_actor_initialization(
        model,
        config,
        config_path,
        algorithm_name,
    )
    logger = configure(str(run_dir / "sb3_logs"), ["stdout", "csv"])
    model.set_logger(logger)

    eval_config = dict(config.get("eval", {}))
    eval_callback = SkillSuccessEvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval"),
        eval_freq=int(eval_config.get("eval_freq", 2_000)),
        n_eval_episodes=int(eval_config.get("episodes", 10)),
        deterministic=bool(eval_config.get("deterministic", True)),
        early_stop_success_rate=(
            float(eval_config["early_stop_success_rate"])
            if "early_stop_success_rate" in eval_config
            else None
        ),
    )
    callbacks: list[BaseCallback] = []
    if curriculum_stages:
        callbacks.append(RandomizationCurriculumCallback(train_skill_env, curriculum_stages, verbose=1))
    callbacks.append(eval_callback)
    train_callback: BaseCallback = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)

    expert_episodes = int(config.get("expert_episodes", 0))
    expert_stats: dict[str, float] | None = None
    if expert_episodes > 0:
        from embodied_agent.expert_warmstart import (
            collect_expert_episodes,
            get_expert_policy,
            prefill_replay_buffer,
        )
        expert_fn = get_expert_policy(skill_name)
        expert_env = PickPlaceSkillEnv(settings=env_settings, gui=False)
        transitions, expert_stats = collect_expert_episodes(
            expert_env,
            expert_fn,
            expert_episodes,
            action_scale=env_settings.action_scale,
            noise_std=float(config.get("expert_noise_std", 0.05)),
        )
        expert_env.close()
        n_added = prefill_replay_buffer(model, transitions)
        print(f"[expert warm-start] {expert_stats}")
        print(f"[expert warm-start] Added {n_added} transitions to replay buffer")

        pretrain_steps = int(config.get("expert_pretrain_steps", 0))
        if pretrain_steps > 0 and n_added > 0:
            model.num_timesteps = model.learning_starts + 1
            model.train(gradient_steps=pretrain_steps, batch_size=algo_kwargs.get("batch_size", 256))
            model.num_timesteps = 0
            print(f"[expert warm-start] Pre-trained {pretrain_steps} gradient steps on expert data")

    resolved_config = {
        **config,
        "algo": algorithm_name,
        "skill": skill_name,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "output_root": str(output_root),
    }
    save_yaml(run_dir / "resolved_config.yaml", resolved_config)
    model.learn(total_timesteps=total_timesteps, callback=train_callback, progress_bar=False)
    if eval_callback.step_history:
        save_json(run_dir / "eval" / "success_eval_log.json", eval_callback.evaluation_log())

    final_model_path = run_dir / "final_model"
    model.save(str(final_model_path))

    policy_manifest = SkillPolicySpec(
        skill_name=skill_name,
        algorithm=algorithm_name,
        model_path=(final_model_path.with_suffix(".zip")).resolve(),
        max_steps=env_settings.max_steps,
        action_repeat=env_settings.action_repeat,
        action_scale=env_settings.action_scale,
        deterministic=True,
        use_staging=env_settings.use_staging,
    )
    save_json(run_dir / "policy_manifest.json", policy_manifest.to_dict())

    best_model_path = run_dir / "best_model" / "best_model.zip"
    best_policy_manifest_path: Path | None = None
    if best_model_path.exists():
        best_policy_manifest = SkillPolicySpec(
            skill_name=skill_name,
            algorithm=algorithm_name,
            model_path=best_model_path.resolve(),
            max_steps=env_settings.max_steps,
            action_repeat=env_settings.action_repeat,
            action_scale=env_settings.action_scale,
            deterministic=True,
            use_staging=env_settings.use_staging,
        )
        best_policy_manifest_path = run_dir / "best_policy_manifest.json"
        save_json(best_policy_manifest_path, best_policy_manifest.to_dict())

    evaluations_path = run_dir / "eval" / "evaluations.npz"
    success_curve_path: Path | None = None
    if evaluations_path.exists():
        success_curve_path = run_dir / "success_rate_curve.png"
        plot_success_rate_curve(evaluations_path, success_curve_path, title=f"{skill_name.upper()} {algorithm_name.upper()} success rate")

    summary = {
        "skill": skill_name,
        "algorithm": algorithm_name,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "run_dir": str(run_dir.resolve()),
        "final_model_path": str(policy_manifest.model_path),
        "policy_manifest_path": str((run_dir / "policy_manifest.json").resolve()),
        "best_model_path": str(best_model_path.resolve()) if best_model_path.exists() else None,
        "best_policy_manifest_path": str(best_policy_manifest_path.resolve()) if best_policy_manifest_path else None,
        "success_curve_path": str(success_curve_path.resolve()) if success_curve_path else None,
        "env_reset_source": env_settings.reset_source,
        "env_post_pick_states_jsonl": (
            str(env_settings.post_pick_states_jsonl)
            if env_settings.post_pick_states_jsonl is not None
            else None
        ),
        "env_task_pool_size": len(env_settings.task_pool),
        "env_task_pool": [task.to_dict() for task in env_settings.task_pool],
        "eval_reset_source": eval_settings.reset_source,
        "eval_post_pick_states_jsonl": (
            str(eval_settings.post_pick_states_jsonl)
            if eval_settings.post_pick_states_jsonl is not None
            else None
        ),
        "eval_task_pool_size": len(eval_settings.task_pool),
        "eval_task_pool": [task.to_dict() for task in eval_settings.task_pool],
        "bc_init": bc_init_stats,
        "best_success_rate": eval_callback.best_success_rate if eval_callback.best_success_rate >= 0.0 else None,
        "best_mean_reward": eval_callback.best_mean_reward if eval_callback.best_success_rate >= 0.0 else None,
        "curriculum_stages": [
            {
                "label": stage.label,
                "until_timestep": stage.until_timestep,
                "randomize_probability": stage.randomize_probability,
            }
            for stage in curriculum_stages
        ],
        "expert_warmstart": expert_stats,
    }
    save_json(run_dir / "training_summary.json", summary)
    train_env.close()
    eval_env.close()
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a low-level RL skill for the embodied agent.")
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Optional override for training timesteps.")
    parser.add_argument("--run-name", default=None, help="Optional label appended to the output directory.")
    args = parser.parse_args(argv)

    summary = train_from_config(
        config_path=Path(args.config).resolve(),
        total_timesteps_override=args.total_timesteps,
        run_name=args.run_name,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _resolve_path(raw_path: object, base_dir: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_curriculum_stages(raw_curriculum: object, total_timesteps: int) -> list[CurriculumStage]:
    if not isinstance(raw_curriculum, dict):
        return []

    raw_stages = raw_curriculum.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        return []

    stages: list[CurriculumStage] = []
    previous_until_timestep = -1
    for index, raw_stage in enumerate(raw_stages):
        if not isinstance(raw_stage, dict):
            raise ValueError("curriculum stages must be dictionaries")

        until_timestep = int(raw_stage.get("until_timestep", total_timesteps))
        if until_timestep <= previous_until_timestep:
            raise ValueError("curriculum until_timestep values must be strictly increasing")

        randomize_probability = float(np.clip(raw_stage.get("randomize_probability", 0.0), 0.0, 1.0))
        label = str(raw_stage.get("label", f"stage_{index + 1}"))
        stages.append(
            CurriculumStage(
                until_timestep=until_timestep,
                randomize_probability=randomize_probability,
                label=label,
            )
        )
        previous_until_timestep = until_timestep

    return stages