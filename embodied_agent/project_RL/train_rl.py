"""
Pure RL training for robotic pick task with dense reward shaping.

Supports:  PPO  /  SAC  /  TD3   (all from stable-baselines3)

Usage examples
--------------
# PPO (recommended first attempt — most stable):
python train_rl.py --algo ppo --timesteps 300000

# SAC with dense reward (was 0% with sparse reward & 30k steps):
python train_rl.py --algo sac --timesteps 200000

# TD3:
python train_rl.py --algo td3 --timesteps 200000

# With randomised object positions:
python train_rl.py --algo ppo --timesteps 500000 --randomize
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# allow `from dense_pick_env import …` when running from project_RL/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense_pick_env import DensePickEnv  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
#  Evaluation callback — tracks success rate (not just mean reward)
# ═══════════════════════════════════════════════════════════════════
class SuccessRateCallback(BaseCallback):
    """Periodically evaluate the policy and save the best model."""

    def __init__(
        self,
        eval_env: DensePickEnv,
        eval_freq: int = 5000,
        n_eval_episodes: int = 50,
        save_path: str | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_success_rate = -1.0
        self.best_step = 0
        self.eval_log: list[dict] = []
        self._last_eval_step = 0
        self._log_file = None
        self._stop_threshold = 0.95   # early stopping threshold

    # ── called every env step ───────────────────────────────────────
    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps

        successes = 0
        total_reward = 0.0
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(
                    action
                )
                ep_reward += reward
                done = terminated or truncated
            total_reward += ep_reward
            if info.get("is_success", False):
                successes += 1

        sr = successes / self.n_eval_episodes
        mr = total_reward / self.n_eval_episodes

        self.eval_log.append(
            {
                "timestep": int(self.num_timesteps),
                "success_rate": round(sr, 4),
                "mean_reward": round(mr, 2),
            }
        )

        msg = (f"[RL] Step {self.num_timesteps:>7d}: "
               f"success={sr * 100:5.1f}%  reward={mr:7.2f}")
        if self.verbose:
            print(msg, flush=True)
        # write to log file for reliable monitoring
        if self._log_file is None and self.save_path:
            self._log_file = open(os.path.join(self.save_path, "train.log"), "a")
        if self._log_file:
            self._log_file.write(msg + "\n")
            self._log_file.flush()

        if sr > self.best_success_rate:
            self.best_success_rate = sr
            self.best_step = int(self.num_timesteps)
            if self.save_path:
                self.model.save(os.path.join(self.save_path, "best_model"))
            best_msg = f"[RL] ★ New best! success={sr * 100:.1f}%"
            if self.verbose:
                print(best_msg, flush=True)
            if self._log_file:
                self._log_file.write(best_msg + "\n")
                self._log_file.flush()

        # early stopping when success rate exceeds threshold
        if sr >= self._stop_threshold:
            stop_msg = (f"[RL] Early stopping at step {self.num_timesteps} "
                        f"(success={sr*100:.1f}% >= {self._stop_threshold*100:.0f}%)")
            if self.verbose:
                print(stop_msg, flush=True)
            if self._log_file:
                self._log_file.write(stop_msg + "\n")
                self._log_file.flush()
            return False   # stop training

        return True
# ═══════════════════════════════════════════════════════════════════
ALGO_REGISTRY: dict[str, type] = {"ppo": PPO, "sac": SAC, "td3": TD3}


def build_model(algo: str, env, seed: int):
    """Create an SB3 model with tuned hyperparameters."""
    common_kw = dict(verbose=0, seed=seed)

    if algo == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,          # small entropy — task needs precision
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            **common_kw,
        )

    if algo == "sac":
        return SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=2,
            ent_coef=0.01,            # fixed entropy — "auto" causes instability
            policy_kwargs=dict(net_arch=[256, 256]),
            **common_kw,
        )

    if algo == "td3":
        return TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=500,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=2,
            policy_kwargs=dict(net_arch=[256, 256]),
            **common_kw,
        )

    raise ValueError(f"Unknown algorithm: {algo}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="RL pick training")
    parser.add_argument(
        "--algo",
        choices=list(ALGO_REGISTRY.keys()),
        default="ppo",
        help="RL algorithm (default: ppo)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=300_000, help="Total training steps"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomise object/zone positions each episode",
    )
    parser.add_argument(
        "--no-staging",
        action="store_true",
        help="Disable gripper staging (start from home instead)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel envs (PPO only, default: 4)",
    )
    parser.add_argument("--eval-freq", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=30,
        help="Physics steps per env step (default: 30, was 60)",
    )
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    staging = not args.no_staging
    n_envs = args.n_envs if args.algo == "ppo" else 1

    # ── output directory ────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_label = args.run_name or f"{args.algo}_pick"
    out_dir = os.path.join("outputs", "rl_training", f"{timestamp}_{run_label}")
    os.makedirs(out_dir, exist_ok=True)

    # ── environments ────────────────────────────────────────────────
    env_kwargs = dict(
        randomize=args.randomize,
        use_staging=staging,
        max_steps=20,
        action_scale=0.04,
        action_repeat=args.action_repeat,
    )

    def make_env(seed: int):
        def _init():
            e = DensePickEnv(**env_kwargs)
            e = Monitor(e)
            return e
        return _init

    train_env = DummyVecEnv([make_env(args.seed + i) for i in range(n_envs)])
    eval_env = DensePickEnv(**env_kwargs)

    # ── model ───────────────────────────────────────────────────────
    model = build_model(args.algo, train_env, args.seed)

    # ── callback ────────────────────────────────────────────────────
    callback = SuccessRateCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        save_path=out_dir,
    )

    # ── train ───────────────────────────────────────────────────────
    print(
        f"[RL] Algorithm : {args.algo.upper()}\n"
        f"[RL] Timesteps : {args.timesteps:,}\n"
        f"[RL] Envs      : {n_envs}\n"
        f"[RL] Randomize : {args.randomize}\n"
        f"[RL] Staging   : {staging}\n"
        f"[RL] Output    : {out_dir}\n",
        flush=True,
    )

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    elapsed = time.time() - t0

    # ── save final model + summary ──────────────────────────────────
    model.save(os.path.join(out_dir, "final_model"))

    summary = {
        "algorithm": args.algo,
        "total_timesteps": args.timesteps,
        "wall_time_seconds": round(elapsed, 1),
        "seed": args.seed,
        "randomize": args.randomize,
        "staging": staging,
        "env_kwargs": {k: v for k, v in env_kwargs.items()},
        "best_success_rate": callback.best_success_rate,
        "best_step": callback.best_step,
        "eval_log": callback.eval_log,
        "run_dir": os.path.abspath(out_dir),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"\n[RL] ──── Training complete ────\n"
        f"[RL] Best success rate : {callback.best_success_rate * 100:.1f}%  "
        f"(step {callback.best_step})\n"
        f"[RL] Wall time         : {elapsed:.0f}s\n"
        f"[RL] Output            : {out_dir}\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
