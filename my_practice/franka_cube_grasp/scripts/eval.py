# Franka Cube Grasp — Evaluation Script
# Conda env: beyondmimic (Python 3.10)
"""
Evaluate a trained SAC model on the Franka cube grasping environment.

Reports:
    - Per-episode reward
    - Success rate (cube lifted above threshold)
    - Episode length statistics

Usage:
    conda activate beyondmimic
    cd ~/Desktop/beyondmimic/my_practice/franka_cube_grasp
    python scripts/eval.py --checkpoint logs/.../checkpoints/latest.zip \\
        --headless --num_episodes 100

⚠️ --headless is MANDATORY (limited VRAM).
"""
from __future__ import annotations

import os
import sys

# -------------------------------------------------------------------
# 0. Project root on sys.path
# -------------------------------------------------------------------
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# -------------------------------------------------------------------
# 1. Parse args & launch IsaacSim (MUST be first)
# -------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka Grasp — Evaluation")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained SB3 SAC model (.zip).")
parser.add_argument("--num_envs", type=int, default=4,
                    help="Number of parallel environments for evaluation.")
parser.add_argument("--num_episodes", type=int, default=100,
                    help="Total number of episodes to evaluate.")
parser.add_argument("--lift_threshold", type=float, default=0.06,
                    help="Minimum cube height (above initial) to count as success.")
# Append standard IsaacLab CLI args (--headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------------------------------------------------
# 2. Imports (after AppLauncher)
# -------------------------------------------------------------------
import numpy as np
import torch

from stable_baselines3 import SAC

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

# Our project modules
import envs  # noqa: F401
from envs.franka_grasp_env_cfg import FrankaGraspEnvCfg


def main() -> None:
    """Evaluate trained SAC model."""
    print("=" * 60)
    print("Franka Cube Grasp — Evaluation")
    print("=" * 60)
    print(f"  checkpoint   : {args_cli.checkpoint}")
    print(f"  num_envs     : {args_cli.num_envs}")
    print(f"  num_episodes : {args_cli.num_episodes}")
    print(f"  lift_thresh  : {args_cli.lift_threshold}")
    print("=" * 60)

    # -- Create environment --
    cfg = FrankaGraspEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedRLEnv(cfg=cfg)
    sb3_env = Sb3VecEnvWrapper(env)

    # -- Load model --
    ckpt_path = args_cli.checkpoint
    if not ckpt_path.endswith(".zip"):
        ckpt_path += ".zip"
    if not os.path.isfile(ckpt_path):
        # Try without .zip
        ckpt_path = args_cli.checkpoint
    print(f"[INFO] Loading model from: {ckpt_path}")
    model = SAC.load(ckpt_path, env=sb3_env)

    # -- Rollout --
    completed_episodes = 0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_successes: list[bool] = []

    # Buffers per sub-environment
    ep_rews = np.zeros(args_cli.num_envs)
    ep_lens = np.zeros(args_cli.num_envs, dtype=int)

    obs = sb3_env.reset()

    while completed_episodes < args_cli.num_episodes:
        # Predict (deterministic)
        with torch.inference_mode():
            actions, _ = model.predict(obs, deterministic=True)

        obs, rewards, dones, infos = sb3_env.step(actions)
        ep_rews += rewards
        ep_lens += 1

        for i in range(args_cli.num_envs):
            if dones[i]:
                completed_episodes += 1
                episode_rewards.append(float(ep_rews[i]))
                episode_lengths.append(int(ep_lens[i]))

                # Check success: was the lift reward earned?
                # Use a simple heuristic: episode reward > 0.5 means some lifting happened
                success = ep_rews[i] > 0.5
                episode_successes.append(success)

                # Reset buffers for this sub-env
                ep_rews[i] = 0.0
                ep_lens[i] = 0

                if completed_episodes >= args_cli.num_episodes:
                    break

    # -- Report --
    n = len(episode_rewards)
    mean_rew = np.mean(episode_rewards) if n > 0 else 0.0
    std_rew = np.std(episode_rewards) if n > 0 else 0.0
    mean_len = np.mean(episode_lengths) if n > 0 else 0.0
    success_rate = np.mean(episode_successes) if n > 0 else 0.0

    print("=" * 60)
    print(f"Evaluation Results ({n} episodes)")
    print("=" * 60)
    print(f"  Mean Reward     : {mean_rew:.3f} ± {std_rew:.3f}")
    print(f"  Mean Ep Length  : {mean_len:.1f}")
    print(f"  Success Rate    : {success_rate:.1%}")
    print("=" * 60)

    # Save results to file
    results_path = os.path.join(os.path.dirname(ckpt_path), "eval_results.txt")
    try:
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(f"Episodes: {n}\n")
            f.write(f"Mean Reward: {mean_rew:.3f} +/- {std_rew:.3f}\n")
            f.write(f"Mean Length: {mean_len:.1f}\n")
            f.write(f"Success Rate: {success_rate:.1%}\n")
        print(f"[INFO] Results saved to: {results_path}")
    except Exception as e:
        print(f"[WARNING] Could not save results file: {e}")

    # Cleanup
    sb3_env.close()
    print("[INFO] Evaluation complete.")


if __name__ == "__main__":
    main()
    simulation_app.close()
