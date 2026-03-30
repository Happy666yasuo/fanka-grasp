# Franka Cube Grasp — HER Buffer Diagnostic
# Conda env: beyondmimic (Python 3.10)
"""
Verify HER replay buffer is filling correctly.

Creates SAC+HER agent, runs a few steps, then inspects the buffer state.

Usage:
    conda activate beyondmimic
    cd ~/Desktop/beyondmimic/my_practice/franka_cube_grasp
    python scripts/check_her_buffer.py --steps 500 --headless

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

parser = argparse.ArgumentParser(description="Franka Grasp — HER Buffer Check")
parser.add_argument("--num_envs", type=int, default=4, help="Number of envs.")
parser.add_argument("--steps", type=int, default=500, help="Number of steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------------------------------------------------
# 2. Imports (after AppLauncher)
# -------------------------------------------------------------------
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import envs  # noqa: F401
from envs.franka_grasp_env_cfg import FrankaGraspEnvCfg, SparseRewardsCfg
from agents.her_wrapper import HERGoalVecEnvWrapper


def main() -> None:
    """Check HER buffer fill status."""
    print("=" * 60)
    print("HER Buffer Diagnostic")
    print("=" * 60)

    # Create environment
    cfg = FrankaGraspEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.rewards = SparseRewardsCfg()  # HER works best with sparse rewards

    env = ManagerBasedRLEnv(cfg=cfg)
    sb3_env = Sb3VecEnvWrapper(env)
    her_env = HERGoalVecEnvWrapper(sb3_env)

    print(f"[INFO] Obs space: {her_env.observation_space}")
    print(f"[INFO] Act space: {her_env.action_space}")

    # Create SAC + HER
    # Episode length = episode_length_s / dt / decimation = 5.0 / 0.01 / 2 = 250
    max_ep_steps = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
    learning_starts = max_ep_steps * args_cli.num_envs

    model = SAC(
        "MultiInputPolicy",
        her_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        buffer_size=10_000,  # Small buffer for diagnostic
        learning_rate=3e-4,
        batch_size=64,
        learning_starts=learning_starts,
        verbose=0,
    )

    # Run enough steps: at least learning_starts + some extra for training
    actual_steps = max(args_cli.steps, learning_starts + 500)
    print(f"[INFO] Running {actual_steps} steps to fill HER buffer "
          f"(learning_starts={learning_starts})...")
    model.learn(total_timesteps=actual_steps, log_interval=None)

    # Inspect buffer
    buffer = model.replay_buffer
    assert isinstance(buffer, HerReplayBuffer), "Buffer is not HerReplayBuffer!"

    buf_size = buffer.size()
    buf_full = buffer.full
    buf_pos = buffer.pos

    print("=" * 60)
    print("HER Buffer Status")
    print("=" * 60)
    print(f"  Buffer class    : {type(buffer).__name__}")
    print(f"  Max size        : {buffer.buffer_size}")
    print(f"  Current size    : {buf_size}")
    print(f"  Position (pos)  : {buf_pos}")
    print(f"  Full            : {buf_full}")
    print(f"  n_sampled_goal  : {buffer.n_sampled_goal}")
    print(f"  goal_strategy   : {buffer.goal_selection_strategy}")

    # Try sampling a batch
    if buf_size > 64:
        try:
            batch = buffer.sample(64)
            print(f"\n  Sample batch OK  : obs keys = {list(batch.observations.keys())}")
            for k, v in batch.observations.items():
                print(f"    {k:20s} shape={v.shape}")
            print(f"    actions          shape={batch.actions.shape}")
            print(f"    rewards          shape={batch.rewards.shape}")
            print(f"    dones            shape={batch.dones.shape}")
            print("  ✅ Buffer sampling successful!")
        except Exception as e:
            print(f"  ❌ Buffer sampling failed: {e}")
    else:
        print(f"  ⚠️ Buffer too small to sample (need > 64, got {buf_size})")

    # Check reward distribution in buffer
    if buf_size > 0:
        try:
            batch = buffer.sample(min(buf_size, 1000))
            rewards = batch.rewards.cpu().numpy().flatten()
            print(f"\n  Reward stats:")
            print(f"    min={rewards.min():.4f}, max={rewards.max():.4f}, "
                  f"mean={rewards.mean():.4f}, std={rewards.std():.4f}")
            n_success = np.sum(rewards == 0.0)
            n_fail = np.sum(rewards == -1.0)
            print(f"    success (r=0): {n_success}/{len(rewards)}")
            print(f"    fail (r=-1):   {n_fail}/{len(rewards)}")
        except Exception as e:
            print(f"  ⚠️ Could not analyze rewards: {e}")

    print("=" * 60)
    print("[INFO] HER buffer diagnostic complete.")

    her_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
