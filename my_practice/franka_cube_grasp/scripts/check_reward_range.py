# Franka Cube Grasp — Reward Range Checker
# Conda env: beyondmimic (Python 3.10)
"""
Diagnostic script: run the environment with random actions for N steps and
report per-term and total reward statistics (min, max, mean, std, NaN, Inf).

Usage:
    conda activate beyondmimic
    cd ~/Desktop/beyondmimic/my_practice/franka_cube_grasp
    python scripts/check_reward_range.py --reward_type shaped --steps 200 --headless

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

parser = argparse.ArgumentParser(description="Franka Grasp — Reward Range Check")
parser.add_argument("--num_envs", type=int, default=4,
                    help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=200,
                    help="Number of environment steps to run.")
parser.add_argument("--reward_type", type=str, default="shaped",
                    choices=["sparse", "shaped", "pbrs"],
                    help="Reward function variant to check.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
# Append standard IsaacLab CLI args (--headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------------------------------------------------
# 2. Imports (after AppLauncher)
# -------------------------------------------------------------------
import torch

from isaaclab.envs import ManagerBasedRLEnv

# Our project modules
import envs  # noqa: F401 — triggers gym.register
from envs.franka_grasp_env_cfg import (
    FrankaGraspEnvCfg,
    SparseRewardsCfg,
    ShapedRewardsCfg,
    PBRSRewardsCfg,
)


def main() -> None:
    """Run environment with random actions and report reward statistics."""

    # -- Create environment --
    cfg = FrankaGraspEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed

    # Switch reward config
    reward_map = {
        "sparse": SparseRewardsCfg,
        "shaped": ShapedRewardsCfg,
        "pbrs": PBRSRewardsCfg,
    }
    if args_cli.reward_type in reward_map:
        cfg.rewards = reward_map[args_cli.reward_type]()

    env = ManagerBasedRLEnv(cfg=cfg)

    print("=" * 60)
    print("Franka Cube Grasp — Reward Range Checker")
    print("=" * 60)
    print(f"  reward_type : {args_cli.reward_type}")
    print(f"  num_envs    : {args_cli.num_envs}")
    print(f"  steps       : {args_cli.steps}")
    print(f"  reward cfg  : {type(cfg.rewards).__name__}")
    print("=" * 60)

    # -- Collect reward data --
    all_rewards: list[torch.Tensor] = []
    per_term_data: dict[str, list[torch.Tensor]] = {}

    env.reset()

    for step_i in range(args_cli.steps):
        # Random actions within action space bounds
        action = torch.randn(args_cli.num_envs, env.action_manager.total_action_dim,
                             device=env.device)
        action = torch.clamp(action, -1.0, 1.0)

        obs, rew, terminated, truncated, info = env.step(action)
        all_rewards.append(rew.clone())

        # Collect per-term rewards from reward manager
        # _step_reward has shape (num_envs, num_terms) — per-step values
        if hasattr(env.reward_manager, "_step_reward"):
            step_rew = env.reward_manager._step_reward  # (num_envs, num_terms)
            term_names = env.reward_manager._term_names
            for idx, term_name in enumerate(term_names):
                if term_name not in per_term_data:
                    per_term_data[term_name] = []
                per_term_data[term_name].append(step_rew[:, idx].clone())

    # -- Aggregate and report --
    all_rewards_t = torch.cat(all_rewards)
    num_nan = torch.isnan(all_rewards_t).sum().item()
    num_inf = torch.isinf(all_rewards_t).sum().item()

    print("\n" + "=" * 60)
    print("TOTAL REWARD STATISTICS")
    print("=" * 60)
    print(f"  Samples   : {all_rewards_t.numel()}")
    print(f"  Min       : {all_rewards_t.min().item():.6f}")
    print(f"  Max       : {all_rewards_t.max().item():.6f}")
    print(f"  Mean      : {all_rewards_t.mean().item():.6f}")
    print(f"  Std       : {all_rewards_t.std().item():.6f}")
    print(f"  NaN count : {num_nan}")
    print(f"  Inf count : {num_inf}")

    if num_nan > 0 or num_inf > 0:
        print("\n⚠️  WARNING: NaN or Inf values detected in rewards!")
    else:
        print("\n✅  No NaN or Inf values — rewards are clean.")

    # -- Per-term breakdown --
    if per_term_data:
        print("\n" + "=" * 60)
        print("PER-TERM REWARD BREAKDOWN")
        print("=" * 60)
        for term_name, term_vals in per_term_data.items():
            if len(term_vals) == 0:
                continue
            term_t = torch.cat(term_vals) if isinstance(term_vals[0], torch.Tensor) else torch.tensor(term_vals)
            t_nan = torch.isnan(term_t).sum().item()
            t_inf = torch.isinf(term_t).sum().item()
            status = "✅" if (t_nan == 0 and t_inf == 0) else "⚠️"
            print(f"  [{status}] {term_name:30s} | "
                  f"min={term_t.min().item():+10.4f}  "
                  f"max={term_t.max().item():+10.4f}  "
                  f"mean={term_t.mean().item():+10.4f}  "
                  f"std={term_t.std().item():8.4f}  "
                  f"NaN={t_nan}  Inf={t_inf}")

    # -- Final verdict --
    print("\n" + "=" * 60)
    if num_nan == 0 and num_inf == 0:
        abs_max = all_rewards_t.abs().max().item()
        if abs_max > 100:
            print(f"⚠️  Reward magnitude is large (|max|={abs_max:.2f}). "
                  "Consider scaling down weights.")
        elif abs_max < 0.001:
            print(f"⚠️  Reward magnitude is very small (|max|={abs_max:.6f}). "
                  "Consider scaling up weights.")
        else:
            print(f"✅  Reward range looks healthy (|max|={abs_max:.4f}).")
    else:
        print("❌  FAILED: NaN/Inf detected — fix reward functions before training.")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
