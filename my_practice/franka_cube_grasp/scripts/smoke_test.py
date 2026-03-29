# Franka Cube Grasp — Smoke Test
# Conda env: beyondmimic (Python 3.10)
"""
Smoke test for the Franka cube grasping project.

Uses IsaacLab AppLauncher to initialize the simulation runtime before importing
any IsaacSim/IsaacLab modules (mandatory pattern).

Usage:
    conda activate beyondmimic
    cd /home/happywindman/Desktop/beyondmimic/my_practice/franka_cube_grasp
    python scripts/smoke_test.py --headless --num_envs 2 --num_steps 100
"""
from __future__ import annotations

import os
import sys

# -------------------------------------------------------------------
# 0. Ensure the project root is importable
# -------------------------------------------------------------------
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# -------------------------------------------------------------------
# 1. Parse args & launch IsaacSim AppLauncher (MUST be done first)
# -------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka Grasp — Smoke Test")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=100, help="Random rollout steps.")
# Append standard IsaacLab CLI args (--headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------------------------------------------------
# 2. Now safe to import IsaacLab / env modules
# -------------------------------------------------------------------
import torch

from isaaclab.envs import ManagerBasedRLEnv

# Trigger gym.register via envs/__init__.py
import envs  # noqa: F401
from envs.franka_grasp_env_cfg import FrankaGraspEnvCfg


def main() -> None:
    """Instantiate env, print spaces, run random steps."""
    print("=" * 60)
    print("Franka Cube Grasp — Smoke Test (Phase 1)")
    print("=" * 60)

    errors: list[str] = []

    # -- Create environment --
    try:
        cfg = FrankaGraspEnvCfg()
        cfg.scene.num_envs = args_cli.num_envs
        env = ManagerBasedRLEnv(cfg=cfg)
        print(f"[✓] Environment created: {args_cli.num_envs} envs")
    except Exception as e:
        errors.append(f"env creation: {e}")
        print(f"[✗] env creation: {e}")
        _report_and_exit(errors)

    # -- Print observation / action spaces --
    try:
        obs_space = env.observation_space
        act_space = env.action_space
        print(f"    obs_space: {obs_space}")
        print(f"    act_space: {act_space}")
    except Exception as e:
        errors.append(f"spaces: {e}")
        print(f"[✗] spaces: {e}")

    # -- Run random rollout --
    try:
        obs, info = env.reset()
        obs_t = obs["policy"]
        print(f"[✓] env.reset() → obs['policy'] shape: {obs_t.shape}")

        for step in range(args_cli.num_steps):
            action = torch.randn(args_cli.num_envs, env.action_space.shape[-1], device=env.device)
            obs, reward, terminated, truncated, info = env.step(action)
            if step == 0:
                print(f"    step 0: reward mean={reward.mean().item():.4f}")

        print(f"[✓] Completed {args_cli.num_steps} random steps")
    except Exception as e:
        errors.append(f"rollout: {e}")
        print(f"[✗] rollout: {e}")
    finally:
        env.close()
        print("[✓] env.close() OK")

    _report_and_exit(errors)


def _report_and_exit(errors: list[str]) -> None:
    """Print summary and exit."""
    print("=" * 60)
    if errors:
        print(f"FAILED — {len(errors)} error(s):")
        for err in errors:
            print(f"  • {err}")
        simulation_app.close()
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED ✅")
        simulation_app.close()
        sys.exit(0)


if __name__ == "__main__":
    main()
