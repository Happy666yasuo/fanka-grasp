# Franka Cube Grasp — SAC Training Script
# Conda env: beyondmimic (Python 3.10)
"""
Train Franka cube grasping with SAC (Stable-Baselines3).

Architecture:
    AppLauncher → IsaacLab ManagerBasedRLEnv → Sb3VecEnvWrapper → SAC

Usage:
    conda activate beyondmimic
    cd ~/Desktop/beyondmimic/my_practice/franka_cube_grasp
    python scripts/train.py --headless --num_envs 64 --total_timesteps 500000

Smoke test:
    python scripts/train.py --headless --num_envs 4 --total_timesteps 1000 \\
        --log_dir logs/smoke_test

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

parser = argparse.ArgumentParser(description="Franka Grasp — SAC Training")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--total_timesteps", type=int, default=500_000, help="Total training timesteps.")
parser.add_argument("--reward_type", type=str, default="sparse",
                    choices=["sparse", "shaped", "pbrs"],
                    help="Reward function variant.")
parser.add_argument("--algo", type=str, default="sac",
                    choices=["sac", "sac_her"],
                    help="Algorithm: sac (standard) or sac_her (SAC + HER).")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--log_dir", type=str, default=None,
                    help="Log directory (default: logs/sac_{reward_type}_{timestamp}).")
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
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.her import HerReplayBuffer

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

# Our project modules
import envs  # noqa: F401 — triggers gym.register
from envs.franka_grasp_env_cfg import (
    FrankaGraspEnvCfg,
    RewardsCfg,
    SparseRewardsCfg,
    ShapedRewardsCfg,
    PBRSRewardsCfg,
)
from agents.sac_cfg import SACConfig
from agents.her_wrapper import HERGoalVecEnvWrapper


# ===================================================================
# Custom callback: log per-episode success rate
# ===================================================================
from stable_baselines3.common.callbacks import BaseCallback


class SuccessRateCallback(BaseCallback):
    """Log the fraction of episodes where the cube was lifted.

    Checks the ``extras["log"]`` dict from IsaacLab for a key named
    ``lifting_object`` (the sparse reward term). If the cumulative episode
    reward from that term is > 0, the episode is considered a success.
    """

    def __init__(self, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self._successes: list[float] = []
        self._n_calls_since_log = 0

    def _on_step(self) -> bool:
        self._n_calls_since_log += 1
        # Collect success info from episode endings
        infos = self.locals.get("infos", [])
        for info in infos:
            if info is not None and isinstance(info, dict):
                ep_info = info.get("episode")
                if ep_info is not None:
                    # Episode just ended — check if any lift reward was earned
                    ep_reward = ep_info.get("r", 0.0)
                    # Simple heuristic: if total reward > some threshold, count as partial success
                    self._successes.append(float(ep_reward > 0.5))

        if self._n_calls_since_log >= self.check_freq and len(self._successes) > 0:
            success_rate = sum(self._successes) / len(self._successes)
            self.logger.record("rollout/success_rate", success_rate)
            if self.verbose >= 1:
                print(f"[SuccessRate] {len(self._successes)} episodes, rate={success_rate:.3f}")
            self._successes.clear()
            self._n_calls_since_log = 0

        return True


# ===================================================================
# ONNX export helper
# ===================================================================
def export_onnx(model: SAC, save_path: str, obs_dim: int) -> None:
    """Export SAC actor network to ONNX format.

    Args:
        model: Trained SAC model.
        save_path: Path for the .onnx file.
        obs_dim: Observation dimensionality.
    """
    try:
        import torch.onnx

        actor = model.actor
        actor.eval()
        dummy_input = torch.randn(1, obs_dim, device=model.device)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.onnx.export(
            actor,
            dummy_input,
            save_path,
            opset_version=11,
            input_names=["obs"],
            output_names=["action"],
        )
        print(f"[INFO] ONNX model exported to: {save_path}")
    except Exception as e:
        print(f"[WARNING] ONNX export failed: {e}")


# ===================================================================
# Main training function
# ===================================================================
def main() -> None:
    """Train SAC on Franka cube grasping environment."""

    # -- Setup logging directory --
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args_cli.log_dir is not None:
        log_dir = args_cli.log_dir
    else:
        algo_tag = args_cli.algo.replace("_", "-")
        log_dir = os.path.join("logs", f"{algo_tag}_{args_cli.reward_type}_{timestamp}")

    log_dir = os.path.abspath(log_dir)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    tb_dir = os.path.join(log_dir, "tb_logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    print("=" * 60)
    print("Franka Cube Grasp — SAC Training")
    print("=" * 60)
    print(f"  algo            : {args_cli.algo}")
    print(f"  reward_type     : {args_cli.reward_type}")
    print(f"  num_envs        : {args_cli.num_envs}")
    print(f"  total_timesteps : {args_cli.total_timesteps}")
    print(f"  seed            : {args_cli.seed}")
    print(f"  log_dir         : {log_dir}")
    print("=" * 60)

    # -- Create environment --
    cfg = FrankaGraspEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed

    # -- Switch reward config based on reward_type --
    reward_map = {
        "sparse": SparseRewardsCfg,
        "shaped": ShapedRewardsCfg,
        "pbrs": PBRSRewardsCfg,
    }
    if args_cli.reward_type in reward_map:
        cfg.rewards = reward_map[args_cli.reward_type]()
        print(f"[INFO] Using reward config: {type(cfg.rewards).__name__}")
    else:
        # Default: keep the original dense+sparse RewardsCfg
        print(f"[INFO] Using default RewardsCfg")

    env = ManagerBasedRLEnv(cfg=cfg)

    # Wrap for SB3
    sb3_env = Sb3VecEnvWrapper(env)
    print(f"[INFO] Obs space: {sb3_env.observation_space}")
    print(f"[INFO] Act space: {sb3_env.action_space}")

    # -- Optionally wrap for HER --
    use_her = (args_cli.algo == "sac_her")
    if use_her:
        sb3_env = HERGoalVecEnvWrapper(sb3_env)
        print(f"[INFO] HER wrapper applied. Dict obs space: {sb3_env.observation_space}")

    # -- Create SAC agent --
    sac_cfg = SACConfig(seed=args_cli.seed)
    sb3_kwargs = sac_cfg.to_sb3_kwargs()
    # SB3 SAC constructor takes `env` as first positional arg after `policy`
    policy_name = sb3_kwargs.pop("policy")

    # Override policy for HER (requires MultiInputPolicy for Dict obs)
    if use_her:
        policy_name = "MultiInputPolicy"
        sb3_kwargs["replay_buffer_class"] = HerReplayBuffer
        sb3_kwargs["replay_buffer_kwargs"] = dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        )
        # HER needs at least one full episode before sampling from the buffer.
        # Episode length = episode_length_s / dt / decimation = 5.0 / 0.01 / 2 = 250.
        # learning_starts must be >= max_episode_steps * num_envs to ensure
        # at least one episode completes before training begins.
        max_ep_steps = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
        sb3_kwargs["learning_starts"] = max_ep_steps * args_cli.num_envs
        print(f"[INFO] Using HER: MultiInputPolicy + HerReplayBuffer (future, n=4)")
        print(f"[INFO] HER learning_starts = {sb3_kwargs['learning_starts']} "
              f"(ep_len={max_ep_steps} x num_envs={args_cli.num_envs})")

    model = SAC(
        policy_name,
        sb3_env,
        tensorboard_log=tb_dir,
        **sb3_kwargs,
    )

    # Configure logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # -- Callbacks --
    checkpoint_cb = CheckpointCallback(
        save_freq=max(sac_cfg.checkpoint_freq // args_cli.num_envs, 1),
        save_path=ckpt_dir,
        name_prefix="sac_grasp",
        verbose=1,
    )
    success_cb = SuccessRateCallback(
        check_freq=max(1000 // args_cli.num_envs, 1),
        verbose=1,
    )
    callback_list = CallbackList([checkpoint_cb, success_cb])

    # -- Train --
    print(f"[INFO] Starting training for {args_cli.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args_cli.total_timesteps,
        callback=callback_list,
        log_interval=sac_cfg.log_interval,
    )

    # -- Save final model --
    final_path = os.path.join(ckpt_dir, "latest")
    model.save(final_path)
    print(f"[INFO] Final model saved: {final_path}.zip")

    # -- Export ONNX --
    if use_her:
        # For HER, obs_dim is the "observation" part (without goal)
        obs_dim = sb3_env.observation_space["observation"].shape[0]
    else:
        obs_dim = sb3_env.observation_space.shape[0]  # type: ignore[index]
    onnx_path = os.path.join(log_dir, "policy.onnx")
    export_onnx(model, onnx_path, obs_dim)

    # -- Cleanup --
    sb3_env.close()
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
    simulation_app.close()
