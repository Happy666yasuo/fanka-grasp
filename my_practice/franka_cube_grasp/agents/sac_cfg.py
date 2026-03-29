# Franka Cube Grasp — SAC Hyperparameter Configuration
# Conda env: beyondmimic (Python 3.10)
"""
Default SAC hyperparameters for Franka cube grasping.

Key constraints (VRAM-limited machine):
    - buffer_size = 100_000 (NOT 1M)
    - batch_size = 256
    - Default num_envs = 64 for training

These are passed to ``stable_baselines3.SAC`` via ``**cfg``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SACConfig:
    """SAC hyperparameters for Franka cube grasping."""

    # -- Core algorithm --
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    buffer_size: int = 100_000  # ⚠️ 本机显存/内存有限
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"  # auto-tune temperature α
    target_entropy: str = "auto"

    # -- Policy network --
    policy_kwargs: dict[str, Any] = field(default_factory=lambda: {
        "net_arch": [256, 256],
        "log_std_init": -2,
    })

    # -- Training defaults --
    total_timesteps: int = 500_000
    num_envs: int = 64

    # -- Logging & checkpointing --
    log_interval: int = 10
    checkpoint_freq: int = 10_000  # save every N steps
    eval_freq: int = 10_000
    eval_episodes: int = 20

    # -- Misc --
    seed: int | None = None
    verbose: int = 1

    def to_sb3_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs dict for ``stable_baselines3.SAC()`` constructor.

        Excludes non-SB3 fields (total_timesteps, num_envs, checkpoint_freq, etc.).
        """
        return {
            "policy": self.policy,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "ent_coef": self.ent_coef,
            "target_entropy": self.target_entropy,
            "policy_kwargs": self.policy_kwargs,
            "verbose": self.verbose,
            "seed": self.seed,
        }


# Singleton default config
SAC_DEFAULT_CFG = SACConfig()
