# Franka Cube Grasp — Termination conditions
# Conda env: beyondmimic (Python 3.10)
"""
Termination conditions for the Franka cube grasping task.

Provides:
    - object_dropped_below_table: terminate if cube falls below the table.

Built-in terminations used via isaaclab.envs.mdp:
    - time_out: episode time limit.
    - root_height_below_minimum: generic height check.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_dropped_below_table(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate if the object has dropped below the table surface.

    Args:
        env: The environment.
        minimum_height: Height below which the object is considered dropped.
        object_cfg: The object scene entity configuration.

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_pos_w[:, 2] < minimum_height
