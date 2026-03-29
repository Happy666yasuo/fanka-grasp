# Franka Cube Grasp — Reward functions
# Conda env: beyondmimic (Python 3.10)
"""
Reward functions for the Franka cube grasping task.

Phase 1: sparse reward only.
Phase 3 will add: shaped, PBRS, curriculum rewards.

Strategies available:
    - object_is_lifted: sparse binary reward when cube is above threshold height.
    - object_ee_distance: dense reach reward using tanh kernel.
    - object_goal_tracking: dense reward for tracking goal position.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Sparse reward: +1 if the object is above minimal_height, else 0.

    Math:
        r = 1  if  z_obj > h_min
        r = 0  otherwise

    Args:
        env: The environment.
        minimal_height: The height threshold for the object.
        object_cfg: The object scene entity configuration.

    Returns:
        Tensor of shape (num_envs,).
    """
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense reach reward using tanh kernel.

    Math:
        r = 1 - tanh(||p_obj - p_ee|| / std)

    Args:
        env: The environment.
        std: The standard deviation for the tanh kernel.
        object_cfg: The object scene entity configuration.
        ee_frame_cfg: The end-effector frame configuration.

    Returns:
        Tensor of shape (num_envs,).
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = object.data.root_pos_w[:, :3]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(distance / std)


def object_goal_tracking(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    target_height: float = 0.2,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense reward for tracking goal height (simplified — no command manager).

    The goal is simply to lift the object to `target_height` above the table.
    Only gives reward if the object is above `minimal_height`.

    Math:
        r = (z_obj > h_min) * (1 - tanh(|z_obj - z_target| / std))

    Args:
        env: The environment.
        std: The standard deviation for the tanh kernel.
        minimal_height: Minimum height to activate reward.
        target_height: Desired object height.
        object_cfg: The object scene entity configuration.

    Returns:
        Tensor of shape (num_envs,).
    """
    object: RigidObject = env.scene[object_cfg.name]
    z = object.data.root_pos_w[:, 2]
    distance = torch.abs(z - target_height)
    return (z > minimal_height).float() * (1 - torch.tanh(distance / std))
