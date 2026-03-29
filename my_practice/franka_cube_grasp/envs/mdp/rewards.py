# Franka Cube Grasp — Reward functions
# Conda env: beyondmimic (Python 3.10)
"""
Reward functions for the Franka cube grasping task.

Strategies (Phase 1 + Phase 3):
    Phase 1 (basic):
        - object_is_lifted: Sparse binary reward.
        - object_ee_distance: Dense reach reward (tanh kernel).
        - object_goal_tracking: Dense reward for tracking goal height.

    Phase 3 (advanced):
        - shaped_multi_stage: 4-stage dense reward (reach -> grasp -> lift -> hold).
        - pbrs_shaping: Potential-Based Reward Shaping (F = gamma*Phi(s') - Phi(s)).
        - curriculum_reward: Difficulty-adaptive reward with 3 tiers.

All functions are GPU-vectorized (pure torch tensor ops, no Python for-loops).
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


# ============================================================================
# Phase 3 — Advanced reward strategies
# ============================================================================


def shaped_multi_stage(
    env: ManagerBasedRLEnv,
    reach_std: float = 0.1,
    grasp_threshold: float = 0.02,
    lift_target: float = 0.2,
    lift_std: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Multi-stage dense shaped reward: reach -> grasp -> lift -> hold.

    Implements a 4-stage reward that progressively guides the agent:

    Stage 1 -- Reach:
        r_reach = 1 - tanh(d_ee_obj / reach_std)

    Stage 2 -- Grasp proxy:
        r_grasp = (d_ee_obj < grasp_threshold) * gripper_closed
        where gripper_closed = (finger_pos < 0.035).all()

    Stage 3 -- Lift:
        r_lift = grasping * clamp(z_obj - z_initial, 0, lift_target) / lift_target

    Stage 4 -- Hold:
        r_hold = grasping * (1 - tanh(|z_obj - lift_target| / lift_std))

    Total:
        r = 1.0 * r_reach + 3.0 * r_grasp + 5.0 * r_lift + 8.0 * r_hold

    Args:
        env: The environment.
        reach_std: Tanh kernel std for reach stage.
        grasp_threshold: Distance threshold to consider "close enough to grasp".
        lift_target: Target lift height above table.
        lift_std: Tanh kernel std for hold stage.
        object_cfg: Object scene entity configuration.
        ee_frame_cfg: End-effector frame configuration.

    Returns:
        Tensor of shape (num_envs,).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    obj_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    z_obj = obj_pos_w[:, 2]

    # EE-object distance
    d_ee_obj = torch.norm(obj_pos_w - ee_pos_w, dim=1)

    # Stage 1: Reach
    r_reach = 1.0 - torch.tanh(d_ee_obj / reach_std)

    # Stage 2: Grasp proxy — EE close + gripper closing
    robot = env.scene["robot"]
    finger_pos = robot.data.joint_pos[:, -2:]  # panda_finger_joint_l, _r
    gripper_closed = (finger_pos < 0.035).all(dim=1).float()
    close_enough = (d_ee_obj < grasp_threshold).float()
    r_grasp = close_enough * gripper_closed

    # Stage 3: Lift (gated by grasp)
    z_initial = 0.055  # cube initial height on table
    lift_progress = torch.clamp(z_obj - z_initial, 0.0, lift_target) / lift_target
    is_grasping = r_grasp  # reuse grasp signal
    r_lift = is_grasping * lift_progress

    # Stage 4: Hold at target
    hold_dist = torch.abs(z_obj - (z_initial + lift_target))
    r_hold = is_grasping * (1.0 - torch.tanh(hold_dist / lift_std))

    # Weighted combination
    reward = 1.0 * r_reach + 3.0 * r_grasp + 5.0 * r_lift + 8.0 * r_hold

    return reward


def pbrs_shaping(
    env: ManagerBasedRLEnv,
    gamma: float = 0.99,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Potential-Based Reward Shaping (PBRS).

    Adds a shaping reward that preserves the optimal policy (Ng et al., 1999):

        F(s, s') = gamma * Phi(s') - Phi(s)

    where the potential function is:

        Phi(s) = -(||p_ee - p_obj|| + ||z_obj - z_target||)

    This encourages both reaching the object and lifting it, while
    guaranteeing that the optimal policy under the original sparse reward
    remains optimal under the shaped reward.

    Note:
        We store the previous potential in ``env._pbrs_prev_potential``.
        On the very first call (or after reset), we initialize it.

    Args:
        env: The environment.
        gamma: Discount factor for the shaping.
        object_cfg: Object scene entity configuration.
        ee_frame_cfg: End-effector frame configuration.

    Returns:
        Tensor of shape (num_envs,) -- the shaping term F(s, s').
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    obj_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    z_obj = obj_pos_w[:, 2]

    # Target height for the object
    z_target = 0.255  # 0.055 (table) + 0.2 (lift)

    # Current potential: Phi(s') = -(d_ee_obj + d_obj_target)
    d_ee_obj = torch.norm(obj_pos_w - ee_pos_w, dim=1)
    d_obj_target = torch.abs(z_obj - z_target)
    phi_current = -(d_ee_obj + d_obj_target)

    # Retrieve previous potential (or initialize)
    if not hasattr(env, "_pbrs_prev_potential") or env._pbrs_prev_potential is None:
        env._pbrs_prev_potential = phi_current.clone()

    phi_prev = env._pbrs_prev_potential

    # F(s, s') = gamma * Phi(s') - Phi(s)
    shaping = gamma * phi_current - phi_prev

    # Store current as previous for next step
    env._pbrs_prev_potential = phi_current.clone()

    # Reset potential for environments that just reset
    # (detected by episode step count == 0 after the first step)
    if hasattr(env, "episode_length_buf"):
        just_reset = (env.episode_length_buf == 0)
        if just_reset.any():
            shaping[just_reset] = 0.0
            env._pbrs_prev_potential[just_reset] = phi_current[just_reset]

    return shaping


def curriculum_reward(
    env: ManagerBasedRLEnv,
    easy_threshold: int = 50_000,
    medium_threshold: int = 200_000,
    reach_std: float = 0.1,
    lift_target: float = 0.2,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Curriculum-aware reward: difficulty increases over training.

    Three tiers based on ``env.common_step_counter``:

    Easy (steps < easy_threshold):
        - Heavy reach bonus: 2.0 * (1 - tanh(d/std))
        - Bonus for any downward finger movement

    Medium (easy_threshold <= steps < medium_threshold):
        - Standard multi-stage reward (reach + grasp + lift)

    Hard (steps >= medium_threshold):
        - Only lift + hold are rewarded (agent should have learned reach/grasp)
        - Small action penalty increases to encourage smooth motions

    Math per tier:
        Easy:   r = 2.0 * r_reach + 1.0 * r_grasp_proxy
        Medium: r = 1.0 * r_reach + 2.0 * r_grasp + 3.0 * r_lift
        Hard:   r = 5.0 * r_lift + 8.0 * r_hold

    Args:
        env: The environment.
        easy_threshold: Step count to transition from easy to medium.
        medium_threshold: Step count to transition from medium to hard.
        reach_std: Tanh kernel std for reach.
        lift_target: Target lift height.
        object_cfg: Object scene entity.
        ee_frame_cfg: EE frame entity.

    Returns:
        Tensor of shape (num_envs,).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    obj_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    z_obj = obj_pos_w[:, 2]

    d_ee_obj = torch.norm(obj_pos_w - ee_pos_w, dim=1)
    z_initial = 0.055

    # Compute building blocks
    r_reach = 1.0 - torch.tanh(d_ee_obj / reach_std)

    robot = env.scene["robot"]
    finger_pos = robot.data.joint_pos[:, -2:]
    gripper_closed = (finger_pos < 0.035).all(dim=1).float()
    close_enough = (d_ee_obj < 0.02).float()
    r_grasp = close_enough * gripper_closed

    lift_progress = torch.clamp(z_obj - z_initial, 0.0, lift_target) / lift_target
    r_lift = r_grasp * lift_progress

    hold_dist = torch.abs(z_obj - (z_initial + lift_target))
    r_hold = r_grasp * (1.0 - torch.tanh(hold_dist / 0.1))

    # Determine curriculum tier
    step_count = env.common_step_counter

    if step_count < easy_threshold:
        # Easy: heavy reach + grasp proxy
        reward = 2.0 * r_reach + 1.0 * r_grasp
    elif step_count < medium_threshold:
        # Medium: full pipeline
        reward = 1.0 * r_reach + 2.0 * r_grasp + 3.0 * r_lift
    else:
        # Hard: only lift + hold
        reward = 5.0 * r_lift + 8.0 * r_hold

    return reward
