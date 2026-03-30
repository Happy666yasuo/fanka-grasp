# Franka Cube Grasp — HER GoalEnv VecEnv Wrapper
# Conda env: beyondmimic (Python 3.10)
"""
GoalEnv-compatible VecEnv wrapper for Hindsight Experience Replay (HER).

Wraps an existing ``Sb3VecEnvWrapper`` (flat obs) into a Dict observation
VecEnv with keys ``observation``, ``achieved_goal``, ``desired_goal``.

SB3 HER requirements:
    1. Dict observation space with 3 keys.
    2. ``env.compute_reward(achieved_goal, desired_goal, info)`` method.
    3. Use ``MultiInputPolicy`` with ``HerReplayBuffer``.

Goal definition for cube grasping:
    - achieved_goal: current cube position (3D world frame).
    - desired_goal: target position (cube lifted to target height).
    - observation: remaining state (joint_pos, joint_vel, ee_object_rel, actions).

The wrapper extracts ``achieved_goal`` from the flat obs (object_position slice)
and provides a fixed ``desired_goal`` (cube at target height above table).
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class HERGoalVecEnvWrapper(VecEnvWrapper):
    """VecEnv wrapper that converts flat obs to Dict obs for HER.

    Observation layout (from FrankaGraspEnvCfg, 32D):
        [0:9]   joint_pos_rel   (7 arm + 2 finger)
        [9:18]  joint_vel_rel   (7 arm + 2 finger)
        [18:21] object_pos_b    (3D — cube in robot base frame)
        [21:24] ee_object_rel   (3D — EE→object vector)
        [24:32] actions         (8D — last action)

    We use ``object_pos_b`` (indices 18:21) as the achieved_goal.
    The desired_goal is a fixed target: [0.0, 0.0, target_height_in_base_frame].

    Args:
        venv: The wrapped SB3 VecEnv (from Sb3VecEnvWrapper).
        obs_dim: Total observation dimensionality. Default 32.
        goal_indices: Slice of obs corresponding to cube position. Default (18, 21).
        desired_goal: The fixed target position for the cube (3D). Default [0.0, 0.0, 0.2].
        distance_threshold: Distance below which goal is considered achieved. Default 0.05.
    """

    def __init__(
        self,
        venv: VecEnv,
        obs_dim: int = 32,
        goal_indices: tuple[int, int] = (18, 21),
        desired_goal: np.ndarray | None = None,
        distance_threshold: float = 0.05,
    ) -> None:
        self.venv = venv
        self.num_envs = venv.num_envs
        self.render_mode = getattr(venv, "render_mode", None)

        self._obs_dim = obs_dim
        self._goal_start = goal_indices[0]
        self._goal_end = goal_indices[1]
        self._goal_dim = self._goal_end - self._goal_start
        self._distance_threshold = distance_threshold

        # Desired goal: fixed target position for the cube
        if desired_goal is not None:
            self._desired_goal = np.array(desired_goal, dtype=np.float32)
        else:
            # Default: cube lifted ~0.2m above initial (in robot base frame)
            self._desired_goal = np.array([0.0, 0.0, 0.2], dtype=np.float32)

        # Build Dict observation space
        flat_space = venv.observation_space
        assert isinstance(flat_space, spaces.Box), \
            f"Expected Box obs space, got {type(flat_space)}"
        low = flat_space.low.flatten()
        high = flat_space.high.flatten()

        # Observation: everything except the goal indices
        obs_indices = list(range(0, self._goal_start)) + \
                      list(range(self._goal_end, obs_dim))
        self._obs_indices = np.array(obs_indices, dtype=np.int64)
        obs_low = low[self._obs_indices]
        obs_high = high[self._obs_indices]

        goal_low = np.full(self._goal_dim, -np.inf, dtype=np.float32)
        goal_high = np.full(self._goal_dim, np.inf, dtype=np.float32)

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(obs_low, obs_high, dtype=np.float32),
            "achieved_goal": spaces.Box(goal_low, goal_high, dtype=np.float32),
            "desired_goal": spaces.Box(goal_low, goal_high, dtype=np.float32),
        })
        self.action_space = venv.action_space

    def _split_obs(self, flat_obs: np.ndarray) -> dict[str, np.ndarray]:
        """Convert flat observation to Dict observation.

        Args:
            flat_obs: Array of shape (num_envs, obs_dim).

        Returns:
            Dict with keys observation, achieved_goal, desired_goal.
        """
        if flat_obs.ndim == 1:
            flat_obs = flat_obs.reshape(1, -1)

        observation = flat_obs[:, self._obs_indices]
        achieved_goal = flat_obs[:, self._goal_start:self._goal_end]
        desired_goal = np.tile(
            self._desired_goal, (flat_obs.shape[0], 1)
        )

        return {
            "observation": observation.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    def reset(self) -> dict[str, np.ndarray]:
        """Reset all environments and return Dict observation."""
        flat_obs = self.venv.reset()
        return self._split_obs(flat_obs)

    def step_wait(self) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
        """Wait for step and return Dict observation.

        Also converts ``terminal_observation`` in infos to dict format,
        which SB3's ``_store_transition`` requires when using Dict obs spaces.
        """
        flat_obs, rewards, dones, infos = self.venv.step_wait()
        dict_obs = self._split_obs(flat_obs)

        # Convert terminal_observation in infos from flat array to dict
        for i, done in enumerate(dones):
            if done and isinstance(infos[i], dict):
                terminal = infos[i].get("terminal_observation")
                if terminal is not None and not isinstance(terminal, dict):
                    # terminal is a flat ndarray of shape (obs_dim,)
                    terminal_dict = self._split_obs(terminal)
                    # _split_obs adds a batch dim → remove it with [0]
                    infos[i]["terminal_observation"] = {
                        key: val[0] for key, val in terminal_dict.items()
                    }

        return dict_obs, rewards, dones, infos

    def step_async(self, actions: np.ndarray) -> None:
        """Send actions to vectorized environment."""
        self.venv.step_async(actions)

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Compute sparse reward for HER goal relabeling.

        Reward is:
            0  if  ||achieved_goal - desired_goal|| < distance_threshold
           -1  otherwise

        This is the standard sparse reward formulation used in HER papers.

        Args:
            achieved_goal: Current cube position. Shape (batch, 3) or (3,).
            desired_goal: Target cube position. Shape (batch, 3) or (3,).
            info: Additional info (unused).

        Returns:
            Reward array. Shape (batch,) or scalar.
        """
        achieved_goal = np.atleast_2d(achieved_goal)
        desired_goal = np.atleast_2d(desired_goal)
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(dist > self._distance_threshold).astype(np.float32)

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if environments are wrapped."""
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call a method on the wrapped environments.

        Intercepts ``compute_reward`` calls (from HER buffer) and handles
        them locally instead of forwarding to the inner env.
        """
        if method_name == "compute_reward":
            # HER calls env_method("compute_reward", achieved, desired, info)
            # We handle this ourselves since the inner env doesn't have it.
            return [self.compute_reward(*method_args, **method_kwargs)]
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def get_attr(self, attr_name, indices=None):
        """Get attribute from wrapped environments."""
        return self.venv.get_attr(attr_name, indices=indices)

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute on wrapped environments."""
        return self.venv.set_attr(attr_name, value, indices=indices)

    def seed(self, seed=None):
        """Seed the environments."""
        if hasattr(self.venv, "seed"):
            return self.venv.seed(seed)
        return None

    def close(self) -> None:
        """Close the environments."""
        self.venv.close()
