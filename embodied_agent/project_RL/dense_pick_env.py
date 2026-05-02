"""
Dense-reward Gymnasium environment for robotic pick task.

Key differences from the original rl_envs.PickPlaceSkillEnv:
  - Multi-phase dense reward (Approach → Grasp → Hold → Lift)
  - Every phase provides continuous gradient signal
  - Designed for pure RL training (PPO / SAC / TD3)

Observation (17D):
  [0:3]   ee position
  [3:6]   object position
  [6:9]   zone position
  [9:12]  ee → object relative vector
  [12:15] object → zone relative vector
  [15]    gripper open ratio (0=closed, 1=open)
  [16]    holding flag (0 or 1)

Action (4D, continuous [-1, 1]):
  [0:3]   delta xyz  (scaled by action_scale)
  [3]     gripper command  (-1=close, +1=open)
"""

import os
import random as _random
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── import simulator from parent project ────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
from embodied_agent.simulator import BulletPickPlaceSimulation  # noqa: E402


class DensePickEnv(gym.Env):
    """Franka Panda pick task with dense, phase-aware reward shaping."""

    metadata = {"render_modes": ["human"]}

    # ── physics constants (match simulator defaults) ────────────────
    PLACED_Z = 0.645          # table_top_z + cube_half_extent
    SUCCESS_Z = 0.705         # PLACED_Z + pick_success_height (0.06)

    def __init__(
        self,
        randomize: bool = False,
        use_staging: bool = True,
        max_steps: int = 20,
        action_scale: float = 0.04,
        action_repeat: int = 60,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.randomize = randomize
        self.use_staging = use_staging
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.action_repeat = action_repeat

        gui = render_mode == "human"
        self.sim = BulletPickPlaceSimulation(gui=gui)
        self._rng = _random.Random()

        self.observation_space = spaces.Box(
            -10.0, 10.0, shape=(17,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(4,), dtype=np.float32
        )

        # internal state
        self._step_count = 0
        self._prev_dist = 0.0

    # ── reset / step ────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        if self.randomize:
            obj_xy, zone_xy = self.sim.sample_task_layout(rng=self._rng)
        else:
            obj_xy = self.sim.config.cube_start_xy
            zone_xy = self.sim.config.goal_center_xy

        self.sim.reset_task(object_xy=obj_xy, zone_xy=zone_xy)

        if self.use_staging:
            self.sim.prepare_pick_staging_pose("red_block")

        obs = self.sim.get_skill_observation(
            "red_block", "green_zone"
        ).astype(np.float32)
        self._step_count = 0
        self._prev_dist = float(np.linalg.norm(obs[9:12]))
        return obs, {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        delta = tuple(float(action[i]) * self.action_scale for i in range(3))
        self.sim.apply_skill_action(
            delta_position=delta,
            gripper_command=float(action[3]),
            action_steps=self.action_repeat,
            object_name="red_block",
        )

        obs = self.sim.get_skill_observation(
            "red_block", "green_zone"
        ).astype(np.float32)
        self._step_count += 1

        reward = self._compute_reward(obs)
        success = self.sim.is_pick_success("red_block")
        terminated = success
        truncated = self._step_count >= self.max_steps

        return obs, reward, terminated, truncated, {"is_success": success}

    # ── dense reward function ───────────────────────────────────────
    def _compute_reward(self, obs: np.ndarray) -> float:
        """
        Phase-aware dense reward — v3 (fix early-termination incentive).

        v2 BUG: hold_reward=10/step × 20 steps = 200 >> success_bonus=50.
        Agent learned to grasp-and-hold without lifting because NOT
        lifting (200 total) > lifting and terminating early (86 total).

        v3 FIX:
          - Reduce per-step hold reward to 2.0 (was 10.0)
          - Increase lift gradient to 100.0 (was 50.0)
          - Increase success bonus to 200.0 (was 50.0)

        Now:  hold 20 steps = 40  vs  succeed in 3 steps = 218   ✓
        """
        dist = float(np.linalg.norm(obs[9:12]))
        holding = obs[16] > 0.5
        obj_z = float(obs[5])
        gripper = float(obs[15])  # 0 = closed, 1 = open

        if holding:
            lift = max(0.0, obj_z - self.PLACED_Z)
            r = 2.0 + 100.0 * lift
            if obj_z >= self.SUCCESS_Z:
                r += 200.0
        else:
            r = -0.5                                        # time penalty
            r += 0.5 * (1.0 - np.tanh(5.0 * dist))         # approach
            if dist < 0.10:
                r += 1.0 * (1.0 - gripper)                  # close gripper

        self._prev_dist = dist
        return float(r)

    # ── cleanup ─────────────────────────────────────────────────────
    def close(self):
        self.sim.shutdown()
