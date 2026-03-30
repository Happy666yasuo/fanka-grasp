# Franka Cube Grasp — Sim2Sim MuJoCo Evaluation
# Conda env: unitree-rl (Python 3.8)
"""
Evaluate an ONNX-exported SAC policy in MuJoCo (Sim2Sim).

Observation alignment with IsaacLab (32D):
    [0:9]   joint_pos_rel   — joint positions relative to default
    [9:18]  joint_vel_rel   — joint velocities (normalized)
    [18:21] object_pos_b    — cube position in robot base frame
    [21:24] ee_object_rel   — EE→object vector in world frame
    [24:32] actions         — last action (7 arm + 1 gripper)

Action space (8D):
    [0:7] arm joint positions (scaled by 0.5, offset by default)
    [7]   gripper binary (>0 → open, ≤0 → close)

MuJoCo actuator mapping:
    actuator0-6 → joint1-7 (position control with gains)
    actuator7   → finger tendon (ctrlrange 0-255, 255=open)

Usage:
    conda activate unitree-rl
    cd ~/Desktop/beyondmimic/my_practice/franka_cube_grasp
    python sim2sim/mujoco_eval.py --episodes 10 --render false
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np
import onnxruntime as ort

# ============================================================================
# Constants matching IsaacLab env
# ============================================================================

# Default joint positions (IsaacLab init_state)
DEFAULT_QPOS = np.array([
    0.0,     # joint1
    -0.569,  # joint2
    0.0,     # joint3
    -2.810,  # joint4
    0.0,     # joint5
    3.037,   # joint6
    0.741,   # joint7
    0.04,    # finger_joint1
    0.04,    # finger_joint2
], dtype=np.float64)

# Action scaling (must match IsaacLab ActionsCfg)
ACTION_SCALE = 0.5  # JointPositionActionCfg scale

# Physics timing (match IsaacLab)
CONTROL_DT = 0.01      # IsaacLab dt=0.01
SIM_DT = 0.002         # MuJoCo timestep
SUBSTEPS = int(CONTROL_DT / SIM_DT)  # = 5
DECIMATION = 2          # IsaacLab decimation
STEPS_PER_ACTION = SUBSTEPS * DECIMATION  # 10 MuJoCo steps per policy step

# Episode params
EPISODE_LENGTH_S = 5.0
MAX_EPISODE_STEPS = int(EPISODE_LENGTH_S / (CONTROL_DT * DECIMATION))  # = 250

# Success criterion
LIFT_THRESHOLD = 0.06  # cube must be lifted this much above initial z
CUBE_INIT_Z = 0.055

# EE offset from panda_hand frame (matching IsaacLab FrameTransformerCfg)
EE_OFFSET = np.array([0.0, 0.0, 0.1034])

# Cube randomization range (matching IsaacLab EventCfg)
CUBE_X_RANGE = (-0.1, 0.1)  # relative to 0.5
CUBE_Y_RANGE = (-0.25, 0.25)


# ============================================================================
# Scene wrapper
# ============================================================================

class FrankaGraspMuJoCoEnv:
    """MuJoCo environment for Franka cube grasping Sim2Sim evaluation."""

    def __init__(self, xml_path: str, render: bool = False):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Cache body/joint IDs
        self.link0_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "link0"
        )
        self.hand_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )
        self.cube_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )

        # Joint indices (hinge joints 0-8, free joint starts at qpos index 9)
        self.arm_qpos_idx = list(range(0, 7))    # joint1-7
        self.finger_qpos_idx = list(range(7, 9)) # finger_joint1-2
        self.cube_qpos_idx = list(range(9, 16))  # cube free joint (pos + quat)

        self.arm_qvel_idx = list(range(0, 7))
        self.finger_qvel_idx = list(range(7, 9))
        self.cube_qvel_idx = list(range(9, 15))  # free joint vel (6 DOF)

        # Actuator indices
        self.arm_ctrl_idx = list(range(0, 7))
        self.finger_ctrl_idx = 7  # tendon actuator

        # Last action buffer
        self.last_action = np.zeros(8, dtype=np.float32)

        # Rendering
        self.render = render
        self.viewer = None

        # Step counter
        self.step_count = 0

        # Verify model structure
        assert self.model.nu == 8, f"Expected 8 actuators, got {self.model.nu}"
        assert self.model.nq == 16, f"Expected 16 qpos, got {self.model.nq}"

    def reset(self, randomize_cube: bool = True) -> np.ndarray:
        """Reset environment and return initial observation."""
        # Reset to keyframe
        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "grasp_home"
        )
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        # Randomize cube position (matching IsaacLab EventCfg)
        if randomize_cube:
            dx = np.random.uniform(*CUBE_X_RANGE)
            dy = np.random.uniform(*CUBE_Y_RANGE)
            self.data.qpos[9] = 0.5 + dx   # cube x
            self.data.qpos[10] = 0.0 + dy   # cube y
            self.data.qpos[11] = CUBE_INIT_Z # cube z

        mujoco.mj_forward(self.model, self.data)

        self.last_action = np.zeros(8, dtype=np.float32)
        self.step_count = 0

        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one policy step (with SUBSTEPS * DECIMATION physics steps).

        Args:
            action: 8D array [arm_delta(7), gripper_binary(1)]

        Returns:
            obs, reward, done, info
        """
        # Decode action → joint targets
        arm_action = action[:7]  # delta from default, scaled
        gripper_action = action[7]

        # Arm: target = default + action * scale
        arm_target = DEFAULT_QPOS[:7] + arm_action * ACTION_SCALE

        # Gripper: binary → open (255) or close (0)
        finger_target = 255.0 if gripper_action > 0 else 0.0

        # Set control
        self.data.ctrl[:7] = arm_target
        self.data.ctrl[7] = finger_target

        # Step physics (SUBSTEPS * DECIMATION times)
        for _ in range(STEPS_PER_ACTION):
            mujoco.mj_step(self.model, self.data)

        self.last_action = action.astype(np.float32).copy()
        self.step_count += 1

        # Get observation
        obs = self._get_obs()

        # Compute reward (sparse: is cube lifted?)
        cube_z = self.data.body(self.cube_id).xpos[2]
        lifted = cube_z > (CUBE_INIT_Z + LIFT_THRESHOLD)
        reward = 1.0 if lifted else 0.0

        # Done: episode length reached
        done = self.step_count >= MAX_EPISODE_STEPS

        info = {
            "cube_z": cube_z,
            "lifted": lifted,
            "step": self.step_count,
        }

        # Render
        if self.render and self.viewer is not None:
            self.viewer.sync()

        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        """Build 32D observation matching IsaacLab layout.

        Layout:
            [0:9]   joint_pos_rel  = qpos[:9] - default_qpos
            [9:18]  joint_vel_rel  = qvel[:9] (normalized)
            [18:21] object_pos_b   = cube pos in robot base frame
            [21:24] ee_object_rel  = cube_pos_w - ee_pos_w
            [24:32] actions        = last_action (8D)
        """
        obs = np.zeros(32, dtype=np.float32)

        # Joint positions relative to default
        obs[0:9] = self.data.qpos[:9] - DEFAULT_QPOS

        # Joint velocities (raw — IsaacLab uses joint_vel_rel which is similar)
        obs[9:18] = self.data.qvel[:9]

        # Object position in robot base frame
        # Robot base is at link0 (world origin in our setup)
        robot_pos = self.data.body(self.link0_id).xpos
        robot_quat = self.data.body(self.link0_id).xquat  # w,x,y,z
        cube_pos = self.data.body(self.cube_id).xpos

        # Transform cube to robot base frame
        # Since robot base is at origin with identity rotation, this simplifies to:
        cube_pos_b = self._world_to_body(robot_pos, robot_quat, cube_pos)
        obs[18:21] = cube_pos_b

        # EE → object relative position (in world frame)
        hand_pos = self.data.body(self.hand_id).xpos
        hand_rot = self.data.body(self.hand_id).xmat.reshape(3, 3)
        ee_pos = hand_pos + hand_rot @ EE_OFFSET
        ee_object_rel = cube_pos - ee_pos
        obs[21:24] = ee_object_rel

        # Last action
        obs[24:32] = self.last_action

        return obs

    @staticmethod
    def _world_to_body(
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        point_w: np.ndarray,
    ) -> np.ndarray:
        """Transform a world-frame point to body frame.

        Uses MuJoCo quaternion convention (w, x, y, z).
        """
        # Rotation matrix from quaternion
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, body_quat)
        rot = rot.reshape(3, 3)
        # point_b = R^T @ (point_w - body_pos)
        return rot.T @ (point_w - body_pos)

    def close(self):
        """Clean up viewer."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ============================================================================
# ONNX Policy Wrapper
# ============================================================================

class ONNXPolicy:
    """Load and run ONNX actor network."""

    def __init__(self, onnx_path: str):
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        print(f"[INFO] ONNX loaded: input={self.input_name}{input_shape}, "
              f"output={self.output_name}{output_shape}")

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            obs: (obs_dim,) or (1, obs_dim) float32

        Returns:
            action: (act_dim,) float32
        """
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
        obs = obs.astype(np.float32)
        result = self.session.run(None, {self.input_name: obs})
        action = result[0][0]  # remove batch dim
        return action


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(
    env: FrankaGraspMuJoCoEnv,
    policy: ONNXPolicy,
    num_episodes: int = 10,
    verbose: bool = True,
) -> Dict:
    """Run evaluation episodes.

    Returns:
        Dict with success_rate, mean_reward, episode_lengths, etc.
    """
    successes = []
    total_rewards = []
    episode_lengths = []
    max_cube_heights = []

    for ep in range(num_episodes):
        obs = env.reset(randomize_cube=True)
        ep_reward = 0.0
        max_z = CUBE_INIT_Z
        done = False
        ever_lifted = False

        while not done:
            action = policy.predict(obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            max_z = max(max_z, info["cube_z"])
            if info["lifted"]:
                ever_lifted = True

        successes.append(float(ever_lifted))
        total_rewards.append(ep_reward)
        episode_lengths.append(env.step_count)
        max_cube_heights.append(max_z)

        if verbose:
            status = "✅ LIFT" if ever_lifted else "❌ fail"
            print(f"  Episode {ep+1:3d}/{num_episodes}: {status} | "
                  f"reward={ep_reward:6.1f} | max_z={max_z:.4f} | "
                  f"steps={env.step_count}")

    results = {
        "success_rate": np.mean(successes),
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_episode_length": np.mean(episode_lengths),
        "mean_max_cube_z": np.mean(max_cube_heights),
        "num_episodes": num_episodes,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Franka Grasp — MuJoCo Sim2Sim Eval")
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Path to ONNX policy file (default: sim2sim/policy_franka_grasp.onnx).",
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=None,
        help="Path to MuJoCo XML scene (default: sim2sim/franka_emika_panda/franka_table.xml).",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes."
    )
    parser.add_argument(
        "--render",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Enable MuJoCo viewer rendering.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    args = parser.parse_args()

    # Resolve paths
    sim2sim_dir = os.path.dirname(os.path.abspath(__file__))

    if args.onnx is None:
        onnx_path = os.path.join(sim2sim_dir, "policy_franka_grasp.onnx")
    else:
        onnx_path = args.onnx

    if args.xml is None:
        xml_path = os.path.join(sim2sim_dir, "franka_emika_panda", "franka_table.xml")
    else:
        xml_path = args.xml

    render = args.render.lower() == "true"
    np.random.seed(args.seed)

    print("=" * 60)
    print("Franka Cube Grasp — Sim2Sim MuJoCo Evaluation")
    print("=" * 60)
    print(f"  onnx_model : {onnx_path}")
    print(f"  xml_scene  : {xml_path}")
    print(f"  episodes   : {args.episodes}")
    print(f"  render     : {render}")
    print(f"  seed       : {args.seed}")
    print("=" * 60)

    # Create env and load policy
    env = FrankaGraspMuJoCoEnv(xml_path, render=render)
    policy = ONNXPolicy(onnx_path)

    # Run evaluation
    print("\n[INFO] Starting evaluation...\n")
    results = evaluate(env, policy, num_episodes=args.episodes)

    # Summary
    n = results["num_episodes"]
    sr = results["success_rate"]
    ns = int(sr * n)
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Success rate: {sr*100:.1f}% ({ns}/{n})")
    print(f"  Mean reward:  {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean ep len:  {results['mean_episode_length']:.0f}")
    print(f"  Mean max z:   {results['mean_max_cube_z']:.4f}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
