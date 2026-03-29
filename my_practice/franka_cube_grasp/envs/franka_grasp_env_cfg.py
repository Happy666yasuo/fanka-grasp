# Franka Cube Grasp — Environment Configuration
# Conda env: beyondmimic (Python 3.10)
"""
Scene and MDP configuration for Franka cube grasping task.

Based on the official IsaacLab Lift task (lift_env_cfg.py + joint_pos_env_cfg.py),
adapted for SAC-based grasping with sparse/dense reward options.

Architecture:
    - ObjectTableSceneCfg: Franka + table + cube + EE frame sensor
    - FrankaGraspEnvCfg: Full MDP (obs, actions, rewards, terminations, events)

Key differences from official Lift:
    - No UniformPoseCommand (fixed height target instead)
    - Simplified rewards for SAC compatibility
    - Configurable sparse vs. dense reward via RewardsCfg
"""
from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

# Import our custom MDP functions (+ re-exported isaaclab.envs.mdp)
from . import mdp


# ============================================================================
# Scene
# ============================================================================


@configclass
class FrankaGraspSceneCfg(InteractiveSceneCfg):
    """Scene: Franka Panda + table + cube + EE frame sensor + ground + light."""

    # -- Robot --
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # -- End-effector frame sensor --
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
        ],
    )

    # -- Object (cube) --
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # -- Table --
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
    )

    # -- Ground plane --
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # -- Light --
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# ============================================================================
# MDP — Actions
# ============================================================================


@configclass
class ActionsCfg:
    """Action specifications: 7 joint positions + binary gripper."""

    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


# ============================================================================
# MDP — Observations
# ============================================================================


@configclass
class ObservationsCfg:
    """Observation specifications (~23-D concatenated).

    Components:
        joint_pos_rel: 9D (7 arm + 2 finger)
        joint_vel_rel: 9D (7 arm + 2 finger)
        object_pos_b: 3D (cube in robot base frame)
        ee_object_rel: 3D (EE → object vector)
        actions: 8D (last action: 7 arm + 1 gripper)
    Total ≈ 32D
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group."""

        # Robot proprioception
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # Object
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # EE → Object
        ee_object_rel = ObsTerm(func=mdp.ee_object_relative_position)
        # Last action
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ============================================================================
# MDP — Rewards
# ============================================================================


@configclass
class RewardsCfg:
    """Reward terms — starts with dense reaching + sparse lift.

    For pure sparse experiments, set reaching_object.weight = 0.
    """

    # Dense: encourage EE to approach the object
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1},
        weight=1.0,
    )

    # Sparse: reward when object is lifted
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.06},
        weight=15.0,
    )

    # Dense: track target height (simplified goal tracking)
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_tracking,
        params={"std": 0.3, "minimal_height": 0.06, "target_height": 0.2},
        weight=10.0,
    )

    # Penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# -- Phase 3: Alternative reward configurations --


@configclass
class SparseRewardsCfg:
    """Sparse-only rewards (baseline for ablation studies).

    Only rewards the agent for successfully lifting the object.
    Small penalties for action rate and joint velocities.
    """

    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.06},
        weight=15.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class ShapedRewardsCfg:
    """Multi-stage shaped reward (Phase 3 — shaped strategy).

    Combines the shaped_multi_stage function with action penalties.
    The multi-stage function internally computes reach + grasp + lift + hold.
    """

    shaped = RewTerm(
        func=mdp.shaped_multi_stage,
        params={
            "reach_std": 0.1,
            "grasp_threshold": 0.02,
            "lift_target": 0.2,
            "lift_std": 0.1,
        },
        weight=1.0,
    )

    # Penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class PBRSRewardsCfg:
    """Potential-Based Reward Shaping (Phase 3 — PBRS strategy).

    Sparse lift reward + PBRS shaping term (F = gamma*Phi(s') - Phi(s)).
    Preserves optimal policy under the original sparse reward.
    """

    # Sparse base reward
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.06},
        weight=15.0,
    )

    # PBRS shaping term
    pbrs = RewTerm(
        func=mdp.pbrs_shaping,
        params={"gamma": 0.99},
        weight=5.0,
    )

    # Penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ============================================================================
# MDP — Terminations
# ============================================================================


@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


# ============================================================================
# MDP — Events (randomization / resets)
# ============================================================================


@configclass
class EventCfg:
    """Event terms: resets."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


# ============================================================================
# Full Environment Config
# ============================================================================


@configclass
class FrankaGraspEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Franka cube grasping environment.

    Inherits ManagerBasedRLEnvCfg and wires up all MDP components.
    """

    # Scene
    scene: FrankaGraspSceneCfg = FrankaGraspSceneCfg(num_envs=4, env_spacing=2.5)

    # MDP
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # No commands manager (we use fixed target height in rewards)
    commands = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Post initialization — simulation parameters."""
        # General settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # Simulation settings
        self.sim.dt = 0.01  # 100 Hz physics
        self.sim.render_interval = self.decimation
        # PhysX settings
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
