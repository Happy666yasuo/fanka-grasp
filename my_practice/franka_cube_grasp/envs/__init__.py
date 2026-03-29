# Franka Cube Grasp — Environment Registration
# Conda env: beyondmimic (Python 3.10)
"""
Gymnasium environment registration for Franka cube grasping.

Registered Environment IDs:
    Isaac-Grasp-Cube-Franka-v0:
        Entry point  : isaaclab.envs:ManagerBasedRLEnv
        Configuration: FrankaGraspEnvCfg (default num_envs=4)

NOTE: env_cfg_entry_point uses a string reference to avoid importing
the config module at registration time (which would require the full
IsaacSim runtime via AppLauncher to be initialized first).
"""
import gymnasium as gym

gym.register(
    id="Isaac-Grasp-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_grasp_env_cfg:FrankaGraspEnvCfg",
    },
)
