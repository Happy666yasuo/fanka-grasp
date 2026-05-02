"""Diagnose place expert and IK reachability for zone positions."""
import sys; sys.path.insert(0, 'src')
from embodied_agent.rl_envs import PickPlaceSkillEnv, SkillEnvSettings
from embodied_agent.expert_warmstart import expert_place_action, collect_expert_episodes
from embodied_agent.simulator import BulletPickPlaceSimulation, SceneConfig
import numpy as np

# Test with randomize=True
settings = SkillEnvSettings(skill_name='place', max_steps=30, action_scale=0.04,
                             action_repeat=12, use_staging=True, randomize=True)
env = PickPlaceSkillEnv(settings=settings, gui=False)
_, stats = collect_expert_episodes(env, expert_place_action, 100, action_scale=0.04, noise_std=0.0)
env.close()
print(f"randomize=True ar=12 ms=30: success={stats['success_rate']:.0%}")

# Test with a closer zone
config = SceneConfig(goal_center_xy=(0.55, -0.10))
sim = BulletPickPlaceSimulation(gui=False, config=config)
settings2 = SkillEnvSettings(skill_name='place', max_steps=30, action_scale=0.04,
                              action_repeat=12, use_staging=True, randomize=False)
env2 = PickPlaceSkillEnv(settings=settings2, gui=False, simulation=sim)
_, stats2 = collect_expert_episodes(env2, expert_place_action, 50, action_scale=0.04, noise_std=0.0)
env2.close()
print(f"closer zone (0.55,-0.10) ar=12 ms=30: success={stats2['success_rate']:.0%}")

# IK reachability map
sim3 = BulletPickPlaceSimulation(gui=False)
sim3.reset()
table_top_z = sim3.config.table_top_z
test_height = table_top_z + 0.085
print(f"\nIK reachability at z={test_height:.3f}:")
for x in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    row = []
    for y in [-0.24, -0.18, -0.10, 0.0, 0.10, 0.18, 0.24]:
        sim3.teleport_end_effector((x, y, test_height))
        sim3._simulate(60)
        ee_pos = sim3.get_end_effector_pose()[0]
        err = np.linalg.norm(np.array(ee_pos) - np.array([x, y, test_height]))
        row.append(f"{err:.3f}")
    print(f"x={x:.2f}: {' '.join(row)}")
sim3.shutdown()
