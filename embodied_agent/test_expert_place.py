"""Quick test for expert place policy."""
import sys
sys.path.insert(0, "src")
import numpy as np
from embodied_agent.rl_envs import PickPlaceSkillEnv, SkillEnvSettings
from embodied_agent.expert_warmstart import expert_place_action, collect_expert_episodes

settings = SkillEnvSettings(
    skill_name="place", max_steps=20, action_scale=0.04,
    action_repeat=60, use_staging=True,
)
env = PickPlaceSkillEnv(settings=settings, gui=False)

print("--- Batch place expert (no noise) ---")
_, stats = collect_expert_episodes(env, expert_place_action, 50, action_scale=0.04, noise_std=0.0)
print(stats)

print("\n--- Batch place expert (noise=0.05) ---")
_, stats2 = collect_expert_episodes(env, expert_place_action, 50, action_scale=0.04, noise_std=0.05)
print(stats2)

env.close()
