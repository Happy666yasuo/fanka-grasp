"""Quick smoke test for expert pick policy."""
import sys
sys.path.insert(0, "src")
import numpy as np
from embodied_agent.rl_envs import PickPlaceSkillEnv, SkillEnvSettings
from embodied_agent.expert_warmstart import expert_pick_action, collect_expert_episodes

settings = SkillEnvSettings(
    skill_name="pick", max_steps=20, action_scale=0.04,
    action_repeat=60, use_staging=True,
)
env = PickPlaceSkillEnv(settings=settings, gui=False)

# Test single episode trace
obs, _ = env.reset()
d = float(np.linalg.norm(obs[9:12]))
print(f"Reset: ee_pos={obs[:3].round(4)}, obj_pos={obs[3:6].round(4)}, dist={d:.4f}")
for step in range(20):
    action = expert_pick_action(obs, 0.04)
    obs, r, term, trunc, info = env.step(action)
    d = float(np.linalg.norm(obs[9:12]))
    h = obs[16]
    suc = info.get("is_success", False)
    print(f"  step {step:2d}: hold={h:.0f} dist={d:.4f} r={r:+.3f} suc={suc} act={action.round(2)}")
    if term or trunc:
        break

# Batch test
print("\n--- Batch expert test (no noise) ---")
_, stats = collect_expert_episodes(env, expert_pick_action, 50, action_scale=0.04, noise_std=0.0)
print(stats)

print("\n--- Batch expert test (noise=0.05) ---")
_, stats2 = collect_expert_episodes(env, expert_pick_action, 50, action_scale=0.04, noise_std=0.05)
print(stats2)

env.close()
