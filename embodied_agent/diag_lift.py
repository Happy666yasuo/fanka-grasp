"""Quick diagnostic: can the arm lift from staging pose?"""
import sys
sys.path.insert(0, "src")
from embodied_agent.simulator import BulletPickPlaceSimulation

sim = BulletPickPlaceSimulation(gui=False)
sim.prepare_pick_staging_pose("red_block")
pos, _ = sim.get_end_effector_pose()
print(f"Staging pose: {pos}")

# Simulate lifting by 0.04 each step
for step in range(10):
    target = (pos[0], pos[1], pos[2] + 0.04 * (step + 1))
    sim.teleport_end_effector(target)
    sim._simulate(60)
    actual, _ = sim.get_end_effector_pose()
    err = abs(actual[2] - target[2])
    print(f"  step {step}: target_z={target[2]:.4f}, actual_z={actual[2]:.4f}, err={err:.4f}")

sim.shutdown()
