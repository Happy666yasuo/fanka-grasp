"""Diagnostic: IK accuracy, link frames, scripted pick."""
import sys
sys.path.insert(0, "src")
import numpy as np
import pybullet as p
from embodied_agent.simulator import BulletPickPlaceSimulation

sim = BulletPickPlaceSimulation(gui=False)
print(f"EE link index: {sim.end_effector_index}")
info = p.getJointInfo(sim.robot_id, sim.end_effector_index, physicsClientId=sim.client_id)
print(f"EE link name: {info[12].decode()}, joint name: {info[1].decode()}")

# Check link_state[0] (COM) vs link_state[4] (URDF frame)
ls = p.getLinkState(sim.robot_id, sim.end_effector_index,
                    computeForwardKinematics=True, physicsClientId=sim.client_id)
print(f"COM pos  [0]: {[round(x,4) for x in ls[0]]}")
print(f"Frame pos[4]: {[round(x,4) for x in ls[4]]}")
print(f"Offset [4]-[0]: {[round(ls[4][i]-ls[0][i],4) for i in range(3)]}")

# Test scripted pick
print("\n=== Scripted pick ===")
sim.reset_task()
sim.pick_object("red_block")
obj_pos = sim._get_body_position(sim.object_ids["red_block"])
held = sim.held_object_name
suc = sim.is_pick_success("red_block")
print(f"held={held}, obj_z={obj_pos[2]:.4f}, success={suc}")

# Check fingertip positions after pick
for link_name in ["panda_hand", "panda_leftfinger", "panda_rightfinger"]:
    try:
        idx = sim._find_link_index(link_name)
        ls2 = p.getLinkState(sim.robot_id, idx, computeForwardKinematics=True,
                             physicsClientId=sim.client_id)
        print(f"  {link_name} (idx={idx}): frame_z={ls2[4][2]:.4f}")
    except KeyError:
        pass

# Test: move_end_effector from hover to approach
print("\n=== move_end_effector accuracy ===")
sim.reset_task()
obj_pos = sim._get_body_position(sim.object_ids["red_block"])
hover_pos = (obj_pos[0], obj_pos[1], sim.config.table_top_z + sim.config.hover_height)
approach_pos = (obj_pos[0], obj_pos[1], sim.config.table_top_z + sim.config.pick_height)
print(f"Object: {[round(x,4) for x in obj_pos]}")
print(f"Hover target: {[round(x,4) for x in hover_pos]}")
print(f"Approach target: {[round(x,4) for x in approach_pos]}")

sim.open_gripper()
sim.move_end_effector(hover_pos, steps=240)
pos_h, _ = sim.get_end_effector_pose()
err_h = np.linalg.norm(np.array(pos_h) - np.array(hover_pos))
print(f"After hover: ee={[round(x,4) for x in pos_h]}, err={err_h:.4f}")

sim.move_end_effector(approach_pos, steps=240)
pos_a, _ = sim.get_end_effector_pose()
err_a = np.linalg.norm(np.array(pos_a) - np.array(approach_pos))
print(f"After approach: ee={[round(x,4) for x in pos_a]}, err={err_a:.4f}")
print(f"ee_to_obj dist: {np.linalg.norm(np.array(pos_a) - np.array(obj_pos)):.4f}")

# Now test teleport to same positions
print("\n=== Teleport accuracy from home ===")
for label, tgt in [("hover", hover_pos), ("approach", approach_pos)]:
    sim.reset_task()
    sim.open_gripper()
    sim.teleport_end_effector(tgt)
    pos, _ = sim.get_end_effector_pose()
    err = np.linalg.norm(np.array(pos) - np.array(tgt))
    print(f"Teleport to {label}: target={[round(x,4) for x in tgt]} actual={[round(x,4) for x in pos]} err={err:.4f}")

# Teleport chain: hover then approach
print("\n=== Teleport chain (hover -> approach) ===")
sim.reset_task()
sim.open_gripper()
sim.teleport_end_effector(hover_pos)
pos1, _ = sim.get_end_effector_pose()
print(f"After teleport hover: {[round(x,4) for x in pos1]}")
sim.teleport_end_effector(approach_pos)
pos2, _ = sim.get_end_effector_pose()
err2 = np.linalg.norm(np.array(pos2) - np.array(approach_pos))
print(f"After teleport approach: {[round(x,4) for x in pos2]} err={err2:.4f}")
print(f"ee_to_obj dist: {np.linalg.norm(np.array(pos2) - np.array(obj_pos)):.4f}")

sim.shutdown()
