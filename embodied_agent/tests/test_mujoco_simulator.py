from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.simulator import create_pick_place_simulation
from embodied_agent.mujoco_simulator import MujocoPickPlaceSimulation


class MujocoPickPlaceSimulationTests(unittest.TestCase):
    def test_creates_requested_scene_entities(self) -> None:
        simulation = MujocoPickPlaceSimulation(
            gui=False,
            object_names=("red_block", "blue_block"),
            zone_names=("green_zone", "blue_zone"),
        )
        try:
            self.assertEqual(set(simulation.object_ids), {"red_block", "blue_block"})
            self.assertEqual(set(simulation.zone_ids), {"green_zone", "blue_zone"})
            scene = simulation.observe_scene("move blocks")
            self.assertEqual(set(scene.object_positions), {"red_block", "blue_block"})
            self.assertEqual(set(scene.zone_positions), {"green_zone", "blue_zone"})
        finally:
            simulation.shutdown()

    def test_pick_place_and_pose_helpers(self) -> None:
        simulation = create_pick_place_simulation("mujoco", gui=False)
        try:
            object_pose = simulation.get_object_pose("red_block")
            self.assertEqual(len(object_pose[0]), 3)
            self.assertEqual(len(object_pose[1]), 4)

            simulation.pick_object("red_block")
            self.assertTrue(simulation.is_pick_success("red_block"))
            simulation.place_object("green_zone")

            self.assertTrue(simulation.is_place_success("red_block", "green_zone"))
            self.assertTrue(simulation.is_object_in_zone("red_block", "green_zone"))
        finally:
            simulation.shutdown()

    def test_capture_and_restore_held_runtime_state(self) -> None:
        simulation = MujocoPickPlaceSimulation(gui=False)
        try:
            simulation.reset_task(holding_object=True)
            simulation.teleport_end_effector((0.62, 0.08, simulation.config.table_top_z + 0.22))
            captured_state = simulation.capture_skill_state("red_block", "green_zone")

            simulation.reset_task(holding_object=False)
            simulation.restore_runtime_state(
                captured_state,
                object_name="red_block",
                zone_name="green_zone",
            )
            restored_state = simulation.capture_skill_state("red_block", "green_zone")

            self.assertTrue(restored_state["holding_target_object"])
            self.assertEqual(simulation.held_object_name, "red_block")
            self.assertLess(
                math.dist(captured_state["object_position"], restored_state["object_position"]),
                1e-6,
            )
        finally:
            simulation.shutdown()

    def test_apply_skill_action_can_attach_and_release(self) -> None:
        simulation = MujocoPickPlaceSimulation(gui=False)
        try:
            object_position = simulation.get_object_position("red_block")
            simulation.teleport_end_effector(
                (object_position[0], object_position[1], object_position[2] + 0.04)
            )
            simulation.apply_skill_action(
                delta_position=(0.0, 0.0, 0.0),
                gripper_command=-1.0,
                object_name="red_block",
            )
            self.assertTrue(simulation.is_object_held("red_block"))

            simulation.apply_skill_action(
                delta_position=(0.0, 0.0, 0.06),
                gripper_command=1.0,
                object_name="red_block",
            )
            self.assertFalse(simulation.is_object_held("red_block"))
        finally:
            simulation.shutdown()


if __name__ == "__main__":
    unittest.main()
