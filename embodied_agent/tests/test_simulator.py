from __future__ import annotations

import math
import os
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.simulator import create_pick_place_simulation


class SimulatorMultiObjectTests(unittest.TestCase):
    def test_factory_uses_backend_environment_default(self) -> None:
        with patch.dict(os.environ, {"EMBODIED_SIM_BACKEND": "isaaclab"}):
            with self.assertRaisesRegex(RuntimeError, "IsaacLab"):
                create_pick_place_simulation(gui=False)

    def test_factory_explicit_backend_overrides_environment_default(self) -> None:
        with patch.dict(os.environ, {"EMBODIED_SIM_BACKEND": "isaaclab"}):
            simulation = create_pick_place_simulation(backend="mujoco", gui=False)
        try:
            self.assertEqual(simulation.__class__.__name__, "MujocoPickPlaceSimulation")
        finally:
            simulation.shutdown()

    def test_mujoco_factory_does_not_import_pybullet(self) -> None:
        sys.modules.pop("pybullet", None)
        sys.modules.pop("pybullet_data", None)

        simulation = create_pick_place_simulation(backend="mujoco", gui=False)
        try:
            self.assertNotIn("pybullet", sys.modules)
            self.assertNotIn("pybullet_data", sys.modules)
        finally:
            simulation.shutdown()

    def test_creates_only_requested_scene_entities(self) -> None:
        simulation = create_pick_place_simulation(
            backend="mujoco",
            gui=False,
            object_names=("red_block", "blue_block"),
            zone_names=("green_zone", "blue_zone"),
        )
        try:
            self.assertEqual(set(simulation.object_ids), {"red_block", "blue_block"})
            self.assertEqual(set(simulation.zone_ids), {"green_zone", "blue_zone"})
        finally:
            simulation.shutdown()

    def test_samples_multi_object_layout_with_pairwise_separation(self) -> None:
        simulation = create_pick_place_simulation(
            backend="mujoco",
            gui=False,
            object_names=("red_block", "blue_block", "yellow_block"),
            zone_names=("green_zone", "blue_zone", "yellow_zone"),
        )
        try:
            object_layout, zone_layout = simulation.sample_scene_layout(
                rng=random.Random(7),
                min_separation=0.12,
            )
            positions = [*object_layout.values(), *zone_layout.values()]
            self.assertEqual(set(object_layout), {"red_block", "blue_block", "yellow_block"})
            self.assertEqual(set(zone_layout), {"green_zone", "blue_zone", "yellow_zone"})
            for index, position in enumerate(positions):
                for other_position in positions[index + 1 :]:
                    self.assertGreaterEqual(math.dist(position, other_position), 0.12)
        finally:
            simulation.shutdown()

    def test_single_task_layout_respects_custom_ranges(self) -> None:
        simulation = create_pick_place_simulation(backend="mujoco", gui=False)
        try:
            object_xy, zone_xy = simulation.sample_task_layout(
                rng=random.Random(11),
                min_separation=0.05,
                object_x_range=(0.62, 0.68),
                object_y_range=(-0.16, -0.12),
                zone_x_range=(0.40, 0.44),
                zone_y_range=(0.20, 0.24),
            )
            self.assertGreaterEqual(object_xy[0], 0.62)
            self.assertLessEqual(object_xy[0], 0.68)
            self.assertGreaterEqual(object_xy[1], -0.16)
            self.assertLessEqual(object_xy[1], -0.12)
            self.assertGreaterEqual(zone_xy[0], 0.40)
            self.assertLessEqual(zone_xy[0], 0.44)
            self.assertGreaterEqual(zone_xy[1], 0.20)
            self.assertLessEqual(zone_xy[1], 0.24)
        finally:
            simulation.shutdown()

    def test_single_task_layout_can_use_candidate_slots(self) -> None:
        simulation = create_pick_place_simulation(backend="mujoco", gui=False)
        object_candidates = ((0.40, -0.16), (0.68, -0.16))
        zone_candidates = ((0.40, 0.12), (0.64, 0.24))
        try:
            object_xy, zone_xy = simulation.sample_task_layout(
                rng=random.Random(5),
                min_separation=0.05,
                object_candidates=object_candidates,
                zone_candidates=zone_candidates,
            )
            self.assertIn(object_xy, object_candidates)
            self.assertIn(zone_xy, zone_candidates)
        finally:
            simulation.shutdown()

    def test_restore_runtime_state_recovers_held_place_entry(self) -> None:
        simulation = create_pick_place_simulation(backend="mujoco", gui=False)
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
                math.dist(captured_state["end_effector_position"], restored_state["end_effector_position"]),
                0.03,
            )
            self.assertLess(
                math.dist(captured_state["object_position"], restored_state["object_position"]),
                0.03,
            )
        finally:
            simulation.shutdown()

    def test_multi_object_reset_capture_restore_pick_place(self) -> None:
        simulation = create_pick_place_simulation(
            backend="mujoco",
            gui=False,
            object_names=("red_block", "blue_block", "yellow_block"),
            zone_names=("green_zone", "blue_zone", "yellow_zone"),
        )
        try:
            simulation.reset_task(
                object_layout={"blue_block": (0.62, -0.14)},
                zone_layout={"yellow_zone": (0.46, 0.22)},
            )
            captured_state = simulation.capture_skill_state("blue_block", "yellow_zone")

            simulation.reset_task(
                object_layout={"blue_block": (0.68, -0.04)},
                zone_layout={"yellow_zone": (0.64, 0.18)},
            )
            simulation.restore_runtime_state(
                captured_state,
                object_name="blue_block",
                zone_name="yellow_zone",
            )
            restored_state = simulation.capture_skill_state("blue_block", "yellow_zone")
            self.assertLess(
                math.dist(captured_state["object_position"], restored_state["object_position"]),
                1e-6,
            )
            self.assertLess(
                math.dist(captured_state["zone_position"], restored_state["zone_position"]),
                1e-6,
            )

            simulation.pick_object("blue_block")
            self.assertTrue(simulation.is_pick_success("blue_block"))
            simulation.place_object("yellow_zone")
            self.assertTrue(simulation.is_place_success("blue_block", "yellow_zone"))
        finally:
            simulation.shutdown()
