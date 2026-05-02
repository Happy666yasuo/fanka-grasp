from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src')))

from src.skills.place_improvements import (
    improve_place_release_timing,
    improve_place_transport,
    improved_place_object,
)
from embodied_agent.contracts import ExecutorResult
from embodied_agent.simulator import create_pick_place_simulation


class TestPlaceReleaseTiming(unittest.TestCase):
    def setUp(self):
        self.sim = create_pick_place_simulation(backend="mujoco", gui=False)

    def tearDown(self):
        self.sim.shutdown()

    def test_release_timing_no_held_object_reports_error(self):
        debug = improve_place_release_timing(self.sim, "green_zone")
        self.assertIn("error", debug)
        self.assertEqual(debug["error"], "no_held_object")

    def test_release_timing_returns_dict_with_expected_keys(self):
        self.sim.pick_object("red_block")
        debug = improve_place_release_timing(self.sim, "green_zone")
        self.assertIn("object_name", debug)
        self.assertIn("zone_name", debug)
        self.assertIn("zone_proximity_ok", debug)
        self.assertIn("adjustments_made", debug)

    def test_release_timing_detects_zone_proximity(self):
        self.sim.pick_object("red_block")
        zone_pos = self.sim.zone_positions["green_zone"]
        config = self.sim.config
        hover_pos = (
            zone_pos[0],
            zone_pos[1],
            config.table_top_z + config.hover_height,
        )
        self.sim.teleport_end_effector(hover_pos)
        self.sim._simulate(30)

        debug = improve_place_release_timing(self.sim, "green_zone", tolerance=0.08)
        self.assertTrue(debug["zone_proximity_ok"])


class TestPlaceTransport(unittest.TestCase):
    def setUp(self):
        self.sim = create_pick_place_simulation(backend="mujoco", gui=False)

    def tearDown(self):
        self.sim.shutdown()

    def test_transport_no_held_object_reports_error(self):
        debug = improve_place_transport(self.sim, "green_zone")
        self.assertIn("error", debug)
        self.assertEqual(debug["error"], "no_held_object")

    def test_transport_with_held_object_succeeds(self):
        self.sim.pick_object("red_block")
        debug = improve_place_transport(self.sim, "green_zone")
        self.assertTrue(debug["transport_success"])
        self.assertEqual(debug["stages_completed"], 4)


class TestImprovedPlaceObject(unittest.TestCase):
    def setUp(self):
        self.sim = create_pick_place_simulation(backend="mujoco", gui=False)

    def tearDown(self):
        self.sim.shutdown()

    def test_improved_place_returns_executor_result(self):
        self.sim.pick_object("red_block")
        result = improved_place_object(self.sim, "green_zone")
        self.assertIsInstance(result, ExecutorResult)

    def test_improved_place_succeeds_with_held_object(self):
        self.sim.pick_object("red_block")
        result = improved_place_object(self.sim, "green_zone")
        self.assertTrue(result.success)
        self.assertEqual(result.reward, 1.0)
        self.assertIn("executed_skill", result.final_state)

    def test_improved_place_fails_without_held_object(self):
        result = improved_place_object(self.sim, "green_zone")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "place_failed: no_held_object")

    def test_improved_place_object_in_zone(self):
        self.sim.pick_object("red_block")
        result = improved_place_object(self.sim, "green_zone")
        self.assertTrue(result.success)
        in_zone = self.sim.is_object_in_zone("red_block", "green_zone")
        self.assertTrue(in_zone)

    def test_multiple_pick_place_sequence(self):
        """Run multiple pick-place cycles to verify robustness."""
        success_count = 0
        for _ in range(5):
            self.sim.reset_task()
            self.sim.pick_object("red_block")
            result = improved_place_object(self.sim, "green_zone")
            if result.success:
                success_count += 1
        self.assertGreaterEqual(success_count, 2)


if __name__ == "__main__":
    unittest.main()
