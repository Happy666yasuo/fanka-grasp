from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.causal_explore.explore_strategies import (
    RandomStrategy,
    CuriosityDrivenStrategy,
    CausalExploreStrategy,
    ExploreHistory,
    ExploreStep,
)
from src.causal_explore.probe_actions import ProbeActionResult


def _make_result(probe_name: str, object_name: str, disp_mag: float, success: bool = True):
    return ProbeActionResult(
        probe_name=probe_name,
        object_name=object_name,
        success=success,
        pre_position=(0.0, 0.0, 0.05),
        post_position=(disp_mag, 0.0, 0.05),
        displacement=(disp_mag, 0.0, 0.0),
        displacement_magnitude=disp_mag,
        contact_detected=disp_mag > 0.001,
    )


class TestExploreHistory(unittest.TestCase):
    def test_empty_history(self):
        h = ExploreHistory()
        self.assertEqual(len(h.steps), 0)
        self.assertIsNone(h.last_result())
        self.assertEqual(h.last_step_index(), -1)

    def test_record_and_retrieve(self):
        h = ExploreHistory()
        result = _make_result("lateral_push", "red_block", 0.01)
        step = ExploreStep("lateral_push", "red_block", result, 0)
        h.record(step)
        self.assertEqual(len(h.steps), 1)
        self.assertEqual(h.last_step_index(), 0)
        self.assertEqual(h.probe_counts["lateral_push:red_block"], 1)
        self.assertEqual(h.object_visits["red_block"], 1)
        self.assertAlmostEqual(h.total_displacement["lateral_push:red_block"], 0.01)

    def test_multiple_steps_accumulate(self):
        h = ExploreHistory()
        h.record(ExploreStep("lateral_push", "red_block", _make_result("lateral_push", "red_block", 0.01), 0))
        h.record(ExploreStep("lateral_push", "red_block", _make_result("lateral_push", "red_block", 0.02), 1))
        h.record(ExploreStep("top_press", "red_block", _make_result("top_press", "red_block", 0.005), 2))
        self.assertEqual(len(h.steps), 3)
        self.assertAlmostEqual(h.total_displacement["lateral_push:red_block"], 0.03)
        self.assertEqual(h.probe_counts["top_press:red_block"], 1)


class TestRandomStrategy(unittest.TestCase):
    def test_select_next_returns_valid_pair(self):
        strategy = RandomStrategy()
        history = ExploreHistory()
        probes = ["lateral_push", "top_press"]
        objects = ["red_block", "blue_block"]
        probe, obj = strategy.select_next(history, probes, objects)
        self.assertIn(probe, probes)
        self.assertIn(obj, objects)

    def test_strategy_name(self):
        self.assertEqual(RandomStrategy().strategy_name, "random")


class TestCuriosityDrivenStrategy(unittest.TestCase):
    def test_select_next_with_empty_history(self):
        strategy = CuriosityDrivenStrategy()
        history = ExploreHistory()
        probes = ["lateral_push", "top_press"]
        objects = ["red_block"]
        probe, obj = strategy.select_next(history, probes, objects)
        self.assertIn(probe, probes)
        self.assertEqual(obj, "red_block")

    def test_prefers_high_displacement_pairs(self):
        strategy = CuriosityDrivenStrategy(temperature=0.1)
        history = ExploreHistory()
        history.record(ExploreStep("lateral_push", "red_block", _make_result("lateral_push", "red_block", 0.001), 0))
        history.record(ExploreStep("top_press", "red_block", _make_result("top_press", "red_block", 0.10), 1))
        probes = ["lateral_push", "top_press"]
        objects = ["red_block"]
        # With low temperature, high-displacement pair should dominate
        top_press_count = 0
        for _ in range(50):
            probe, _ = strategy.select_next(history, probes, objects)
            if probe == "top_press":
                top_press_count += 1
        self.assertGreater(top_press_count, 25)

    def test_strategy_name(self):
        self.assertEqual(CuriosityDrivenStrategy().strategy_name, "curiosity")


class TestCausalExploreStrategy(unittest.TestCase):
    def test_select_next_empty_history(self):
        from src.causal_explore.probe_executor import ProbeExecutor
        import sys, os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src')))
        from embodied_agent.simulator import create_pick_place_simulation

        sim = create_pick_place_simulation(backend="mujoco", gui=False)
        try:
            executor = ProbeExecutor(sim)
            strategy = CausalExploreStrategy(executor)
            history = ExploreHistory()
            probes = ["lateral_push", "top_press"]
            objects = ["red_block"]
            probe, obj = strategy.select_next(history, probes, objects)
            self.assertEqual(probe, "lateral_push")
            self.assertIn(obj, objects)
        finally:
            sim.shutdown()

    def test_select_next_with_history_explores_new_probes(self):
        from src.causal_explore.probe_executor import ProbeExecutor
        import sys, os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src')))
        from embodied_agent.simulator import create_pick_place_simulation

        sim = create_pick_place_simulation(backend="mujoco", gui=False)
        try:
            executor = ProbeExecutor(sim)
            strategy = CausalExploreStrategy(executor)
            history = ExploreHistory()
            history.record(ExploreStep("lateral_push", "red_block", _make_result("lateral_push", "red_block", 0.01), 0))
            probes = ["lateral_push", "top_press", "side_pull"]
            objects = ["red_block"]
            probe, _ = strategy.select_next(history, probes, objects)
            self.assertNotEqual(probe, "lateral_push")
        finally:
            sim.shutdown()

    def test_strategy_name(self):
        from src.causal_explore.probe_executor import ProbeExecutor
        import sys, os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src')))
        from embodied_agent.simulator import create_pick_place_simulation

        sim = create_pick_place_simulation(backend="mujoco", gui=False)
        try:
            executor = ProbeExecutor(sim)
            self.assertEqual(CausalExploreStrategy(executor).strategy_name, "causal_explore")
        finally:
            sim.shutdown()

    def test_mujoco_probe_executes_for_non_default_object(self):
        from src.causal_explore.probe_executor import ProbeExecutor
        import sys, os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src')))
        from embodied_agent.simulator import create_pick_place_simulation

        sim = create_pick_place_simulation(
            backend="mujoco",
            gui=False,
            object_names=("blue_block",),
            zone_names=("yellow_zone",),
        )
        try:
            executor = ProbeExecutor(sim)
            result = executor.execute_probe("lateral_push", "blue_block")
            self.assertEqual(result.object_name, "blue_block")
            self.assertEqual(result.probe_name, "lateral_push")
            self.assertTrue(result.success)
        finally:
            sim.shutdown()


if __name__ == "__main__":
    unittest.main()
