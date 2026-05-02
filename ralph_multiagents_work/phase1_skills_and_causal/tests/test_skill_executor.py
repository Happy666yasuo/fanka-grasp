from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src')))

from src.skills.skill_registry import SkillRegistry
from src.skills.skill_executor import UnifiedSkillExecutor
from embodied_agent.contracts import PlannerStep, ExecutorResult, ContactState


class TestUnifiedSkillExecutor(unittest.TestCase):
    def setUp(self):
        self.registry = SkillRegistry()

    def test_executor_creation(self):
        # Test that we can create executor without simulation (unit test mode)
        executor = UnifiedSkillExecutor.__new__(UnifiedSkillExecutor)
        executor.registry = self.registry
        executor.headless = True
        self.assertEqual(executor.headless, True)
        self.assertEqual(len(executor.registry), 6)

    def test_execute_all_with_empty_list(self):
        executor = UnifiedSkillExecutor.__new__(UnifiedSkillExecutor)
        executor.registry = self.registry
        results = executor.execute_all([])
        self.assertEqual(results, [])

    def test_planner_step_maps_to_registered_skill(self):
        """Verify that PlannerStep skill names match registry skills."""
        for skill_name in ["press", "push", "pull", "rotate"]:
            self.assertIn(skill_name, self.registry)

    def test_planner_step_factory(self):
        """Test that PlannerStep can be created for all new skills."""
        test_cases = [
            ("press", "red_block", {"press_direction": "down", "force": 0.8}),
            ("push", "blue_block", {"push_direction": "left", "distance": 0.15}),
            ("pull", "yellow_block", {"pull_direction": "backward", "distance": 0.12}),
            ("rotate", "red_block", {"rotation_axis": "z", "angle": 90.0}),
        ]

        for skill_name, target, args in test_cases:
            step = PlannerStep(
                task_id=f"test_{skill_name}",
                step_index=0,
                selected_skill=skill_name,
                target_object=target,
                skill_args=args,
            )
            self.assertEqual(step.selected_skill, skill_name)
            self.assertEqual(step.target_object, target)
            for key, value in args.items():
                self.assertEqual(step.skill_args[key], value)

    def test_planner_step_rejects_invalid_skill(self):
        with self.assertRaises(ValueError):
            PlannerStep(
                task_id="test",
                step_index=0,
                selected_skill="invalid_skill",
                target_object=None,
            )

    def test_planner_step_rejects_continuous_control_args(self):
        with self.assertRaises(ValueError):
            PlannerStep(
                task_id="test",
                step_index=0,
                selected_skill="push",
                target_object="red_block",
                skill_args={"joint_positions": [1.0, 2.0]},
            )


if __name__ == "__main__":
    unittest.main()
