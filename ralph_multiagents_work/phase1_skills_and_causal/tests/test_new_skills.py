from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src')))

from src.skills.new_skills import PressSkill, PushSkill, PullSkill, RotateSkill
from embodied_agent.contracts import ExecutorResult, PlannerStep
from embodied_agent.simulator import create_pick_place_simulation


class TestPressSkill(unittest.TestCase):
    def setUp(self):
        self.skill = PressSkill()

    def test_execute_returns_executor_result(self):
        step = PlannerStep(
            task_id="test_press",
            step_index=0,
            selected_skill="press",
            target_object="red_block",
            skill_args={"press_direction": "down", "force": 0.5},
        )
        self.assertEqual(step.selected_skill, "press")
        self.assertEqual(step.target_object, "red_block")

    def test_skill_params_default_values(self):
        self.assertEqual(self.skill.skill_name, "press")

    def test_build_result_structure(self):
        result = self.skill._build_result(
            success=True,
            reward=1.0,
            final_state={"test": True},
            contact_region="top",
        )
        self.assertIsInstance(result, ExecutorResult)
        self.assertTrue(result.success)
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.contact_state.contact_region, "top")

    def test_build_result_failure_structure(self):
        result = self.skill._build_result(
            success=False,
            reward=0.0,
            final_state={"test": False},
            contact_region=None,
            error_code="press_failed",
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "press_failed")

    def test_smoke_execute_on_simulator(self):
        sim = create_pick_place_simulation(backend="mujoco", gui=False)
        try:
            result = self.skill.execute(sim, target_object="red_block", press_direction="down", force=0.5)
            self.assertIsInstance(result, ExecutorResult)
            self.assertTrue(result.success)
            self.assertEqual(result.final_state["executed_skill"], "press")
        finally:
            sim.shutdown()


class TestPushSkill(unittest.TestCase):
    def setUp(self):
        self.skill = PushSkill()

    def test_execute_returns_executor_result_type(self):
        self.assertEqual(self.skill.skill_name, "push")

    def test_planner_step_for_push(self):
        step = PlannerStep(
            task_id="test_push",
            step_index=0,
            selected_skill="push",
            target_object="red_block",
            skill_args={"push_direction": "forward", "distance": 0.10},
        )
        self.assertEqual(step.selected_skill, "push")
        self.assertEqual(step.skill_args["push_direction"], "forward")
        self.assertEqual(step.skill_args["distance"], 0.10)

    def test_smoke_execute_on_simulator(self):
        sim = create_pick_place_simulation(backend="mujoco", gui=False)
        try:
            result = self.skill.execute(sim, target_object="red_block", push_direction="forward", distance=0.10)
            self.assertIsInstance(result, ExecutorResult)
            self.assertTrue(result.success)
            self.assertEqual(result.final_state["executed_skill"], "push")
        finally:
            sim.shutdown()


class TestPullSkill(unittest.TestCase):
    def setUp(self):
        self.skill = PullSkill()

    def test_execute_returns_executor_result_type(self):
        self.assertEqual(self.skill.skill_name, "pull")

    def test_planner_step_for_pull(self):
        step = PlannerStep(
            task_id="test_pull",
            step_index=0,
            selected_skill="pull",
            target_object="red_block",
            skill_args={"pull_direction": "backward", "distance": 0.10},
        )
        self.assertEqual(step.selected_skill, "pull")
        self.assertEqual(step.skill_args["pull_direction"], "backward")

    def test_smoke_execute_on_simulator(self):
        sim = create_pick_place_simulation(backend="mujoco", gui=False)
        try:
            result = self.skill.execute(sim, target_object="red_block", pull_direction="backward", distance=0.10)
            self.assertIsInstance(result, ExecutorResult)
            self.assertTrue(result.success)
            self.assertEqual(result.final_state["executed_skill"], "pull")
        finally:
            sim.shutdown()


class TestRotateSkill(unittest.TestCase):
    def setUp(self):
        self.skill = RotateSkill()

    def test_execute_returns_executor_result_type(self):
        self.assertEqual(self.skill.skill_name, "rotate")

    def test_planner_step_for_rotate(self):
        step = PlannerStep(
            task_id="test_rotate",
            step_index=0,
            selected_skill="rotate",
            target_object="red_block",
            skill_args={"rotation_axis": "z", "angle": 45.0},
        )
        self.assertEqual(step.selected_skill, "rotate")
        self.assertEqual(step.skill_args["rotation_axis"], "z")
        self.assertEqual(step.skill_args["angle"], 45.0)

    def test_smoke_execute_on_simulator(self):
        sim = create_pick_place_simulation(backend="mujoco", gui=False)
        try:
            result = self.skill.execute(sim, target_object="red_block", rotation_axis="z", angle=45.0)
            self.assertIsInstance(result, ExecutorResult)
            self.assertTrue(result.success)
            self.assertEqual(result.final_state["executed_skill"], "rotate")
        finally:
            sim.shutdown()


class TestNewSkillsIntegration(unittest.TestCase):
    """Integration-style tests that validate all skills produce valid ExecutorResults."""

    def test_all_skills_build_valid_results(self):
        skills = [
            PressSkill(),
            PushSkill(),
            PullSkill(),
            RotateSkill(),
        ]
        for skill in skills:
            result = skill._build_result(
                success=True,
                reward=0.8,
                final_state={"executed_skill": skill.skill_name},
                contact_region="side",
            )
            self.assertIsInstance(result, ExecutorResult)
            self.assertIn("executed_skill", result.final_state)

    def test_planner_steps_for_all_new_skills(self):
        test_cases = [
            ("press", "red_block", {"press_direction": "down", "force": 0.5}),
            ("push", "red_block", {"push_direction": "forward", "distance": 0.10}),
            ("pull", "red_block", {"pull_direction": "backward", "distance": 0.10}),
            ("rotate", "red_block", {"rotation_axis": "z", "angle": 45.0}),
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
            self.assertIsNotNone(step.target_object)


if __name__ == "__main__":
    unittest.main()
