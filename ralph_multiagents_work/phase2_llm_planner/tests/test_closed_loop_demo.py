"""Tests for closed_loop_demo.py — integration test for full pipeline."""

import json
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src'))

from src.demo.closed_loop_demo import (
    ClosedLoopDemo,
    DemoConfig,
    DemoResult,
    make_mock_causal_output,
    make_mock_world_state,
    _mock_instruction_llm,
)


class TestMockInstructionLLM(unittest.TestCase):

    def test_mock_llm_parses_chinese_pick_place(self):
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "把红色方块移到绿色区域"},
        ]
        response = _mock_instruction_llm(messages)
        data = json.loads(response)
        steps = data["steps"]
        skills = [s["selected_skill"] for s in steps]
        self.assertIn("observe", skills)
        self.assertIn("pick", skills)
        self.assertIn("place", skills)

    def test_mock_llm_keeps_target_object_separate_from_target_zone(self):
        messages = [
            {"role": "user", "content": "把蓝色方块移到黄色区域"},
        ]
        response = _mock_instruction_llm(messages)
        data = json.loads(response)
        steps = data["steps"]

        self.assertEqual(
            [s["selected_skill"] for s in steps],
            ["observe", "pick", "place"],
        )
        self.assertEqual(steps[1]["target_object"], "blue_block")
        self.assertEqual(steps[2]["target_object"], "blue_block")
        self.assertEqual(steps[2]["skill_args"]["target_zone"], "yellow_zone")

    def test_mock_llm_parses_mouse_interaction(self):
        messages = [
            {"role": "user", "content": "探索鼠标的功能：先按压，然后向前拖动"},
        ]
        response = _mock_instruction_llm(messages)
        data = json.loads(response)
        steps = data["steps"]
        skills = [s["selected_skill"] for s in steps]
        self.assertIn("probe", skills)
        self.assertIn("press", skills)
        self.assertIn("push", skills)

    def test_mock_llm_multi_block(self):
        messages = [
            {"role": "user", "content": "把红色方块移到绿色区域，把蓝色方块移到黄色区域"},
        ]
        response = _mock_instruction_llm(messages)
        data = json.loads(response)
        steps = data["steps"]
        pick_steps = [s for s in steps if s["selected_skill"] == "pick"]
        self.assertGreaterEqual(len(pick_steps), 2)

    def test_mock_llm_no_continuous_control(self):
        messages = [
            {"role": "user", "content": "把红色方块移到绿色区域"},
        ]
        response = _mock_instruction_llm(messages)
        data = json.loads(response)
        forbidden = [
            "joint_positions", "joint_position", "motor_torques",
            "torques", "cartesian_trajectory", "raw_actions", "continuous_action",
        ]
        for step in data["steps"]:
            args = step.get("skill_args", {})
            for key in forbidden:
                self.assertNotIn(key, args)

    def test_mock_llm_replan_mode(self):
        messages = [
            {"role": "user", "content": "上次执行失败了。把红色方块移到绿色区域"},
        ]
        response = _mock_instruction_llm(messages)
        data = json.loads(response)
        self.assertIsNotNone(data)


class TestMockHelpers(unittest.TestCase):

    def test_make_mock_causal_output(self):
        co = make_mock_causal_output("red_block", uncertainty=0.45)
        self.assertEqual(co.object_id, "red_block")
        self.assertEqual(co.uncertainty_score, 0.45)
        self.assertFalse(co.requires_probe())

    def test_make_mock_causal_output_high_uncertainty(self):
        co = make_mock_causal_output("blue_block", uncertainty=0.60)
        self.assertTrue(co.requires_probe())

    def test_make_mock_world_state(self):
        ws = make_mock_world_state("test")
        self.assertIn("red_block", ws.object_positions)
        self.assertIn("green_zone", ws.zone_positions)

    def test_make_mock_world_state_custom(self):
        ws = make_mock_world_state(
            instruction="custom",
            objects={"cube": (0.5, 0.5, 0.6)},
            zones={"target": (0.6, 0.6, 0.6)},
        )
        self.assertIn("cube", ws.object_positions)
        self.assertIn("target", ws.zone_positions)


class TestClosedLoopDemo(unittest.TestCase):

    def test_dry_run_success(self):
        config = DemoConfig(dry_run=True, headless=True)
        demo = ClosedLoopDemo(config)
        result = demo.run("把红色方块移到绿色区域")
        self.assertTrue(result.success)
        self.assertGreater(len(result.executed_steps), 0)

    def test_dry_run_multi_task(self):
        config = DemoConfig(dry_run=True, headless=True)
        demo = ClosedLoopDemo(config)
        for instruction in [
            "把红色方块移到绿色区域",
            "把蓝色方块移到黄色区域",
        ]:
            result = demo.run(instruction)
            self.assertTrue(result.success)

    def test_dry_run_mouse_interaction(self):
        config = DemoConfig(dry_run=True, headless=True)
        demo = ClosedLoopDemo(config)
        result = demo.run("探索鼠标的功能：先按压，然后向前拖动")
        self.assertTrue(result.success)
        skills = result.executed_steps
        self.assertIn("observe", skills)

    def test_demo_result_structure(self):
        config = DemoConfig(dry_run=True)
        demo = ClosedLoopDemo(config)
        result = demo.run("test")
        self.assertIsInstance(result, DemoResult)
        self.assertIsNotNone(result.instruction)
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.plan_steps, list)

    def test_language_config(self):
        config = DemoConfig(language="en", dry_run=True)
        self.assertEqual(config.language, "en")
        demo = ClosedLoopDemo(config)
        result = demo.run("move red block to green zone")
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
