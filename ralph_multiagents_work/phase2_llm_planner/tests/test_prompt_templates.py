"""Tests for prompt_templates.py — template output format validation."""

import json
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src'))

from src.planner.prompt_templates import (
    build_system_prompt,
    build_task_prompt,
    build_replan_prompt,
    parse_llm_json_response,
    validate_step_dict,
    ALLOWED_SKILLS,
    FORBIDDEN_CONTROL_ARGS,
)


class TestPromptTemplates(unittest.TestCase):

    def test_build_system_prompt_chinese(self):
        prompt = build_system_prompt("zh")
        self.assertIn("具身智能体", prompt)
        self.assertIn("严禁输出", prompt)
        self.assertIn("关节位置", prompt)
        self.assertIn("step_index", prompt)
        self.assertIn("selected_skill", prompt)
        for skill in ALLOWED_SKILLS:
            self.assertIn(skill, prompt)

    def test_build_system_prompt_english(self):
        prompt = build_system_prompt("en")
        self.assertIn("embodied agent", prompt)
        self.assertIn("Strictly Forbidden", prompt)
        self.assertIn("joint_positions", prompt)
        for skill in ALLOWED_SKILLS:
            self.assertIn(skill, prompt)

    def test_build_system_prompt_forbids_continuous_control(self):
        for lang in ("zh", "en"):
            prompt = build_system_prompt(lang)
            self.assertIn("joint_positions", prompt)
            self.assertIn("motor_torques", prompt)
            self.assertIn("cartesian_trajectory", prompt)
            self.assertIn("continuous_action", prompt)

    def test_build_task_prompt_with_causal_outputs(self):
        causal = {
            "red_block": {
                "uncertainty_score": 0.35,
                "affordance_candidates": [
                    {"name": "pushable", "confidence": 0.85},
                    {"name": "graspable", "confidence": 0.90},
                ],
                "property_belief": {
                    "movable": {"label": "movable", "confidence": 0.90},
                },
                "recommended_probe": None,
                "contact_region": "top",
            },
        }
        prompt = build_task_prompt(
            "把红色方块移到绿色区域",
            causal_outputs=causal,
            language="zh",
        )
        self.assertIn("把红色方块移到绿色区域", prompt)
        self.assertIn("red_block", prompt)
        self.assertIn("0.35", prompt)
        self.assertIn("pushable", prompt)
        self.assertIn("graspable", prompt)

    def test_build_task_prompt_with_world_state(self):
        state = {
            "object_positions": {"red_block": [0.55, 0.10, 0.645]},
            "zone_positions": {"green_zone": [0.50, -0.10, 0.622]},
            "held_object_name": None,
            "end_effector_position": [0.45, 0.0, 0.80],
        }
        prompt = build_task_prompt(
            "pick red block",
            world_state=state,
            language="en",
        )
        self.assertIn("pick red block", prompt)
        self.assertIn("red_block", prompt)
        self.assertIn("green_zone", prompt)

    def test_build_replan_prompt(self):
        failures = [
            {
                "selected_skill": "pick",
                "failure_source": "execution_error",
                "reason": "gripper missed object",
                "replan_attempt": 1,
            },
        ]
        prompt = build_replan_prompt(
            "把红色方块移到绿色区域",
            failure_history=failures,
            language="zh",
        )
        self.assertIn("把红色方块移到绿色区域", prompt)
        self.assertIn("pick", prompt)
        self.assertIn("gripper missed object", prompt)
        self.assertIn("失败", prompt)

    def test_parse_llm_json_response_direct_json(self):
        response = json.dumps({
            "task_id": "test",
            "steps": [
                {
                    "step_index": 0,
                    "selected_skill": "observe",
                    "target_object": None,
                    "skill_args": {},
                    "preconditions": [],
                    "expected_effect": None,
                    "fallback_action": "probe_or_replan",
                },
            ],
        })
        steps = parse_llm_json_response(response)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["selected_skill"], "observe")

    def test_parse_llm_json_response_code_fence(self):
        response = """```json
{
  "task_id": "test",
  "steps": [
    {
      "step_index": 0,
      "selected_skill": "pick",
      "target_object": "red_block",
      "skill_args": {"zone": "green_zone"},
      "preconditions": ["object_visible"],
      "expected_effect": "holding(red_block)",
      "fallback_action": "probe_or_replan"
    }
  ]
}
```"""
        steps = parse_llm_json_response(response)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["selected_skill"], "pick")
        self.assertEqual(steps[0]["target_object"], "red_block")

    def test_parse_llm_json_response_list_format(self):
        response = json.dumps([
            {
                "step_index": 0,
                "selected_skill": "observe",
                "target_object": None,
                "skill_args": {},
                "preconditions": [],
                "expected_effect": None,
                "fallback_action": "probe_or_replan",
            },
            {
                "step_index": 1,
                "selected_skill": "pick",
                "target_object": "red_block",
                "skill_args": {},
                "preconditions": [],
                "expected_effect": None,
                "fallback_action": "probe_or_replan",
            },
        ])
        steps = parse_llm_json_response(response)
        self.assertEqual(len(steps), 2)

    def test_validate_step_dict_valid(self):
        step = {
            "step_index": 0,
            "selected_skill": "pick",
            "target_object": "red_block",
            "skill_args": {"zone": "green_zone"},
            "preconditions": [],
            "expected_effect": None,
            "fallback_action": "probe_or_replan",
        }
        validate_step_dict(step)

    def test_validate_step_dict_invalid_skill(self):
        step = {
            "step_index": 0,
            "selected_skill": "invalid_skill",
            "target_object": None,
            "skill_args": {},
            "preconditions": [],
            "expected_effect": None,
            "fallback_action": "probe_or_replan",
        }
        with self.assertRaises(ValueError):
            validate_step_dict(step)

    def test_validate_step_dict_forbidden_continuous_control(self):
        step = {
            "step_index": 0,
            "selected_skill": "pick",
            "target_object": "red_block",
            "skill_args": {"joint_positions": [0.1, 0.2, 0.3]},
            "preconditions": [],
            "expected_effect": None,
            "fallback_action": "probe_or_replan",
        }
        with self.assertRaises(ValueError):
            validate_step_dict(step)

    def test_validate_step_dict_motor_torques_forbidden(self):
        step = {
            "step_index": 0,
            "selected_skill": "push",
            "target_object": "red_block",
            "skill_args": {"motor_torques": [1.0, 2.0]},
            "preconditions": [],
            "expected_effect": None,
            "fallback_action": "probe_or_replan",
        }
        with self.assertRaises(ValueError):
            validate_step_dict(step)

    def test_validate_step_dict_missing_step_index(self):
        step = {
            "selected_skill": "observe",
            "target_object": None,
            "skill_args": {},
            "preconditions": [],
            "expected_effect": None,
            "fallback_action": "probe_or_replan",
        }
        with self.assertRaises(ValueError):
            validate_step_dict(step)

    def test_all_allowed_skills_present_in_system_prompt(self):
        prompt = build_system_prompt("zh")
        for skill in ALLOWED_SKILLS:
            with self.subTest(skill=skill):
                self.assertIn(skill, prompt)

    def test_no_continuous_control_in_system_prompt_allowed_skills(self):
        for skill in ALLOWED_SKILLS:
            self.assertNotIn(skill, FORBIDDEN_CONTROL_ARGS)


if __name__ == "__main__":
    unittest.main()
