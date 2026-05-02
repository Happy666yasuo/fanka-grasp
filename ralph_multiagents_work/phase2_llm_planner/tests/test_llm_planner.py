"""Tests for llm_planner.py — LLMPlanner schema validation and protocol compliance."""

import json
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src'))

from embodied_agent.types import PlanStep, PostCondition, WorldState, StepFailure

from src.planner.llm_planner import LLMPlanner, _default_llm_call
from src.planner.prompt_templates import ALLOWED_SKILLS, FORBIDDEN_CONTROL_ARGS


def _make_mock_llm(steps_config: list[dict] | None = None):
    """Create a mock LLM callable that returns the given steps."""
    default_steps = [
        {
            "step_index": 0,
            "selected_skill": "observe",
            "target_object": None,
            "skill_args": {},
            "preconditions": ["object_visible"],
            "expected_effect": "scene_observed",
            "fallback_action": "probe_or_replan",
        },
        {
            "step_index": 1,
            "selected_skill": "pick",
            "target_object": "red_block",
            "skill_args": {"zone": "green_zone"},
            "preconditions": ["object_visible"],
            "expected_effect": "holding(red_block)",
            "fallback_action": "probe_or_replan",
        },
        {
            "step_index": 2,
            "selected_skill": "place",
            "target_object": "red_block",
            "skill_args": {"object": "red_block", "target_zone": "green_zone"},
            "preconditions": ["holding(red_block)"],
            "expected_effect": "placed(red_block,green_zone)",
            "fallback_action": "probe_or_replan",
        },
    ]
    steps = steps_config if steps_config is not None else default_steps

    def mock_llm(messages: list[dict[str, str]]) -> str:
        return json.dumps({"task_id": "test_task", "steps": steps}, ensure_ascii=False)

    return mock_llm


def _make_world_state():
    return WorldState(
        instruction="把红色方块移到绿色区域",
        object_positions={"red_block": (0.55, 0.10, 0.645)},
        zone_positions={"green_zone": (0.50, -0.10, 0.622)},
        end_effector_position=(0.45, 0.0, 0.80),
        held_object_name=None,
    )


class TestLLMPlanner(unittest.TestCase):

    def test_default_llm_call_returns_valid_json(self):
        response = _default_llm_call([])
        data = json.loads(response)
        self.assertIn("steps", data)
        self.assertEqual(len(data["steps"]), 1)
        self.assertEqual(data["steps"][0]["selected_skill"], "observe")

    def test_plan_returns_list_of_plan_steps(self):
        planner = LLMPlanner(llm_callable=_make_mock_llm())
        state = _make_world_state()
        plan = planner.plan("把红色方块移到绿色区域", state)
        self.assertIsInstance(plan, list)
        self.assertGreater(len(plan), 0)
        for step in plan:
            self.assertIsInstance(step, PlanStep)

    def test_plan_includes_observe_first(self):
        planner = LLMPlanner(llm_callable=_make_mock_llm())
        state = _make_world_state()
        plan = planner.plan("把红色方块移到绿色区域", state)
        self.assertEqual(plan[0].action, "observe")

    def test_plan_pick_place_sequence(self):
        planner = LLMPlanner(llm_callable=_make_mock_llm())
        state = _make_world_state()
        plan = planner.plan("把红色方块移到绿色区域", state)
        actions = [s.action for s in plan]
        self.assertIn("observe", actions)
        self.assertIn("pick", actions)
        self.assertIn("place", actions)
        observe_idx = actions.index("observe")
        pick_idx = actions.index("pick")
        place_idx = actions.index("place")
        self.assertLess(observe_idx, pick_idx)
        self.assertLess(pick_idx, place_idx)

    def test_plan_step_has_valid_post_condition(self):
        planner = LLMPlanner(llm_callable=_make_mock_llm())
        state = _make_world_state()
        plan = planner.plan("把红色方块移到绿色区域", state)
        pick_step = [s for s in plan if s.action == "pick"][0]
        self.assertIsNotNone(pick_step.post_condition)
        self.assertEqual(pick_step.post_condition.kind, "holding")

    def test_plan_rejects_invalid_skill(self):
        bad_steps = [{
            "step_index": 0,
            "selected_skill": "fly",
            "target_object": None,
            "skill_args": {},
            "preconditions": [],
            "expected_effect": None,
            "fallback_action": "probe_or_replan",
        }]
        planner = LLMPlanner(llm_callable=_make_mock_llm(bad_steps))
        state = _make_world_state()
        with self.assertRaises((ValueError, KeyError)):
            planner.plan("test", state)

    def test_plan_rejects_continuous_control_args(self):
        bad_steps = [{
            "step_index": 0,
            "selected_skill": "push",
            "target_object": "red_block",
            "skill_args": {"joint_positions": [0.1, 0.2]},
            "preconditions": [],
            "expected_effect": None,
            "fallback_action": "probe_or_replan",
        }]
        planner = LLMPlanner(llm_callable=_make_mock_llm(bad_steps))
        state = _make_world_state()
        with self.assertRaises((ValueError, KeyError)):
            planner.plan("test", state)

    def test_replan_returns_new_plan(self):
        planner = LLMPlanner(llm_callable=_make_mock_llm())
        state = _make_world_state()
        failed_step = PlanStep(action="pick", target="red_block")
        failure = StepFailure(
            failed_step=failed_step,
            source="execution_error",
            reason="gripper missed",
            replan_attempt=1,
        )
        new_plan = planner.replan("test", state, failed_step, [], failure)
        self.assertIsInstance(new_plan, list)
        self.assertGreater(len(new_plan), 0)

    def test_update_causal_outputs(self):
        planner = LLMPlanner()
        self.assertEqual(planner.causal_outputs, {})
        planner.update_causal_outputs({"red_block": {"uncertainty_score": 0.5}})
        self.assertIn("red_block", planner.causal_outputs)

    def test_plan_with_causal_outputs(self):
        causal = {
            "red_block": {
                "uncertainty_score": 0.35,
                "affordance_candidates": [
                    {"name": "graspable", "confidence": 0.90},
                ],
                "property_belief": {},
                "recommended_probe": None,
                "contact_region": "top",
            },
        }
        planner = LLMPlanner(
            llm_callable=_make_mock_llm(),
            causal_outputs=causal,
        )
        state = _make_world_state()
        plan = planner.plan("把红色方块移到绿色区域", state)
        self.assertIsInstance(plan, list)
        self.assertGreater(len(plan), 0)

    def test_world_state_to_dict(self):
        state = _make_world_state()
        d = LLMPlanner._world_state_to_dict(state)
        self.assertEqual(d["instruction"], "把红色方块移到绿色区域")
        self.assertIn("red_block", d["object_positions"])
        self.assertIn("green_zone", d["zone_positions"])
        self.assertEqual(d["held_object_name"], None)

    def test_plan_all_steps_use_allowed_skills(self):
        planner = LLMPlanner(llm_callable=_make_mock_llm())
        state = _make_world_state()
        plan = planner.plan("把红色方块移到绿色区域", state)
        for step in plan:
            self.assertIn(step.action, ALLOWED_SKILLS)


if __name__ == "__main__":
    unittest.main()
