"""Tests for uncertainty_handler.py — probe injection on high uncertainty."""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src'))

from embodied_agent.contracts import (
    CausalExploreOutput,
    PlannerStep,
    PropertyBelief,
    AffordanceCandidate,
    PROBE_UNCERTAINTY_THRESHOLD,
    LOW_AFFORDANCE_CONFIDENCE_THRESHOLD,
)

from src.planner.uncertainty_handler import UncertaintyHandler


def _make_output(
    object_id: str = "red_block",
    uncertainty: float = 0.35,
    affordances: list[tuple[str, float]] | None = None,
) -> CausalExploreOutput:
    if affordances is None:
        affordances = [("pushable", 0.85), ("graspable", 0.90)]
    return CausalExploreOutput(
        scene_id="test_scene",
        object_id=object_id,
        object_category="block",
        property_belief={
            "movable": PropertyBelief(label="movable", confidence=0.90),
        },
        affordance_candidates=[
            AffordanceCandidate(name=n, confidence=c) for n, c in affordances
        ],
        uncertainty_score=uncertainty,
        recommended_probe="lateral_push" if uncertainty >= 0.50 else None,
        contact_region="top",
        skill_constraints={},
        artifact_path="mock://test",
    )


def _make_plan(task_id: str = "test") -> list[PlannerStep]:
    return [
        PlannerStep(
            task_id=task_id, step_index=0,
            selected_skill="observe", target_object=None,
            skill_args={}, preconditions=[],
            expected_effect=None, fallback_action="probe_or_replan",
        ),
        PlannerStep(
            task_id=task_id, step_index=1,
            selected_skill="pick", target_object="red_block",
            skill_args={"zone": "green_zone"},
            preconditions=["object_visible"],
            expected_effect="holding(red_block)",
            fallback_action="probe_or_replan",
        ),
        PlannerStep(
            task_id=task_id, step_index=2,
            selected_skill="place", target_object="red_block",
            skill_args={"object": "red_block", "target_zone": "green_zone"},
            preconditions=["holding(red_block)"],
            expected_effect="placed(red_block,green_zone)",
            fallback_action="probe_or_replan",
        ),
    ]


class TestUncertaintyHandler(unittest.TestCase):

    def test_low_uncertainty_no_probe_needed(self):
        output = _make_output(uncertainty=0.30)
        self.assertFalse(UncertaintyHandler.requires_probe(output))

    def test_high_uncertainty_probe_needed(self):
        output = _make_output(uncertainty=0.55)
        self.assertTrue(UncertaintyHandler.requires_probe(output))

    def test_exact_threshold_probe_needed(self):
        output = _make_output(uncertainty=0.50)
        self.assertTrue(UncertaintyHandler.requires_probe(output))

    def test_low_affordance_confidence_probe_needed(self):
        output = _make_output(
            uncertainty=0.30,
            affordances=[("graspable", 0.60), ("pushable", 0.85)],
        )
        self.assertTrue(UncertaintyHandler.requires_probe(output))

    def test_high_confidence_no_probe_needed(self):
        output = _make_output(
            uncertainty=0.25,
            affordances=[("graspable", 0.90), ("pushable", 0.88)],
        )
        self.assertFalse(UncertaintyHandler.requires_probe(output))

    def test_requires_probe_from_dict(self):
        d = {
            "uncertainty_score": 0.60,
            "affordance_candidates": [
                {"name": "graspable", "confidence": 0.85},
            ],
        }
        self.assertTrue(UncertaintyHandler.requires_probe_from_dict(d))

    def test_requires_probe_from_dict_low_affordance(self):
        d = {
            "uncertainty_score": 0.30,
            "affordance_candidates": [
                {"name": "graspable", "confidence": 0.60},
            ],
        }
        self.assertTrue(UncertaintyHandler.requires_probe_from_dict(d))

    def test_requires_probe_from_dict_all_good(self):
        d = {
            "uncertainty_score": 0.25,
            "affordance_candidates": [
                {"name": "graspable", "confidence": 0.90},
            ],
        }
        self.assertFalse(UncertaintyHandler.requires_probe_from_dict(d))

    def test_get_recommended_probe(self):
        output = _make_output(uncertainty=0.55)
        self.assertEqual(
            UncertaintyHandler.get_recommended_probe(output), "lateral_push",
        )

    def test_get_recommended_probe_none_when_low(self):
        output = _make_output(uncertainty=0.30)
        self.assertIsNone(UncertaintyHandler.get_recommended_probe(output))

    def test_evaluate_returns_probe_decision_payload(self):
        output = _make_output(uncertainty=0.55)
        decision = UncertaintyHandler.evaluate(output)

        self.assertTrue(decision["needs_probe"])
        self.assertEqual(decision["recommended_probe"], "lateral_push")
        self.assertEqual(decision["object_id"], "red_block")
        self.assertEqual(decision["uncertainty_score"], 0.55)
        self.assertEqual(decision["reason"], "high_uncertainty")

    def test_evaluate_reports_low_confidence_affordance_reason(self):
        output = _make_output(
            uncertainty=0.25,
            affordances=[("graspable", 0.60), ("pushable", 0.85)],
        )
        decision = UncertaintyHandler.evaluate(output)

        self.assertTrue(decision["needs_probe"])
        self.assertEqual(decision["reason"], "low_affordance_confidence")
        self.assertEqual(decision["low_confidence_affordances"], ["graspable"])

    def test_should_probe_any(self):
        outputs = {
            "red_block": _make_output("red_block", uncertainty=0.30),
            "blue_block": _make_output("blue_block", uncertainty=0.55),
        }
        uncertain = UncertaintyHandler.should_probe_any(outputs)
        self.assertEqual(uncertain, ["blue_block"])

    def test_should_probe_any_none(self):
        outputs = {
            "red_block": _make_output("red_block", uncertainty=0.25),
        }
        uncertain = UncertaintyHandler.should_probe_any(outputs)
        self.assertEqual(uncertain, [])

    def test_inject_probe_step(self):
        plan = _make_plan()
        output = _make_output(uncertainty=0.55)
        new_plan = UncertaintyHandler.inject_probe_step(
            plan, "test", 1, output,
        )
        self.assertEqual(len(new_plan), len(plan) + 1)
        self.assertEqual(new_plan[1].selected_skill, "probe")
        self.assertEqual(new_plan[1].target_object, "red_block")

    def test_inject_probe_step_preserves_order(self):
        plan = _make_plan()
        output = _make_output(uncertainty=0.55)
        new_plan = UncertaintyHandler.inject_probe_step(
            plan, "test", 1, output,
        )
        self.assertEqual(new_plan[0].selected_skill, "observe")
        self.assertEqual(new_plan[1].selected_skill, "probe")
        self.assertEqual(new_plan[2].selected_skill, "pick")
        self.assertEqual(new_plan[3].selected_skill, "place")

    def test_inject_probe_step_reindexes(self):
        plan = _make_plan()
        output = _make_output(uncertainty=0.55)
        new_plan = UncertaintyHandler.inject_probe_step(
            plan, "test", 1, output,
        )
        for i, step in enumerate(new_plan):
            self.assertEqual(step.step_index, i)

    def test_inject_probes_for_all_uncertain(self):
        plan = _make_plan()
        outputs = {
            "red_block": _make_output("red_block", uncertainty=0.55),
        }
        new_plan = UncertaintyHandler.inject_probes_for_all_uncertain(
            plan, "test", outputs,
        )
        self.assertGreater(len(new_plan), len(plan))
        probe_steps = [s for s in new_plan if s.selected_skill == "probe"]
        self.assertEqual(len(probe_steps), 1)

    def test_get_uncertainty_summary(self):
        outputs = {
            "red_block": _make_output("red_block", uncertainty=0.55),
            "blue_block": _make_output("blue_block", uncertainty=0.25),
        }
        summary = UncertaintyHandler.get_uncertainty_summary(outputs)
        self.assertIn("red_block", summary)
        self.assertIn("blue_block", summary)
        self.assertTrue(summary["red_block"]["requires_probe"])
        self.assertFalse(summary["blue_block"]["requires_probe"])

    def test_inject_probe_step_invalid_index(self):
        plan = _make_plan()
        output = _make_output(uncertainty=0.55)
        result = UncertaintyHandler.inject_probe_step(plan, "test", -1, output)
        self.assertEqual(len(result), len(plan))
        result = UncertaintyHandler.inject_probe_step(plan, "test", 100, output)
        self.assertEqual(len(result), len(plan))

    def test_probe_threshold_constant(self):
        self.assertEqual(PROBE_UNCERTAINTY_THRESHOLD, 0.50)
        self.assertEqual(LOW_AFFORDANCE_CONFIDENCE_THRESHOLD, 0.70)


if __name__ == "__main__":
    unittest.main()
