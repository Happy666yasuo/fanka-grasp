"""Tests for replan_handler.py — failure classification and bounded retry logic."""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src'))

from embodied_agent.contracts import FailureRecord
from embodied_agent.types import PlanStep, PostCondition, StepFailure

from src.planner.replan_handler import ReplanHandler, MAX_REPLANS, MAX_SKILL_RETRIES


def _make_failure_record(
    skill: str = "pick",
    source: str = "execution_error",
    reason: str = "gripper missed",
    attempt: int = 0,
) -> FailureRecord:
    return FailureRecord(
        step_index=0,
        selected_skill=skill,
        failure_source=source,
        reason=reason,
        replan_attempt=attempt,
    )


def _make_step_failure(
    action: str = "pick",
    source: str = "execution_error",
    reason: str = "gripper missed",
    attempt: int = 1,
) -> StepFailure:
    return StepFailure(
        failed_step=PlanStep(action=action, target="red_block"),
        source=source,
        reason=reason,
        replan_attempt=attempt,
    )


class TestReplanHandler(unittest.TestCase):

    def test_classify_execution_error(self):
        failure = _make_failure_record(source="execution_error")
        self.assertEqual(
            ReplanHandler.classify_failure_source(failure), "execution_error",
        )

    def test_classify_planning_error(self):
        failure = _make_failure_record(source="planning_error")
        self.assertEqual(
            ReplanHandler.classify_failure_source(failure), "planning_error",
        )

    def test_classify_from_dict(self):
        failure_dict = {"failure_source": "execution_error"}
        self.assertEqual(
            ReplanHandler.classify_failure_source(failure_dict), "execution_error",
        )

    def test_is_execution_error_true(self):
        failure = _make_step_failure(source="execution_error")
        self.assertTrue(ReplanHandler.is_execution_error(failure))

    def test_is_execution_error_false_for_planning(self):
        failure = _make_step_failure(source="planning_error")
        self.assertFalse(ReplanHandler.is_execution_error(failure))

    def test_is_planning_error_true(self):
        failure = _make_step_failure(source="planning_error")
        self.assertTrue(ReplanHandler.is_planning_error(failure))

    def test_is_planning_error_keyword_match(self):
        failure = _make_step_failure(
            source="execution_error",
            reason="The plan is unreasonable for this scene",
        )
        self.assertTrue(ReplanHandler.is_planning_error(failure))

    def test_can_retry_skill_within_limit(self):
        history = [
            _make_failure_record(skill="pick"),
        ]
        self.assertTrue(ReplanHandler.can_retry_skill(history, "pick"))

    def test_can_retry_skill_exceeded(self):
        history = [
            _make_failure_record(skill="pick"),
            _make_failure_record(skill="pick"),
        ]
        self.assertFalse(ReplanHandler.can_retry_skill(history, "pick"))

    def test_can_retry_skill_different_skills(self):
        history = [
            _make_failure_record(skill="pick"),
            _make_failure_record(skill="pick"),
        ]
        self.assertTrue(ReplanHandler.can_retry_skill(history, "place"))

    def test_can_replan_within_limit(self):
        self.assertTrue(ReplanHandler.can_replan(0))
        self.assertTrue(ReplanHandler.can_replan(2))

    def test_can_replan_exceeded(self):
        self.assertFalse(ReplanHandler.can_replan(3))
        self.assertFalse(ReplanHandler.can_replan(5))

    def test_can_replan_exact_boundary(self):
        self.assertTrue(ReplanHandler.can_replan(2))
        self.assertFalse(ReplanHandler.can_replan(3))

    def test_extract_relevant_failures(self):
        history = [
            _make_failure_record(skill="observe", source="execution_error"),
            _make_failure_record(skill="pick", source="execution_error"),
            _make_failure_record(skill="place", source="planning_error"),
        ]
        extracted = ReplanHandler.extract_relevant_failures(history)
        self.assertEqual(len(extracted), 3)
        self.assertEqual(extracted[0]["selected_skill"], "observe")
        self.assertEqual(extracted[1]["selected_skill"], "pick")

    def test_extract_relevant_failures_from_dicts(self):
        history = [
            {"selected_skill": "pick", "failure_source": "execution_error",
             "reason": "missed", "replan_attempt": 0, "step_index": 0},
        ]
        extracted = ReplanHandler.extract_relevant_failures(history)
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0]["selected_skill"], "pick")

    def test_extract_relevant_failures_filter_by_step(self):
        f1 = _make_failure_record(skill="observe")
        f2 = FailureRecord(
            step_index=1, selected_skill="pick",
            failure_source="execution_error", reason="missed", replan_attempt=0,
        )
        extracted = ReplanHandler.extract_relevant_failures([f1, f2], current_step_index=0)
        self.assertEqual(len(extracted), 1)

    def test_determine_recovery_execution_error_retry(self):
        failure = _make_step_failure(action="pick", source="execution_error")
        strategy = ReplanHandler.determine_recovery_strategy(
            failure, [], 0,
        )
        self.assertEqual(strategy["action"], "retry_skill")

    def test_determine_recovery_execution_error_exhausted(self):
        failure = _make_step_failure(action="pick", source="execution_error")
        history = [
            _make_failure_record(skill="pick", source="execution_error"),
            _make_failure_record(skill="pick", source="execution_error"),
        ]
        strategy = ReplanHandler.determine_recovery_strategy(
            failure, history, 0,
        )
        self.assertEqual(strategy["action"], "replan")

    def test_determine_recovery_planning_error(self):
        failure = _make_step_failure(action="pick", source="planning_error")
        strategy = ReplanHandler.determine_recovery_strategy(
            failure, [], 0,
        )
        self.assertEqual(strategy["action"], "replan")

    def test_determine_recovery_max_replans_exhausted(self):
        failure = _make_step_failure(action="pick", source="execution_error")
        history = [
            _make_failure_record(skill="pick", source="execution_error"),
            _make_failure_record(skill="pick", source="execution_error"),
        ]
        strategy = ReplanHandler.determine_recovery_strategy(
            failure, history, 3,
        )
        self.assertEqual(strategy["action"], "abort")

    def test_max_replans_constant(self):
        self.assertEqual(MAX_REPLANS, 3)

    def test_max_skill_retries_constant(self):
        self.assertEqual(MAX_SKILL_RETRIES, 2)

    def test_determine_recovery_includes_remaining_replans(self):
        failure = _make_step_failure(action="pick", source="execution_error")
        strategy = ReplanHandler.determine_recovery_strategy(
            failure, [], 1,
        )
        self.assertIn("remaining_replans", strategy)
        self.assertEqual(strategy["remaining_replans"], 2)

    def test_determine_recovery_post_condition_failed(self):
        failure = _make_step_failure(
            action="place",
            source="post_condition_failed",
            reason="Object not in target zone",
        )
        strategy = ReplanHandler.determine_recovery_strategy(
            failure, [], 0,
        )
        self.assertEqual(strategy["action"], "replan")


if __name__ == "__main__":
    unittest.main()
