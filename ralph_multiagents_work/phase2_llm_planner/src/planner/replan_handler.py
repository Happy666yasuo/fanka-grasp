"""Failure-Driven Replanning — classify failures and apply bounded retry/replan.

Distinguishes "execution error" (skill execution failed) from "planning error"
(plan itself was unreasonable), and enforces retry limits.
"""

from __future__ import annotations

import sys
import os
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src'))
from embodied_agent.contracts import FailureRecord
from embodied_agent.types import PlanStep, StepFailure, WorldState

MAX_REPLANS = 3
MAX_SKILL_RETRIES = 2


class ReplanHandler:
    """Handles failure classification and bounded replanning logic.

    Rules:
    - "execution_error": skill didn't execute well → retry same skill (up to 2x)
    - "planning_error": plan itself unreasonable → full replan (up to 3x)
    - "post_condition_failed": step completed but didn't achieve goal → replan
    """

    @staticmethod
    def classify_failure_source(
        failure: StepFailure | FailureRecord | dict[str, Any],
    ) -> str:
        """Classify the failure source.

        Returns one of: 'execution_error', 'planning_error', 'post_condition_failed'
        """
        if isinstance(failure, dict):
            source = failure.get("failure_source", failure.get("source", ""))
        else:
            source = getattr(failure, "failure_source", None) or getattr(failure, "source", "")
        return str(source)

    @staticmethod
    def is_execution_error(failure: StepFailure | FailureRecord | dict[str, Any]) -> bool:
        """True if the failure is due to skill execution, not planning."""
        source = ReplanHandler.classify_failure_source(failure)
        return source in ("execution_error", "executor_error")

    @staticmethod
    def is_planning_error(failure: StepFailure | FailureRecord | dict[str, Any]) -> bool:
        """True if the failure is due to unreasonable planning."""
        source = ReplanHandler.classify_failure_source(failure)
        reason = ""
        if isinstance(failure, dict):
            reason = str(failure.get("reason", "")).lower()
        else:
            reason = str(getattr(failure, "reason", "")).lower()
        planning_keywords = [
            "unreasonable", "invalid", "infeasible",
            "不合理", "不可行", "无法执行",
            "not found", "does not exist",
        ]
        if source in ("planning_error",):
            return True
        if source == "execution_error" and any(kw in reason for kw in planning_keywords):
            return True
        return False

    @staticmethod
    def can_retry_skill(
        failure_history: list[FailureRecord | dict[str, Any]],
        skill_name: str,
        max_retries: int = MAX_SKILL_RETRIES,
    ) -> bool:
        """Check if a skill can be retried based on failure history."""
        retry_count = 0
        for failure in failure_history:
            if isinstance(failure, dict):
                f_skill = failure.get("selected_skill", "")
                f_source = failure.get("failure_source", "")
            else:
                f_skill = failure.selected_skill
                f_source = failure.failure_source
            if f_skill == skill_name and f_source == "execution_error":
                retry_count += 1
        return retry_count < max_retries

    @staticmethod
    def can_replan(
        replan_count: int,
        max_replans: int = MAX_REPLANS,
    ) -> bool:
        """Check if another replan is allowed."""
        return replan_count < max_replans

    @staticmethod
    def extract_relevant_failures(
        failure_history: list[FailureRecord | dict[str, Any]],
        current_step_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """Extract and normalize failure records for LLM context."""
        records: list[dict[str, Any]] = []
        for failure in failure_history:
            if isinstance(failure, dict):
                records.append({
                    "selected_skill": failure.get("selected_skill", "unknown"),
                    "failure_source": failure.get("failure_source", "unknown"),
                    "reason": failure.get("reason", "unknown"),
                    "replan_attempt": failure.get("replan_attempt", 0),
                    "step_index": failure.get("step_index", 0),
                })
            else:
                records.append({
                    "selected_skill": failure.selected_skill,
                    "failure_source": failure.failure_source,
                    "reason": failure.reason,
                    "replan_attempt": failure.replan_attempt,
                    "step_index": failure.step_index,
                })
        if current_step_index is not None:
            records = [r for r in records if r["step_index"] <= current_step_index]
        return records

    @staticmethod
    def determine_recovery_strategy(
        failure: StepFailure,
        failure_history: list[FailureRecord | dict[str, Any]],
        replan_count: int,
    ) -> dict[str, Any]:
        """Determine the recovery strategy based on failure analysis.

        Returns a dict with:
        - action: 'retry_skill' | 'replan' | 'abort'
        - reason: explanation
        - remaining_replans: how many replans left
        """
        if ReplanHandler.is_execution_error(failure):
            skill_name = failure.failed_step.action if hasattr(failure, 'failed_step') else "unknown"
            if ReplanHandler.can_retry_skill(failure_history, str(skill_name)):
                return {
                    "action": "retry_skill",
                    "reason": f"Execution error on {skill_name}, retrying with same parameters",
                    "remaining_replans": MAX_REPLANS - replan_count,
                }
            elif ReplanHandler.can_replan(replan_count):
                return {
                    "action": "replan",
                    "reason": f"Skill {skill_name} exhausted retries, triggering replan",
                    "remaining_replans": MAX_REPLANS - replan_count - 1,
                }
            else:
                return {
                    "action": "abort",
                    "reason": "Max replans exhausted",
                    "remaining_replans": 0,
                }

        if ReplanHandler.is_planning_error(failure):
            if ReplanHandler.can_replan(replan_count):
                return {
                    "action": "replan",
                    "reason": "Planning error detected, full replan required",
                    "remaining_replans": MAX_REPLANS - replan_count - 1,
                }
            else:
                return {
                    "action": "abort",
                    "reason": "Max replans exhausted on planning error",
                    "remaining_replans": 0,
                }

        if ReplanHandler.can_replan(replan_count):
            return {
                "action": "replan",
                "reason": f"Unknown failure source '{ReplanHandler.classify_failure_source(failure)}', defaulting to replan",
                "remaining_replans": MAX_REPLANS - replan_count - 1,
            }
        return {"action": "abort", "reason": "Max replans exhausted", "remaining_replans": 0}
