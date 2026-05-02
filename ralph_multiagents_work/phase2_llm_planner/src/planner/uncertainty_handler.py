"""Uncertainty-Driven Planning — inject probe steps when CausalExplore confidence is low.

Reuses ContractPlanningBridge probe injection logic.
"""

from __future__ import annotations

import sys
import os
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src'))
from embodied_agent.contracts import (
    CausalExploreOutput,
    PlannerStep,
    PROBE_UNCERTAINTY_THRESHOLD,
    LOW_AFFORDANCE_CONFIDENCE_THRESHOLD,
    build_planner_step_from_causal_output,
)

PROBE_UNCERTAINTY_THRESHOLD = 0.50


class UncertaintyHandler:
    """Decides whether to insert probe steps based on CausalExplore uncertainty.

    Key rules:
    - uncertainty_score >= 0.50 → must probe
    - any affordance confidence < 0.70 → should probe
    - probe before execution, not blind planning
    """

    @staticmethod
    def requires_probe(causal_output: CausalExploreOutput) -> bool:
        """Check if a probe step should be inserted before executing on this object."""
        return causal_output.requires_probe()

    @staticmethod
    def requires_probe_from_dict(output_dict: dict[str, Any]) -> bool:
        """Check probe requirement from a dict representation."""
        uncertainty = float(output_dict.get("uncertainty_score", 0.0))
        if uncertainty >= PROBE_UNCERTAINTY_THRESHOLD:
            return True
        affordances = output_dict.get("affordance_candidates", [])
        return any(
            float(c.get("confidence", 0.0)) < LOW_AFFORDANCE_CONFIDENCE_THRESHOLD
            for c in affordances
        )

    @staticmethod
    def get_recommended_probe(causal_output: CausalExploreOutput) -> str | None:
        """Get the recommended probe action name."""
        return causal_output.recommended_probe

    @staticmethod
    def get_recommended_probe_from_dict(output_dict: dict[str, Any]) -> str | None:
        """Get recommended probe from dict representation."""
        return output_dict.get("recommended_probe")

    @staticmethod
    def evaluate(causal_output: CausalExploreOutput) -> dict[str, Any]:
        """Return a structured probe decision for experiment/reporting code."""
        low_confidence_affordances = [
            candidate.name
            for candidate in causal_output.affordance_candidates
            if candidate.confidence < LOW_AFFORDANCE_CONFIDENCE_THRESHOLD
        ]
        high_uncertainty = causal_output.uncertainty_score >= PROBE_UNCERTAINTY_THRESHOLD
        needs_probe = high_uncertainty or bool(low_confidence_affordances)
        if high_uncertainty:
            reason = "high_uncertainty"
        elif low_confidence_affordances:
            reason = "low_affordance_confidence"
        else:
            reason = "sufficient_confidence"

        return {
            "object_id": causal_output.object_id,
            "needs_probe": needs_probe,
            "recommended_probe": causal_output.recommended_probe,
            "uncertainty_score": causal_output.uncertainty_score,
            "low_confidence_affordances": low_confidence_affordances,
            "reason": reason,
        }

    @staticmethod
    def should_probe_any(
        causal_outputs: dict[str, CausalExploreOutput],
    ) -> list[str]:
        """Return list of object IDs that need probing."""
        return [
            obj_id
            for obj_id, output in causal_outputs.items()
            if output.requires_probe()
        ]

    @staticmethod
    def should_probe_any_from_dict(
        causal_outputs: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Return list of object IDs that need probing (dict version)."""
        return [
            obj_id
            for obj_id, output in causal_outputs.items()
            if UncertaintyHandler.requires_probe_from_dict(output)
        ]

    @staticmethod
    def inject_probe_step(
        plan: list[PlannerStep],
        task_id: str,
        target_index: int,
        causal_output: CausalExploreOutput,
    ) -> list[PlannerStep]:
        """Insert a probe step before the target step in the plan.

        Reuses the contract helper for building probe steps.
        """
        if target_index < 0 or target_index >= len(plan):
            return plan
        requested_skill = plan[target_index].selected_skill
        probe_step = build_planner_step_from_causal_output(
            task_id=task_id,
            step_index=target_index,
            causal_output=causal_output,
            requested_skill=requested_skill,
        )
        updated = list(plan)
        updated.insert(target_index, probe_step)
        for i, step in enumerate(updated):
            updated[i] = PlannerStep(
                task_id=step.task_id,
                step_index=i,
                selected_skill=step.selected_skill,
                target_object=step.target_object,
                skill_args=dict(step.skill_args),
                preconditions=list(step.preconditions),
                expected_effect=step.expected_effect,
                fallback_action=step.fallback_action,
            )
        return updated

    @staticmethod
    def inject_probes_for_all_uncertain(
        plan: list[PlannerStep],
        task_id: str,
        causal_outputs: dict[str, CausalExploreOutput],
    ) -> list[PlannerStep]:
        """Insert probe steps for all objects with high uncertainty.

        Probes are inserted before the first step that targets each uncertain object.
        """
        uncertain_ids = UncertaintyHandler.should_probe_any(causal_outputs)
        if not uncertain_ids:
            return plan

        result = list(plan)
        insertion_offset = 0
        for obj_id in uncertain_ids:
            output = causal_outputs[obj_id]
            target_index = UncertaintyHandler._find_first_target_index(result, obj_id)
            if target_index is None:
                continue
            actual_index = target_index + insertion_offset
            probe_step = build_planner_step_from_causal_output(
                task_id=task_id,
                step_index=actual_index,
                causal_output=output,
                requested_skill=result[actual_index].selected_skill
                if actual_index < len(result)
                else "observe",
            )
            result.insert(actual_index, probe_step)
            insertion_offset += 1
        for i, step in enumerate(result):
            result[i] = PlannerStep(
                task_id=step.task_id,
                step_index=i,
                selected_skill=step.selected_skill,
                target_object=step.target_object,
                skill_args=dict(step.skill_args),
                preconditions=list(step.preconditions),
                expected_effect=step.expected_effect,
                fallback_action=step.fallback_action,
            )
        return result

    @staticmethod
    def _find_first_target_index(
        plan: list[PlannerStep],
        object_id: str,
    ) -> int | None:
        for i, step in enumerate(plan):
            if step.target_object == object_id:
                return i
        return None

    @staticmethod
    def get_uncertainty_summary(
        causal_outputs: dict[str, CausalExploreOutput],
    ) -> dict[str, Any]:
        """Get a summary of uncertainty status for all objects."""
        summary: dict[str, Any] = {}
        for obj_id, output in causal_outputs.items():
            summary[obj_id] = {
                "uncertainty_score": output.uncertainty_score,
                "requires_probe": output.requires_probe(),
                "recommended_probe": output.recommended_probe,
                "low_confidence_affordances": [
                    c.name
                    for c in output.affordance_candidates
                    if c.confidence < LOW_AFFORDANCE_CONFIDENCE_THRESHOLD
                ],
            }
        return summary
