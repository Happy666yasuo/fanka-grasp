"""LLMPlanner — structured LLM-based task planner following the Planner protocol.

Uses pluggable LLM backend (Anthropic, OpenAI, or mock for testing).
Output is strictly validated against PlannerStep schema constraints.
"""

from __future__ import annotations

import json
import sys
import os
from typing import Any, Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src'))
from embodied_agent.types import PlanStep, PostCondition, WorldState, StepFailure

from .prompt_templates import (
    build_system_prompt,
    build_task_prompt,
    build_replan_prompt,
    parse_llm_json_response,
    validate_step_dict,
    ALLOWED_SKILLS,
)

# LLM callable signature: (messages: list[dict]) -> str
LlmCallable = Callable[[list[dict[str, str]]], str]


def _default_llm_call(messages: list[dict[str, str]]) -> str:
    """Default LLM callable for environments without a real API key.
    Returns a minimal observe-only plan so the planner works out of the box.
    """
    return json.dumps({
        "task_id": "default",
        "steps": [{
            "step_index": 0,
            "selected_skill": "observe",
            "target_object": None,
            "skill_args": {},
            "preconditions": ["object_visible"],
            "expected_effect": "scene_observed",
            "fallback_action": "probe_or_replan",
        }],
    })


class LLMPlanner:
    """LLM-based planner following the Planner protocol.

    Plugs into ContractPlannerAdapter and ContractPlanningBridge
    for unified contract-based planning with probe injection.

    Usage:
        planner = LLMPlanner(llm_callable=my_anthropic_call)
        steps = planner.plan("put red block in green zone", world_state)
    """

    def __init__(
        self,
        llm_callable: LlmCallable | None = None,
        language: str = "zh",
        causal_outputs: dict[str, Any] | None = None,
    ) -> None:
        self._llm_call = llm_callable or _default_llm_call
        self.language = language
        self.causal_outputs = causal_outputs or {}

    def plan(self, instruction: str, state: WorldState) -> list[PlanStep]:
        """Plan a sequence of steps for the given instruction.

        Args:
            instruction: Natural language task instruction.
            state: Current WorldState from the simulator.

        Returns:
            List of PlanStep objects.
        """
        sys_msg = build_system_prompt(self.language)
        world_state_dict = self._world_state_to_dict(state)
        user_msg = build_task_prompt(
            instruction=instruction,
            causal_outputs=self.causal_outputs or None,
            world_state=world_state_dict,
            language=self.language,
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]
        response = self._llm_call(messages)
        return self._parse_response(response, instruction)

    def replan(
        self,
        instruction: str,
        state: WorldState,
        failed_step: PlanStep,
        remaining_plan: list[PlanStep],
        failure: StepFailure,
    ) -> list[PlanStep]:
        """Replan after a step failure.

        Args:
            instruction: Original task instruction.
            state: Current WorldState after failure.
            failed_step: The PlanStep that failed.
            remaining_plan: Remaining steps from the original plan.
            failure: The StepFailure record.

        Returns:
            New list of PlanStep objects.
        """
        failure_history = [{
            "selected_skill": failed_step.action,
            "failure_source": failure.source,
            "reason": failure.reason,
            "replan_attempt": failure.replan_attempt,
            "step_index": 0,
        }]
        for i, step in enumerate(remaining_plan):
            failure_history.append({
                "selected_skill": step.action,
                "failure_source": "pending",
                "reason": f"Remaining step {i + 1}: {step.action}",
                "replan_attempt": failure.replan_attempt,
                "step_index": i + 1,
            })

        sys_msg = build_system_prompt(self.language)
        world_state_dict = self._world_state_to_dict(state)
        user_msg = build_replan_prompt(
            instruction=instruction,
            failure_history=failure_history,
            causal_outputs=self.causal_outputs or None,
            state=world_state_dict,
            language=self.language,
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]
        response = self._llm_call(messages)
        return self._parse_response(response, instruction)

    def update_causal_outputs(self, causal_outputs: dict[str, Any]) -> None:
        """Update causal explore outputs for informed planning."""
        self.causal_outputs = causal_outputs

    def _parse_response(self, response: str, instruction: str) -> list[PlanStep]:
        step_dicts = parse_llm_json_response(response)
        for step_dict in step_dicts:
            validate_step_dict(step_dict)
        return [self._dict_to_plan_step(sd, instruction) for sd in step_dicts]

    def _dict_to_plan_step(self, step_dict: dict[str, Any], instruction: str) -> PlanStep:
        skill_args = step_dict.get("skill_args", {})
        selected_skill = step_dict["selected_skill"]
        target = step_dict.get("target_object")
        expected_effect = step_dict.get("expected_effect")

        post_condition = None
        if expected_effect:
            if "holding" in str(expected_effect).lower():
                obj = target or "unknown"
                post_condition = PostCondition(kind="holding", object_name=obj)
            elif "placed" in str(expected_effect).lower():
                obj = skill_args.get("object", target or "unknown")
                zone = skill_args.get("target_zone", "unknown")
                post_condition = PostCondition(
                    kind="placed", object_name=str(obj), zone_name=str(zone),
                )

        return PlanStep(
            action=selected_skill,
            target=str(target) if target else None,
            parameters=skill_args,
            post_condition=post_condition,
        )

    @staticmethod
    def _world_state_to_dict(state: WorldState) -> dict[str, Any]:
        return {
            "instruction": state.instruction,
            "object_positions": dict(state.object_positions),
            "zone_positions": dict(state.zone_positions),
            "end_effector_position": list(state.end_effector_position),
            "held_object_name": state.held_object_name,
        }
