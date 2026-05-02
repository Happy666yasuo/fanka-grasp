from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PROBE_UNCERTAINTY_THRESHOLD = 0.50
LOW_AFFORDANCE_CONFIDENCE_THRESHOLD = 0.70

ALLOWED_PLANNER_SKILLS = {
    "observe",
    "probe",
    "pick",
    "place",
    "press",
    "push",
    "pull",
    "rotate",
    "replan",
}

FORBIDDEN_CONTINUOUS_CONTROL_ARGS = {
    "joint_positions",
    "joint_position",
    "motor_torques",
    "torques",
    "cartesian_trajectory",
    "raw_actions",
    "continuous_action",
}


@dataclass(frozen=True)
class PropertyBelief:
    label: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "confidence": self.confidence}


@dataclass(frozen=True)
class AffordanceCandidate:
    name: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "confidence": self.confidence}


@dataclass(frozen=True)
class CausalExploreOutput:
    scene_id: str
    object_id: str
    object_category: str
    property_belief: dict[str, PropertyBelief]
    affordance_candidates: list[AffordanceCandidate]
    uncertainty_score: float
    recommended_probe: str | None
    contact_region: str | None
    skill_constraints: dict[str, Any]
    artifact_path: str

    def requires_probe(self) -> bool:
        if self.uncertainty_score >= PROBE_UNCERTAINTY_THRESHOLD:
            return True
        return any(
            candidate.confidence < LOW_AFFORDANCE_CONFIDENCE_THRESHOLD
            for candidate in self.affordance_candidates
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "object_id": self.object_id,
            "object_category": self.object_category,
            "property_belief": {
                name: belief.to_dict() for name, belief in self.property_belief.items()
            },
            "affordance_candidates": [
                candidate.to_dict() for candidate in self.affordance_candidates
            ],
            "uncertainty_score": self.uncertainty_score,
            "recommended_probe": self.recommended_probe,
            "contact_region": self.contact_region,
            "skill_constraints": dict(self.skill_constraints),
            "artifact_path": self.artifact_path,
        }


def causal_output_from_dict(
    payload: dict[str, Any],
    *,
    artifact_path: str | None = None,
) -> CausalExploreOutput:
    property_belief = {
        str(name): PropertyBelief(
            label=str(value["label"]),
            confidence=float(value["confidence"]),
        )
        for name, value in dict(payload.get("property_belief", {})).items()
    }
    affordance_candidates = [
        AffordanceCandidate(
            name=str(candidate["name"]),
            confidence=float(candidate["confidence"]),
        )
        for candidate in list(payload.get("affordance_candidates", []))
    ]
    resolved_artifact_path = artifact_path or str(payload.get("artifact_path", ""))
    return CausalExploreOutput(
        scene_id=str(payload["scene_id"]),
        object_id=str(payload["object_id"]),
        object_category=str(payload["object_category"]),
        property_belief=property_belief,
        affordance_candidates=affordance_candidates,
        uncertainty_score=float(payload["uncertainty_score"]),
        recommended_probe=(
            str(payload["recommended_probe"])
            if payload.get("recommended_probe") is not None
            else None
        ),
        contact_region=(
            str(payload["contact_region"])
            if payload.get("contact_region") is not None
            else None
        ),
        skill_constraints=dict(payload.get("skill_constraints", {})),
        artifact_path=resolved_artifact_path,
    )


@dataclass(frozen=True)
class PlannerStep:
    task_id: str
    step_index: int
    selected_skill: str
    target_object: str | None
    skill_args: dict[str, Any] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    expected_effect: str | None = None
    fallback_action: str = "probe_or_replan"

    def __post_init__(self) -> None:
        if self.selected_skill not in ALLOWED_PLANNER_SKILLS:
            raise ValueError(f"Unsupported planner skill: {self.selected_skill}")
        forbidden_args = FORBIDDEN_CONTINUOUS_CONTROL_ARGS.intersection(self.skill_args)
        if forbidden_args:
            names = ", ".join(sorted(forbidden_args))
            raise ValueError(f"PlannerStep cannot contain continuous control arguments: {names}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step_index": self.step_index,
            "selected_skill": self.selected_skill,
            "target_object": self.target_object,
            "skill_args": dict(self.skill_args),
            "preconditions": list(self.preconditions),
            "expected_effect": self.expected_effect,
            "fallback_action": self.fallback_action,
        }


@dataclass(frozen=True)
class ContactState:
    has_contact: bool
    contact_region: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_contact": self.has_contact,
            "contact_region": self.contact_region,
        }


@dataclass(frozen=True)
class FailureRecord:
    step_index: int
    selected_skill: str
    failure_source: str
    reason: str
    replan_attempt: int
    selected_recovery_policy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "selected_skill": self.selected_skill,
            "failure_source": self.failure_source,
            "reason": self.reason,
            "replan_attempt": self.replan_attempt,
            "selected_recovery_policy": self.selected_recovery_policy,
        }


@dataclass(frozen=True)
class ExecutorResult:
    success: bool
    reward: float
    final_state: dict[str, Any]
    contact_state: ContactState
    error_code: str | None
    rollout_path: str
    failure_history: list[FailureRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "reward": self.reward,
            "final_state": dict(self.final_state),
            "contact_state": self.contact_state.to_dict(),
            "error_code": self.error_code,
            "rollout_path": self.rollout_path,
            "failure_history": [failure.to_dict() for failure in self.failure_history],
        }


def build_planner_step_from_causal_output(
    *,
    task_id: str,
    step_index: int,
    causal_output: CausalExploreOutput,
    requested_skill: str,
    skill_args: dict[str, Any] | None = None,
) -> PlannerStep:
    if causal_output.requires_probe():
        return PlannerStep(
            task_id=task_id,
            step_index=step_index,
            selected_skill="probe",
            target_object=causal_output.object_id,
            skill_args={
                "probe": causal_output.recommended_probe,
                "contact_region": causal_output.contact_region,
                "artifact_path": causal_output.artifact_path,
            },
            preconditions=["object_visible"],
            expected_effect=f"uncertainty_reduced({causal_output.object_id})",
            fallback_action="probe_or_replan",
        )

    preconditions = [
        f"affordance.{candidate.name}.confidence >= {LOW_AFFORDANCE_CONFIDENCE_THRESHOLD:.2f}"
        for candidate in causal_output.affordance_candidates
    ]
    merged_skill_args = dict(skill_args or {})
    merged_skill_args.setdefault("contact_region", causal_output.contact_region)

    return PlannerStep(
        task_id=task_id,
        step_index=step_index,
        selected_skill=requested_skill,
        target_object=causal_output.object_id,
        skill_args=merged_skill_args,
        preconditions=preconditions,
        expected_effect=None,
        fallback_action="probe_or_replan",
    )


def simulate_contract_step(
    *,
    task_id: str,
    step_index: int,
    causal_output: CausalExploreOutput,
    requested_skill: str,
    skill_args: dict[str, Any] | None = None,
) -> ExecutorResult:
    step = build_planner_step_from_causal_output(
        task_id=task_id,
        step_index=step_index,
        causal_output=causal_output,
        requested_skill=requested_skill,
        skill_args=skill_args,
    )
    contact_region = step.skill_args.get("contact_region")
    return ExecutorResult(
        success=True,
        reward=1.0,
        final_state={
            "task_id": step.task_id,
            "step_index": step.step_index,
            "executed_skill": step.selected_skill,
            "target_object": step.target_object,
            "expected_effect": step.expected_effect,
        },
        contact_state=ContactState(
            has_contact=contact_region is not None,
            contact_region=contact_region,
        ),
        error_code=None,
        rollout_path=f"mock://contract_flow/{task_id}/{step_index}",
        failure_history=[],
    )


def build_planner_step_from_plan_step(
    *,
    task_id: str,
    step_index: int,
    plan_step: Any,
) -> PlannerStep:
    skill_args = dict(getattr(plan_step, "parameters", {}) or {})
    selected_skill = str(getattr(plan_step, "action"))
    target = getattr(plan_step, "target", None)
    post_condition = getattr(plan_step, "post_condition", None)

    target_object = str(target) if target is not None else None
    if selected_skill == "place":
        object_name = skill_args.get("object")
        target_object = str(object_name) if object_name is not None else None
        if target is not None:
            skill_args.setdefault("target_zone", target)

    return PlannerStep(
        task_id=task_id,
        step_index=step_index,
        selected_skill=selected_skill,
        target_object=target_object,
        skill_args=skill_args,
        preconditions=["object_visible"],
        expected_effect=_format_post_condition(post_condition),
        fallback_action="probe_or_replan",
    )


def _format_post_condition(post_condition: Any) -> str | None:
    if post_condition is None:
        return None
    kind = str(getattr(post_condition, "kind"))
    object_name = getattr(post_condition, "object_name", None)
    zone_name = getattr(post_condition, "zone_name", None)
    if kind == "holding" and object_name is not None:
        return f"holding({object_name})"
    if kind == "placed" and object_name is not None and zone_name is not None:
        return f"placed({object_name},{zone_name})"
    return kind
