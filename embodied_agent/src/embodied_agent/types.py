from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class PostCondition:
    kind: str
    object_name: str | None = None
    zone_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "object_name": self.object_name,
            "zone_name": self.zone_name,
        }


@dataclass(frozen=True)
class PlanStep:
    action: str
    target: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    post_condition: PostCondition | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target": self.target,
            "parameters": dict(self.parameters),
            "post_condition": self.post_condition.to_dict() if self.post_condition is not None else None,
        }


@dataclass(frozen=True)
class WorldState:
    instruction: str
    object_positions: dict[str, Vec3]
    zone_positions: dict[str, Vec3]
    end_effector_position: Vec3
    held_object_name: str | None = None


@dataclass(frozen=True)
class StepFailure:
    failed_step: PlanStep
    source: str
    reason: str
    replan_attempt: int
    recovery_policy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "failed_step": self.failed_step.to_dict(),
            "source": self.source,
            "reason": self.reason,
            "replan_attempt": self.replan_attempt,
            "recovery_policy": self.recovery_policy,
        }


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    plan: list[PlanStep]
    executed_actions: list[str]
    final_object_positions: dict[str, Vec3]
    metrics: dict[str, float]
    error: str | None = None
    replan_count: int = 0
    failure_history: list[StepFailure] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "plan": [step.to_dict() for step in self.plan],
            "executed_actions": list(self.executed_actions),
            "final_object_positions": {
                name: list(position) for name, position in self.final_object_positions.items()
            },
            "metrics": dict(self.metrics),
            "error": self.error,
            "replan_count": self.replan_count,
            "failure_history": [failure.to_dict() for failure in self.failure_history],
        }