from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .probe_actions import ProbeActionResult, PROBE_ACTION_REGISTRY
from .probe_executor import ObjectManifest, ProbeExecutor


@dataclass
class ExploreStep:
    probe_name: str
    object_name: str
    result: ProbeActionResult
    step_index: int


@dataclass
class ExploreHistory:
    steps: list[ExploreStep] = field(default_factory=list)
    probe_counts: dict[str, int] = field(default_factory=dict)
    object_visits: dict[str, int] = field(default_factory=dict)
    total_displacement: dict[str, float] = field(default_factory=dict)

    def record(self, step: ExploreStep) -> None:
        self.steps.append(step)
        pair_key = f"{step.probe_name}:{step.object_name}"
        self.probe_counts[pair_key] = self.probe_counts.get(pair_key, 0) + 1
        self.object_visits[step.object_name] = self.object_visits.get(step.object_name, 0) + 1
        self.total_displacement[pair_key] = (
            self.total_displacement.get(pair_key, 0.0) + step.result.displacement_magnitude
        )

    def last_result(self) -> ProbeActionResult | None:
        if not self.steps:
            return None
        return self.steps[-1].result

    def last_step_index(self) -> int:
        if not self.steps:
            return -1
        return self.steps[-1].step_index


class BaseExploreStrategy(ABC):
    """Abstract base for exploration strategies."""

    strategy_name: str = "base"

    @abstractmethod
    def select_next(
        self,
        history: ExploreHistory,
        available_probes: list[str],
        available_objects: list[str],
    ) -> tuple[str, str]:
        """Return (probe_name, object_name) for the next exploration step."""

    def reset(self) -> None:
        pass


class RandomStrategy(BaseExploreStrategy):
    """Randomly sample probe-object pairs with uniform probability."""

    strategy_name = "random"

    def select_next(
        self,
        history: ExploreHistory,
        available_probes: list[str],
        available_objects: list[str],
    ) -> tuple[str, str]:
        probe = random.choice(available_probes)
        obj = random.choice(available_objects)
        return probe, obj


class CuriosityDrivenStrategy(BaseExploreStrategy):
    """Prioritize probe-object pairs that produced larger displacement magnitudes.

    Uses softmax-weighted sampling over accumulated displacement. Unexplored
    pairs get a small bonus to encourage coverage.
    """

    strategy_name = "curiosity"

    def __init__(self, temperature: float = 0.5, unexplored_bonus: float = 0.01) -> None:
        self.temperature = temperature
        self.unexplored_bonus = unexplored_bonus

    def select_next(
        self,
        history: ExploreHistory,
        available_probes: list[str],
        available_objects: list[str],
    ) -> tuple[str, str]:
        pairs = [(p, o) for p in available_probes for o in available_objects]

        if not history.steps:
            return random.choice(pairs)

        weights: list[float] = []
        for probe, obj in pairs:
            pair_key = f"{probe}:{obj}"
            acc_disp = history.total_displacement.get(pair_key, 0.0)
            weight = acc_disp + self.unexplored_bonus
            weights.append(weight)

        total = sum(weights)
        if total == 0.0:
            return random.choice(pairs)

        probs = [w / total for w in weights]

        if self.temperature != 1.0:
            probs = self._softmax([w / self.temperature for w in weights])

        chosen = random.choices(pairs, weights=probs, k=1)[0]
        return chosen

    @staticmethod
    def _softmax(logits: list[float]) -> list[float]:
        max_logit = max(logits)
        exps = [math.exp(l - max_logit) for l in logits]
        total = sum(exps)
        return [e / total for e in exps]


class CausalExploreStrategy(BaseExploreStrategy):
    """Select probes that minimize uncertainty in property/affordance beliefs.

    Uses the ProbeExecutor's belief estimation to compute uncertainty and
    prefers probes that target the least-certain properties.
    """

    strategy_name = "causal_explore"

    def __init__(self, probe_executor: ProbeExecutor) -> None:
        self.probe_executor = probe_executor

    def select_next(
        self,
        history: ExploreHistory,
        available_probes: list[str],
        available_objects: list[str],
    ) -> tuple[str, str]:
        if not history.steps:
            return "lateral_push", available_objects[0] if available_objects else "red_block"

        latest_object = history.steps[-1].object_name

        completed_probes = {s.probe_name for s in history.steps}

        for obj in available_objects:
            obj_results = [s.result for s in history.steps if s.object_name == obj]
            if obj_results:
                uncertainty = self._estimate_remaining_uncertainty(obj_results)
                if uncertainty > 0.3:
                    for probe in available_probes:
                        if probe not in completed_probes:
                            return probe, obj

        for probe in available_probes:
            if probe not in completed_probes:
                return probe, latest_object

        return available_probes[0], latest_object

    def _estimate_remaining_uncertainty(self, results: list[ProbeActionResult]) -> float:
        if not results:
            return 1.0
        success_rate = sum(1 for r in results if r.success) / len(results)
        avg_disp = sum(r.displacement_magnitude for r in results) / len(results)
        disp_confidence = min(1.0, avg_disp / 0.02)
        combined = 0.4 * success_rate + 0.6 * disp_confidence
        return 1.0 - combined


STRATEGY_REGISTRY: dict[str, type[BaseExploreStrategy]] = {
    "random": RandomStrategy,
    "curiosity": CuriosityDrivenStrategy,
    "causal_explore": CausalExploreStrategy,
}
