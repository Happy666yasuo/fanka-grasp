from __future__ import annotations

import json
import math
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../embodied_agent/src')))
from embodied_agent.contracts import (
    CausalExploreOutput,
    PropertyBelief,
    AffordanceCandidate,
    PROBE_UNCERTAINTY_THRESHOLD,
    LOW_AFFORDANCE_CONFIDENCE_THRESHOLD,
)
from embodied_agent.mujoco_simulator import MujocoPickPlaceSimulation

from .probe_actions import (
    ProbeAction,
    ProbeActionResult,
    PROBE_ACTION_REGISTRY,
)


@dataclass
class ObjectManifest:
    """Static description of an object to explore."""
    object_id: str
    object_category: str
    scene_id: str = "default_scene"
    expected_properties: list[str] = field(default_factory=list)
    candidate_affordances: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_category": self.object_category,
            "scene_id": self.scene_id,
            "expected_properties": list(self.expected_properties),
            "candidate_affordances": list(self.candidate_affordances),
        }


class ProbeExecutor:
    """Execute probe actions in MuJoCo and produce CausalExploreOutput artifacts."""

    def __init__(
        self,
        simulation: MujocoPickPlaceSimulation,
        output_dir: str | None = None,
    ) -> None:
        self.simulation = simulation
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute_probe(
        self,
        probe_name: str,
        object_name: str = "red_block",
        **probe_kwargs: Any,
    ) -> ProbeActionResult:
        probe_fn = PROBE_ACTION_REGISTRY.get(probe_name)
        if probe_fn is None:
            available = ", ".join(sorted(PROBE_ACTION_REGISTRY.keys()))
            raise KeyError(f"Unknown probe '{probe_name}'. Available: {available}")
        return probe_fn(self.simulation, object_name, **probe_kwargs)

    def run_probe_sequence(
        self,
        object_name: str = "red_block",
        probes: list[str] | None = None,
        reset_between: bool = True,
    ) -> list[ProbeActionResult]:
        probe_names = probes or list(PROBE_ACTION_REGISTRY.keys())
        results: list[ProbeActionResult] = []
        for probe_name in probe_names:
            if reset_between:
                self.simulation.fast_reset()
            result = self.execute_probe(probe_name, object_name)
            results.append(result)
        return results

    def build_causal_output(
        self,
        manifest: ObjectManifest,
        probe_results: list[ProbeActionResult],
    ) -> CausalExploreOutput:
        property_beliefs: dict[str, PropertyBelief] = {}
        for prop_name in manifest.expected_properties:
            confidence = self._estimate_property_confidence(prop_name, probe_results)
            property_beliefs[prop_name] = PropertyBelief(
                label=prop_name,
                confidence=confidence,
            )

        affordance_candidates: list[AffordanceCandidate] = []
        for affordance_name in manifest.candidate_affordances:
            confidence = self._estimate_affordance_confidence(affordance_name, probe_results)
            affordance_candidates.append(
                AffordanceCandidate(name=affordance_name, confidence=confidence)
            )

        uncertainty = self._compute_uncertainty(property_beliefs, affordance_candidates)
        recommended = self._recommend_next_probe(probe_results, uncertainty)

        contact_region = self._infer_contact_region(probe_results)

        artifact_path = str(self.output_dir / f"causal_{manifest.object_id}.json")

        return CausalExploreOutput(
            scene_id=manifest.scene_id,
            object_id=manifest.object_id,
            object_category=manifest.object_category,
            property_belief=property_beliefs,
            affordance_candidates=affordance_candidates,
            uncertainty_score=uncertainty,
            recommended_probe=recommended,
            contact_region=contact_region,
            skill_constraints={},
            artifact_path=artifact_path,
        )

    def save_artifact(
        self,
        causal_output: CausalExploreOutput,
        probe_results: list[ProbeActionResult],
    ) -> str:
        artifact = {
            "causal_output": causal_output.to_dict(),
            "probe_results": [r.to_dict() for r in probe_results],
        }
        artifact_path = self.output_dir / f"causal_{causal_output.object_id}.json"
        artifact_path.write_text(json.dumps(artifact, indent=2, default=str))
        return str(artifact_path)

    def _estimate_property_confidence(
        self,
        prop_name: str,
        results: list[ProbeActionResult],
    ) -> float:
        """Estimate property confidence from probe results.

        Uses displacement magnitude as primary signal.
        """
        prop_keywords: dict[str, list[str]] = {
            "movable": ["lateral_push", "side_pull"],
            "pressable": ["top_press", "surface_tap"],
            "graspable": ["grasp_attempt"],
            "rigid": ["surface_tap"],
        }
        relevant = prop_keywords.get(prop_name, [prop_name])
        relevant_results = [r for r in results if r.probe_name in relevant]

        if not relevant_results:
            return 0.5

        success_rate = sum(1 for r in relevant_results if r.success) / len(relevant_results)
        avg_displacement = (
            sum(r.displacement_magnitude for r in relevant_results) / len(relevant_results)
            if relevant_results else 0.0
        )

        displacement_confidence = min(1.0, avg_displacement / 0.02)
        combined = 0.4 * success_rate + 0.6 * displacement_confidence
        return min(1.0, max(0.0, combined))

    def _estimate_affordance_confidence(
        self,
        affordance_name: str,
        results: list[ProbeActionResult],
    ) -> float:
        """Estimate affordance confidence from probe results."""
        affordance_probe_map: dict[str, str] = {
            "pushable": "lateral_push",
            "pressable": "top_press",
            "pullable": "side_pull",
            "graspable": "grasp_attempt",
            "tappable": "surface_tap",
        }
        probe_name = affordance_probe_map.get(affordance_name, affordance_name)
        matching = [r for r in results if r.probe_name == probe_name]

        if not matching:
            return 0.5

        result = matching[0]
        if result.success and result.displacement_magnitude > 0.005:
            return 0.85
        elif result.contact_detected:
            return 0.6
        return 0.3

    def _compute_uncertainty(
        self,
        property_beliefs: dict[str, PropertyBelief],
        affordance_candidates: list[AffordanceCandidate],
    ) -> float:
        prop_uncertainty = (
            1.0 - sum(b.confidence for b in property_beliefs.values()) / max(1, len(property_beliefs))
        )
        afford_uncertainty = (
            1.0 - sum(c.confidence for c in affordance_candidates) / max(1, len(affordance_candidates))
        )
        return 0.5 * prop_uncertainty + 0.5 * afford_uncertainty

    def _recommend_next_probe(
        self,
        results: list[ProbeActionResult],
        uncertainty: float,
    ) -> str | None:
        if uncertainty < PROBE_UNCERTAINTY_THRESHOLD:
            return None

        completed = {r.probe_name for r in results}
        probe_priority = ["lateral_push", "top_press", "side_pull", "surface_tap", "grasp_attempt"]
        for probe in probe_priority:
            if probe not in completed:
                return probe
        return probe_priority[0] if probe_priority else None

    def _infer_contact_region(
        self,
        results: list[ProbeActionResult],
    ) -> str | None:
        if not results:
            return None

        region_map: dict[str, str] = {
            "lateral_push": "side",
            "top_press": "top",
            "side_pull": "side",
            "surface_tap": "top",
            "grasp_attempt": "top",
        }
        for result in results:
            if result.contact_detected:
                return region_map.get(result.probe_name)
        return None
