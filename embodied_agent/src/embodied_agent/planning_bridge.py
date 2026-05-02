from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from embodied_agent.contracts import (
    CausalExploreOutput,
    PlannerStep,
    build_planner_step_from_causal_output,
    causal_output_from_dict,
)
from embodied_agent.planner import Planner
from embodied_agent.planner_contracts import ContractPlannerAdapter
from embodied_agent.types import PlanStep, StepFailure, WorldState


class CausalOutputProvider(Protocol):
    def get_outputs(self, object_ids: list[str]) -> dict[str, CausalExploreOutput]:
        ...


@dataclass(frozen=True)
class ArtifactRegistryCausalOutputProvider:
    registry_path: Path

    def get_outputs(self, object_ids: list[str]) -> dict[str, CausalExploreOutput]:
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        object_registry = dict(registry.get("objects", {}))
        outputs: dict[str, CausalExploreOutput] = {}
        for object_id in object_ids:
            artifact_ref = object_registry.get(object_id)
            if artifact_ref is None:
                continue
            artifact_path = self._resolve_artifact_path(str(artifact_ref))
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            outputs[object_id] = causal_output_from_dict(
                payload,
                artifact_path=str(artifact_path.resolve()),
            )
        return outputs

    def _resolve_artifact_path(self, artifact_ref: str) -> Path:
        artifact_path = Path(artifact_ref)
        if artifact_path.is_absolute():
            return artifact_path
        return (self.registry_path.parent / artifact_path).resolve()


@dataclass(frozen=True)
class SimulatorArtifactCatalogCausalOutputProvider:
    catalog_path: Path

    def get_outputs(self, object_ids: list[str]) -> dict[str, CausalExploreOutput]:
        catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        expected_scene_id = str(catalog.get("scene_id", ""))
        object_manifest = self._load_object_manifest(catalog)
        object_registry = dict(catalog.get("objects", {}))
        outputs: dict[str, CausalExploreOutput] = {}
        for object_id in object_ids:
            self._validate_object_membership(object_id, object_manifest)
            object_entry = object_registry.get(object_id)
            if object_entry is None:
                continue
            artifact_ref = object_entry.get("artifact_path", object_entry)
            artifact_path = self._resolve_relative_path(str(artifact_ref))
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            if expected_scene_id and str(payload.get("scene_id", "")) != expected_scene_id:
                raise ValueError(
                    f"Artifact scene_id mismatch for {object_id}: "
                    f"expected {expected_scene_id}, got {payload.get('scene_id')}"
                )
            outputs[object_id] = causal_output_from_dict(
                payload,
                artifact_path=str(artifact_path.resolve()),
            )
        return outputs

    def _load_object_manifest(self, catalog: dict[str, object]) -> set[str]:
        manifest_ref = catalog.get("object_manifest_path")
        if not isinstance(manifest_ref, str) or not manifest_ref:
            return set()
        manifest_path = self._resolve_relative_path(manifest_ref)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        objects = payload.get("objects", [])
        if isinstance(objects, dict):
            return {str(object_id) for object_id in objects.keys()}
        manifest_ids: set[str] = set()
        for item in objects:
            if isinstance(item, str):
                manifest_ids.add(item)
                continue
            if isinstance(item, dict) and "object_id" in item:
                manifest_ids.add(str(item["object_id"]))
        return manifest_ids

    def _validate_object_membership(self, object_id: str, manifest_ids: set[str]) -> None:
        if manifest_ids and object_id not in manifest_ids:
            raise ValueError(f"{object_id} is not present in the simulator object manifest.")

    def _resolve_relative_path(self, path_ref: str) -> Path:
        candidate = Path(path_ref)
        if candidate.is_absolute():
            return candidate
        return (self.catalog_path.parent / candidate).resolve()


@dataclass
class ContractPlanningBridge:
    planner: Planner
    causal_output_provider: CausalOutputProvider | None = None

    def plan_contract(
        self,
        *,
        task_id: str,
        instruction: str,
        state: WorldState,
        causal_outputs: dict[str, CausalExploreOutput] | None = None,
    ) -> list[PlannerStep]:
        contract_plan = ContractPlannerAdapter(self.planner).plan_contract(
            task_id=task_id,
            instruction=instruction,
            state=state,
        )
        return self._inject_probe_if_needed(
            task_id=task_id,
            contract_plan=contract_plan,
            causal_outputs=causal_outputs,
        )

    def replan_contract(
        self,
        *,
        task_id: str,
        instruction: str,
        state: WorldState,
        failed_step: PlanStep,
        remaining_plan: list[PlanStep],
        failure: StepFailure,
        causal_outputs: dict[str, CausalExploreOutput] | None = None,
    ) -> list[PlannerStep]:
        contract_plan = ContractPlannerAdapter(self.planner).replan_contract(
            task_id=task_id,
            instruction=instruction,
            state=state,
            failed_step=failed_step,
            remaining_plan=remaining_plan,
            failure=failure,
        )
        return self._inject_probe_if_needed(
            task_id=task_id,
            contract_plan=contract_plan,
            causal_outputs=causal_outputs,
        )

    def _inject_probe_if_needed(
        self,
        *,
        task_id: str,
        contract_plan: list[PlannerStep],
        causal_outputs: dict[str, CausalExploreOutput] | None,
    ) -> list[PlannerStep]:
        resolved_outputs = causal_outputs
        if resolved_outputs is None:
            resolved_outputs = self._load_causal_outputs(contract_plan)

        if not resolved_outputs:
            return contract_plan

        target_index = self._find_first_target_step_index(contract_plan)
        if target_index is None:
            return contract_plan

        target_object = contract_plan[target_index].target_object
        if target_object is None:
            return contract_plan

        causal_output = resolved_outputs.get(target_object)
        if causal_output is None or not causal_output.requires_probe():
            return contract_plan

        probe_step = build_planner_step_from_causal_output(
            task_id=task_id,
            step_index=target_index,
            causal_output=causal_output,
            requested_skill=contract_plan[target_index].selected_skill,
        )
        return self._insert_probe(contract_plan, target_index, probe_step)

    def _load_causal_outputs(self, contract_plan: list[PlannerStep]) -> dict[str, CausalExploreOutput]:
        if self.causal_output_provider is None:
            return {}

        object_ids = self._collect_target_object_ids(contract_plan)
        if not object_ids:
            return {}

        return self.causal_output_provider.get_outputs(object_ids)

    def _collect_target_object_ids(self, contract_plan: list[PlannerStep]) -> list[str]:
        object_ids: list[str] = []
        seen: set[str] = set()
        for step in contract_plan:
            if step.target_object is None or step.target_object in seen:
                continue
            seen.add(step.target_object)
            object_ids.append(step.target_object)
        return object_ids

    def _find_first_target_step_index(self, contract_plan: list[PlannerStep]) -> int | None:
        for index, step in enumerate(contract_plan):
            if step.selected_skill != "observe" and step.target_object is not None:
                return index
        return None

    def _insert_probe(
        self,
        contract_plan: list[PlannerStep],
        target_index: int,
        probe_step: PlannerStep,
    ) -> list[PlannerStep]:
        updated_plan = list(contract_plan)
        updated_plan.insert(target_index, probe_step)
        return [
            PlannerStep(
                task_id=step.task_id,
                step_index=index,
                selected_skill=step.selected_skill,
                target_object=step.target_object,
                skill_args=step.skill_args,
                preconditions=step.preconditions,
                expected_effect=step.expected_effect,
                fallback_action=step.fallback_action,
            )
            for index, step in enumerate(updated_plan)
        ]
