from __future__ import annotations

import json
from pathlib import Path

from embodied_agent.planner import RuleBasedPlanner
from embodied_agent.planning_bridge import (
    ArtifactRegistryCausalOutputProvider,
    ContractPlanningBridge,
    SimulatorArtifactCatalogCausalOutputProvider,
)
from embodied_agent.types import WorldState


DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "causal_explore"
    / "mock_registry_v1"
    / "registry.json"
)

PLANNING_SCENARIOS: dict[str, str] = {
    "uncertain_red_to_green": "put the red block in the green zone",
    "confident_blue_to_yellow": "put the blue block in the yellow zone",
}


def build_default_world_state(instruction: str) -> WorldState:
    return WorldState(
        instruction=instruction,
        object_positions={
            "red_block": (0.55, 0.10, 0.645),
            "blue_block": (0.63, 0.02, 0.645),
            "yellow_block": (0.44, 0.16, 0.645),
        },
        zone_positions={
            "green_zone": (0.70, -0.18, 0.623),
            "blue_zone": (0.58, 0.18, 0.623),
            "yellow_zone": (0.40, 0.13, 0.623),
        },
        end_effector_position=(0.35, 0.00, 0.75),
    )


def run_mock_planning_bridge(
    instruction: str,
    *,
    registry_path: str | Path | None = None,
    catalog_path: str | Path | None = None,
    task_id: str = "mock_task",
    scenario: str | None = None,
) -> dict[str, object]:
    provider, resolved_registry_path, resolved_catalog_path = _build_provider(
        registry_path=registry_path,
        catalog_path=catalog_path,
    )
    bridge = ContractPlanningBridge(
        RuleBasedPlanner(),
        causal_output_provider=provider,
    )
    plan = bridge.plan_contract(
        task_id=task_id,
        instruction=instruction,
        state=build_default_world_state(instruction),
    )
    return {
        "task_id": task_id,
        "scenario": scenario,
        "instruction": instruction,
        "registry_path": str(resolved_registry_path) if resolved_registry_path is not None else None,
        "catalog_path": str(resolved_catalog_path) if resolved_catalog_path is not None else None,
        "planner_steps": [step.to_dict() for step in plan],
    }


def render_mock_planning_bridge(
    instruction: str,
    *,
    registry_path: str | Path | None = None,
    catalog_path: str | Path | None = None,
    task_id: str = "mock_task",
    scenario: str | None = None,
) -> str:
    return json.dumps(
        run_mock_planning_bridge(
            instruction,
            registry_path=registry_path,
            catalog_path=catalog_path,
            task_id=task_id,
            scenario=scenario,
        ),
        indent=2,
        ensure_ascii=False,
    )


def _build_provider(
    *,
    registry_path: str | Path | None,
    catalog_path: str | Path | None,
) -> tuple[object, Path | None, Path | None]:
    if catalog_path is not None:
        resolved_catalog_path = Path(catalog_path).resolve()
        return (
            SimulatorArtifactCatalogCausalOutputProvider(resolved_catalog_path),
            None,
            resolved_catalog_path,
        )

    resolved_registry_path = Path(registry_path or DEFAULT_REGISTRY_PATH).resolve()
    return (
        ArtifactRegistryCausalOutputProvider(resolved_registry_path),
        resolved_registry_path,
        None,
    )
