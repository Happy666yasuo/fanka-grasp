from __future__ import annotations

import math
import sys
import os
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../embodied_agent/src')))
from embodied_agent.contracts import ContactState, ExecutorResult, FailureRecord
from embodied_agent.mujoco_simulator import MujocoPickPlaceSimulation


def improve_place_release_timing(
    simulation: MujocoPickPlaceSimulation,
    zone_name: str,
    tolerance: float = 0.08,
    check_steps: int = 30,
) -> dict[str, Any]:
    """Check zone proximity before release and adjust if needed.

    Returns debug info about the release check.
    """
    debug: dict[str, Any] = {
        "object_name": simulation.held_object_name,
        "zone_name": zone_name,
        "zone_proximity_ok": False,
        "adjustments_made": 0,
        "final_ee_zone_distance_xy": float("inf"),
    }

    if simulation.held_object_name is None:
        debug["error"] = "no_held_object"
        return debug

    zone_position = simulation.zone_positions[zone_name]
    config = simulation.config

    for attempt in range(3):
        ee_pos, _ = simulation.get_end_effector_pose()
        xy_distance = math.dist(ee_pos[:2], zone_position[:2])

        if xy_distance <= tolerance:
            debug["zone_proximity_ok"] = True
            debug["final_ee_zone_distance_xy"] = xy_distance
            break

        target_xy = (
            zone_position[0],
            zone_position[1],
            config.table_top_z + config.hover_height,
        )
        simulation.teleport_end_effector(target_xy)
        simulation._simulate(check_steps)
        debug["adjustments_made"] = attempt + 1

    if not debug["zone_proximity_ok"]:
        ee_pos, _ = simulation.get_end_effector_pose()
        debug["final_ee_zone_distance_xy"] = math.dist(ee_pos[:2], zone_position[:2])

    return debug


def improve_place_transport(
    simulation: MujocoPickPlaceSimulation,
    zone_name: str,
    descend_slow_factor: float = 2.0,
) -> dict[str, Any]:
    """Move to zone centroid using controlled descent for more reliable placement.

    Returns debug info about the transport operation.
    """
    debug: dict[str, Any] = {
        "object_name": simulation.held_object_name,
        "zone_name": zone_name,
        "transport_success": False,
        "stages_completed": 0,
    }

    if simulation.held_object_name is None:
        debug["error"] = "no_held_object"
        return debug

    zone_position = simulation.zone_positions[zone_name]
    config = simulation.config

    hover_pos = (
        zone_position[0],
        zone_position[1],
        config.table_top_z + config.hover_height,
    )
    place_pos = (
        zone_position[0],
        zone_position[1],
        config.table_top_z + config.place_height,
    )
    final_object_pos = (
        zone_position[0],
        zone_position[1],
        config.placed_object_z,
    )

    simulation.move_end_effector(hover_pos)
    debug["stages_completed"] = 1

    simulation.move_end_effector(
        place_pos,
        steps=int(240 * descend_slow_factor),
    )
    debug["stages_completed"] = 2

    held_object_name = simulation.held_object_name
    simulation._release_object(final_object_pos)
    simulation.open_gripper()
    simulation._simulate(120)
    debug["stages_completed"] = 3

    simulation.move_end_effector(hover_pos)
    debug["stages_completed"] = 4

    debug["transport_success"] = True
    debug["released_object"] = held_object_name
    return debug


def improved_place_object(
    simulation: MujocoPickPlaceSimulation,
    zone_name: str,
    object_name: str | None = None,
    tolerance: float = 0.08,
) -> ExecutorResult:
    """Improved place that combines zone-centered transport with release timing checks.

    This is a drop-in improvement wrapper around the simulation's place capabilities.
    """
    try:
        if simulation.held_object_name is None:
            return ExecutorResult(
                success=False,
                reward=0.0,
                final_state={"zone_name": zone_name},
                contact_state=ContactState(has_contact=False, contact_region=None),
                error_code="place_failed: no_held_object",
                rollout_path="scripted://improved_place",
                failure_history=[FailureRecord(
                    step_index=0,
                    selected_skill="place",
                    failure_source="execution_error",
                    reason="No object is currently attached to the gripper.",
                    replan_attempt=0,
                )],
            )

        held_object = simulation.held_object_name

        zone_position = simulation.zone_positions[zone_name]
        config = simulation.config

        hover_pos = (
            zone_position[0],
            zone_position[1],
            config.table_top_z + config.hover_height,
        )
        place_pos = (
            zone_position[0],
            zone_position[1],
            config.table_top_z + config.place_height,
        )
        final_object_pos = (
            zone_position[0],
            zone_position[1],
            config.placed_object_z,
        )

        simulation.move_end_effector(hover_pos)
        release_debug = improve_place_release_timing(
            simulation, zone_name, tolerance=tolerance,
        )

        simulation.move_end_effector(place_pos)
        simulation._release_object(final_object_pos)
        simulation.open_gripper()
        simulation._simulate(120)
        simulation.move_end_effector(hover_pos)

        success = simulation.is_place_success(
            held_object, zone_name, tolerance=tolerance,
        )

        return ExecutorResult(
            success=success,
            reward=1.0 if success else 0.0,
            final_state={
                "executed_skill": "place",
                "target_zone": zone_name,
                "held_object": held_object,
                "release_debug": release_debug,
            },
            contact_state=ContactState(
                has_contact=False,
                contact_region=None,
            ),
            error_code=None if success else "place_failed: released_outside_zone",
            rollout_path="scripted://improved_place",
            failure_history=[] if success else [FailureRecord(
                step_index=0,
                selected_skill="place",
                failure_source="post_condition_failed",
                reason="Object released outside zone after improved placement.",
                replan_attempt=0,
            )],
        )

    except Exception as exc:
        return ExecutorResult(
            success=False,
            reward=0.0,
            final_state={"zone_name": zone_name},
            contact_state=ContactState(has_contact=False, contact_region=None),
            error_code=f"place_failed: {exc}",
            rollout_path="scripted://improved_place",
            failure_history=[FailureRecord(
                step_index=0,
                selected_skill="place",
                failure_source="execution_error",
                reason=str(exc),
                replan_attempt=0,
            )],
        )
