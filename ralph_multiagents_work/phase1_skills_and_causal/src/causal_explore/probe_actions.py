from __future__ import annotations

import math
import sys
import os
from dataclasses import dataclass, field
from typing import Any, Callable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../embodied_agent/src')))
from embodied_agent.mujoco_simulator import MujocoPickPlaceSimulation


@dataclass
class ProbeActionResult:
    probe_name: str
    object_name: str
    success: bool
    pre_position: tuple[float, float, float]
    post_position: tuple[float, float, float]
    displacement: tuple[float, float, float]
    displacement_magnitude: float
    contact_detected: bool
    observations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_name": self.probe_name,
            "object_name": self.object_name,
            "success": self.success,
            "pre_position": list(self.pre_position),
            "post_position": list(self.post_position),
            "displacement": list(self.displacement),
            "displacement_magnitude": self.displacement_magnitude,
            "contact_detected": self.contact_detected,
            "observations": dict(self.observations),
        }


ProbeAction = Callable[[MujocoPickPlaceSimulation, str], ProbeActionResult]


def _get_object_position(simulation: MujocoPickPlaceSimulation, object_name: str) -> tuple[float, float, float]:
    return tuple(simulation.get_object_position(object_name))


def lateral_push(
    simulation: MujocoPickPlaceSimulation,
    object_name: str = "red_block",
    direction: str = "right",
    push_distance: float = 0.08,
) -> ProbeActionResult:
    """Push object laterally from the side and measure displacement."""
    pre_pos = _get_object_position(simulation, object_name)
    table_top_z = simulation.config.table_top_z

    dir_map: dict[str, tuple[float, float, float]] = {
        "right": (1.0, 0.0, 0.0),
        "left": (-1.0, 0.0, 0.0),
        "forward": (0.0, 1.0, 0.0),
        "backward": (0.0, -1.0, 0.0),
    }
    dir_vec = dir_map.get(direction, (1.0, 0.0, 0.0))

    try:
        approach_pos = (
            pre_pos[0] - dir_vec[0] * 0.06,
            pre_pos[1] - dir_vec[1] * 0.06,
            table_top_z + 0.09,
        )
        push_target = (
            pre_pos[0] + dir_vec[0] * push_distance,
            pre_pos[1] + dir_vec[1] * push_distance,
            table_top_z + 0.09,
        )

        simulation.open_gripper()
        simulation.move_end_effector(approach_pos)
        simulation.close_gripper()
        simulation.move_end_effector(push_target, steps=120)
        simulation.open_gripper()

        post_pos = _get_object_position(simulation, object_name)
        displacement = tuple(post_pos[i] - pre_pos[i] for i in range(3))
        disp_mag = math.sqrt(sum(d * d for d in displacement))
        contact = disp_mag > 0.001

        return ProbeActionResult(
            probe_name="lateral_push",
            object_name=object_name,
            success=contact,
            pre_position=pre_pos,
            post_position=post_pos,
            displacement=displacement,
            displacement_magnitude=disp_mag,
            contact_detected=contact,
            observations={
                "direction": direction,
                "push_distance": push_distance,
            },
        )
    except Exception as exc:
        post_pos = _get_object_position(simulation, object_name)
        return ProbeActionResult(
            probe_name="lateral_push",
            object_name=object_name,
            success=False,
            pre_position=pre_pos,
            post_position=post_pos,
            displacement=(0.0, 0.0, 0.0),
            displacement_magnitude=0.0,
            contact_detected=False,
            observations={"error": str(exc)},
        )


def top_press(
    simulation: MujocoPickPlaceSimulation,
    object_name: str = "red_block",
    press_depth: float = 0.02,
) -> ProbeActionResult:
    """Press down on top of object and measure resistance."""
    pre_pos = _get_object_position(simulation, object_name)
    table_top_z = simulation.config.table_top_z

    try:
        hover_pos = (pre_pos[0], pre_pos[1], table_top_z + 0.20)
        press_pos = (pre_pos[0], pre_pos[1], table_top_z + 0.06)

        simulation.open_gripper()
        simulation.move_end_effector(hover_pos)
        simulation.move_end_effector(press_pos, steps=120)
        simulation._simulate(60)
        simulation.move_end_effector(hover_pos)

        post_pos = _get_object_position(simulation, object_name)
        displacement = tuple(post_pos[i] - pre_pos[i] for i in range(3))
        disp_mag = math.sqrt(sum(d * d for d in displacement))
        z_delta = abs(post_pos[2] - pre_pos[2])
        contact = z_delta > 0.0005

        return ProbeActionResult(
            probe_name="top_press",
            object_name=object_name,
            success=contact,
            pre_position=pre_pos,
            post_position=post_pos,
            displacement=displacement,
            displacement_magnitude=disp_mag,
            contact_detected=contact,
            observations={
                "press_depth": press_depth,
                "z_delta": z_delta,
            },
        )
    except Exception as exc:
        post_pos = _get_object_position(simulation, object_name)
        return ProbeActionResult(
            probe_name="top_press",
            object_name=object_name,
            success=False,
            pre_position=pre_pos,
            post_position=post_pos,
            displacement=(0.0, 0.0, 0.0),
            displacement_magnitude=0.0,
            contact_detected=False,
            observations={"error": str(exc)},
        )


def side_pull(
    simulation: MujocoPickPlaceSimulation,
    object_name: str = "red_block",
    direction: str = "left",
    pull_distance: float = 0.06,
) -> ProbeActionResult:
    """Pull object from the side and measure resistance/displacement."""
    pre_pos = _get_object_position(simulation, object_name)
    table_top_z = simulation.config.table_top_z

    dir_map: dict[str, tuple[float, float, float]] = {
        "left": (-1.0, 0.0, 0.0),
        "right": (1.0, 0.0, 0.0),
        "backward": (0.0, -1.0, 0.0),
        "forward": (0.0, 1.0, 0.0),
    }
    dir_vec = dir_map.get(direction, (-1.0, 0.0, 0.0))

    try:
        grasp_pos = (
            pre_pos[0] + dir_vec[0] * 0.08,
            pre_pos[1] + dir_vec[1] * 0.08,
            table_top_z + 0.08,
        )
        hover_pos = (
            grasp_pos[0],
            grasp_pos[1],
            table_top_z + 0.18,
        )
        pull_target = (
            pre_pos[0] + dir_vec[0] * (0.08 + pull_distance),
            pre_pos[1] + dir_vec[1] * (0.08 + pull_distance),
            table_top_z + 0.10,
        )

        simulation.open_gripper()
        simulation.move_end_effector(hover_pos)
        simulation.move_end_effector(grasp_pos)
        simulation.close_gripper()
        simulation.move_end_effector(pull_target, steps=120)
        simulation.open_gripper()

        post_pos = _get_object_position(simulation, object_name)
        displacement = tuple(post_pos[i] - pre_pos[i] for i in range(3))
        disp_mag = math.sqrt(sum(d * d for d in displacement))
        contact = disp_mag > 0.001

        return ProbeActionResult(
            probe_name="side_pull",
            object_name=object_name,
            success=contact,
            pre_position=pre_pos,
            post_position=post_pos,
            displacement=displacement,
            displacement_magnitude=disp_mag,
            contact_detected=contact,
            observations={
                "direction": direction,
                "pull_distance": pull_distance,
            },
        )
    except Exception as exc:
        post_pos = _get_object_position(simulation, object_name)
        return ProbeActionResult(
            probe_name="side_pull",
            object_name=object_name,
            success=False,
            pre_position=pre_pos,
            post_position=post_pos,
            displacement=(0.0, 0.0, 0.0),
            displacement_magnitude=0.0,
            contact_detected=False,
            observations={"error": str(exc)},
        )


def surface_tap(
    simulation: MujocoPickPlaceSimulation,
    object_name: str = "red_block",
    tap_height: float = 0.10,
) -> ProbeActionResult:
    """Lightly tap the object surface from above and measure vibration/displacement."""
    pre_pos = _get_object_position(simulation, object_name)
    table_top_z = simulation.config.table_top_z

    try:
        hover_pos = (pre_pos[0], pre_pos[1], table_top_z + 0.20)
        tap_pos = (pre_pos[0], pre_pos[1], table_top_z + tap_height)

        simulation.open_gripper()
        simulation.move_end_effector(hover_pos)
        simulation.move_end_effector(tap_pos, steps=60)
        simulation.move_end_effector(hover_pos, steps=60)

        post_pos = _get_object_position(simulation, object_name)
        displacement = tuple(post_pos[i] - pre_pos[i] for i in range(3))
        disp_mag = math.sqrt(sum(d * d for d in displacement))
        z_delta = abs(post_pos[2] - pre_pos[2])
        contact = z_delta > 0.0005

        return ProbeActionResult(
            probe_name="surface_tap",
            object_name=object_name,
            success=contact,
            pre_position=pre_pos,
            post_position=post_pos,
            displacement=displacement,
            displacement_magnitude=disp_mag,
            contact_detected=contact,
            observations={
                "tap_height": tap_height,
                "z_delta": z_delta,
            },
        )
    except Exception as exc:
        post_pos = _get_object_position(simulation, object_name)
        return ProbeActionResult(
            probe_name="surface_tap",
            object_name=object_name,
            success=False,
            pre_position=pre_pos,
            post_position=post_pos,
            displacement=(0.0, 0.0, 0.0),
            displacement_magnitude=0.0,
            contact_detected=False,
            observations={"error": str(exc)},
        )


def grasp_attempt(
    simulation: MujocoPickPlaceSimulation,
    object_name: str = "red_block",
) -> ProbeActionResult:
    """Attempt to grasp the object and check if graspable."""
    pre_pos = _get_object_position(simulation, object_name)
    table_top_z = simulation.config.table_top_z

    try:
        simulation.pick_object(object_name)
        post_pos = _get_object_position(simulation, object_name)
        graspable = simulation.held_object_name == object_name

        displacement = tuple(post_pos[i] - pre_pos[i] for i in range(3))
        disp_mag = math.sqrt(sum(d * d for d in displacement))

        if graspable:
            simulation.place_object("green_zone")

        return ProbeActionResult(
            probe_name="grasp_attempt",
            object_name=object_name,
            success=graspable,
            pre_position=pre_pos,
            post_position=post_pos,
            displacement=displacement,
            displacement_magnitude=disp_mag,
            contact_detected=graspable,
            observations={
                "graspable": graspable,
                "lift_height": post_pos[2] - pre_pos[2] if graspable else 0.0,
            },
        )
    except Exception as exc:
        post_pos = _get_object_position(simulation, object_name)
        return ProbeActionResult(
            probe_name="grasp_attempt",
            object_name=object_name,
            success=False,
            pre_position=pre_pos,
            post_position=post_pos,
            displacement=(0.0, 0.0, 0.0),
            displacement_magnitude=0.0,
            contact_detected=False,
            observations={"error": str(exc)},
        )


PROBE_ACTION_REGISTRY: dict[str, ProbeAction] = {
    "lateral_push": lateral_push,
    "top_press": top_press,
    "side_pull": side_pull,
    "surface_tap": surface_tap,
    "grasp_attempt": grasp_attempt,
}
