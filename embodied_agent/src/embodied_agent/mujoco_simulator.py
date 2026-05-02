from __future__ import annotations

import math
import random
from typing import Any, Sequence

import mujoco
import numpy as np

from embodied_agent.simulator import SceneConfig
from embodied_agent.types import Vec3, WorldState


def _quaternion_from_euler(euler: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = euler
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _quat_conjugate(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (-quat[0], -quat[1], -quat[2], quat[3])


def _quat_multiply(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = lhs
    bx, by, bz, bw = rhs
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _rotate_vector(
    quat: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    rotated = _quat_multiply(
        _quat_multiply(quat, (vector[0], vector[1], vector[2], 0.0)),
        _quat_conjugate(quat),
    )
    return (rotated[0], rotated[1], rotated[2])


class MujocoPickPlaceSimulation:
    """MuJoCo-backed kinematic pick/place simulator.

    This backend intentionally matches the high-level test contract of the
    legacy PyBullet simulator. It uses MuJoCo for scene/model state while the
    scripted end-effector remains kinematic until full robot assets exist.
    """

    def __init__(
        self,
        gui: bool = False,
        config: SceneConfig | None = None,
        object_names: Sequence[str] | None = None,
        zone_names: Sequence[str] | None = None,
    ) -> None:
        self.gui = gui
        self.config = config or SceneConfig()
        self.object_names = self._resolve_active_names(
            names=object_names,
            supported_names=tuple(self.config.object_specs.keys()),
            default_names=("red_block",),
            entity_label="object",
        )
        self.zone_names = self._resolve_active_names(
            names=zone_names,
            supported_names=tuple(self.config.zone_specs.keys()),
            default_names=("green_zone",),
            entity_label="zone",
        )
        self.model = mujoco.MjModel.from_xml_string(self._build_xml())
        self.data = mujoco.MjData(self.model)
        self.object_ids = {name: index for index, name in enumerate(self.object_names)}
        self.zone_ids = {name: index for index, name in enumerate(self.zone_names)}
        self.object_positions: dict[str, Vec3] = {}
        self.object_orientations: dict[str, tuple[float, float, float, float]] = {}
        self.zone_positions: dict[str, Vec3] = {}
        self.end_effector_position: Vec3 = self.config.carry_pose
        self.end_effector_orientation = _quaternion_from_euler((math.pi, 0.0, -math.pi / 2.0))
        self.downward_orientation = self.end_effector_orientation
        self.gripper_target = 0.04
        self.held_object_name: str | None = None
        self.held_local_position: tuple[float, float, float] | None = None
        self.held_local_orientation: tuple[float, float, float, float] | None = None
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.object_positions = {
            object_name: self.config.object_start_for(object_name)
            for object_name in self.object_names
        }
        self.object_orientations = {
            object_name: (0.0, 0.0, 0.0, 1.0)
            for object_name in self.object_names
        }
        self.zone_positions = {
            zone_name: self.config.zone_center_for(zone_name)
            for zone_name in self.zone_names
        }
        self.held_object_name = None
        self.held_local_position = None
        self.held_local_orientation = None
        self.move_home()
        self._sync_all_to_mujoco()

    def fast_reset(self) -> None:
        self.held_object_name = None
        self.held_local_position = None
        self.held_local_orientation = None
        self.move_home()
        for object_name, object_xy in self.default_object_layout().items():
            self.set_object_xy(object_name, object_xy)
        for zone_name, zone_xy in self.default_zone_layout().items():
            self.set_zone_xy(zone_name, zone_xy)
        self._simulate(1)

    def move_home(self) -> None:
        self.end_effector_position = self.config.carry_pose
        self.end_effector_orientation = self.downward_orientation
        self.open_gripper()

    def reset_task(
        self,
        object_xy: tuple[float, float] | None = None,
        zone_xy: tuple[float, float] | None = None,
        holding_object: bool = False,
        object_layout: dict[str, tuple[float, float]] | None = None,
        zone_layout: dict[str, tuple[float, float]] | None = None,
        held_object_name: str | None = None,
    ) -> None:
        self.fast_reset()
        resolved_object_layout = self.default_object_layout()
        resolved_zone_layout = self.default_zone_layout()
        if object_xy is not None:
            resolved_object_layout["red_block"] = object_xy
        if zone_xy is not None:
            resolved_zone_layout["green_zone"] = zone_xy
        if object_layout is not None:
            resolved_object_layout.update(object_layout)
        if zone_layout is not None:
            resolved_zone_layout.update(zone_layout)
        for object_name, position_xy in resolved_object_layout.items():
            if object_name in self.object_ids:
                self.set_object_xy(object_name, position_xy)
        for zone_name, position_xy in resolved_zone_layout.items():
            if zone_name in self.zone_ids:
                self.set_zone_xy(zone_name, position_xy)
        resolved_held_object_name = held_object_name
        if resolved_held_object_name is None and holding_object:
            resolved_held_object_name = "red_block"
        if resolved_held_object_name is not None:
            self.prepare_object_for_place(resolved_held_object_name)

    def default_object_layout(self) -> dict[str, tuple[float, float]]:
        return {
            object_name: self.config.object_start_xy_for(object_name)
            for object_name in self.object_names
        }

    def default_zone_layout(self) -> dict[str, tuple[float, float]]:
        return {
            zone_name: self.config.zone_center_xy_for(zone_name)
            for zone_name in self.zone_names
        }

    def sample_task_layout(
        self,
        rng: random.Random | None = None,
        min_separation: float = 0.18,
        object_x_range: tuple[float, float] | None = None,
        object_y_range: tuple[float, float] | None = None,
        zone_x_range: tuple[float, float] | None = None,
        zone_y_range: tuple[float, float] | None = None,
        object_candidates: Sequence[tuple[float, float]] | None = None,
        zone_candidates: Sequence[tuple[float, float]] | None = None,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        object_name = self._single_active_name(self.object_names, "object")
        zone_name = self._single_active_name(self.zone_names, "zone")
        object_layout, zone_layout = self.sample_scene_layout(
            rng=rng,
            object_names=(object_name,),
            zone_names=(zone_name,),
            min_separation=min_separation,
            object_x_range=object_x_range,
            object_y_range=object_y_range,
            zone_x_range=zone_x_range,
            zone_y_range=zone_y_range,
            object_candidates=object_candidates,
            zone_candidates=zone_candidates,
        )
        return object_layout[object_name], zone_layout[zone_name]

    def sample_scene_layout(
        self,
        rng: random.Random | None = None,
        object_names: Sequence[str] | None = None,
        zone_names: Sequence[str] | None = None,
        min_separation: float = 0.12,
        object_x_range: tuple[float, float] | None = None,
        object_y_range: tuple[float, float] | None = None,
        zone_x_range: tuple[float, float] | None = None,
        zone_y_range: tuple[float, float] | None = None,
        object_candidates: Sequence[tuple[float, float]] | None = None,
        zone_candidates: Sequence[tuple[float, float]] | None = None,
    ) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
        sampler = rng or random.Random()
        selected_object_names = tuple(object_names or self.object_names)
        selected_zone_names = tuple(zone_names or self.zone_names)
        if len(selected_object_names) == 1 and len(selected_zone_names) == 1:
            return self._sample_single_task_layout(
                sampler=sampler,
                object_names=selected_object_names,
                zone_names=selected_zone_names,
                min_separation=min_separation,
                object_x_range=object_x_range,
                object_y_range=object_y_range,
                zone_x_range=zone_x_range,
                zone_y_range=zone_y_range,
                object_candidates=object_candidates,
                zone_candidates=zone_candidates,
            )
        object_layout = self._sample_layout_from_candidates(
            names=selected_object_names,
            candidates=((0.40, -0.16), (0.56, -0.16), (0.68, -0.16), (0.48, -0.02), (0.64, -0.02)),
            sampler=sampler,
            min_separation=min_separation,
        )
        zone_layout = self._sample_layout_from_candidates(
            names=selected_zone_names,
            candidates=((0.40, 0.12), (0.56, 0.12), (0.68, 0.12), (0.48, 0.24), (0.64, 0.24)),
            sampler=sampler,
            min_separation=min_separation,
        )
        return object_layout, zone_layout

    def observe_scene(self, instruction: str = "") -> WorldState:
        return WorldState(
            instruction=instruction,
            object_positions=dict(self.object_positions),
            zone_positions=dict(self.zone_positions),
            end_effector_position=self.end_effector_position,
            held_object_name=self.held_object_name,
        )

    def get_object_position(self, object_name: str) -> Vec3:
        return self.object_positions[object_name]

    def get_object_pose(self, object_name: str) -> tuple[Vec3, tuple[float, float, float, float]]:
        return self.object_positions[object_name], self.object_orientations[object_name]

    def get_skill_observation(self, object_name: str, zone_name: str) -> np.ndarray:
        end_effector_position, _ = self.get_end_effector_pose()
        object_position = self.object_positions[object_name]
        zone_position = self.zone_positions[zone_name]
        ee_to_object = tuple(object_position[index] - end_effector_position[index] for index in range(3))
        object_to_zone = tuple(zone_position[index] - object_position[index] for index in range(3))
        return np.array(
            [
                *end_effector_position,
                *object_position,
                *zone_position,
                *ee_to_object,
                *object_to_zone,
                self.get_gripper_open_ratio(),
                1.0 if self.held_object_name == object_name else 0.0,
            ],
            dtype=np.float32,
        )

    def get_skill_metrics(self, object_name: str, zone_name: str) -> dict[str, float]:
        end_effector_position, _ = self.get_end_effector_pose()
        object_position = self.object_positions[object_name]
        zone_position = self.zone_positions[zone_name]
        return {
            "ee_object_distance": math.dist(end_effector_position, object_position),
            "object_zone_distance_xy": math.dist(object_position[:2], zone_position[:2]),
            "object_height": object_position[2],
            "holding_flag": 1.0 if self.held_object_name == object_name else 0.0,
            "lift_progress": max(0.0, object_position[2] - self.config.placed_object_z),
            "gripper_open_ratio": self.get_gripper_open_ratio(),
        }

    def capture_skill_state(self, object_name: str, zone_name: str) -> dict[str, object]:
        end_effector_position, end_effector_orientation = self.get_end_effector_pose()
        object_position, object_orientation = self.get_object_pose(object_name)
        zone_position = self.zone_positions[zone_name]
        ee_to_object = tuple(float(object_position[index] - end_effector_position[index]) for index in range(3))
        object_to_zone = tuple(float(zone_position[index] - object_position[index]) for index in range(3))
        return {
            "object_name": object_name,
            "zone_name": zone_name,
            "held_object_name": self.held_object_name,
            "holding_target_object": self.held_object_name == object_name,
            "gripper_open_ratio": float(self.get_gripper_open_ratio()),
            "end_effector_position": [float(value) for value in end_effector_position],
            "end_effector_orientation": [float(value) for value in end_effector_orientation],
            "object_position": [float(value) for value in object_position],
            "object_orientation": [float(value) for value in object_orientation],
            "zone_position": [float(value) for value in zone_position],
            "ee_to_object": [float(value) for value in ee_to_object],
            "object_to_zone": [float(value) for value in object_to_zone],
            "held_local_position": (
                [float(value) for value in self.held_local_position]
                if self.held_object_name == object_name and self.held_local_position is not None
                else None
            ),
            "held_local_orientation": (
                [float(value) for value in self.held_local_orientation]
                if self.held_object_name == object_name and self.held_local_orientation is not None
                else None
            ),
            "metrics": {
                key: float(value)
                for key, value in self.get_skill_metrics(object_name, zone_name).items()
            },
        }

    def simulate_steps(self, steps: int) -> None:
        self._simulate(max(0, int(steps)))

    def is_object_held(self, object_name: str) -> bool:
        return self.held_object_name == object_name

    def is_pick_success(self, object_name: str) -> bool:
        object_position = self.object_positions[object_name]
        lift_threshold = self.config.placed_object_z + self.config.pick_success_height
        return self.held_object_name == object_name and object_position[2] >= lift_threshold

    def apply_skill_action(
        self,
        delta_position: Vec3,
        gripper_command: float,
        action_steps: int = 24,
        object_name: str = "red_block",
    ) -> WorldState:
        current_position, _ = self.get_end_effector_pose()
        target_position = tuple(current_position[index] + delta_position[index] for index in range(3))
        self.set_gripper_command(gripper_command)
        self.teleport_end_effector(self._clamp_position(target_position))
        self._simulate(action_steps)
        self._maybe_attach_object(object_name)
        self._maybe_release_held_object()
        return self.observe_scene()

    def set_gripper_command(self, gripper_command: float) -> None:
        clipped_command = max(-1.0, min(1.0, gripper_command))
        self.gripper_target = 0.5 * (clipped_command + 1.0) * 0.04

    def get_gripper_open_ratio(self) -> float:
        return max(0.0, min(1.0, self.gripper_target / 0.04))

    def set_object_xy(self, object_name: str, object_xy: tuple[float, float]) -> None:
        self.set_object_position(object_name, (object_xy[0], object_xy[1], self.config.placed_object_z))

    def set_object_position(self, object_name: str, position: Vec3) -> None:
        self.set_object_pose(object_name, position)

    def set_object_pose(
        self,
        object_name: str,
        position: Vec3,
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        self.object_positions[object_name] = tuple(float(value) for value in position)
        self.object_orientations[object_name] = tuple(float(value) for value in orientation)
        self._sync_object_to_mujoco(object_name)

    def set_zone_xy(self, zone_name: str, zone_xy: tuple[float, float]) -> None:
        self.zone_positions[zone_name] = (
            float(zone_xy[0]),
            float(zone_xy[1]),
            self.config.goal_center[2],
        )

    def restore_runtime_state(self, state: dict[str, Any], object_name: str, zone_name: str) -> None:
        record_object_name = str(state.get("object_name", object_name))
        record_zone_name = str(state.get("zone_name", zone_name))
        if record_object_name != object_name:
            raise ValueError(f"Runtime state object mismatch: expected {object_name}, got {record_object_name}.")
        if record_zone_name != zone_name:
            raise ValueError(f"Runtime state zone mismatch: expected {zone_name}, got {record_zone_name}.")
        object_position = self._coerce_vec3(state.get("object_position"), "object_position")
        object_orientation = self._coerce_quaternion(state.get("object_orientation"), "object_orientation")
        zone_position = self._coerce_vec3(state.get("zone_position"), "zone_position")
        end_effector_position = self._coerce_vec3(state.get("end_effector_position"), "end_effector_position")
        end_effector_orientation = self._coerce_quaternion(state.get("end_effector_orientation"), "end_effector_orientation")
        gripper_open_ratio = float(state.get("gripper_open_ratio", 0.0))
        self.reset_task(
            object_layout={object_name: (object_position[0], object_position[1])},
            zone_layout={zone_name: (zone_position[0], zone_position[1])},
            holding_object=False,
        )
        self.set_gripper_command(2.0 * gripper_open_ratio - 1.0)
        self.teleport_end_effector_pose(end_effector_position, end_effector_orientation)
        if bool(state.get("holding_target_object", False)):
            held_local_position = self._coerce_vec3(state.get("held_local_position"), "held_local_position")
            held_local_orientation = self._coerce_quaternion(state.get("held_local_orientation"), "held_local_orientation")
            self._set_held_object_offset(object_name, held_local_position, held_local_orientation)
        else:
            self.held_object_name = None
            self.held_local_position = None
            self.held_local_orientation = None
            self.set_object_pose(object_name, object_position, object_orientation)
        self._simulate(1)

    def prepare_object_for_place(self, object_name: str) -> None:
        self.open_gripper()
        self.teleport_end_effector(self.config.carry_pose)
        end_effector_position, _ = self.get_end_effector_pose()
        preload_position = (end_effector_position[0], end_effector_position[1], end_effector_position[2] - 0.06)
        self.set_object_position(object_name, preload_position)
        self.close_gripper()
        self._attach_object(object_name)
        self._simulate(1)

    def normalize_held_object_for_place(self, object_name: str) -> None:
        if self.held_object_name != object_name:
            raise RuntimeError("The requested object is not currently held by the gripper.")
        self.close_gripper()
        self.teleport_end_effector(self.config.carry_pose)
        end_effector_position, _ = self.get_end_effector_pose()
        self.set_object_position(
            object_name,
            (end_effector_position[0], end_effector_position[1], end_effector_position[2] - 0.06),
        )
        self._attach_object(object_name)
        self._simulate(1)

    def prepare_pick_staging_pose(self, object_name: str) -> None:
        object_position = self.object_positions[object_name]
        self.open_gripper()
        self.teleport_end_effector((object_position[0], object_position[1], self.config.table_top_z + self.config.pick_height))

    def prepare_place_staging_pose(self, zone_name: str) -> None:
        zone_position = self.zone_positions[zone_name]
        self.teleport_end_effector((zone_position[0], zone_position[1], self.config.table_top_z + 0.085))

    def canonicalize_held_object_for_place(
        self,
        object_name: str,
        zone_name: str,
        height_offset: float = 0.0,
    ) -> None:
        if self.held_object_name != object_name:
            raise RuntimeError("The requested object is not currently held by the gripper.")
        zone_position = self.zone_positions[zone_name]
        target_object_position = (zone_position[0], zone_position[1], self.config.placed_object_z + height_offset)
        self.close_gripper()
        self._set_held_object_offset(
            object_name,
            self._world_to_local(target_object_position, (0.0, 0.0, 0.0, 1.0))[0],
            (0.0, 0.0, 0.0, 1.0),
        )
        self._simulate(1)

    def pick_object(self, object_name: str) -> None:
        object_position = self.object_positions[object_name]
        hover_position = (object_position[0], object_position[1], self.config.table_top_z + self.config.hover_height)
        approach_position = (object_position[0], object_position[1], self.config.table_top_z + self.config.pick_height)
        self.open_gripper()
        self.move_end_effector(hover_position)
        self.move_end_effector(approach_position)
        self.close_gripper()
        self._attach_object(object_name)
        self.move_end_effector(hover_position)

    def place_object(self, zone_name: str) -> None:
        if self.held_object_name is None:
            raise RuntimeError("No object is currently attached to the gripper.")
        zone_position = self.zone_positions[zone_name]
        hover_position = (zone_position[0], zone_position[1], self.config.table_top_z + self.config.hover_height)
        place_position = (zone_position[0], zone_position[1], self.config.table_top_z + self.config.place_height)
        final_object_position = (zone_position[0], zone_position[1], self.config.placed_object_z)
        self.move_end_effector(hover_position)
        self.move_end_effector(place_position)
        self._release_object(final_object_position)
        self.open_gripper()
        self._simulate(1)
        self.move_end_effector(hover_position)

    def is_object_in_zone(self, object_name: str, zone_name: str, tolerance: float = 0.08) -> bool:
        object_position = self.object_positions[object_name]
        zone_position = self.zone_positions[zone_name]
        return (
            math.dist(object_position[:2], zone_position[:2]) <= tolerance
            and abs(object_position[2] - self.config.placed_object_z) <= 0.05
        )

    def is_place_success(self, object_name: str, zone_name: str, tolerance: float = 0.08) -> bool:
        return self.held_object_name != object_name and self.is_object_in_zone(object_name, zone_name, tolerance=tolerance)

    def move_end_effector(self, target_position: Vec3, steps: int = 240) -> None:
        previous_position = self.end_effector_position
        self.end_effector_position = tuple(float(value) for value in target_position)
        self._apply_contact_motion(previous_position, self.end_effector_position)
        if self.held_object_name is not None:
            self._update_held_object_pose()
        self._simulate(max(1, min(int(steps), 24)))

    def teleport_end_effector(self, target_position: Vec3) -> None:
        self.teleport_end_effector_pose(target_position, self.downward_orientation)

    def teleport_end_effector_pose(
        self,
        target_position: Vec3,
        target_orientation: tuple[float, float, float, float],
    ) -> None:
        self.end_effector_position = tuple(float(value) for value in target_position)
        self.end_effector_orientation = tuple(float(value) for value in target_orientation)
        if self.held_object_name is not None:
            self._update_held_object_pose()
        self._simulate(1)

    def open_gripper(self) -> None:
        self.gripper_target = 0.04

    def close_gripper(self) -> None:
        self.gripper_target = 0.0

    def get_end_effector_pose(self) -> tuple[Vec3, tuple[float, float, float, float]]:
        return self.end_effector_position, self.end_effector_orientation

    def get_quaternion_from_euler(self, euler: tuple[float, float, float]) -> tuple[float, float, float, float]:
        return _quaternion_from_euler(euler)

    def shutdown(self) -> None:
        return None

    def _attach_object(self, object_name: str) -> None:
        object_position, object_orientation = self.get_object_pose(object_name)
        local_position, local_orientation = self._world_to_local(object_position, object_orientation)
        self.held_object_name = object_name
        self.held_local_position = local_position
        self.held_local_orientation = local_orientation
        self._update_held_object_pose()

    def _set_held_object_offset(
        self,
        object_name: str,
        local_position: tuple[float, float, float],
        local_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        self.held_object_name = object_name
        self.held_local_position = local_position
        self.held_local_orientation = local_orientation
        self._update_held_object_pose()

    def _release_object(self, final_object_position: Vec3 | None = None) -> None:
        if self.held_object_name is None:
            return
        object_name = self.held_object_name
        if final_object_position is not None:
            self.set_object_pose(object_name, final_object_position)
        self.held_object_name = None
        self.held_local_position = None
        self.held_local_orientation = None

    def _update_held_object_pose(self) -> None:
        if self.held_object_name is None or self.held_local_position is None:
            return
        local_orientation = self.held_local_orientation or (0.0, 0.0, 0.0, 1.0)
        rotated_offset = _rotate_vector(self.end_effector_orientation, self.held_local_position)
        world_position = tuple(self.end_effector_position[index] + rotated_offset[index] for index in range(3))
        world_orientation = _quat_multiply(self.end_effector_orientation, local_orientation)
        self.set_object_pose(self.held_object_name, world_position, world_orientation)

    def _simulate(self, steps: int) -> None:
        self._sync_all_to_mujoco()
        for _ in range(max(0, int(steps))):
            mujoco.mj_step(self.model, self.data)

    def _maybe_attach_object(self, object_name: str) -> None:
        if self.held_object_name is not None:
            return
        end_effector_position, _ = self.get_end_effector_pose()
        object_position = self.object_positions[object_name]
        close_enough = math.dist(end_effector_position, object_position) <= 0.12
        low_gripper = self.get_gripper_open_ratio() <= 0.25
        aligned_height = end_effector_position[2] <= object_position[2] + 0.12
        if close_enough and low_gripper and aligned_height:
            self._attach_object(object_name)

    def _maybe_release_held_object(self) -> None:
        if self.held_object_name is not None and self.get_gripper_open_ratio() >= 0.70:
            self._release_object(None)

    def _apply_contact_motion(self, previous_position: Vec3, target_position: Vec3) -> None:
        if self.held_object_name is not None:
            return
        delta = tuple(target_position[index] - previous_position[index] for index in range(3))
        if math.dist((0.0, 0.0, 0.0), delta) == 0.0:
            return
        for object_name, object_position in list(self.object_positions.items()):
            near_xy = math.dist(target_position[:2], object_position[:2]) <= 0.09
            near_height = abs(target_position[2] - object_position[2]) <= 0.08
            closed = self.get_gripper_open_ratio() <= 0.25
            if closed and near_xy and near_height:
                self.set_object_position(
                    object_name,
                    (
                        object_position[0] + delta[0],
                        object_position[1] + delta[1],
                        max(self.config.placed_object_z, object_position[2] + delta[2]),
                    ),
                )
            elif near_xy and target_position[2] <= object_position[2] + 0.04:
                self.set_object_position(
                    object_name,
                    (object_position[0], object_position[1], max(self.config.table_top_z, object_position[2] - 0.001)),
                )

    def _world_to_local(
        self,
        world_position: Vec3,
        world_orientation: tuple[float, float, float, float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        inverse_orientation = _quat_conjugate(self.end_effector_orientation)
        position_delta = tuple(world_position[index] - self.end_effector_position[index] for index in range(3))
        local_position = _rotate_vector(inverse_orientation, position_delta)
        local_orientation = _quat_multiply(inverse_orientation, world_orientation)
        return local_position, local_orientation

    def _sync_object_to_mujoco(self, object_name: str) -> None:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_name}_joint")
        qpos_address = self.model.jnt_qposadr[joint_id]
        position = self.object_positions[object_name]
        orientation = self.object_orientations[object_name]
        self.data.qpos[qpos_address : qpos_address + 7] = [
            position[0],
            position[1],
            position[2],
            orientation[3],
            orientation[0],
            orientation[1],
            orientation[2],
        ]
        self.data.qvel[self.model.jnt_dofadr[joint_id] : self.model.jnt_dofadr[joint_id] + 6] = 0.0

    def _sync_all_to_mujoco(self) -> None:
        for object_name in self.object_names:
            self._sync_object_to_mujoco(object_name)
        mujoco.mj_forward(self.model, self.data)

    def _build_xml(self) -> str:
        object_bodies = []
        for object_name in self.object_names:
            spec = self.config.object_specs[object_name]
            position = self.config.object_start_for(object_name)
            object_bodies.append(
                f"""
                <body name="{object_name}" pos="{position[0]} {position[1]} {position[2]}">
                    <freejoint name="{object_name}_joint"/>
                    <geom name="{object_name}_geom" type="box"
                          size="{self.config.cube_half_extent} {self.config.cube_half_extent} {self.config.cube_half_extent}"
                          rgba="{spec.rgba[0]} {spec.rgba[1]} {spec.rgba[2]} {spec.rgba[3]}"
                          mass="0.05"/>
                </body>
                """
            )
        zone_bodies = []
        for zone_name in self.zone_names:
            spec = self.config.zone_specs[zone_name]
            position = self.config.zone_center_for(zone_name)
            zone_bodies.append(
                f"""
                <body name="{zone_name}" pos="{position[0]} {position[1]} {position[2]}">
                    <geom name="{zone_name}_geom" type="box"
                          size="{self.config.goal_half_extents[0]} {self.config.goal_half_extents[1]} {self.config.goal_half_extents[2]}"
                          rgba="{spec.rgba[0]} {spec.rgba[1]} {spec.rgba[2]} {spec.rgba[3]}"
                          contype="0" conaffinity="0"/>
                </body>
                """
            )
        return f"""
        <mujoco model="pick_place_kinematic">
            <option timestep="0.01" gravity="0 0 -9.81"/>
            <worldbody>
                <geom name="floor" type="plane" size="2 2 0.01" rgba="0.92 0.92 0.92 1"/>
                <body name="table" pos="{self.config.table_center[0]} {self.config.table_center[1]} {self.config.table_center[2]}">
                    <geom name="table_geom" type="box"
                          size="{self.config.table_half_extents[0]} {self.config.table_half_extents[1]} {self.config.table_half_extents[2]}"
                          rgba="0.55 0.43 0.32 1"/>
                </body>
                {''.join(zone_bodies)}
                {''.join(object_bodies)}
            </worldbody>
        </mujoco>
        """

    def _clamp_position(self, target_position: Vec3) -> Vec3:
        return tuple(
            max(self.config.workspace_low[index], min(self.config.workspace_high[index], target_position[index]))
            for index in range(3)
        )

    def _resolve_active_names(
        self,
        names: Sequence[str] | None,
        supported_names: tuple[str, ...],
        default_names: tuple[str, ...],
        entity_label: str,
    ) -> tuple[str, ...]:
        candidate_names = tuple(dict.fromkeys(names or default_names))
        unsupported_names = [name for name in candidate_names if name not in supported_names]
        if unsupported_names:
            raise ValueError(f"Unsupported {entity_label} names: {', '.join(sorted(unsupported_names))}.")
        return candidate_names

    def _single_active_name(self, names: tuple[str, ...], entity_label: str) -> str:
        if len(names) != 1:
            raise ValueError(f"Expected exactly one active {entity_label}, but found {len(names)}.")
        return names[0]

    def _sample_layout_from_candidates(
        self,
        names: Sequence[str],
        candidates: Sequence[tuple[float, float]],
        sampler: random.Random,
        min_separation: float,
    ) -> dict[str, tuple[float, float]]:
        if len(names) > len(candidates):
            raise RuntimeError("Not enough candidate positions to place all requested scene entities.")
        for _ in range(128):
            shuffled_candidates = list(candidates)
            sampler.shuffle(shuffled_candidates)
            selected_candidates = shuffled_candidates[: len(names)]
            if all(
                math.dist(position, other_position) >= min_separation
                for index, position in enumerate(selected_candidates)
                for other_position in selected_candidates[index + 1 :]
            ):
                return {name: position for name, position in zip(names, selected_candidates)}
        raise RuntimeError("Candidate layout does not satisfy the requested minimum separation.")

    def _sample_single_task_layout(
        self,
        sampler: random.Random,
        object_names: Sequence[str],
        zone_names: Sequence[str],
        min_separation: float,
        object_x_range: tuple[float, float] | None = None,
        object_y_range: tuple[float, float] | None = None,
        zone_x_range: tuple[float, float] | None = None,
        zone_y_range: tuple[float, float] | None = None,
        object_candidates: Sequence[tuple[float, float]] | None = None,
        zone_candidates: Sequence[tuple[float, float]] | None = None,
    ) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
        object_name = self._single_active_name(tuple(object_names), "object")
        zone_name = self._single_active_name(tuple(zone_names), "zone")
        if object_candidates is not None or zone_candidates is not None:
            resolved_object_candidates = tuple(object_candidates or ()) or (self.config.object_start_xy_for(object_name),)
            resolved_zone_candidates = tuple(zone_candidates or ()) or (self.config.zone_center_xy_for(zone_name),)
            for _ in range(128):
                object_xy = resolved_object_candidates[sampler.randrange(len(resolved_object_candidates))]
                zone_xy = resolved_zone_candidates[sampler.randrange(len(resolved_zone_candidates))]
                if math.dist(object_xy, zone_xy) >= min_separation:
                    return {object_name: object_xy}, {zone_name: zone_xy}
        resolved_object_x_range = object_x_range or (0.45, 0.58)
        resolved_object_y_range = object_y_range or (-0.15, 0.15)
        resolved_zone_x_range = zone_x_range or (0.45, 0.56)
        resolved_zone_y_range = zone_y_range or (-0.15, 0.15)
        for _ in range(128):
            object_xy = (sampler.uniform(*resolved_object_x_range), sampler.uniform(*resolved_object_y_range))
            zone_xy = (sampler.uniform(*resolved_zone_x_range), sampler.uniform(*resolved_zone_y_range))
            if math.dist(object_xy, zone_xy) >= min_separation:
                return {object_name: object_xy}, {zone_name: zone_xy}
        return (
            {object_name: self.config.object_start_xy_for(object_name)},
            {zone_name: self.config.zone_center_xy_for(zone_name)},
        )

    def _coerce_vec3(self, value: object, label: str) -> tuple[float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{label} must be a 3D position.")
        return (float(value[0]), float(value[1]), float(value[2]))

    def _coerce_quaternion(self, value: object, label: str) -> tuple[float, float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError(f"{label} must be a quaternion.")
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
