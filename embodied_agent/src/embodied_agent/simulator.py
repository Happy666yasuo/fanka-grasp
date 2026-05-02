from __future__ import annotations

import math
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from embodied_agent.types import Vec3, WorldState


pybullet: Any | None = None
pybullet_data: Any | None = None


def _load_pybullet_modules() -> None:
    global pybullet, pybullet_data
    if pybullet is not None and pybullet_data is not None:
        return
    import pybullet as loaded_pybullet
    import pybullet_data as loaded_pybullet_data

    pybullet = loaded_pybullet
    pybullet_data = loaded_pybullet_data


@dataclass(frozen=True)
class SceneObjectSpec:
    name: str
    rgba: tuple[float, float, float, float]
    start_xy: tuple[float, float]


@dataclass(frozen=True)
class SceneZoneSpec:
    name: str
    rgba: tuple[float, float, float, float]
    center_xy: tuple[float, float]


@dataclass(frozen=True)
class SceneConfig:
    robot_base: Vec3 = (0.0, 0.0, 0.0)
    table_center: Vec3 = (0.55, 0.0, 0.60)
    table_half_extents: Vec3 = (0.35, 0.45, 0.02)
    cube_half_extent: float = 0.025
    cube_start_xy: tuple[float, float] = (0.55, 0.10)
    blue_block_start_xy: tuple[float, float] = (0.64, 0.00)
    yellow_block_start_xy: tuple[float, float] = (0.46, -0.12)
    goal_center_xy: tuple[float, float] = (0.50, -0.10)
    blue_zone_center_xy: tuple[float, float] = (0.65, 0.13)
    yellow_zone_center_xy: tuple[float, float] = (0.42, 0.14)
    goal_half_extents: Vec3 = (0.08, 0.08, 0.002)
    hover_height: float = 0.20
    pick_height: float = 0.08
    place_height: float = 0.08
    carry_height: float = 0.18
    pick_success_height: float = 0.06

    @property
    def table_top_z(self) -> float:
        return self.table_center[2] + self.table_half_extents[2]

    @property
    def cube_start(self) -> Vec3:
        return self.object_start_for("red_block")

    @property
    def goal_center(self) -> Vec3:
        return self.zone_center_for("green_zone")

    @property
    def placed_object_z(self) -> float:
        return self.table_top_z + self.cube_half_extent

    @property
    def object_specs(self) -> dict[str, SceneObjectSpec]:
        return {
            "red_block": SceneObjectSpec(
                name="red_block",
                rgba=(0.90, 0.15, 0.15, 1.0),
                start_xy=self.cube_start_xy,
            ),
            "blue_block": SceneObjectSpec(
                name="blue_block",
                rgba=(0.20, 0.35, 0.90, 1.0),
                start_xy=self.blue_block_start_xy,
            ),
            "yellow_block": SceneObjectSpec(
                name="yellow_block",
                rgba=(0.92, 0.75, 0.16, 1.0),
                start_xy=self.yellow_block_start_xy,
            ),
        }

    @property
    def zone_specs(self) -> dict[str, SceneZoneSpec]:
        return {
            "green_zone": SceneZoneSpec(
                name="green_zone",
                rgba=(0.20, 0.75, 0.25, 0.65),
                center_xy=self.goal_center_xy,
            ),
            "blue_zone": SceneZoneSpec(
                name="blue_zone",
                rgba=(0.20, 0.35, 0.85, 0.65),
                center_xy=self.blue_zone_center_xy,
            ),
            "yellow_zone": SceneZoneSpec(
                name="yellow_zone",
                rgba=(0.88, 0.75, 0.18, 0.65),
                center_xy=self.yellow_zone_center_xy,
            ),
        }

    def object_start_xy_for(self, object_name: str) -> tuple[float, float]:
        return self.object_specs[object_name].start_xy

    def object_start_for(self, object_name: str) -> Vec3:
        object_xy = self.object_start_xy_for(object_name)
        return (
            object_xy[0],
            object_xy[1],
            self.table_top_z + self.cube_half_extent,
        )

    def zone_center_xy_for(self, zone_name: str) -> tuple[float, float]:
        return self.zone_specs[zone_name].center_xy

    def zone_center_for(self, zone_name: str) -> Vec3:
        zone_xy = self.zone_center_xy_for(zone_name)
        return (
            zone_xy[0],
            zone_xy[1],
            self.table_top_z + self.goal_half_extents[2] + 0.001,
        )

    @property
    def carry_pose(self) -> Vec3:
        return (0.60, 0.0, self.table_top_z + self.carry_height)

    @property
    def workspace_low(self) -> Vec3:
        return (0.35, -0.35, self.table_top_z + 0.02)

    @property
    def workspace_high(self) -> Vec3:
        return (0.82, 0.35, self.table_top_z + 0.35)


class BulletPickPlaceSimulation:
    def __init__(
        self,
        gui: bool = False,
        config: SceneConfig | None = None,
        object_names: Sequence[str] | None = None,
        zone_names: Sequence[str] | None = None,
    ) -> None:
        _load_pybullet_modules()
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
        self.asset_root = self._prepare_asset_root()
        self.client_id = pybullet.connect(pybullet.GUI if gui else pybullet.DIRECT)
        self.robot_id = -1
        self.end_effector_index = -1
        self.arm_joint_indices: list[int] = []
        self.finger_joint_indices: list[int] = []
        self.object_ids: dict[str, int] = {}
        self.zone_ids: dict[str, int] = {}
        self.zone_positions: dict[str, Vec3] = {}
        self.held_object_name: str | None = None
        self.held_local_position: tuple[float, float, float] | None = None
        self.held_local_orientation: tuple[float, float, float, float] | None = None
        self.downward_orientation = pybullet.getQuaternionFromEuler(
            (math.pi, 0.0, -math.pi / 2.0)
        )
        self.gripper_target = 0.04
        self.reset()

    def reset(self) -> None:
        pybullet.resetSimulation(physicsClientId=self.client_id)
        pybullet.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        pybullet.setTimeStep(1.0 / 240.0, physicsClientId=self.client_id)
        pybullet.setAdditionalSearchPath(str(self.asset_root), physicsClientId=self.client_id)
        self._create_floor()
        self._create_table()
        self._load_robot()
        self._create_scene_objects()
        self.held_object_name = None
        self.held_local_position = None
        self.held_local_orientation = None
        self.move_home()

    def fast_reset(self) -> None:
        """Reset robot and objects without reloading URDFs (much faster)."""
        self.held_object_name = None
        self.held_local_position = None
        self.held_local_orientation = None
        self.move_home()
        for object_name, object_xy in self.default_object_layout().items():
            self.set_object_xy(object_name, object_xy)
        for zone_name, zone_xy in self.default_zone_layout().items():
            self.set_zone_xy(zone_name, zone_xy)
        self._simulate(60)

    def move_home(self) -> None:
        home_joint_positions = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
        for joint_index, joint_position in zip(self.arm_joint_indices, home_joint_positions):
            pybullet.resetJointState(
                self.robot_id,
                joint_index,
                targetValue=joint_position,
                physicsClientId=self.client_id,
            )
        self.open_gripper()
        self._simulate(60)

    def reset_task(
        self,
        object_xy: tuple[float, float] | None = None,
        zone_xy: tuple[float, float] | None = None,
        holding_object: bool = False,
        object_layout: dict[str, tuple[float, float]] | None = None,
        zone_layout: dict[str, tuple[float, float]] | None = None,
        held_object_name: str | None = None,
    ) -> None:
        if self.robot_id >= 0:
            self.fast_reset()
        else:
            self.reset()

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
            candidates=(
                (0.40, -0.16),
                (0.56, -0.16),
                (0.68, -0.16),
                (0.48, -0.02),
                (0.64, -0.02),
            ),
            sampler=sampler,
            min_separation=min_separation,
        )
        zone_layout = self._sample_layout_from_candidates(
            names=selected_zone_names,
            candidates=(
                (0.40, 0.12),
                (0.56, 0.12),
                (0.68, 0.12),
                (0.48, 0.24),
                (0.64, 0.24),
            ),
            sampler=sampler,
            min_separation=min_separation,
        )
        return object_layout, zone_layout

    def observe_scene(self, instruction: str = "") -> WorldState:
        end_effector_position, _ = self.get_end_effector_pose()
        object_positions = {
            name: self._get_body_position(body_id) for name, body_id in self.object_ids.items()
        }
        return WorldState(
            instruction=instruction,
            object_positions=object_positions,
            zone_positions=dict(self.zone_positions),
            end_effector_position=end_effector_position,
            held_object_name=self.held_object_name,
        )

    def get_skill_observation(self, object_name: str, zone_name: str) -> np.ndarray:
        end_effector_position, _ = self.get_end_effector_pose()
        object_position = self._get_body_position(self.object_ids[object_name])
        zone_position = self.zone_positions[zone_name]
        ee_to_object = tuple(object_position[index] - end_effector_position[index] for index in range(3))
        object_to_zone = tuple(zone_position[index] - object_position[index] for index in range(3))
        observation = np.array(
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
        return observation

    def get_object_position(self, object_name: str) -> Vec3:
        return self._get_body_position(self.object_ids[object_name])

    def get_object_pose(self, object_name: str) -> tuple[Vec3, tuple[float, float, float, float]]:
        position, orientation = pybullet.getBasePositionAndOrientation(
            self.object_ids[object_name],
            physicsClientId=self.client_id,
        )
        return tuple(position), tuple(orientation)

    def get_quaternion_from_euler(
        self,
        euler: tuple[float, float, float],
    ) -> tuple[float, float, float, float]:
        return tuple(pybullet.getQuaternionFromEuler(euler))

    def get_skill_metrics(self, object_name: str, zone_name: str) -> dict[str, float]:
        end_effector_position, _ = self.get_end_effector_pose()
        object_position = self._get_body_position(self.object_ids[object_name])
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
        object_position, object_orientation = pybullet.getBasePositionAndOrientation(
            self.object_ids[object_name],
            physicsClientId=self.client_id,
        )
        zone_position = self.zone_positions[zone_name]
        ee_to_object = tuple(
            float(object_position[index] - end_effector_position[index]) for index in range(3)
        )
        object_to_zone = tuple(
            float(zone_position[index] - object_position[index]) for index in range(3)
        )
        held_local_position = None
        held_local_orientation = None
        if self.held_object_name == object_name:
            if self.held_local_position is not None:
                held_local_position = [float(value) for value in self.held_local_position]
            if self.held_local_orientation is not None:
                held_local_orientation = [float(value) for value in self.held_local_orientation]

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
            "held_local_position": held_local_position,
            "held_local_orientation": held_local_orientation,
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
        object_position = self._get_body_position(self.object_ids[object_name])
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
        target_position = tuple(
            current_position[index] + delta_position[index] for index in range(3)
        )
        clamped_target_position = self._clamp_position(target_position)
        self.set_gripper_command(gripper_command)
        self.teleport_end_effector(clamped_target_position)
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
        self.set_object_position(
            object_name,
            (object_xy[0], object_xy[1], self.config.placed_object_z),
        )

    def set_object_position(self, object_name: str, position: Vec3) -> None:
        self.set_object_pose(object_name, position)

    def set_object_pose(
        self,
        object_name: str,
        position: Vec3,
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        pybullet.resetBasePositionAndOrientation(
            self.object_ids[object_name],
            posObj=position,
            ornObj=orientation,
            physicsClientId=self.client_id,
        )
        pybullet.resetBaseVelocity(
            self.object_ids[object_name],
            linearVelocity=(0.0, 0.0, 0.0),
            angularVelocity=(0.0, 0.0, 0.0),
            physicsClientId=self.client_id,
        )

    def set_zone_xy(self, zone_name: str, zone_xy: tuple[float, float]) -> None:
        zone_position = (
            zone_xy[0],
            zone_xy[1],
            self.config.goal_center[2],
        )
        self.zone_positions[zone_name] = zone_position
        pybullet.resetBasePositionAndOrientation(
            self.zone_ids[zone_name],
            posObj=zone_position,
            ornObj=(0.0, 0.0, 0.0, 1.0),
            physicsClientId=self.client_id,
        )

    def restore_runtime_state(
        self,
        state: dict[str, Any],
        object_name: str,
        zone_name: str,
    ) -> None:
        record_object_name = str(state.get("object_name", object_name))
        record_zone_name = str(state.get("zone_name", zone_name))
        if record_object_name != object_name:
            raise ValueError(
                "Runtime state object mismatch: "
                f"expected {object_name}, got {record_object_name}."
            )
        if record_zone_name != zone_name:
            raise ValueError(
                "Runtime state zone mismatch: "
                f"expected {zone_name}, got {record_zone_name}."
            )

        object_position = self._coerce_vec3(state.get("object_position"), label="object_position")
        object_orientation = self._coerce_quaternion(
            state.get("object_orientation"),
            label="object_orientation",
        )
        zone_position = self._coerce_vec3(state.get("zone_position"), label="zone_position")
        end_effector_position = self._coerce_vec3(
            state.get("end_effector_position"),
            label="end_effector_position",
        )
        end_effector_orientation = self._coerce_quaternion(
            state.get("end_effector_orientation"),
            label="end_effector_orientation",
        )
        gripper_open_ratio = float(state.get("gripper_open_ratio", 0.0))

        self.reset_task(
            object_layout={object_name: (object_position[0], object_position[1])},
            zone_layout={zone_name: (zone_position[0], zone_position[1])},
            holding_object=False,
        )
        self.set_gripper_command(2.0 * gripper_open_ratio - 1.0)
        self.teleport_end_effector_pose(end_effector_position, end_effector_orientation)

        holding_target_object = bool(state.get("holding_target_object", False))
        held_local_position_raw = state.get("held_local_position")
        held_local_orientation_raw = state.get("held_local_orientation")
        if holding_target_object:
            if held_local_position_raw is None or held_local_orientation_raw is None:
                raise ValueError(
                    "Runtime state is missing held-object local transform for a held object."
                )
            held_local_position = self._coerce_vec3(
                held_local_position_raw,
                label="held_local_position",
            )
            held_local_orientation = self._coerce_quaternion(
                held_local_orientation_raw,
                label="held_local_orientation",
            )
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
        preload_position = (
            end_effector_position[0],
            end_effector_position[1],
            end_effector_position[2] - 0.06,
        )
        self.set_object_position(object_name, preload_position)
        self.close_gripper()
        self._attach_object(object_name)
        self._simulate(60)

    def normalize_held_object_for_place(self, object_name: str) -> None:
        if self.held_object_name != object_name:
            raise RuntimeError("The requested object is not currently held by the gripper.")

        self.close_gripper()
        self.teleport_end_effector(self.config.carry_pose)
        end_effector_position, _ = self.get_end_effector_pose()
        preload_position = (
            end_effector_position[0],
            end_effector_position[1],
            end_effector_position[2] - 0.06,
        )
        self.set_object_position(object_name, preload_position)
        self._attach_object(object_name)
        self._simulate(60)

    def prepare_pick_staging_pose(self, object_name: str) -> None:
        object_position = self._get_body_position(self.object_ids[object_name])
        staging_position = (
            object_position[0],
            object_position[1],
            self.config.table_top_z + self.config.pick_height,
        )
        self.open_gripper()
        self.teleport_end_effector(staging_position)

    def prepare_place_staging_pose(self, zone_name: str) -> None:
        zone_position = self.zone_positions[zone_name]
        staging_position = (
            zone_position[0],
            zone_position[1],
            self.config.table_top_z + 0.085,
        )
        self.teleport_end_effector(staging_position)

    def canonicalize_held_object_for_place(
        self,
        object_name: str,
        zone_name: str,
        height_offset: float = 0.0,
    ) -> None:
        if self.held_object_name != object_name:
            raise RuntimeError("The requested object is not currently held by the gripper.")

        zone_position = self.zone_positions[zone_name]
        target_object_position = (
            zone_position[0],
            zone_position[1],
            self.config.placed_object_z + height_offset,
        )
        end_effector_position, end_effector_orientation = self.get_end_effector_pose()
        inverted_position, inverted_orientation = pybullet.invertTransform(
            end_effector_position,
            end_effector_orientation,
        )
        local_position, local_orientation = pybullet.multiplyTransforms(
            inverted_position,
            inverted_orientation,
            target_object_position,
            (0.0, 0.0, 0.0, 1.0),
        )
        self.close_gripper()
        self._set_held_object_offset(
            object_name,
            tuple(local_position),
            tuple(local_orientation),
        )
        self._simulate(60)

    def pick_object(self, object_name: str) -> None:
        object_id = self.object_ids[object_name]
        object_position = self._get_body_position(object_id)
        hover_position = (
            object_position[0],
            object_position[1],
            self.config.table_top_z + self.config.hover_height,
        )
        approach_position = (
            object_position[0],
            object_position[1],
            self.config.table_top_z + self.config.pick_height,
        )
        lift_position = (
            object_position[0],
            object_position[1],
            self.config.table_top_z + self.config.hover_height,
        )

        self.open_gripper()
        self.move_end_effector(hover_position)
        self.move_end_effector(approach_position)
        self.close_gripper()
        self._attach_object(object_name)
        self.move_end_effector(lift_position)

    def place_object(self, zone_name: str) -> None:
        if self.held_object_name is None:
            raise RuntimeError("No object is currently attached to the gripper.")

        zone_position = self.zone_positions[zone_name]
        hover_position = (
            zone_position[0],
            zone_position[1],
            self.config.table_top_z + self.config.hover_height,
        )
        place_position = (
            zone_position[0],
            zone_position[1],
            self.config.table_top_z + self.config.place_height,
        )
        final_object_position = (
            zone_position[0],
            zone_position[1],
            self.config.placed_object_z,
        )

        self.move_end_effector(hover_position)
        self.move_end_effector(place_position)
        held_object_name = self.held_object_name
        self._release_object(final_object_position)
        self.open_gripper()
        self._simulate(120)
        self.move_end_effector(hover_position)

        if held_object_name is None:
            raise RuntimeError("The place skill released an invalid object.")

    def is_object_in_zone(self, object_name: str, zone_name: str, tolerance: float = 0.08) -> bool:
        object_position = self._get_body_position(self.object_ids[object_name])
        zone_position = self.zone_positions[zone_name]
        xy_distance = math.dist(object_position[:2], zone_position[:2])
        z_delta = abs(object_position[2] - self.config.placed_object_z)
        return xy_distance <= tolerance and z_delta <= 0.05

    def is_place_success(self, object_name: str, zone_name: str, tolerance: float = 0.08) -> bool:
        return self.held_object_name != object_name and self.is_object_in_zone(
            object_name,
            zone_name,
            tolerance=tolerance,
        )

    def move_end_effector(self, target_position: Vec3, steps: int = 240) -> None:
        joint_targets = pybullet.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            targetPosition=target_position,
            targetOrientation=self.downward_orientation,
            maxNumIterations=100,
            residualThreshold=1e-4,
            physicsClientId=self.client_id,
        )

        for joint_index, joint_target in zip(self.arm_joint_indices, joint_targets[:7]):
            pybullet.setJointMotorControl2(
                self.robot_id,
                joint_index,
                pybullet.POSITION_CONTROL,
                targetPosition=float(joint_target),
                force=200.0,
                positionGain=0.05,
                velocityGain=1.0,
                physicsClientId=self.client_id,
            )

        self._simulate(steps)

    def teleport_end_effector(self, target_position: Vec3) -> None:
        self.teleport_end_effector_pose(target_position, self.downward_orientation)

    def teleport_end_effector_pose(
        self,
        target_position: Vec3,
        target_orientation: tuple[float, float, float, float],
    ) -> None:
        num_all_joints = pybullet.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        lower_limits = []
        upper_limits = []
        joint_ranges = []
        rest_poses = []
        home_positions = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
        arm_set = set(self.arm_joint_indices)
        arm_i = 0
        for j in range(num_all_joints):
            info = pybullet.getJointInfo(self.robot_id, j, physicsClientId=self.client_id)
            if info[3] > -1:
                ll, ul = float(info[8]), float(info[9])
                if ll >= ul:
                    ll, ul = -2 * math.pi, 2 * math.pi
                lower_limits.append(ll)
                upper_limits.append(ul)
                joint_ranges.append(ul - ll)
                if j in arm_set and arm_i < len(home_positions):
                    rest_poses.append(home_positions[arm_i])
                    arm_i += 1
                else:
                    rest_poses.append(0.0)

        for _ik_iter in range(5):
            joint_targets = pybullet.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_index,
                targetPosition=target_position,
                targetOrientation=target_orientation,
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
                maxNumIterations=200,
                residualThreshold=1e-5,
                physicsClientId=self.client_id,
            )
            for joint_index, joint_target in zip(self.arm_joint_indices, joint_targets[:7]):
                pybullet.resetJointState(
                    self.robot_id,
                    joint_index,
                    targetValue=float(joint_target),
                    physicsClientId=self.client_id,
                )

        for joint_index, joint_target in zip(self.arm_joint_indices, joint_targets[:7]):
            pybullet.setJointMotorControl2(
                self.robot_id,
                joint_index,
                pybullet.POSITION_CONTROL,
                targetPosition=float(joint_target),
                force=200.0,
                positionGain=0.3,
                velocityGain=1.0,
                physicsClientId=self.client_id,
            )

        if self.held_object_name is not None:
            self._update_held_object_pose()
        self._simulate(24)

    def open_gripper(self) -> None:
        self.gripper_target = 0.04
        self._apply_gripper_target()
        self._simulate(90)

    def close_gripper(self) -> None:
        self.gripper_target = 0.0
        self._apply_gripper_target()
        self._simulate(90)

    def get_end_effector_pose(self) -> tuple[Vec3, tuple[float, float, float, float]]:
        link_state = pybullet.getLinkState(
            self.robot_id,
            self.end_effector_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        return tuple(link_state[4]), tuple(link_state[5])

    def shutdown(self) -> None:
        if pybullet.isConnected(self.client_id):
            pybullet.disconnect(self.client_id)

    def _coerce_vec3(self, value: object, label: str) -> tuple[float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{label} must be a 3D position.")
        return (float(value[0]), float(value[1]), float(value[2]))

    def _coerce_quaternion(
        self,
        value: object,
        label: str,
    ) -> tuple[float, float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError(f"{label} must be a quaternion.")
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))

    def _create_table(self) -> None:
        collision_shape = pybullet.createCollisionShape(
            pybullet.GEOM_BOX,
            halfExtents=self.config.table_half_extents,
            physicsClientId=self.client_id,
        )
        visual_shape = pybullet.createVisualShape(
            pybullet.GEOM_BOX,
            halfExtents=self.config.table_half_extents,
            rgbaColor=(0.55, 0.43, 0.32, 1.0),
            physicsClientId=self.client_id,
        )
        pybullet.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.config.table_center,
            physicsClientId=self.client_id,
        )

    def _create_floor(self) -> None:
        collision_shape = pybullet.createCollisionShape(
            pybullet.GEOM_PLANE,
            physicsClientId=self.client_id,
        )
        visual_shape = pybullet.createVisualShape(
            pybullet.GEOM_PLANE,
            rgbaColor=(0.92, 0.92, 0.92, 1.0),
            physicsClientId=self.client_id,
        )
        pybullet.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=(0.0, 0.0, 0.0),
            physicsClientId=self.client_id,
        )

    def _load_robot(self) -> None:
        self.robot_id = pybullet.loadURDF(
            str(self.asset_root / "franka_panda" / "panda.urdf"),
            basePosition=self.config.robot_base,
            useFixedBase=True,
            physicsClientId=self.client_id,
        )
        self.arm_joint_indices = [
            self._find_joint_index(f"panda_joint{joint_number}") for joint_number in range(1, 8)
        ]
        self.finger_joint_indices = [
            self._find_joint_index("panda_finger_joint1"),
            self._find_joint_index("panda_finger_joint2"),
        ]
        for link_name in ("panda_grasptarget_hand", "panda_hand"):
            try:
                self.end_effector_index = self._find_link_index(link_name)
                break
            except KeyError:
                continue
        if self.end_effector_index == -1:
            raise RuntimeError("Could not find a supported Franka Panda end-effector link.")

    def _create_scene_objects(self) -> None:
        cube_collision_shape = pybullet.createCollisionShape(
            pybullet.GEOM_BOX,
            halfExtents=(
                self.config.cube_half_extent,
                self.config.cube_half_extent,
                self.config.cube_half_extent,
            ),
            physicsClientId=self.client_id,
        )
        self.object_ids = {}
        for object_name in self.object_names:
            object_spec = self.config.object_specs[object_name]
            cube_visual_shape = pybullet.createVisualShape(
                pybullet.GEOM_BOX,
                halfExtents=(
                    self.config.cube_half_extent,
                    self.config.cube_half_extent,
                    self.config.cube_half_extent,
                ),
                rgbaColor=object_spec.rgba,
                physicsClientId=self.client_id,
            )
            object_id = pybullet.createMultiBody(
                baseMass=0.05,
                baseCollisionShapeIndex=cube_collision_shape,
                baseVisualShapeIndex=cube_visual_shape,
                basePosition=self.config.object_start_for(object_name),
                physicsClientId=self.client_id,
            )
            self.object_ids[object_name] = object_id

        self.zone_ids = {}
        self.zone_positions = {}
        for zone_name in self.zone_names:
            zone_spec = self.config.zone_specs[zone_name]
            zone_visual_shape = pybullet.createVisualShape(
                pybullet.GEOM_BOX,
                halfExtents=self.config.goal_half_extents,
                rgbaColor=zone_spec.rgba,
                physicsClientId=self.client_id,
            )
            zone_id = pybullet.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=zone_visual_shape,
                basePosition=self.config.zone_center_for(zone_name),
                physicsClientId=self.client_id,
            )
            self.zone_ids[zone_name] = zone_id
            self.zone_positions[zone_name] = self.config.zone_center_for(zone_name)

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
            raise ValueError(
                f"Unsupported {entity_label} names: {', '.join(sorted(unsupported_names))}."
            )
        return candidate_names

    def _single_active_name(self, names: tuple[str, ...], entity_label: str) -> str:
        if len(names) != 1:
            raise ValueError(
                f"Expected exactly one active {entity_label}, but found {len(names)}."
            )
        return names[0]

    def _sample_named_layout(
        self,
        names: Sequence[str],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        sampler: random.Random,
        min_separation: float,
        existing_positions: Sequence[tuple[float, float]] = (),
    ) -> dict[str, tuple[float, float]]:
        positions: dict[str, tuple[float, float]] = {}
        placed_positions = list(existing_positions)
        for name in names:
            for _ in range(256):
                candidate = (
                    sampler.uniform(*x_range),
                    sampler.uniform(*y_range),
                )
                if all(math.dist(candidate, position) >= min_separation for position in placed_positions):
                    positions[name] = candidate
                    placed_positions.append(candidate)
                    break
            else:
                raise RuntimeError(f"Could not sample a collision-free layout for {name}.")
        return positions

    def _sample_layout_from_candidates(
        self,
        names: Sequence[str],
        candidates: Sequence[tuple[float, float]],
        sampler: random.Random,
        min_separation: float,
    ) -> dict[str, tuple[float, float]]:
        if len(names) > len(candidates):
            raise RuntimeError("Not enough candidate positions to place all requested scene entities.")

        shuffled_candidates = list(candidates)
        sampler.shuffle(shuffled_candidates)
        selected_candidates = shuffled_candidates[: len(names)]

        for index, position in enumerate(selected_candidates):
            for other_position in selected_candidates[index + 1 :]:
                if math.dist(position, other_position) < min_separation:
                    raise RuntimeError("Candidate layout does not satisfy the requested minimum separation.")

        return {
            name: position
            for name, position in zip(names, selected_candidates)
        }

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
            resolved_object_candidates = tuple(object_candidates or ()) or (
                self.config.object_start_xy_for(object_name),
            )
            resolved_zone_candidates = tuple(zone_candidates or ()) or (
                self.config.zone_center_xy_for(zone_name),
            )
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
            object_xy = (
                sampler.uniform(*resolved_object_x_range),
                sampler.uniform(*resolved_object_y_range),
            )
            zone_xy = (
                sampler.uniform(*resolved_zone_x_range),
                sampler.uniform(*resolved_zone_y_range),
            )
            if math.dist(object_xy, zone_xy) >= min_separation:
                return {object_name: object_xy}, {zone_name: zone_xy}

        return (
            {object_name: self.config.object_start_xy_for(object_name)},
            {zone_name: self.config.zone_center_xy_for(zone_name)},
        )

    def _find_joint_index(self, joint_name: str) -> int:
        for joint_index in range(pybullet.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            current_joint_name = pybullet.getJointInfo(
                self.robot_id,
                joint_index,
                physicsClientId=self.client_id,
            )[1].decode("utf-8")
            if current_joint_name == joint_name:
                return joint_index
        raise KeyError(f"Joint not found: {joint_name}")

    def _find_link_index(self, link_name: str) -> int:
        for joint_index in range(pybullet.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            current_link_name = pybullet.getJointInfo(
                self.robot_id,
                joint_index,
                physicsClientId=self.client_id,
            )[12].decode("utf-8")
            if current_link_name == link_name:
                return joint_index
        raise KeyError(f"Link not found: {link_name}")

    def _apply_gripper_target(self) -> None:
        for joint_index in self.finger_joint_indices:
            pybullet.setJointMotorControl2(
                self.robot_id,
                joint_index,
                pybullet.POSITION_CONTROL,
                targetPosition=self.gripper_target,
                force=40.0,
                physicsClientId=self.client_id,
            )

    def _attach_object(self, object_name: str) -> None:
        object_id = self.object_ids[object_name]
        object_position, object_orientation = pybullet.getBasePositionAndOrientation(
            object_id,
            physicsClientId=self.client_id,
        )
        end_effector_position, end_effector_orientation = self.get_end_effector_pose()
        inverted_position, inverted_orientation = pybullet.invertTransform(
            end_effector_position,
            end_effector_orientation,
        )
        local_position, local_orientation = pybullet.multiplyTransforms(
            inverted_position,
            inverted_orientation,
            object_position,
            object_orientation,
        )
        self.held_object_name = object_name
        self.held_local_position = tuple(local_position)
        self.held_local_orientation = tuple(local_orientation)
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
        object_id = self.object_ids[self.held_object_name]
        current_position, current_orientation = pybullet.getBasePositionAndOrientation(
            object_id,
            physicsClientId=self.client_id,
        )
        pybullet.resetBaseVelocity(
            object_id,
            linearVelocity=(0.0, 0.0, 0.0),
            angularVelocity=(0.0, 0.0, 0.0),
            physicsClientId=self.client_id,
        )
        pybullet.resetBasePositionAndOrientation(
            object_id,
            posObj=final_object_position or current_position,
            ornObj=current_orientation if final_object_position is None else (0.0, 0.0, 0.0, 1.0),
            physicsClientId=self.client_id,
        )
        self.held_object_name = None
        self.held_local_position = None
        self.held_local_orientation = None

    def _update_held_object_pose(self) -> None:
        if self.held_object_name is None:
            return
        if self.held_local_position is None or self.held_local_orientation is None:
            return

        object_id = self.object_ids[self.held_object_name]
        end_effector_position, end_effector_orientation = self.get_end_effector_pose()
        world_position, world_orientation = pybullet.multiplyTransforms(
            end_effector_position,
            end_effector_orientation,
            self.held_local_position,
            self.held_local_orientation,
        )
        pybullet.resetBasePositionAndOrientation(
            object_id,
            posObj=world_position,
            ornObj=world_orientation,
            physicsClientId=self.client_id,
        )
        pybullet.resetBaseVelocity(
            object_id,
            linearVelocity=(0.0, 0.0, 0.0),
            angularVelocity=(0.0, 0.0, 0.0),
            physicsClientId=self.client_id,
        )

    def _simulate(self, steps: int) -> None:
        for _ in range(steps):
            self._apply_gripper_target()
            if self.held_object_name is not None:
                self._update_held_object_pose()
            pybullet.stepSimulation(physicsClientId=self.client_id)

    def _get_body_position(self, body_id: int) -> Vec3:
        position, _ = pybullet.getBasePositionAndOrientation(body_id, physicsClientId=self.client_id)
        return tuple(position)

    def _clamp_position(self, target_position: Vec3) -> Vec3:
        return tuple(
            max(self.config.workspace_low[index], min(self.config.workspace_high[index], target_position[index]))
            for index in range(3)
        )

    def _maybe_attach_object(self, object_name: str) -> None:
        if self.held_object_name is not None:
            return

        end_effector_position, _ = self.get_end_effector_pose()
        object_position = self._get_body_position(self.object_ids[object_name])
        close_enough = math.dist(end_effector_position, object_position) <= 0.12
        low_gripper = self.get_gripper_open_ratio() <= 0.25
        aligned_height = end_effector_position[2] <= object_position[2] + 0.12
        if close_enough and low_gripper and aligned_height:
            self._attach_object(object_name)

    def _maybe_release_held_object(self) -> None:
        if self.held_object_name is None:
            return
        if self.get_gripper_open_ratio() >= 0.70:
            self._release_object(None)
            self._simulate(120)

    def _prepare_asset_root(self) -> Path:
        source_root = Path(pybullet_data.getDataPath())
        target_root = Path(tempfile.gettempdir()) / "pybullet_ascii_assets"
        target_root.mkdir(parents=True, exist_ok=True)

        panda_source = source_root / "franka_panda"
        panda_target = target_root / "franka_panda"
        if not panda_target.exists():
            shutil.copytree(panda_source, panda_target)

        return target_root


def create_pick_place_simulation(
    backend: str | None = None,
    **kwargs: Any,
) -> "PickPlaceSimulationProtocol":
    selected_backend = backend or os.environ.get("EMBODIED_SIM_BACKEND", "mujoco")
    normalized_backend = selected_backend.strip().lower().replace("-", "_")
    if normalized_backend in {"mujoco", "mjcf"}:
        from embodied_agent.mujoco_simulator import MujocoPickPlaceSimulation

        return MujocoPickPlaceSimulation(**kwargs)
    if normalized_backend in {"pybullet", "bullet", "legacy_pybullet"}:
        return BulletPickPlaceSimulation(**kwargs)
    if normalized_backend in {"isaaclab", "isaac_lab", "isaac"}:
        from embodied_agent.isaaclab_simulator import IsaacLabPickPlaceSimulation

        return IsaacLabPickPlaceSimulation(**kwargs)
    raise ValueError(f"Unsupported pick/place simulation backend: {selected_backend}")
