from __future__ import annotations

import math
import sys
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../embodied_agent/src')))
from embodied_agent.contracts import ContactState, ExecutorResult, FailureRecord
from embodied_agent.mujoco_simulator import MujocoPickPlaceSimulation


def _abs_import_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src')
    )


@dataclass
class SkillParams:
    target_object: str
    press_direction: str = "down"
    force: float = 0.5
    push_direction: str = "forward"
    distance: float = 0.10
    pull_direction: str = "backward"
    rotation_axis: str = "z"
    angle: float = 45.0


def _resolve_direction_vector(direction_name: str, distance: float = 1.0) -> tuple[float, float, float]:
    direction_map: dict[str, tuple[float, float, float]] = {
        "forward": (distance, 0.0, 0.0),
        "backward": (-distance, 0.0, 0.0),
        "left": (0.0, distance, 0.0),
        "right": (0.0, -distance, 0.0),
        "down": (0.0, 0.0, -distance),
        "up": (0.0, 0.0, distance),
        "forward_left": (distance, distance, 0.0),
        "forward_right": (distance, -distance, 0.0),
        "backward_left": (-distance, distance, 0.0),
        "backward_right": (-distance, -distance, 0.0),
    }
    return direction_map.get(direction_name, (distance, 0.0, 0.0))


class BaseScriptedSkill(ABC):
    """Base class for scripted/heuristic skills. RL versions will replace these later."""

    skill_name: str = "base"

    @abstractmethod
    def execute(self, simulation: MujocoPickPlaceSimulation, **kwargs: Any) -> ExecutorResult:
        ...

    def _get_object_position(self, simulation: MujocoPickPlaceSimulation, object_name: str) -> tuple[float, float, float]:
        if object_name not in simulation.object_ids:
            raise ValueError(f"Object '{object_name}' not found in simulation.")
        return tuple(simulation.get_object_position(object_name))

    def _build_result(
        self,
        success: bool,
        reward: float,
        final_state: dict[str, Any],
        contact_region: str | None,
        error_code: str | None = None,
        failure_history: list[FailureRecord] | None = None,
    ) -> ExecutorResult:
        return ExecutorResult(
            success=success,
            reward=reward,
            final_state=final_state,
            contact_state=ContactState(
                has_contact=contact_region is not None,
                contact_region=contact_region,
            ),
            error_code=error_code,
            rollout_path=f"mock://{self.skill_name}/scripted",
            failure_history=failure_history or [],
        )

    def _clamp_delta(self, target: tuple[float, float, float], current: tuple[float, float, float], max_step: float = 0.02) -> tuple[float, float, float]:
        total_dist = math.dist(target, current)
        if total_dist <= max_step:
            return target
        ratio = max_step / total_dist
        return tuple(current[i] + (target[i] - current[i]) * ratio for i in range(3))


class PressSkill(BaseScriptedSkill):
    """Press down on an object from above using heuristic control."""

    skill_name = "press"

    def execute(self, simulation: MujocoPickPlaceSimulation, **kwargs: Any) -> ExecutorResult:
        target_object = str(kwargs.get("target_object", "red_block"))
        press_direction = str(kwargs.get("press_direction", "down"))
        force = float(kwargs.get("force", 0.5))

        try:
            obj_pos = self._get_object_position(simulation, target_object)
            table_top_z = simulation.config.table_top_z
            hover_height = table_top_z + simulation.config.hover_height
            press_height = table_top_z + 0.06

            hover_pos = (obj_pos[0], obj_pos[1], hover_height)
            press_pos = (obj_pos[0], obj_pos[1], press_height)

            simulation.open_gripper()
            simulation.move_end_effector(hover_pos)
            simulation.move_end_effector(press_pos, steps=int(120 * force))

            simulation.move_end_effector(hover_pos)

            return self._build_result(
                success=True,
                reward=1.0,
                final_state={
                    "executed_skill": "press",
                    "target_object": target_object,
                    "press_direction": press_direction,
                    "force": force,
                    "object_position": list(obj_pos),
                },
                contact_region="top",
            )
        except Exception as exc:
            return self._build_result(
                success=False,
                reward=0.0,
                final_state={"executed_skill": "press", "target_object": target_object},
                contact_region=None,
                error_code=f"press_failed: {exc}",
                failure_history=[FailureRecord(
                    step_index=0, selected_skill="press",
                    failure_source="execution_error", reason=str(exc), replan_attempt=0,
                )],
            )


class PushSkill(BaseScriptedSkill):
    """Push an object in a direction using heuristic control."""

    skill_name = "push"

    def execute(self, simulation: MujocoPickPlaceSimulation, **kwargs: Any) -> ExecutorResult:
        target_object = str(kwargs.get("target_object", "red_block"))
        push_direction = str(kwargs.get("push_direction", "forward"))
        distance = float(kwargs.get("distance", 0.10))

        try:
            obj_pos = self._get_object_position(simulation, target_object)
            table_top_z = simulation.config.table_top_z
            approach_height = table_top_z + 0.09

            dir_vec = _resolve_direction_vector(push_direction, 0.05)
            approach_pos = (
                obj_pos[0] - dir_vec[0],
                obj_pos[1] - dir_vec[1],
                approach_height,
            )

            push_delta = _resolve_direction_vector(push_direction, distance)
            push_target = (
                obj_pos[0] + push_delta[0],
                obj_pos[1] + push_delta[1],
                approach_height,
            )

            simulation.open_gripper()
            simulation.move_end_effector(approach_pos)
            simulation.close_gripper()
            simulation.move_end_effector(push_target, steps=int(240 * distance / 0.10))
            simulation.open_gripper()

            return self._build_result(
                success=True,
                reward=1.0,
                final_state={
                    "executed_skill": "push",
                    "target_object": target_object,
                    "push_direction": push_direction,
                    "distance": distance,
                    "object_position": list(obj_pos),
                },
                contact_region="side",
            )
        except Exception as exc:
            return self._build_result(
                success=False,
                reward=0.0,
                final_state={"executed_skill": "push", "target_object": target_object},
                contact_region=None,
                error_code=f"push_failed: {exc}",
                failure_history=[FailureRecord(
                    step_index=0, selected_skill="push",
                    failure_source="execution_error", reason=str(exc), replan_attempt=0,
                )],
            )


class PullSkill(BaseScriptedSkill):
    """Pull an object towards the robot using heuristic control."""

    skill_name = "pull"

    def execute(self, simulation: MujocoPickPlaceSimulation, **kwargs: Any) -> ExecutorResult:
        target_object = str(kwargs.get("target_object", "red_block"))
        pull_direction = str(kwargs.get("pull_direction", "backward"))
        distance = float(kwargs.get("distance", 0.10))

        try:
            obj_pos = self._get_object_position(simulation, target_object)
            table_top_z = simulation.config.table_top_z

            pull_dir_vec = _resolve_direction_vector(pull_direction, 0.06)
            grasp_pos = (
                obj_pos[0] + pull_dir_vec[0],
                obj_pos[1] + pull_dir_vec[1],
                table_top_z + 0.08,
            )
            hover_pos = (
                obj_pos[0] + pull_dir_vec[0],
                obj_pos[1] + pull_dir_vec[1],
                table_top_z + simulation.config.hover_height,
            )

            pull_delta = _resolve_direction_vector(pull_direction, -distance)
            pull_target = (
                obj_pos[0] + pull_delta[0],
                obj_pos[1] + pull_delta[1],
                table_top_z + 0.10,
            )

            simulation.open_gripper()
            simulation.move_end_effector(hover_pos)
            simulation.move_end_effector(grasp_pos)
            simulation.close_gripper()
            simulation.move_end_effector(pull_target, steps=int(240 * distance / 0.10))
            simulation.open_gripper()

            return self._build_result(
                success=True,
                reward=1.0,
                final_state={
                    "executed_skill": "pull",
                    "target_object": target_object,
                    "pull_direction": pull_direction,
                    "distance": distance,
                    "object_position": list(obj_pos),
                },
                contact_region="side",
            )
        except Exception as exc:
            return self._build_result(
                success=False,
                reward=0.0,
                final_state={"executed_skill": "pull", "target_object": target_object},
                contact_region=None,
                error_code=f"pull_failed: {exc}",
                failure_history=[FailureRecord(
                    step_index=0, selected_skill="pull",
                    failure_source="execution_error", reason=str(exc), replan_attempt=0,
                )],
            )


class RotateSkill(BaseScriptedSkill):
    """Rotate an object by grasping and turning the end-effector."""

    skill_name = "rotate"

    def execute(self, simulation: MujocoPickPlaceSimulation, **kwargs: Any) -> ExecutorResult:
        target_object = str(kwargs.get("target_object", "red_block"))
        rotation_axis = str(kwargs.get("rotation_axis", "z"))
        angle = float(kwargs.get("angle", 45.0))

        try:
            obj_pos = self._get_object_position(simulation, target_object)
            table_top_z = simulation.config.table_top_z
            grasp_height = table_top_z + 0.08
            lift_height = table_top_z + 0.15

            simulation.open_gripper()
            simulation.move_end_effector((obj_pos[0], obj_pos[1], lift_height))
            simulation.move_end_effector((obj_pos[0], obj_pos[1], grasp_height))
            simulation.close_gripper()

            angle_rad = math.radians(angle)
            if rotation_axis == "z":
                rot_quat = simulation.get_quaternion_from_euler((0.0, 0.0, angle_rad))
            elif rotation_axis == "y":
                rot_quat = simulation.get_quaternion_from_euler((0.0, angle_rad, 0.0))
            elif rotation_axis == "x":
                rot_quat = simulation.get_quaternion_from_euler((angle_rad, 0.0, 0.0))
            else:
                rot_quat = simulation.get_quaternion_from_euler((0.0, 0.0, angle_rad))

            ee_pos, _ = simulation.get_end_effector_pose()
            simulation.teleport_end_effector_pose((ee_pos[0], ee_pos[1], lift_height), rot_quat)
            simulation._simulate(120)

            simulation.open_gripper()
            simulation.move_end_effector((obj_pos[0], obj_pos[1], lift_height))

            return self._build_result(
                success=True,
                reward=1.0,
                final_state={
                    "executed_skill": "rotate",
                    "target_object": target_object,
                    "rotation_axis": rotation_axis,
                    "angle": angle,
                    "object_position": list(obj_pos),
                },
                contact_region="top",
            )
        except Exception as exc:
            return self._build_result(
                success=False,
                reward=0.0,
                final_state={"executed_skill": "rotate", "target_object": target_object},
                contact_region=None,
                error_code=f"rotate_failed: {exc}",
                failure_history=[FailureRecord(
                    step_index=0, selected_skill="rotate",
                    failure_source="execution_error", reason=str(exc), replan_attempt=0,
                )],
            )
