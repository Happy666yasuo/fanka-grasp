from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from embodied_agent.failure_analysis import classify_pick_failure, classify_place_failure
from embodied_agent.rl_support import SkillPolicySpec, load_policy_model, policy_spec_from_dict
from embodied_agent.simulation_protocol import PickPlaceSimulationProtocol
from embodied_agent.types import PlanStep, PostCondition, WorldState


def _stabilize_pick_action(
    skill_name: str,
    holding_object: bool,
    delta_position: tuple[float, float, float],
    gripper_command: float,
) -> tuple[tuple[float, float, float], float, bool]:
    if skill_name != "pick" or not holding_object:
        return delta_position, gripper_command, False

    stabilized_delta_position = (
        delta_position[0],
        delta_position[1],
        max(delta_position[2], 0.0),
    )
    stabilized_gripper_command = min(gripper_command, -0.8)
    return stabilized_delta_position, stabilized_gripper_command, True


@dataclass
class LearnedSkillPolicy:
    spec: SkillPolicySpec

    def __post_init__(self) -> None:
        self.model = load_policy_model(self.spec.algorithm, self.spec.model_path)
        self.last_debug: dict[str, Any] = {}

    def execute(self, simulation: PickPlaceSimulationProtocol, object_name: str, zone_name: str, gui_delay: float = 0.0) -> bool:
        initial_metrics = simulation.get_skill_metrics(object_name, zone_name)
        initial_state = simulation.observe_scene()
        debug: dict[str, Any] = {
            "skill_name": self.spec.skill_name,
            "controller": "learned",
            "success": False,
            "steps_executed": 0,
            "max_steps": self.spec.max_steps,
            "initial_metrics": dict(initial_metrics),
            "final_metrics": dict(initial_metrics),
            "initial_object_position": list(initial_state.object_positions.get(object_name, (0.0, 0.0, 0.0))),
            "initial_zone_position": list(initial_state.zone_positions.get(zone_name, (0.0, 0.0, 0.0))),
            "initial_end_effector_position": list(initial_state.end_effector_position),
            "final_object_position": list(initial_state.object_positions.get(object_name, (0.0, 0.0, 0.0))),
            "final_end_effector_position": list(initial_state.end_effector_position),
            "min_ee_object_distance": float(initial_metrics.get("ee_object_distance", float("inf"))),
            "min_object_zone_distance_xy": float(initial_metrics.get("object_zone_distance_xy", float("inf"))),
            "max_lift_progress": float(initial_metrics.get("lift_progress", 0.0)),
            "min_gripper_open_ratio": float(initial_metrics.get("gripper_open_ratio", 1.0)),
            "max_gripper_open_ratio": float(initial_metrics.get("gripper_open_ratio", 0.0)),
            "ever_holding": float(initial_metrics.get("holding_flag", 0.0)) >= 0.5,
            "ever_release": False,
            "pick_stabilization_steps": 0,
            "failure_category": None,
        }

        for _ in range(self.spec.max_steps):
            observation = simulation.get_skill_observation(object_name, zone_name)
            action, _ = self.model.predict(observation, deterministic=self.spec.deterministic)
            flattened_action = np.asarray(action, dtype=np.float32).reshape(-1)
            if flattened_action.shape[0] != 4:
                raise ValueError("A learned skill policy must output four action values.")
            delta_position = tuple(
                float(flattened_action[index]) * self.spec.action_scale for index in range(3)
            )
            previous_holding = simulation.held_object_name == object_name
            delta_position, gripper_command, stabilized = _stabilize_pick_action(
                self.spec.skill_name,
                previous_holding,
                delta_position,
                float(flattened_action[3]),
            )
            if stabilized:
                debug["pick_stabilization_steps"] = int(debug["pick_stabilization_steps"]) + 1
            simulation.apply_skill_action(
                delta_position=delta_position,
                gripper_command=gripper_command,
                action_steps=self.spec.action_repeat,
                object_name=object_name,
            )
            metrics = simulation.get_skill_metrics(object_name, zone_name)
            state = simulation.observe_scene()
            debug["steps_executed"] = int(debug["steps_executed"]) + 1
            debug["final_metrics"] = dict(metrics)
            debug["final_object_position"] = list(state.object_positions.get(object_name, (0.0, 0.0, 0.0)))
            debug["final_end_effector_position"] = list(state.end_effector_position)
            debug["min_ee_object_distance"] = min(float(debug["min_ee_object_distance"]), float(metrics["ee_object_distance"]))
            debug["min_object_zone_distance_xy"] = min(
                float(debug["min_object_zone_distance_xy"]),
                float(metrics["object_zone_distance_xy"]),
            )
            debug["max_lift_progress"] = max(float(debug["max_lift_progress"]), float(metrics["lift_progress"]))
            debug["min_gripper_open_ratio"] = min(
                float(debug["min_gripper_open_ratio"]),
                float(metrics["gripper_open_ratio"]),
            )
            debug["max_gripper_open_ratio"] = max(
                float(debug["max_gripper_open_ratio"]),
                float(metrics["gripper_open_ratio"]),
            )
            debug["ever_holding"] = bool(debug["ever_holding"]) or float(metrics["holding_flag"]) >= 0.5
            debug["ever_release"] = bool(debug["ever_release"]) or (previous_holding and float(metrics["holding_flag"]) < 0.5)
            if gui_delay > 0:
                time.sleep(gui_delay)
            if self.spec.skill_name == "pick" and simulation.is_pick_success(object_name):
                debug["success"] = True
                self.last_debug = debug
                return True
            if self.spec.skill_name == "place" and simulation.is_place_success(object_name, zone_name):
                debug["success"] = True
                self.last_debug = debug
                return True

        if self.spec.skill_name == "pick":
            debug["failure_category"] = classify_pick_failure(debug)
        elif self.spec.skill_name == "place":
            debug["failure_category"] = classify_place_failure(debug)
        self.last_debug = debug
        return False


def build_learned_skill_policies(
    configs: dict[str, SkillPolicySpec | dict[str, object]] | None,
    base_dir: Path | None = None,
) -> dict[str, LearnedSkillPolicy]:
    if not configs:
        return {}

    learned_policies: dict[str, LearnedSkillPolicy] = {}
    for skill_name, spec_or_dict in configs.items():
        if isinstance(spec_or_dict, SkillPolicySpec):
            spec = spec_or_dict
        else:
            spec = policy_spec_from_dict(skill_name, spec_or_dict, base_dir=base_dir)
        learned_policies[skill_name] = LearnedSkillPolicy(spec)
    return learned_policies


class SkillLibrary:
    def __init__(
        self,
        simulation: PickPlaceSimulationProtocol,
        learned_skill_policies: dict[str, LearnedSkillPolicy] | None = None,
        fallback_to_scripted: bool = False,
        gui_delay: float = 0.0,
    ) -> None:
        self.simulation = simulation
        self.current_instruction = ""
        self.learned_skill_policies = learned_skill_policies or {}
        self.fallback_to_scripted = fallback_to_scripted
        self.gui_delay = gui_delay
        self.last_pick_controller: str | None = None
        self.skill_debug_log: list[dict[str, Any]] = []
        self.post_pick_state_log: list[dict[str, Any]] = []
        self.place_entry_state_log: list[dict[str, Any]] = []

    def observe_scene(self, instruction: str = "") -> WorldState:
        if instruction:
            self.current_instruction = instruction
        return self.simulation.observe_scene(self.current_instruction)

    def execute(self, step: PlanStep) -> str:
        if step.action == "observe":
            self.observe_scene(self.current_instruction)
            return "observe"
        if step.action == "pick":
            if step.target is None:
                raise ValueError("The pick step requires an object target.")
            zone_name = self._resolve_pick_zone_name(step)
            self._execute_pick(step.target, zone_name)
            return f"pick:{step.target}"
        if step.action == "place":
            if step.target is None:
                raise ValueError("The place step requires a zone target.")
            object_name = self._resolve_place_object_name(step)
            self._execute_place(object_name=object_name, zone_name=step.target)
            return f"place:{step.target}"
        if step.action == "reset_home":
            self.simulation.move_home()
            return "reset_home"
        raise ValueError(f"Unsupported plan step: {step.action}")

    def verify(self, object_name: str, zone_name: str) -> bool:
        return self.simulation.is_place_success(object_name, zone_name)

    def check_post_condition(self, post_condition: PostCondition) -> bool:
        if post_condition.kind == "holding":
            object_name = post_condition.object_name or "red_block"
            return self.simulation.is_object_held(object_name)

        if post_condition.kind == "placed":
            object_name = post_condition.object_name or "red_block"
            zone_name = post_condition.zone_name or "green_zone"
            return self.simulation.is_place_success(object_name, zone_name)

        raise ValueError(f"Unsupported post-condition: {post_condition.kind}")

    def shutdown(self) -> None:
        self.simulation.shutdown()

    def debug_snapshot(self) -> dict[str, Any]:
        return {
            "last_pick_controller": self.last_pick_controller,
            "skills": [dict(skill_debug) for skill_debug in self.skill_debug_log],
            "runtime_alignment": {
                "post_pick_states": [dict(state) for state in self.post_pick_state_log],
                "place_entry_states": [dict(state) for state in self.place_entry_state_log],
            },
        }

    def _policy_debug_info(self, learned_policy: LearnedSkillPolicy, success: bool) -> dict[str, Any]:
        raw_debug = getattr(learned_policy, "last_debug", None)
        if isinstance(raw_debug, dict) and raw_debug:
            return dict(raw_debug)
        return {
            "skill_name": learned_policy.spec.skill_name,
            "controller": "learned",
            "success": success,
        }

    def _record_runtime_alignment_state(
        self,
        target_log: list[dict[str, Any]],
        capture_stage: str,
        object_name: str,
        zone_name: str,
        place_controller: str | None = None,
        use_staging: bool | None = None,
    ) -> None:
        if not hasattr(self.simulation, "capture_skill_state"):
            return

        raw_state = self.simulation.capture_skill_state(object_name, zone_name)
        state_record = {
            "capture_index": len(target_log) + 1,
            "capture_stage": capture_stage,
            "pick_controller": self.last_pick_controller,
            "place_controller": place_controller,
            "use_staging": use_staging,
            **raw_state,
        }
        target_log.append(state_record)

    def _resolve_pick_zone_name(self, step: PlanStep) -> str:
        zone_name = step.parameters.get("zone")
        if isinstance(zone_name, str):
            return zone_name

        available_zones = getattr(self.simulation, "zone_positions", None)
        if isinstance(available_zones, dict) and len(available_zones) == 1:
            return next(iter(available_zones.keys()))

        raise ValueError("The pick step requires an explicit zone target when multiple zones are active.")

    def _resolve_place_object_name(self, step: PlanStep) -> str:
        object_name = step.parameters.get("object")
        if isinstance(object_name, str):
            return object_name

        available_objects = getattr(self.simulation, "object_ids", None)
        if isinstance(available_objects, dict) and len(available_objects) == 1:
            return next(iter(available_objects.keys()))

        raise ValueError("The place step requires an explicit object target when multiple objects are active.")

    def _execute_pick(self, object_name: str, zone_name: str) -> None:
        learned_policy = self.learned_skill_policies.get("pick")
        if learned_policy is None:
            self.simulation.pick_object(object_name)
            self.last_pick_controller = "scripted"
            self.skill_debug_log.append(
                {
                    "skill_name": "pick",
                    "controller": "scripted",
                    "success": True,
                }
            )
            self._record_runtime_alignment_state(
                self.post_pick_state_log,
                capture_stage="post_pick_success",
                object_name=object_name,
                zone_name=zone_name,
            )
            return

        self._prepare_learned_skill_entry(learned_policy, object_name=object_name, zone_name="green_zone")

        success = learned_policy.execute(
            simulation=self.simulation,
            object_name=object_name,
            zone_name=zone_name,
            gui_delay=self.gui_delay,
        )
        self.skill_debug_log.append(self._policy_debug_info(learned_policy, success))
        if not success:
            if self.fallback_to_scripted:
                self.simulation.pick_object(object_name)
                self.last_pick_controller = "scripted"
                self._record_runtime_alignment_state(
                    self.post_pick_state_log,
                    capture_stage="post_pick_success",
                    object_name=object_name,
                    zone_name=zone_name,
                )
                return
            raise RuntimeError("The learned pick policy did not finish successfully.")
        self.last_pick_controller = "learned"
        self._record_runtime_alignment_state(
            self.post_pick_state_log,
            capture_stage="post_pick_success",
            object_name=object_name,
            zone_name=zone_name,
        )

    def _execute_place(self, object_name: str, zone_name: str) -> None:
        learned_policy = self.learned_skill_policies.get("place")
        place_controller = "learned" if learned_policy is not None else "scripted"
        place_use_staging = learned_policy.spec.use_staging if learned_policy is not None else False
        self._record_runtime_alignment_state(
            self.place_entry_state_log,
            capture_stage="pre_place_runtime_raw",
            object_name=object_name,
            zone_name=zone_name,
            place_controller=place_controller,
            use_staging=place_use_staging,
        )
        if learned_policy is None:
            self.simulation.place_object(zone_name)
            self.skill_debug_log.append(
                {
                    "skill_name": "place",
                    "controller": "scripted",
                    "success": True,
                }
            )
            return

        self._prepare_learned_skill_entry(learned_policy, object_name=object_name, zone_name=zone_name)
        self._record_runtime_alignment_state(
            self.place_entry_state_log,
            capture_stage="pre_place_policy_entry",
            object_name=object_name,
            zone_name=zone_name,
            place_controller="learned",
            use_staging=learned_policy.spec.use_staging,
        )

        success = learned_policy.execute(
            simulation=self.simulation,
            object_name=object_name,
            zone_name=zone_name,
            gui_delay=self.gui_delay,
        )
        self.skill_debug_log.append(self._policy_debug_info(learned_policy, success))
        if not success:
            if self.fallback_to_scripted:
                self.simulation.place_object(zone_name)
                return
            raise RuntimeError("The learned place policy did not finish successfully.")

    def _prepare_learned_skill_entry(
        self,
        learned_policy: LearnedSkillPolicy,
        object_name: str,
        zone_name: str,
    ) -> None:
        if not learned_policy.spec.use_staging:
            return

        if learned_policy.spec.skill_name == "pick":
            self.simulation.prepare_pick_staging_pose(object_name)
            return

        if learned_policy.spec.skill_name == "place":
            if self.last_pick_controller == "scripted" and self.simulation.held_object_name == object_name:
                self.simulation.normalize_held_object_for_place(object_name)
            self.simulation.prepare_place_staging_pose(zone_name)
            if self.last_pick_controller == "scripted" and self.simulation.held_object_name == object_name:
                self.simulation.canonicalize_held_object_for_place(object_name, zone_name)
