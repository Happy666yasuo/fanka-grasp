from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from embodied_agent.simulation_protocol import PickPlaceSimulationProtocol
from embodied_agent.simulator import create_pick_place_simulation


def _normalize_optional_range(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Layout ranges must be two-element lists or tuples.")
    low = float(value[0])
    high = float(value[1])
    if low > high:
        low, high = high, low
    return (low, high)


def _normalize_optional_candidates(value: object) -> tuple[tuple[float, float], ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("Layout candidates must be a list or tuple of 2D positions.")
    normalized: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each layout candidate must be a two-element list or tuple.")
        normalized.append((float(item[0]), float(item[1])))
    return tuple(normalized)


def _normalize_reset_source(value: object) -> str:
    normalized = str(value or "default").strip().lower().replace("-", "_")
    if normalized not in {"default", "post_pick_states"}:
        raise ValueError(f"Unsupported reset_source: {value}")
    return normalized


def _resolve_optional_path(value: object, base_dir: Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


@dataclass(frozen=True)
class SkillTask:
    task_id: str
    instruction: str
    object_name: str
    zone_name: str

    def to_dict(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "object_name": self.object_name,
            "zone_name": self.zone_name,
        }


def _normalize_task_pool(
    value: object,
    *,
    default_object_name: str,
    default_zone_name: str,
) -> tuple[SkillTask, ...]:
    if value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
        return (
            SkillTask(
                task_id="task_0",
                instruction="",
                object_name=default_object_name,
                zone_name=default_zone_name,
            ),
        )

    if not isinstance(value, (list, tuple)):
        raise ValueError("task_pool must be a list or tuple of task mappings.")

    normalized: list[SkillTask] = []
    for index, raw_task in enumerate(value):
        if isinstance(raw_task, SkillTask):
            normalized.append(raw_task)
            continue
        if not isinstance(raw_task, dict):
            raise ValueError("Each task_pool entry must be a mapping.")

        object_name = str(raw_task.get("object_name", "")).strip()
        zone_name = str(raw_task.get("zone_name", "")).strip()
        if not object_name or not zone_name:
            raise ValueError("Each task_pool entry must define object_name and zone_name.")

        normalized.append(
            SkillTask(
                task_id=str(raw_task.get("task_id", raw_task.get("name", f"task_{index}"))),
                instruction=str(raw_task.get("instruction", "")).strip(),
                object_name=object_name,
                zone_name=zone_name,
            )
        )

    if not normalized:
        raise ValueError("task_pool must contain at least one task entry.")
    return tuple(normalized)


@dataclass(frozen=True)
class SkillEnvSettings:
    skill_name: str
    object_name: str = "red_block"
    zone_name: str = "green_zone"
    max_steps: int = 24
    action_scale: float = 0.02
    action_repeat: int = 24
    randomize: bool = True
    randomize_probability: float | None = None
    seed: int = 7
    use_staging: bool = True
    post_release_settle_steps: int = 0
    task_pool: tuple[SkillTask, ...] = ()
    object_x_range: tuple[float, float] | None = None
    object_y_range: tuple[float, float] | None = None
    zone_x_range: tuple[float, float] | None = None
    zone_y_range: tuple[float, float] | None = None
    object_candidates: tuple[tuple[float, float], ...] | None = None
    zone_candidates: tuple[tuple[float, float], ...] | None = None
    reset_source: str = "default"
    post_pick_states_jsonl: Path | None = None
    reset_state_limit: int | None = None

    def __post_init__(self) -> None:
        probability = self.randomize_probability
        if probability is None:
            probability = 1.0 if self.randomize else 0.0
        reset_source = _normalize_reset_source(self.reset_source)
        if reset_source == "post_pick_states" and self.skill_name != "place":
            raise ValueError("post_pick_states reset_source is only supported for place envs.")
        resolved_post_pick_states_jsonl = _resolve_optional_path(self.post_pick_states_jsonl)
        normalized_task_pool = _normalize_task_pool(
            self.task_pool,
            default_object_name=self.object_name,
            default_zone_name=self.zone_name,
        )
        if reset_source == "post_pick_states" and resolved_post_pick_states_jsonl is None:
            raise ValueError("post_pick_states reset_source requires post_pick_states_jsonl.")
        object.__setattr__(self, "randomize_probability", float(np.clip(probability, 0.0, 1.0)))
        object.__setattr__(self, "post_release_settle_steps", max(0, int(self.post_release_settle_steps)))
        object.__setattr__(self, "task_pool", normalized_task_pool)
        object.__setattr__(self, "object_x_range", _normalize_optional_range(self.object_x_range))
        object.__setattr__(self, "object_y_range", _normalize_optional_range(self.object_y_range))
        object.__setattr__(self, "zone_x_range", _normalize_optional_range(self.zone_x_range))
        object.__setattr__(self, "zone_y_range", _normalize_optional_range(self.zone_y_range))
        object.__setattr__(self, "object_candidates", _normalize_optional_candidates(self.object_candidates))
        object.__setattr__(self, "zone_candidates", _normalize_optional_candidates(self.zone_candidates))
        object.__setattr__(self, "reset_source", reset_source)
        object.__setattr__(self, "post_pick_states_jsonl", resolved_post_pick_states_jsonl)
        object.__setattr__(
            self,
            "reset_state_limit",
            None if self.reset_state_limit is None else max(1, int(self.reset_state_limit)),
        )

    @property
    def active_object_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(task.object_name for task in self.task_pool))

    @property
    def active_zone_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(task.zone_name for task in self.task_pool))

    @classmethod
    def from_config(
        cls,
        skill_name: str,
        env_config: dict[str, Any] | None = None,
        base_dir: Path | None = None,
    ) -> "SkillEnvSettings":
        config = env_config or {}
        return cls(
            skill_name=skill_name,
            object_name=str(config.get("object_name", "red_block")),
            zone_name=str(config.get("zone_name", "green_zone")),
            max_steps=int(config.get("max_steps", 48)),
            action_scale=float(config.get("action_scale", 0.04)),
            action_repeat=int(config.get("action_repeat", 24)),
            randomize=bool(config.get("randomize", True)),
            randomize_probability=(
                float(config["randomize_probability"]) if "randomize_probability" in config else None
            ),
            seed=int(config.get("seed", 7)),
            use_staging=bool(config.get("use_staging", True)),
            post_release_settle_steps=int(
                config.get("post_release_settle_steps", 3 if skill_name == "place" else 0)
            ),
            task_pool=config.get("task_pool"),
            object_x_range=config.get("object_x_range"),
            object_y_range=config.get("object_y_range"),
            zone_x_range=config.get("zone_x_range"),
            zone_y_range=config.get("zone_y_range"),
            object_candidates=config.get("object_candidates"),
            zone_candidates=config.get("zone_candidates"),
            reset_source=config.get("reset_source", "default"),
            post_pick_states_jsonl=_resolve_optional_path(
                config.get("post_pick_states_jsonl"),
                base_dir=base_dir,
            ),
            reset_state_limit=(
                int(config["reset_state_limit"]) if "reset_state_limit" in config else None
            ),
        )


class PickPlaceSkillEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        settings: SkillEnvSettings,
        gui: bool = False,
        simulation: PickPlaceSimulationProtocol | None = None,
    ) -> None:
        self.settings = settings
        self.owns_simulation = simulation is None
        self.simulation = simulation or create_pick_place_simulation(
            backend="mujoco",
            gui=gui,
            object_names=self.settings.active_object_names,
            zone_names=self.settings.active_zone_names,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(17,), dtype=np.float32)
        self.episode_steps = 0
        self.rng = random.Random(self.settings.seed)
        self.randomize_probability = float(self.settings.randomize_probability)
        self.current_randomize_layout = False
        self.current_task = self.settings.task_pool[0]
        self.release_settling_steps_remaining = 0
        self.last_reset_source = "default"
        self.last_runtime_reset_metadata: dict[str, Any] | None = None
        self.runtime_reset_samples = self._load_runtime_reset_samples()
        self.runtime_reset_samples_by_task = self._group_runtime_reset_samples(self.runtime_reset_samples)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)

        options = options or {}
        if self.settings.reset_source == "post_pick_states":
            selected_task = self._select_task(options)
            state = self._sample_runtime_reset_state(selected_task)
            runtime_task = self._resolve_task_from_state(state)
            self.current_task = runtime_task
            self.current_randomize_layout = False
            self.simulation.restore_runtime_state(
                state,
                object_name=runtime_task.object_name,
                zone_name=runtime_task.zone_name,
            )
            self.last_reset_source = "post_pick_states"
            self.last_runtime_reset_metadata = {
                "task_id": runtime_task.task_id,
                "instruction": runtime_task.instruction,
                "object_name": runtime_task.object_name,
                "zone_name": runtime_task.zone_name,
                "episode_index": state.get("episode_index", state.get("episode")),
                "capture_stage": state.get("capture_stage"),
            }
        else:
            self.current_task = self._select_task(options)
            if "randomize" in options:
                randomize = bool(options["randomize"])
            else:
                randomize = self.rng.random() < self.randomize_probability
            self.current_randomize_layout = randomize
            if randomize:
                object_layout, zone_layout = self.simulation.sample_scene_layout(
                    rng=self.rng,
                    object_names=(self.current_task.object_name,),
                    zone_names=(self.current_task.zone_name,),
                    object_x_range=self.settings.object_x_range,
                    object_y_range=self.settings.object_y_range,
                    zone_x_range=self.settings.zone_x_range,
                    zone_y_range=self.settings.zone_y_range,
                    object_candidates=self.settings.object_candidates,
                    zone_candidates=self.settings.zone_candidates,
                )
            else:
                object_layout = {
                    self.current_task.object_name: self._fixed_object_xy(self.current_task.object_name)
                }
                zone_layout = {
                    self.current_task.zone_name: self._fixed_zone_xy(self.current_task.zone_name)
                }

            self.simulation.reset_task(
                object_layout=object_layout,
                zone_layout=zone_layout,
                holding_object=False,
                held_object_name=(
                    self.current_task.object_name if self.settings.skill_name == "place" else None
                ),
            )
            self.last_reset_source = "default"
            self.last_runtime_reset_metadata = None
        if self.settings.use_staging:
            if self.settings.skill_name == "pick":
                self.simulation.prepare_pick_staging_pose(self.current_task.object_name)
            else:
                self.simulation.prepare_place_staging_pose(self.current_task.zone_name)
        self.episode_steps = 0
        self.release_settling_steps_remaining = 0
        observation = self._get_observation()
        return observation, self._build_info(success=False)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._is_place_release_settling():
            return self._step_place_release_settling()

        previous_holding = self.simulation.held_object_name == self.current_task.object_name
        clipped_action = np.asarray(action, dtype=np.float32).reshape(-1)
        delta_position = tuple(
            float(clipped_action[index]) * self.settings.action_scale for index in range(3)
        )
        self.simulation.apply_skill_action(
            delta_position=delta_position,
            gripper_command=float(clipped_action[3]),
            action_steps=self.settings.action_repeat,
            object_name=self.current_task.object_name,
        )
        self.episode_steps += 1
        observation = self._get_observation()
        metrics = self.simulation.get_skill_metrics(
            object_name=self.current_task.object_name,
            zone_name=self.current_task.zone_name,
        )
        success = self._is_success()
        release_event = (
            self.settings.skill_name == "place"
            and previous_holding
            and metrics["holding_flag"] < 0.5
        )

        if release_event and self.settings.post_release_settle_steps > 0:
            self.release_settling_steps_remaining = self.settings.post_release_settle_steps
            reward = self._compute_reward(
                metrics,
                success=False,
                release_event=True,
                release_resolved=False,
            )
            terminated = False
            truncated = False
            return observation, reward, terminated, truncated, self._build_info(
                success=False,
                metrics=metrics,
                release_event=True,
                release_resolved=False,
                settling_active=True,
                settling_steps_remaining=self.release_settling_steps_remaining,
            )

        reward = self._compute_reward(
            metrics,
            success,
            release_event,
            release_resolved=release_event,
        )
        terminated = success or (self.settings.skill_name == "place" and release_event)
        truncated = not terminated and self.episode_steps >= self.settings.max_steps
        if truncated and self.settings.skill_name == "place" and metrics["holding_flag"] > 0.5:
            reward -= self._compute_place_timeout_penalty(metrics)
        return observation, reward, terminated, truncated, self._build_info(
            success,
            metrics,
            release_event,
            release_resolved=release_event,
        )

    def close(self) -> None:
        if self.owns_simulation:
            self.simulation.shutdown()

    def set_randomize_probability(self, probability: float) -> None:
        self.randomize_probability = float(np.clip(probability, 0.0, 1.0))

    def get_randomize_probability(self) -> float:
        return self.randomize_probability

    def _get_observation(self) -> np.ndarray:
        return self.simulation.get_skill_observation(
            object_name=self.current_task.object_name,
            zone_name=self.current_task.zone_name,
        )

    def _fixed_object_xy(self, object_name: str) -> tuple[float, float]:
        object_start_xy_for = getattr(self.simulation.config, "object_start_xy_for", None)
        if callable(object_start_xy_for):
            return tuple(object_start_xy_for(object_name))
        return tuple(getattr(self.simulation.config, "cube_start_xy"))

    def _fixed_zone_xy(self, zone_name: str) -> tuple[float, float]:
        zone_center_xy_for = getattr(self.simulation.config, "zone_center_xy_for", None)
        if callable(zone_center_xy_for):
            return tuple(zone_center_xy_for(zone_name))
        return tuple(getattr(self.simulation.config, "goal_center_xy"))

    def _is_place_release_settling(self) -> bool:
        return self.settings.skill_name == "place" and self.release_settling_steps_remaining > 0

    def _step_place_release_settling(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.simulation.simulate_steps(self.settings.action_repeat)
        observation = self._get_observation()
        metrics = self.simulation.get_skill_metrics(
            object_name=self.current_task.object_name,
            zone_name=self.current_task.zone_name,
        )
        self.release_settling_steps_remaining -= 1
        settling_complete = self.release_settling_steps_remaining == 0
        success = self._is_success() if settling_complete else False
        reward = self._compute_reward(
            metrics,
            success,
            release_event=settling_complete,
            release_resolved=settling_complete,
        )
        return observation, reward, settling_complete, False, self._build_info(
            success=success,
            metrics=metrics,
            release_event=False,
            release_resolved=settling_complete,
            settling_active=not settling_complete,
            settling_steps_remaining=self.release_settling_steps_remaining,
        )

    def _compute_reward(
        self,
        metrics: dict[str, float],
        success: bool,
        release_event: bool = False,
        release_resolved: bool = False,
    ) -> float:
        if self.settings.skill_name == "pick":
            return self._compute_pick_reward(metrics, success)

        return self._compute_place_reward(metrics, success, release_event, release_resolved)

    def _compute_pick_reward(self, metrics: dict[str, float], success: bool) -> float:
        distance = metrics["ee_object_distance"]
        holding_flag = metrics["holding_flag"]
        lift_progress = metrics["lift_progress"]
        gripper_open_ratio = metrics["gripper_open_ratio"]

        if holding_flag > 0.5:
            reward = 2.0 + 100.0 * lift_progress
            if success:
                reward += 200.0
            return float(reward)

        reward = -0.5
        reward += 0.5 * (1.0 - np.tanh(5.0 * distance))
        if distance < 0.10:
            reward += 1.0 * (1.0 - gripper_open_ratio)
        return float(reward)

    def _compute_place_reward(
        self,
        metrics: dict[str, float],
        success: bool,
        release_event: bool,
        release_resolved: bool,
    ) -> float:
        zone_distance = metrics["object_zone_distance_xy"]
        object_height = metrics["object_height"]
        holding_flag = metrics["holding_flag"]
        gripper_open_ratio = metrics["gripper_open_ratio"]
        placed_height = self.simulation.config.placed_object_z
        height_error = abs(object_height - placed_height)

        aligned_score = max(0.0, 1.0 - zone_distance / 0.12)
        height_score = max(0.0, 1.0 - height_error / 0.04)
        placement_xy_score = max(0.0, 1.0 - zone_distance / 0.08)
        placement_height_score = max(0.0, 1.0 - height_error / 0.05)
        placement_score = placement_xy_score * placement_height_score
        release_ready_score = aligned_score * height_score
        delay_progress = min(1.0, self.episode_steps / max(float(self.settings.max_steps), 1.0))

        reward = -1.0 - 2.5 * zone_distance - 3.0 * height_error
        if holding_flag > 0.5:
            reward += 1.5 * aligned_score
            reward += 1.0 * height_score
            reward += 2.0 * placement_score
            reward += 10.0 * release_ready_score * gripper_open_ratio
            reward -= 8.0 * release_ready_score * (1.0 - gripper_open_ratio)
            reward -= 4.0 * release_ready_score * delay_progress
        else:
            reward += 8.0 * aligned_score
            reward += 6.0 * height_score
            reward += 8.0 * placement_score
            if release_event:
                reward += 8.0 * release_ready_score
            if release_resolved and not success:
                reward -= 8.0 + 6.0 * zone_distance + 6.0 * height_error

        if success:
            reward += 30.0
        return float(reward)

    def _compute_place_timeout_penalty(self, metrics: dict[str, float]) -> float:
        zone_distance = metrics["object_zone_distance_xy"]
        object_height = metrics["object_height"]
        placed_height = self.simulation.config.placed_object_z
        height_error = abs(object_height - placed_height)
        aligned_score = max(0.0, 1.0 - zone_distance / 0.12)
        height_score = max(0.0, 1.0 - height_error / 0.04)
        release_ready_score = aligned_score * height_score
        return float(6.0 + 6.0 * release_ready_score)

    def _is_success(self) -> bool:
        if self.settings.skill_name == "pick":
            return self.simulation.is_pick_success(self.current_task.object_name)
        return self.simulation.is_place_success(
            self.current_task.object_name,
            self.current_task.zone_name,
        )

    def _build_info(
        self,
        success: bool,
        metrics: dict[str, float] | None = None,
        release_event: bool = False,
        release_resolved: bool = False,
        settling_active: bool = False,
        settling_steps_remaining: int = 0,
    ) -> dict[str, Any]:
        metrics = metrics or self.simulation.get_skill_metrics(
            object_name=self.current_task.object_name,
            zone_name=self.current_task.zone_name,
        )
        return {
            "skill_name": self.settings.skill_name,
            "task_id": self.current_task.task_id,
            "instruction": self.current_task.instruction,
            "object_name": self.current_task.object_name,
            "zone_name": self.current_task.zone_name,
            "success": success,
            "is_success": success,
            "release_event": release_event,
            "release_resolved": release_resolved,
            "released_failure": release_resolved and not success,
            "settling_active": settling_active,
            "settling_steps_remaining": settling_steps_remaining,
            "reset_source": self.last_reset_source,
            "runtime_reset_metadata": self.last_runtime_reset_metadata,
            "randomize_probability": self.randomize_probability,
            "randomize_layout": self.current_randomize_layout,
            **metrics,
        }

    def _load_runtime_reset_samples(self) -> list[dict[str, Any]]:
        if self.settings.reset_source != "post_pick_states":
            return []
        assert self.settings.post_pick_states_jsonl is not None
        samples: list[dict[str, Any]] = []
        allowed_pairs = {
            (task.object_name, task.zone_name): task
            for task in self.settings.task_pool
        }
        with self.settings.post_pick_states_jsonl.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if str(payload.get("capture_stage")) != "post_pick_success":
                    continue
                object_name = str(payload.get("object_name", "")).strip()
                zone_name = str(payload.get("zone_name", "")).strip()
                if (object_name, zone_name) not in allowed_pairs:
                    continue
                if not bool(payload.get("holding_target_object", False)):
                    continue
                samples.append(payload)
                if (
                    self.settings.reset_state_limit is not None
                    and len(samples) >= self.settings.reset_state_limit
                ):
                    break
        missing_tasks = [
            task.task_id
            for task in self.settings.task_pool
            if not any(
                str(sample.get("object_name", "")).strip() == task.object_name
                and str(sample.get("zone_name", "")).strip() == task.zone_name
                for sample in samples
            )
        ]
        if not samples:
            raise ValueError(
                "No matching post-pick runtime reset samples found in "
                f"{self.settings.post_pick_states_jsonl}."
            )
        if missing_tasks:
            raise ValueError(
                "Missing post-pick runtime reset samples for task_pool entries: "
                + ", ".join(missing_tasks)
            )
        return samples

    def _group_runtime_reset_samples(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for sample in samples:
            key = (
                str(sample.get("object_name", "")).strip(),
                str(sample.get("zone_name", "")).strip(),
            )
            grouped.setdefault(key, []).append(sample)
        return grouped

    def _sample_runtime_reset_state(self, task: SkillTask | None = None) -> dict[str, Any]:
        if not self.runtime_reset_samples:
            raise RuntimeError("Runtime reset dataset was not loaded.")
        selected_task = task or self._select_task({})
        task_key = (selected_task.object_name, selected_task.zone_name)
        task_samples = self.runtime_reset_samples_by_task.get(task_key, [])
        if not task_samples:
            raise RuntimeError(
                "Runtime reset dataset does not contain samples for task "
                f"{selected_task.task_id}."
            )
        return dict(self.rng.choice(task_samples))

    def _select_task(self, options: dict[str, Any]) -> SkillTask:
        requested_task_id = options.get("task_id")
        if requested_task_id is not None:
            task_id = str(requested_task_id).strip()
            for task in self.settings.task_pool:
                if task.task_id == task_id:
                    return task
            raise ValueError(f"Unknown task_id requested in env reset: {task_id}")

        requested_object_name = options.get("object_name")
        requested_zone_name = options.get("zone_name")
        if requested_object_name is not None or requested_zone_name is not None:
            object_name = str(requested_object_name or "").strip()
            zone_name = str(requested_zone_name or "").strip()
            for task in self.settings.task_pool:
                if task.object_name == object_name and task.zone_name == zone_name:
                    return task
            raise ValueError(
                "Unknown object_name/zone_name requested in env reset: "
                f"{object_name}/{zone_name}"
            )

        if len(self.settings.task_pool) == 1:
            return self.settings.task_pool[0]
        return self.rng.choice(self.settings.task_pool)

    def _resolve_task_from_state(self, state: dict[str, Any]) -> SkillTask:
        task_id = str(state.get("task_id", "")).strip()
        object_name = str(state.get("object_name", "")).strip()
        zone_name = str(state.get("zone_name", "")).strip()
        instruction = str(state.get("instruction", "")).strip()

        for task in self.settings.task_pool:
            if (
                task_id
                and task.task_id == task_id
                and task.object_name == object_name
                and task.zone_name == zone_name
            ):
                return task

        for task in self.settings.task_pool:
            if task.object_name == object_name and task.zone_name == zone_name:
                if task.task_id == "task_0" and (task_id or instruction):
                    return SkillTask(
                        task_id=task_id or task.task_id,
                        instruction=instruction or task.instruction,
                        object_name=task.object_name,
                        zone_name=task.zone_name,
                    )
                return task

        return SkillTask(
            task_id=task_id or "task_0",
            instruction=instruction,
            object_name=object_name,
            zone_name=zone_name,
        )
