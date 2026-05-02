from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.rl_envs import PickPlaceSkillEnv, SkillEnvSettings


class FakePlaceSimulation:
    def __init__(self, metrics_sequence: list[dict[str, float]]) -> None:
        self.metrics_sequence = metrics_sequence
        self.metric_index = 0
        object_xy_map = {
            "red_block": (0.5, -0.1),
            "blue_block": (0.42, 0.14),
            "yellow_block": (0.36, -0.18),
        }
        zone_xy_map = {
            "green_zone": (0.6, 0.2),
            "yellow_zone": (0.53, -0.04),
            "blue_zone": (0.47, 0.22),
        }
        self.config = SimpleNamespace(
            cube_start_xy=(0.5, -0.1),
            goal_center_xy=(0.6, 0.2),
            placed_object_z=0.645,
            object_start_xy_for=lambda name: object_xy_map[name],
            zone_center_xy_for=lambda name: zone_xy_map[name],
        )
        self.held_object_name = "red_block"
        self.settle_calls: list[int] = []
        self.restored_states: list[dict[str, object]] = []
        self.restore_calls: list[dict[str, object]] = []
        self.reset_calls: list[dict[str, object]] = []

    def sample_task_layout(
        self,
        rng: object,
        object_x_range: tuple[float, float] | None = None,
        object_y_range: tuple[float, float] | None = None,
        zone_x_range: tuple[float, float] | None = None,
        zone_y_range: tuple[float, float] | None = None,
        object_candidates: tuple[tuple[float, float], ...] | None = None,
        zone_candidates: tuple[tuple[float, float], ...] | None = None,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        del rng, object_x_range, object_y_range, zone_x_range, zone_y_range, object_candidates, zone_candidates
        return self.config.cube_start_xy, self.config.goal_center_xy

    def sample_scene_layout(
        self,
        rng: object,
        object_names: tuple[str, ...],
        zone_names: tuple[str, ...],
        object_x_range: tuple[float, float] | None = None,
        object_y_range: tuple[float, float] | None = None,
        zone_x_range: tuple[float, float] | None = None,
        zone_y_range: tuple[float, float] | None = None,
        object_candidates: tuple[tuple[float, float], ...] | None = None,
        zone_candidates: tuple[tuple[float, float], ...] | None = None,
    ) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
        del rng, object_x_range, object_y_range, zone_x_range, zone_y_range, object_candidates, zone_candidates
        return (
            {name: tuple(self.config.object_start_xy_for(name)) for name in object_names},
            {name: tuple(self.config.zone_center_xy_for(name)) for name in zone_names},
        )

    def reset_task(
        self,
        object_xy: tuple[float, float] | None = None,
        zone_xy: tuple[float, float] | None = None,
        holding_object: bool = False,
        held_object_name: str | None = None,
        object_layout: dict[str, tuple[float, float]] | None = None,
        zone_layout: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.metric_index = 0
        self.reset_calls.append(
            {
                "object_xy": object_xy,
                "zone_xy": zone_xy,
                "holding_object": holding_object,
                "held_object_name": held_object_name,
                "object_layout": dict(object_layout or {}),
                "zone_layout": dict(zone_layout or {}),
            }
        )
        self.held_object_name = held_object_name if held_object_name is not None else ("red_block" if holding_object else None)

    def prepare_pick_staging_pose(self, object_name: str) -> None:
        del object_name

    def prepare_place_staging_pose(self, zone_name: str) -> None:
        del zone_name

    def restore_runtime_state(
        self,
        state: dict[str, object],
        object_name: str,
        zone_name: str,
    ) -> None:
        self.metric_index = 0
        self.restored_states.append(dict(state))
        self.restore_calls.append(
            {
                "state": dict(state),
                "object_name": object_name,
                "zone_name": zone_name,
            }
        )
        self.held_object_name = object_name if bool(state.get("holding_target_object", False)) else None

    def get_skill_observation(self, object_name: str, zone_name: str) -> np.ndarray:
        del object_name, zone_name
        return np.zeros(17, dtype=np.float32)

    def apply_skill_action(
        self,
        delta_position: tuple[float, float, float],
        gripper_command: float,
        action_steps: int,
        object_name: str,
    ) -> dict[str, object]:
        del delta_position, gripper_command, action_steps, object_name
        self.held_object_name = None
        return {}

    def get_skill_metrics(self, object_name: str, zone_name: str) -> dict[str, float]:
        del object_name, zone_name
        return dict(self.metrics_sequence[self.metric_index])

    def is_pick_success(self, object_name: str) -> bool:
        del object_name
        return False

    def is_place_success(self, object_name: str, zone_name: str) -> bool:
        del object_name, zone_name
        metrics = self.metrics_sequence[self.metric_index]
        return bool(metrics["holding_flag"] < 0.5 and metrics["object_zone_distance_xy"] <= 0.08)

    def simulate_steps(self, steps: int) -> None:
        self.settle_calls.append(steps)
        if self.metric_index < len(self.metrics_sequence) - 1:
            self.metric_index += 1


class SkillEnvSmokeTests(unittest.TestCase):
    def test_skill_env_settings_resolve_post_pick_state_dataset_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            dataset_path = base_dir / "post_pick_states.jsonl"
            dataset_path.write_text("{}\n", encoding="utf-8")

            settings = SkillEnvSettings.from_config(
                "place",
                {
                    "reset_source": "post_pick_states",
                    "post_pick_states_jsonl": "post_pick_states.jsonl",
                    "use_staging": False,
                },
                base_dir=base_dir,
            )

            self.assertEqual(settings.reset_source, "post_pick_states")
            self.assertEqual(settings.post_pick_states_jsonl, dataset_path.resolve())

    def test_skill_env_settings_normalize_task_pool(self) -> None:
        settings = SkillEnvSettings.from_config(
            "place",
            {
                "task_pool": [
                    {
                        "task_id": "red_to_green",
                        "instruction": "put the red block in the green zone",
                        "object_name": "red_block",
                        "zone_name": "green_zone",
                    },
                    {
                        "task_id": "blue_to_yellow",
                        "instruction": "put the blue block in the yellow zone",
                        "object_name": "blue_block",
                        "zone_name": "yellow_zone",
                    },
                ]
            },
        )

        self.assertEqual([task.task_id for task in settings.task_pool], ["red_to_green", "blue_to_yellow"])
        self.assertEqual(settings.active_object_names, ("red_block", "blue_block"))
        self.assertEqual(settings.active_zone_names, ("green_zone", "yellow_zone"))

    def test_pick_env_reset_and_step(self) -> None:
        env = PickPlaceSkillEnv(
            settings=SkillEnvSettings(skill_name="pick", randomize=False, max_steps=4),
            gui=False,
        )
        try:
            observation, info = env.reset(seed=7)
            self.assertEqual(observation.shape, (17,))
            self.assertIn("success", info)

            next_observation, reward, terminated, truncated, next_info = env.step(
                np.zeros(4, dtype=np.float32)
            )
            self.assertEqual(next_observation.shape, (17,))
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIn("ee_object_distance", next_info)
        finally:
            env.close()

    def test_place_env_starts_with_held_object(self) -> None:
        env = PickPlaceSkillEnv(
            settings=SkillEnvSettings(skill_name="place", randomize=False, max_steps=4),
            gui=False,
        )
        try:
            observation, _ = env.reset(seed=11)
            self.assertEqual(observation.shape, (17,))
            self.assertEqual(env.simulation.held_object_name, "red_block")
        finally:
            env.close()

    def test_place_success_requires_release(self) -> None:
        env = PickPlaceSkillEnv(
            settings=SkillEnvSettings(skill_name="place", randomize=False, max_steps=4),
            gui=False,
        )
        try:
            env.reset(seed=13)
            self.assertEqual(env.simulation.held_object_name, "red_block")
            self.assertFalse(env._is_success())
        finally:
            env.close()

    def test_place_reward_prefers_release_over_holding_when_aligned(self) -> None:
        env = PickPlaceSkillEnv(
            settings=SkillEnvSettings(skill_name="place", randomize=False, max_steps=4),
            gui=False,
        )
        try:
            holding_closed_reward = env._compute_place_reward(
                {
                    "object_zone_distance_xy": 0.01,
                    "object_height": env.simulation.config.placed_object_z,
                    "holding_flag": 1.0,
                    "lift_progress": 0.0,
                    "gripper_open_ratio": 0.0,
                    "ee_object_distance": 0.06,
                },
                success=False,
                release_event=False,
                release_resolved=False,
            )
            released_success_reward = env._compute_place_reward(
                {
                    "object_zone_distance_xy": 0.01,
                    "object_height": env.simulation.config.placed_object_z,
                    "holding_flag": 0.0,
                    "lift_progress": 0.0,
                    "gripper_open_ratio": 1.0,
                    "ee_object_distance": 0.06,
                },
                success=True,
                release_event=True,
                release_resolved=True,
            )

            self.assertGreater(released_success_reward, holding_closed_reward)
        finally:
            env.close()

    def test_place_reward_penalizes_delaying_release_when_ready(self) -> None:
        env = PickPlaceSkillEnv(
            settings=SkillEnvSettings(skill_name="place", randomize=False, max_steps=8),
            gui=False,
        )
        try:
            env.episode_steps = env.settings.max_steps
            closed_reward = env._compute_place_reward(
                {
                    "object_zone_distance_xy": 0.01,
                    "object_height": env.simulation.config.placed_object_z,
                    "holding_flag": 1.0,
                    "lift_progress": 0.0,
                    "gripper_open_ratio": 0.0,
                    "ee_object_distance": 0.06,
                },
                success=False,
                release_event=False,
                release_resolved=False,
            )
            open_reward = env._compute_place_reward(
                {
                    "object_zone_distance_xy": 0.01,
                    "object_height": env.simulation.config.placed_object_z,
                    "holding_flag": 1.0,
                    "lift_progress": 0.0,
                    "gripper_open_ratio": 1.0,
                    "ee_object_distance": 0.06,
                },
                success=False,
                release_event=False,
                release_resolved=False,
            )

            self.assertGreater(open_reward, closed_reward)
        finally:
            env.close()

    def test_place_reward_penalizes_failed_release_resolution(self) -> None:
        env = PickPlaceSkillEnv(
            settings=SkillEnvSettings(skill_name="place", randomize=False, max_steps=8),
            gui=False,
        )
        try:
            unresolved_reward = env._compute_place_reward(
                {
                    "object_zone_distance_xy": 0.10,
                    "object_height": env.simulation.config.placed_object_z + 0.03,
                    "holding_flag": 0.0,
                    "lift_progress": 0.0,
                    "gripper_open_ratio": 1.0,
                    "ee_object_distance": 0.12,
                },
                success=False,
                release_event=False,
                release_resolved=False,
            )
            resolved_failed_reward = env._compute_place_reward(
                {
                    "object_zone_distance_xy": 0.10,
                    "object_height": env.simulation.config.placed_object_z + 0.03,
                    "holding_flag": 0.0,
                    "lift_progress": 0.0,
                    "gripper_open_ratio": 1.0,
                    "ee_object_distance": 0.12,
                },
                success=False,
                release_event=False,
                release_resolved=True,
            )

            self.assertLess(resolved_failed_reward, unresolved_reward)
        finally:
            env.close()

    def test_place_open_gripper_starts_settling_before_episode_termination(self) -> None:
        env = PickPlaceSkillEnv(
            settings=SkillEnvSettings(
                skill_name="place",
                randomize=False,
                max_steps=4,
                post_release_settle_steps=2,
            ),
            gui=False,
        )
        try:
            env.reset(seed=17)
            _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["release_event"])
            self.assertFalse(info["release_resolved"])
            self.assertTrue(info["settling_active"])
            self.assertEqual(info["settling_steps_remaining"], 2)
            self.assertEqual(info["holding_flag"], 0.0)
        finally:
            env.close()

    def test_place_release_resolution_uses_final_settled_landing_state(self) -> None:
        simulation = FakePlaceSimulation(
            metrics_sequence=[
                {
                    "object_zone_distance_xy": 0.24,
                    "object_height": 0.66,
                    "holding_flag": 0.0,
                    "lift_progress": 0.0,
                    "gripper_open_ratio": 1.0,
                    "ee_object_distance": 0.20,
                },
                {
                    "object_zone_distance_xy": 0.12,
                    "object_height": 0.65,
                    "holding_flag": 0.0,
                    "lift_progress": 0.0,
                    "gripper_open_ratio": 1.0,
                    "ee_object_distance": 0.18,
                },
                {
                    "object_zone_distance_xy": 0.02,
                    "object_height": 0.645,
                    "holding_flag": 0.0,
                    "lift_progress": 0.0,
                    "gripper_open_ratio": 1.0,
                    "ee_object_distance": 0.16,
                },
            ]
        )
        env = PickPlaceSkillEnv(
            settings=SkillEnvSettings(
                skill_name="place",
                randomize=False,
                max_steps=4,
                post_release_settle_steps=2,
            ),
            gui=False,
            simulation=simulation,
        )
        try:
            env.reset(seed=5)

            _, first_reward, terminated, truncated, info = env.step(
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            )
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["release_event"])
            self.assertFalse(info["release_resolved"])

            _, second_reward, terminated, truncated, info = env.step(
                np.zeros(4, dtype=np.float32)
            )
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["settling_active"])
            self.assertEqual(info["settling_steps_remaining"], 1)

            _, final_reward, terminated, truncated, info = env.step(
                np.zeros(4, dtype=np.float32)
            )
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertFalse(info["release_event"])
            self.assertTrue(info["release_resolved"])
            self.assertFalse(info["settling_active"])
            self.assertEqual(info["settling_steps_remaining"], 0)
            self.assertTrue(info["success"])
            self.assertEqual(simulation.settle_calls, [env.settings.action_repeat, env.settings.action_repeat])
            self.assertLess(first_reward, final_reward)
            self.assertLess(second_reward, final_reward)
        finally:
            env.close()

    def test_pick_dense_reward_prefers_progressive_skill_states(self) -> None:
        env = PickPlaceSkillEnv(
            settings=SkillEnvSettings(skill_name="pick", randomize=False, max_steps=4),
            gui=False,
        )
        try:
            far_reward = env._compute_pick_reward(
                {
                    "ee_object_distance": 0.30,
                    "holding_flag": 0.0,
                    "lift_progress": 0.0,
                    "gripper_open_ratio": 1.0,
                },
                success=False,
            )
            near_close_reward = env._compute_pick_reward(
                {
                    "ee_object_distance": 0.05,
                    "holding_flag": 0.0,
                    "lift_progress": 0.0,
                    "gripper_open_ratio": 0.0,
                },
                success=False,
            )
            lifted_success_reward = env._compute_pick_reward(
                {
                    "ee_object_distance": 0.01,
                    "holding_flag": 1.0,
                    "lift_progress": 0.07,
                    "gripper_open_ratio": 0.0,
                },
                success=True,
            )

            self.assertGreater(near_close_reward, far_reward)
            self.assertGreater(lifted_success_reward, near_close_reward)
        finally:
            env.close()

    def test_randomize_probability_can_override_fixed_layout_defaults(self) -> None:
        env = PickPlaceSkillEnv(
            settings=SkillEnvSettings(skill_name="pick", randomize=False, max_steps=4),
            gui=False,
        )
        try:
            self.assertEqual(env.get_randomize_probability(), 0.0)

            env.set_randomize_probability(1.0)
            _, randomized_info = env.reset(seed=3)
            self.assertEqual(randomized_info["randomize_probability"], 1.0)
            self.assertTrue(randomized_info["randomize_layout"])

            env.set_randomize_probability(0.0)
            _, fixed_info = env.reset(seed=3)
            self.assertEqual(fixed_info["randomize_probability"], 0.0)
            self.assertFalse(fixed_info["randomize_layout"])
        finally:
            env.close()

    def test_place_env_can_reset_from_post_pick_runtime_dataset(self) -> None:
        simulation = FakePlaceSimulation(
            metrics_sequence=[
                {
                    "object_zone_distance_xy": 0.18,
                    "object_height": 0.66,
                    "holding_flag": 1.0,
                    "lift_progress": 0.0,
                    "gripper_open_ratio": 0.0,
                    "ee_object_distance": 0.04,
                }
            ]
        )
        sample = {
            "task_id": "red_to_green",
            "instruction": "place red block in green zone",
            "episode_index": 3,
            "capture_stage": "post_pick_success",
            "object_name": "red_block",
            "zone_name": "green_zone",
            "end_effector_position": [0.62, 0.05, 0.79],
            "end_effector_orientation": [1.0, 0.0, 0.0, 0.0],
            "object_position": [0.62, 0.05, 0.73],
            "object_orientation": [0.0, 0.0, 0.0, 1.0],
            "zone_position": [0.50, -0.10, 0.623],
            "holding_target_object": True,
            "held_local_position": [0.0, 0.0, -0.06],
            "held_local_orientation": [0.0, 0.0, 0.0, 1.0],
            "gripper_open_ratio": 0.0,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "post_pick_states.jsonl"
            dataset_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            env = PickPlaceSkillEnv(
                settings=SkillEnvSettings(
                    skill_name="place",
                    randomize=False,
                    max_steps=4,
                    use_staging=False,
                    reset_source="post_pick_states",
                    post_pick_states_jsonl=dataset_path,
                ),
                gui=False,
                simulation=simulation,
            )
            try:
                observation, info = env.reset(seed=19)
                self.assertEqual(observation.shape, (17,))
                self.assertEqual(len(simulation.restored_states), 1)
                self.assertEqual(simulation.restored_states[0]["task_id"], "red_to_green")
                self.assertEqual(info["reset_source"], "post_pick_states")
                self.assertIsNotNone(info["runtime_reset_metadata"])
                self.assertEqual(info["runtime_reset_metadata"]["task_id"], "red_to_green")
                self.assertFalse(info["randomize_layout"])
                self.assertEqual(env.simulation.held_object_name, "red_block")
            finally:
                env.close()

    def test_place_env_runtime_reset_can_select_task_from_task_pool(self) -> None:
        simulation = FakePlaceSimulation(
            metrics_sequence=[
                {
                    "object_zone_distance_xy": 0.22,
                    "object_height": 0.66,
                    "holding_flag": 1.0,
                    "lift_progress": 0.0,
                    "gripper_open_ratio": 0.0,
                    "ee_object_distance": 0.05,
                }
            ]
        )
        samples = [
            {
                "task_id": "red_to_green",
                "instruction": "put the red block in the green zone",
                "episode_index": 3,
                "capture_stage": "post_pick_success",
                "object_name": "red_block",
                "zone_name": "green_zone",
                "holding_target_object": True,
                "held_local_position": [0.0, 0.0, 0.09],
            },
            {
                "task_id": "blue_to_yellow",
                "instruction": "put the blue block in the yellow zone",
                "episode_index": 7,
                "capture_stage": "post_pick_success",
                "object_name": "blue_block",
                "zone_name": "yellow_zone",
                "holding_target_object": True,
                "held_local_position": [0.0, 0.0, 0.09],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "post_pick_states.jsonl"
            dataset_path.write_text(
                "\n".join(json.dumps(sample) for sample in samples) + "\n",
                encoding="utf-8",
            )
            env = PickPlaceSkillEnv(
                settings=SkillEnvSettings(
                    skill_name="place",
                    randomize=False,
                    max_steps=4,
                    use_staging=False,
                    reset_source="post_pick_states",
                    post_pick_states_jsonl=dataset_path,
                    task_pool=(
                        {
                            "task_id": "red_to_green",
                            "instruction": "put the red block in the green zone",
                            "object_name": "red_block",
                            "zone_name": "green_zone",
                        },
                        {
                            "task_id": "blue_to_yellow",
                            "instruction": "put the blue block in the yellow zone",
                            "object_name": "blue_block",
                            "zone_name": "yellow_zone",
                        },
                    ),
                ),
                gui=False,
                simulation=simulation,
            )
            try:
                observation, info = env.reset(seed=23, options={"task_id": "blue_to_yellow"})
                self.assertEqual(observation.shape, (17,))
                self.assertEqual(env.current_task.task_id, "blue_to_yellow")
                self.assertEqual(info["task_id"], "blue_to_yellow")
                self.assertEqual(info["object_name"], "blue_block")
                self.assertEqual(simulation.restore_calls[0]["object_name"], "blue_block")
                self.assertEqual(simulation.restore_calls[0]["zone_name"], "yellow_zone")
                self.assertEqual(info["runtime_reset_metadata"]["episode_index"], 7)
            finally:
                env.close()

    def test_place_env_runtime_reset_requires_samples_for_all_task_pool_entries(self) -> None:
        sample = {
            "task_id": "red_to_green",
            "instruction": "put the red block in the green zone",
            "episode_index": 1,
            "capture_stage": "post_pick_success",
            "object_name": "red_block",
            "zone_name": "green_zone",
            "holding_target_object": True,
            "held_local_position": [0.0, 0.0, 0.09],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "post_pick_states.jsonl"
            dataset_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "blue_to_yellow"):
                PickPlaceSkillEnv(
                    settings=SkillEnvSettings(
                        skill_name="place",
                        randomize=False,
                        max_steps=4,
                        use_staging=False,
                        reset_source="post_pick_states",
                        post_pick_states_jsonl=dataset_path,
                        task_pool=(
                            {
                                "task_id": "red_to_green",
                                "instruction": "put the red block in the green zone",
                                "object_name": "red_block",
                                "zone_name": "green_zone",
                            },
                            {
                                "task_id": "blue_to_yellow",
                                "instruction": "put the blue block in the yellow zone",
                                "object_name": "blue_block",
                                "zone_name": "yellow_zone",
                            },
                        ),
                    ),
                    gui=False,
                    simulation=FakePlaceSimulation(metrics_sequence=[
                        {
                            "object_zone_distance_xy": 0.18,
                            "object_height": 0.66,
                            "holding_flag": 1.0,
                            "lift_progress": 0.0,
                            "gripper_open_ratio": 0.0,
                            "ee_object_distance": 0.04,
                        }
                    ]),
                )

    def test_place_env_default_reset_can_select_task_from_task_pool(self) -> None:
        simulation = FakePlaceSimulation(
            metrics_sequence=[
                {
                    "object_zone_distance_xy": 0.18,
                    "object_height": 0.66,
                    "holding_flag": 1.0,
                    "lift_progress": 0.0,
                    "gripper_open_ratio": 0.0,
                    "ee_object_distance": 0.04,
                }
            ]
        )
        env = PickPlaceSkillEnv(
            settings=SkillEnvSettings(
                skill_name="place",
                randomize=False,
                max_steps=4,
                task_pool=(
                    {
                        "task_id": "red_to_green",
                        "instruction": "put the red block in the green zone",
                        "object_name": "red_block",
                        "zone_name": "green_zone",
                    },
                    {
                        "task_id": "yellow_to_blue",
                        "instruction": "把黄色方块放到蓝色区域",
                        "object_name": "yellow_block",
                        "zone_name": "blue_zone",
                    },
                ),
            ),
            gui=False,
            simulation=simulation,
        )
        try:
            observation, info = env.reset(seed=29, options={"task_id": "yellow_to_blue"})
            self.assertEqual(observation.shape, (17,))
            self.assertEqual(env.current_task.task_id, "yellow_to_blue")
            self.assertEqual(info["task_id"], "yellow_to_blue")
            self.assertEqual(info["object_name"], "yellow_block")
            self.assertEqual(info["zone_name"], "blue_zone")
            self.assertEqual(simulation.reset_calls[0]["held_object_name"], "yellow_block")
            self.assertEqual(
                simulation.reset_calls[0]["object_layout"],
                {"yellow_block": (0.36, -0.18)},
            )
            self.assertEqual(
                simulation.reset_calls[0]["zone_layout"],
                {"blue_zone": (0.47, 0.22)},
            )
            self.assertEqual(env.simulation.held_object_name, "yellow_block")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()