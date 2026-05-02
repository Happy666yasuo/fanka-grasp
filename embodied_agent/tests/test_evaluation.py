from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.evaluation import _build_task_pool, _resolve_scene_entities, _summarize_recovery_metrics
from embodied_agent.evaluation import _append_runtime_alignment_records


class EvaluationRecoverySummaryTests(unittest.TestCase):
    def test_summarizes_recovery_metrics_from_episode_records(self) -> None:
        summary = _summarize_recovery_metrics(
            [
                {
                    "success": True,
                    "replan_count": 1,
                    "failure_history": [
                        {
                            "source": "post_condition_failed",
                            "failed_step": {"action": "pick"},
                            "recovery_policy": "retry_pick_then_continue",
                        }
                    ],
                },
                {
                    "success": False,
                    "replan_count": 1,
                    "failure_history": [
                        {
                            "source": "execution_error",
                            "failed_step": {"action": "place"},
                            "recovery_policy": "repick_then_place",
                        }
                    ],
                },
                {
                    "success": True,
                    "replan_count": 0,
                    "failure_history": [],
                },
            ]
        )

        self.assertEqual(summary["episodes_with_replan"], 2)
        self.assertEqual(summary["successful_recovery_episodes"], 1)
        self.assertEqual(summary["recovery_success_rate"], 0.5)
        self.assertEqual(summary["total_step_failures"], 2)
        self.assertEqual(summary["failure_source_counts"], {"post_condition_failed": 1, "execution_error": 1})
        self.assertEqual(summary["failure_action_counts"], {"pick": 1, "place": 1})
        self.assertEqual(
            summary["recovery_policy_counts"],
            {"retry_pick_then_continue": 1, "repick_then_place": 1},
        )


class EvaluationTaskPoolTests(unittest.TestCase):
    def test_builds_task_pool_from_multi_object_entries(self) -> None:
        task_pool = _build_task_pool(
            {
                "task_pool": [
                    {
                        "task_id": "red_to_green",
                        "instruction": "put the red block in the green zone",
                        "object_name": "red_block",
                        "zone_name": "green_zone",
                    },
                    {
                        "instruction": "把蓝色方块放到黄色区域",
                        "object_name": "blue_block",
                        "zone_name": "yellow_zone",
                    },
                ]
            }
        )

        self.assertEqual(
            task_pool,
            [
                {
                    "task_id": "red_to_green",
                    "instruction": "put the red block in the green zone",
                    "object_name": "red_block",
                    "zone_name": "green_zone",
                },
                {
                    "task_id": "task_1",
                    "instruction": "把蓝色方块放到黄色区域",
                    "object_name": "blue_block",
                    "zone_name": "yellow_zone",
                },
            ],
        )

    def test_resolves_scene_entities_from_task_pool(self) -> None:
        object_names, zone_names = _resolve_scene_entities(
            [
                {"task_id": "a", "instruction": "one", "object_name": "red_block", "zone_name": "green_zone"},
                {"task_id": "b", "instruction": "two", "object_name": "blue_block", "zone_name": "yellow_zone"},
                {"task_id": "c", "instruction": "three", "object_name": "red_block", "zone_name": "blue_zone"},
            ]
        )

        self.assertEqual(object_names, ("red_block", "blue_block"))
        self.assertEqual(zone_names, ("green_zone", "yellow_zone", "blue_zone"))


class EvaluationRuntimeAlignmentTests(unittest.TestCase):
    def test_appends_runtime_alignment_records_to_jsonl_outputs(self) -> None:
        with self.subTest("writes post-pick and place-entry records"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp_dir:
                run_dir = Path(tmp_dir)
                info = _append_runtime_alignment_records(
                    run_dir=run_dir,
                    experiment_name="learned_place_runtime_alignment_50",
                    episode_record={
                        "experiment": "learned_place_runtime_alignment_50",
                        "episode": 3,
                        "task_id": "red_to_green",
                        "instruction": "put the red block in the green zone",
                        "object_name": "red_block",
                        "zone_name": "green_zone",
                        "success": False,
                        "replan_count": 1,
                        "debug": {
                            "runtime_alignment": {
                                "post_pick_states": [
                                    {
                                        "capture_stage": "post_pick_success",
                                        "pick_controller": "scripted",
                                        "place_controller": None,
                                        "use_staging": None,
                                        "metrics": {"holding_flag": 1.0},
                                    }
                                ],
                                "place_entry_states": [
                                    {
                                        "capture_stage": "pre_place_runtime_raw",
                                        "pick_controller": "scripted",
                                        "place_controller": "learned",
                                        "use_staging": True,
                                        "metrics": {"object_zone_distance_xy": 0.14},
                                    },
                                    {
                                        "capture_stage": "pre_place_policy_entry",
                                        "pick_controller": "scripted",
                                        "place_controller": "learned",
                                        "use_staging": True,
                                        "metrics": {"object_zone_distance_xy": 0.08},
                                    },
                                ],
                            }
                        },
                    },
                )

                self.assertTrue(info["has_post_pick_states"])
                self.assertTrue(info["has_place_entry_states"])
                self.assertEqual(
                    info["stage_counts"],
                    {
                        "post_pick_success": 1,
                        "pre_place_runtime_raw": 1,
                        "pre_place_policy_entry": 1,
                    },
                )

                post_pick_lines = (run_dir / "learned_place_runtime_alignment_50_post_pick_states.jsonl").read_text(
                    encoding="utf-8"
                ).strip().splitlines()
                place_entry_lines = (
                    run_dir / "learned_place_runtime_alignment_50_place_entry_states.jsonl"
                ).read_text(encoding="utf-8").strip().splitlines()

                self.assertEqual(len(post_pick_lines), 1)
                self.assertEqual(len(place_entry_lines), 2)


if __name__ == "__main__":
    unittest.main()