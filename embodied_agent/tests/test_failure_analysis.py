from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.failure_analysis import (
    classify_pick_failure,
    format_recovery_tables,
    summarize_pick_failures,
    summarize_recovery_failures,
)


class FailureAnalysisTests(unittest.TestCase):
    def test_classify_pick_failure_as_approach_failed(self) -> None:
        category = classify_pick_failure(
            {
                "initial_metrics": {"ee_object_distance": 0.15, "gripper_open_ratio": 1.0, "lift_progress": 0.0},
                "final_metrics": {"holding_flag": 0.0},
                "min_ee_object_distance": 0.11,
                "min_gripper_open_ratio": 0.0,
                "max_lift_progress": 0.0,
                "ever_holding": False,
            }
        )
        self.assertEqual(category, "approach_failed")

    def test_classify_pick_failure_as_attach_failed(self) -> None:
        category = classify_pick_failure(
            {
                "initial_metrics": {"ee_object_distance": 0.06, "gripper_open_ratio": 1.0, "lift_progress": 0.0},
                "final_metrics": {"holding_flag": 0.0},
                "min_ee_object_distance": 0.03,
                "min_gripper_open_ratio": 0.0,
                "max_lift_progress": 0.0,
                "ever_holding": False,
            }
        )
        self.assertEqual(category, "attach_failed")

    def test_classify_pick_failure_as_dropped_during_lift(self) -> None:
        category = classify_pick_failure(
            {
                "initial_metrics": {"ee_object_distance": 0.08, "gripper_open_ratio": 1.0, "lift_progress": 0.0},
                "final_metrics": {"holding_flag": 0.0},
                "min_ee_object_distance": 0.08,
                "min_gripper_open_ratio": 0.0,
                "max_lift_progress": 0.03,
                "ever_holding": True,
                "ever_release": True,
            }
        )
        self.assertEqual(category, "dropped_during_lift")

    def test_summarize_pick_failures_counts_categories(self) -> None:
        records = [
            {
                "episode": 3,
                "error": "The learned pick policy did not finish successfully.",
                "initial_object_positions": {"red_block": [0.5, 0.1, 0.645]},
                "initial_zone_positions": {"green_zone": [0.5, -0.1, 0.623]},
                "final_object_positions": {"red_block": [0.5, 0.1, 0.645]},
                "debug": {
                    "skills": [
                        {
                            "skill_name": "pick",
                            "controller": "learned",
                            "success": False,
                            "initial_metrics": {
                                "ee_object_distance": 0.06,
                                "gripper_open_ratio": 1.0,
                                "lift_progress": 0.0,
                            },
                            "final_metrics": {"holding_flag": 0.0},
                            "min_ee_object_distance": 0.03,
                            "min_gripper_open_ratio": 0.0,
                            "max_lift_progress": 0.0,
                            "ever_holding": False,
                        }
                    ]
                },
            }
        ]

        summary = summarize_pick_failures(records)

        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(summary["category_counts"], {"attach_failed": 1})
        self.assertEqual(summary["failed_episodes"][0]["episode"], 3)

    def test_summarize_recovery_failures_builds_tables(self) -> None:
        records = [
            {
                "experiment": "learned_place_hybrid_50",
                "episode": 7,
                "success": True,
                "replan_count": 1,
                "error": None,
                "failure_history": [
                    {
                        "source": "execution_error",
                        "reason": "The learned place policy did not finish successfully.",
                        "failed_step": {"action": "place"},
                        "recovery_policy": "repick_then_place",
                    }
                ],
            },
            {
                "experiment": "learned_place_hybrid_50",
                "episode": 8,
                "success": True,
                "replan_count": 0,
                "error": None,
                "failure_history": [],
            },
            {
                "experiment": "scripted_baseline_20",
                "episode": 0,
                "success": False,
                "replan_count": 1,
                "error": "Step 'pick' failed post-condition 'holding'.",
                "failure_history": [
                    {
                        "source": "post_condition_failed",
                        "reason": "Step 'pick' failed post-condition 'holding'.",
                        "failed_step": {"action": "pick"},
                        "recovery_policy": "retry_pick_then_continue",
                    },
                    {
                        "source": "post_condition_failed",
                        "reason": "Step 'pick' failed post-condition 'holding'.",
                        "failed_step": {"action": "pick"},
                        "recovery_policy": "retry_pick_then_continue",
                    },
                ],
            },
        ]

        summary = summarize_recovery_failures(records)

        self.assertEqual(
            summary["experiment_summary_rows"],
            [
                {
                    "experiment": "learned_place_hybrid_50",
                    "episodes": 2,
                    "episodes_with_failures": 1,
                    "episodes_with_replan": 1,
                    "successful_recovery_episodes": 1,
                    "total_step_failures": 1,
                    "recovery_success_rate": 1.0,
                },
                {
                    "experiment": "scripted_baseline_20",
                    "episodes": 1,
                    "episodes_with_failures": 1,
                    "episodes_with_replan": 1,
                    "successful_recovery_episodes": 0,
                    "total_step_failures": 2,
                    "recovery_success_rate": 0.0,
                },
            ],
        )
        self.assertEqual(
            summary["event_rows"],
            [
                {
                    "experiment": "learned_place_hybrid_50",
                    "source": "execution_error",
                    "action": "place",
                    "recovery_policy": "repick_then_place",
                    "count": 1,
                },
                {
                    "experiment": "scripted_baseline_20",
                    "source": "post_condition_failed",
                    "action": "pick",
                    "recovery_policy": "retry_pick_then_continue",
                    "count": 2,
                },
            ],
        )
        self.assertEqual(
            summary["reason_rows"],
            [
                {
                    "experiment": "learned_place_hybrid_50",
                    "reason": "The learned place policy did not finish successfully.",
                    "count": 1,
                },
                {
                    "experiment": "scripted_baseline_20",
                    "reason": "Step 'pick' failed post-condition 'holding'.",
                    "count": 2,
                },
            ],
        )
        self.assertEqual(
            summary["episode_rows"][0]["failure_signature"],
            "place/execution_error->repick_then_place",
        )

    def test_format_recovery_tables_renders_markdown_sections(self) -> None:
        rendered = format_recovery_tables(
            {
                "experiment_summary_rows": [
                    {
                        "experiment": "learned_place_hybrid_50",
                        "episodes": 50,
                        "episodes_with_failures": 2,
                        "episodes_with_replan": 2,
                        "successful_recovery_episodes": 2,
                        "total_step_failures": 2,
                        "recovery_success_rate": 1.0,
                    }
                ],
                "event_rows": [
                    {
                        "experiment": "learned_place_hybrid_50",
                        "source": "execution_error",
                        "action": "place",
                        "recovery_policy": "repick_then_place",
                        "count": 2,
                    }
                ],
                "reason_rows": [
                    {
                        "experiment": "learned_place_hybrid_50",
                        "reason": "The learned place policy did not finish successfully.",
                        "count": 2,
                    }
                ],
                "episode_rows": [
                    {
                        "experiment": "learned_place_hybrid_50",
                        "episode": 7,
                        "success": True,
                        "replan_count": 1,
                        "failure_events": 1,
                        "failure_signature": "place/execution_error->repick_then_place",
                        "reasons": "The learned place policy did not finish successfully.",
                        "error": "",
                    }
                ],
            }
        )

        self.assertIn("## Experiment Summary", rendered)
        self.assertIn("## Failure Events", rendered)
        self.assertIn("## Failure Episodes", rendered)
        self.assertIn("place/execution_error->repick_then_place", rendered)


if __name__ == "__main__":
    unittest.main()