"""Tests for ablation experiment module."""

import json
import sys
import os
import unittest
import tempfile
import pathlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../phase1_skills_and_causal/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../phase2_llm_planner/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from src.experiments.ablation_experiment import (
    AblationDimension,
    AblationCondition,
    AblationTrialResult,
    AblationExperimentRunner,
    AblationExperimentReport,
    run_ablation_experiment,
    _build_all_ablation_conditions,
    _mock_llm_call,
)


class TestAblationDimension(unittest.TestCase):
    def test_enum_values(self):
        self.assertEqual(AblationDimension.EXPLORE_STRATEGY.value, "explore_strategy")
        self.assertEqual(AblationDimension.UNCERTAINTY.value, "uncertainty")
        self.assertEqual(AblationDimension.RECOVERY.value, "recovery")

    def test_three_dimensions(self):
        self.assertEqual(len(AblationDimension), 3)


class TestAblationCondition(unittest.TestCase):
    def test_label_format(self):
        cond = AblationCondition(
            strategy="causal_explore",
            use_uncertainty=True,
            use_recovery=True,
        )
        self.assertIn("causal_explore", cond.label)
        self.assertIn("U+", cond.label)
        self.assertIn("R+", cond.label)

    def test_label_without_uncertainty(self):
        cond = AblationCondition(
            strategy="random",
            use_uncertainty=False,
            use_recovery=False,
        )
        self.assertIn("U-", cond.label)
        self.assertIn("R-", cond.label)

    def test_to_dict(self):
        cond = AblationCondition(
            strategy="curiosity",
            use_uncertainty=True,
            use_recovery=False,
        )
        d = cond.to_dict()
        self.assertEqual(d["strategy"], "curiosity")
        self.assertTrue(d["use_uncertainty"])
        self.assertFalse(d["use_recovery"])


class TestAblationTrialResult(unittest.TestCase):
    def test_to_dict(self):
        result = AblationTrialResult(
            condition_label="causal_explore_U+_R+",
            strategy="causal_explore",
            use_uncertainty=True,
            use_recovery=True,
            task="把红色方块移到绿色区域",
            success=True,
            total_steps=5,
            explore_steps=2,
            uncertainty_score=0.3,
            replan_count=0,
            recovery_count=0,
            execution_time_seconds=0.5,
        )
        d = result.to_dict()
        self.assertEqual(d["condition_label"], "causal_explore_U+_R+")
        self.assertTrue(d["success"])
        self.assertFalse(d["fallback_used"])


class TestAblationExperimentReport(unittest.TestCase):
    def test_empty_report(self):
        report = AblationExperimentReport()
        d = report.to_dict()
        self.assertIn("results", d)
        self.assertIn("summary", d)

    def test_summary_grouping(self):
        report = AblationExperimentReport()
        report.results = [
            AblationTrialResult(
                condition_label="causal_explore_U+_R+", strategy="causal_explore",
                use_uncertainty=True, use_recovery=True, task="task1",
                success=True, total_steps=5, explore_steps=2,
                uncertainty_score=0.3, replan_count=0, recovery_count=0,
                execution_time_seconds=0.5,
            ),
            AblationTrialResult(
                condition_label="causal_explore_U+_R+", strategy="causal_explore",
                use_uncertainty=True, use_recovery=True, task="task2",
                success=True, total_steps=5, explore_steps=2,
                uncertainty_score=0.3, replan_count=0, recovery_count=0,
                execution_time_seconds=0.5,
            ),
        ]
        summary = report._build_summary()
        self.assertIn("causal_explore_U+_R+", summary)
        self.assertEqual(summary["causal_explore_U+_R+"]["success_rate"], 1.0)
        self.assertEqual(summary["causal_explore_U+_R+"]["num_trials"], 2)


class TestAblationExperimentRunner(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = pathlib.Path(self._tmp_dir.name)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_runner_initialization(self):
        runner = AblationExperimentRunner(
            tasks=["把红色方块移到绿色区域"],
            output_dir=str(self.tmp_path),
            headless=True,
        )
        self.assertEqual(len(runner.tasks), 1)
        self.assertTrue(runner.headless)

    def test_run_single_task(self):
        runner = AblationExperimentRunner(
            tasks=["把红色方块移到绿色区域"],
            output_dir=str(self.tmp_path),
            headless=True,
            random_seed=42,
        )
        report = runner.run()
        conditions = _build_all_ablation_conditions()
        self.assertEqual(len(report.results), len(conditions))  # 12 conditions x 1 task

    def test_save_results(self):
        runner = AblationExperimentRunner(
            tasks=["把红色方块移到绿色区域"],
            output_dir=str(self.tmp_path),
            headless=True,
        )
        report = runner.run()
        json_path = runner.save_results(report)
        self.assertTrue(os.path.exists(json_path))
        data = json.loads(open(json_path).read())
        self.assertIn("results", data)
        self.assertIn("summary", data)

    def test_build_all_ablation_conditions(self):
        conditions = _build_all_ablation_conditions()
        self.assertEqual(len(conditions), 12)  # 3 x 2 x 2

        strategies = {c.strategy for c in conditions}
        self.assertEqual(strategies, {"random", "curiosity", "causal_explore"})

        u_plus = sum(1 for c in conditions if c.use_uncertainty)
        u_minus = sum(1 for c in conditions if not c.use_uncertainty)
        self.assertEqual(u_plus, 6)
        self.assertEqual(u_minus, 6)

        r_plus = sum(1 for c in conditions if c.use_recovery)
        r_minus = sum(1 for c in conditions if not c.use_recovery)
        self.assertEqual(r_plus, 6)
        self.assertEqual(r_minus, 6)

    def test_run_ablation_experiment_convenience(self):
        report, json_path = run_ablation_experiment(
            tasks=["把红色方块移到绿色区域"],
            output_dir=str(self.tmp_path),
            headless=True,
        )
        self.assertIsInstance(report, AblationExperimentReport)
        self.assertTrue(os.path.exists(json_path))

    def test_mock_llm_single_target_color_zone_tasks(self):
        cases = [
            ("把红色方块移到绿色区域", "red_block", "green_zone"),
            ("把蓝色方块移到黄色区域", "blue_block", "yellow_zone"),
            ("把黄色方块移到蓝色区域", "yellow_block", "blue_zone"),
        ]
        for instruction, expected_object, expected_zone in cases:
            with self.subTest(instruction=instruction):
                response = _mock_llm_call([
                    {"role": "user", "content": instruction},
                ])
                steps = json.loads(response)["steps"]
                self.assertEqual(
                    [s["selected_skill"] for s in steps],
                    ["observe", "pick", "place"],
                )
                self.assertEqual(steps[1]["target_object"], expected_object)
                self.assertEqual(steps[2]["target_object"], expected_object)
                self.assertEqual(steps[2]["skill_args"]["target_zone"], expected_zone)

    def test_uncertainty_enabled_condition_does_not_raise_missing_evaluate(self):
        runner = AblationExperimentRunner(
            tasks=["把红色方块移到绿色区域"],
            output_dir=str(self.tmp_path),
            headless=True,
            random_seed=42,
        )
        result = runner._run_trial(
            "把红色方块移到绿色区域",
            AblationCondition(strategy="random", use_uncertainty=True, use_recovery=True),
        )
        self.assertNotIn("evaluate", result.error or "")

    def test_causal_explore_condition_supports_non_default_target_object(self):
        runner = AblationExperimentRunner(
            tasks=["把黄色方块移到蓝色区域"],
            output_dir=str(self.tmp_path),
            headless=True,
            random_seed=42,
        )
        result = runner._run_trial(
            "把黄色方块移到蓝色区域",
            AblationCondition(strategy="causal_explore", use_uncertainty=True, use_recovery=True),
        )
        self.assertTrue(result.success)
        self.assertGreater(result.explore_steps, 0)
        self.assertFalse(result.fallback_used)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.uncertainty_score, 0.0)
