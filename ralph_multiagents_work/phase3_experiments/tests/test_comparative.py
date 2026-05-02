"""Tests for comparative experiment module."""

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

from src.experiments.comparative_experiment import (
    ComparativeCondition,
    ComparativeTrialResult,
    ComparativeExperimentRunner,
    ComparativeExperimentReport,
    run_comparative_experiment,
    DEFAULT_TASKS,
    _mock_llm_call,
)


class TestComparativeCondition(unittest.TestCase):
    def test_enum_values(self):
        self.assertEqual(ComparativeCondition.NO_CAUSAL.value, "no_causal")
        self.assertEqual(ComparativeCondition.METADATA_BACKED.value, "metadata_backed")
        self.assertEqual(ComparativeCondition.SIMULATOR_BACKED.value, "simulator_backed")

    def test_three_conditions(self):
        self.assertEqual(len(ComparativeCondition), 3)


class TestComparativeTrialResult(unittest.TestCase):
    def test_to_dict(self):
        result = ComparativeTrialResult(
            condition="no_causal",
            task="把红色方块移到绿色区域",
            success=True,
            total_steps=5,
            explore_steps=0,
            planning_quality=0.85,
            replan_count=0,
            execution_time_seconds=0.5,
        )
        d = result.to_dict()
        self.assertEqual(d["condition"], "no_causal")
        self.assertTrue(d["success"])
        self.assertEqual(d["total_steps"], 5)
        self.assertFalse(d["fallback_used"])

    def test_to_dict_with_error(self):
        result = ComparativeTrialResult(
            condition="simulator_backed",
            task="test task",
            success=False,
            total_steps=0,
            explore_steps=0,
            planning_quality=0.0,
            replan_count=0,
            execution_time_seconds=0.0,
            error="Simulation error",
            fallback_used=True,
        )
        d = result.to_dict()
        self.assertEqual(d["error"], "Simulation error")
        self.assertTrue(d["fallback_used"])


class TestComparativeExperimentReport(unittest.TestCase):
    def test_empty_report(self):
        report = ComparativeExperimentReport()
        d = report.to_dict()
        self.assertIn("results", d)
        self.assertIn("summary", d)
        self.assertEqual(d["results"], [])

    def test_summary_with_results(self):
        report = ComparativeExperimentReport()
        report.results = [
            ComparativeTrialResult(
                condition="no_causal", task="task1", success=True,
                total_steps=5, explore_steps=0, planning_quality=0.8,
                replan_count=0, execution_time_seconds=0.5,
            ),
            ComparativeTrialResult(
                condition="no_causal", task="task2", success=False,
                total_steps=3, explore_steps=0, planning_quality=0.5,
                replan_count=1, execution_time_seconds=0.3,
            ),
            ComparativeTrialResult(
                condition="metadata_backed", task="task1", success=True,
                total_steps=6, explore_steps=1, planning_quality=0.9,
                replan_count=0, execution_time_seconds=0.6,
            ),
        ]
        summary = report._build_summary()
        self.assertIn("no_causal", summary)
        self.assertIn("metadata_backed", summary)
        self.assertEqual(summary["no_causal"]["success_rate"], 0.5)
        self.assertEqual(summary["no_causal"]["num_trials"], 2)


class TestComparativeExperimentRunner(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = pathlib.Path(self._tmp_dir.name)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_runner_initialization(self):
        runner = ComparativeExperimentRunner(
            tasks=["把红色方块移到绿色区域"],
            output_dir=str(self.tmp_path),
            headless=True,
        )
        self.assertEqual(len(runner.tasks), 1)
        self.assertTrue(runner.headless)

    def test_run_single_task(self):
        runner = ComparativeExperimentRunner(
            tasks=["把红色方块移到绿色区域"],
            output_dir=str(self.tmp_path),
            headless=True,
            random_seed=42,
        )
        report = runner.run()
        self.assertEqual(len(report.results), 3)  # 3 conditions x 1 task
        conditions = {r.condition for r in report.results}
        self.assertEqual(conditions, {"no_causal", "metadata_backed", "simulator_backed"})

    def test_save_results(self):
        runner = ComparativeExperimentRunner(
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

    def test_save_artifact_catalog(self):
        runner = ComparativeExperimentRunner(
            tasks=["把红色方块移到绿色区域"],
            output_dir=str(self.tmp_path),
            headless=True,
        )
        report = runner.run()
        catalog_path = runner.save_artifact_catalog(report)
        self.assertTrue(os.path.exists(catalog_path))
        data = json.loads(open(catalog_path).read())
        self.assertEqual(data["experiment"], "comparative_experiment")

    def test_run_multiple_tasks(self):
        runner = ComparativeExperimentRunner(
            tasks=["把红色方块移到绿色区域", "把蓝色方块移到黄色区域"],
            output_dir=str(self.tmp_path),
            headless=True,
            random_seed=42,
        )
        report = runner.run()
        self.assertEqual(len(report.results), 6)  # 3 conditions x 2 tasks

    def test_no_causal_has_zero_explore_steps(self):
        runner = ComparativeExperimentRunner(
            tasks=["把红色方块移到绿色区域"],
            output_dir=str(self.tmp_path),
            headless=True,
        )
        report = runner.run()
        for r in report.results:
            if r.condition == "no_causal":
                self.assertEqual(r.explore_steps, 0)

    def test_run_comparative_experiment_convenience(self):
        report, json_path, catalog_path = run_comparative_experiment(
            tasks=["把红色方块移到绿色区域"],
            output_dir=str(self.tmp_path),
            headless=True,
        )
        self.assertIsInstance(report, ComparativeExperimentReport)
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(catalog_path))

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

    def test_simulator_condition_reports_no_fallback_on_successful_probe(self):
        runner = ComparativeExperimentRunner(
            tasks=["把红色方块移到绿色区域"],
            output_dir=str(self.tmp_path),
            headless=True,
        )
        report = runner.run()
        simulator_result = next(
            r for r in report.results
            if r.condition == ComparativeCondition.SIMULATOR_BACKED.value
        )
        self.assertIn("fallback_used", simulator_result.to_dict())
        self.assertFalse(simulator_result.fallback_used)
        self.assertIsNone(simulator_result.error)

    def test_simulator_condition_supports_non_default_target_object(self):
        runner = ComparativeExperimentRunner(
            tasks=["把蓝色方块移到黄色区域"],
            output_dir=str(self.tmp_path),
            headless=True,
        )
        report = runner.run()
        simulator_result = next(
            r for r in report.results
            if r.condition == ComparativeCondition.SIMULATOR_BACKED.value
        )
        self.assertTrue(simulator_result.success)
        self.assertGreater(simulator_result.explore_steps, 0)
        self.assertFalse(simulator_result.fallback_used)
        self.assertIsNone(simulator_result.error)
