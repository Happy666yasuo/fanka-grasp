"""Tests for reporting module."""

import json
import sys
import os
import unittest
import tempfile
import pathlib
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../phase1_skills_and_causal/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../phase2_llm_planner/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from src.reporting.experiment_reporter import ExperimentReporter
from src.reporting.chart_generator import ChartGenerator


COMPARATIVE_JSON = {
    "config": {
        "tasks": ["task1"],
        "headless": True,
        "max_replans": 3,
        "conditions": ["no_causal", "metadata_backed", "simulator_backed"],
    },
    "results": [
        {
            "condition": "no_causal", "task": "task1", "success": True,
            "total_steps": 4, "explore_steps": 0, "planning_quality": 0.8,
            "replan_count": 0, "execution_time_seconds": 0.5,
            "plan_actions": ["observe", "pick", "place"], "failure_details": [],
        },
        {
            "condition": "metadata_backed", "task": "task1", "success": True,
            "total_steps": 5, "explore_steps": 1, "planning_quality": 0.85,
            "replan_count": 0, "execution_time_seconds": 0.6,
            "plan_actions": ["observe", "probe", "pick", "place"], "failure_details": [],
        },
        {
            "condition": "simulator_backed", "task": "task1", "success": True,
            "total_steps": 6, "explore_steps": 2, "planning_quality": 0.9,
            "replan_count": 0, "execution_time_seconds": 0.8,
            "plan_actions": ["observe", "probe", "probe", "pick", "place"], "failure_details": [],
            "error": None, "fallback_used": False,
        },
    ],
    "summary": {
        "no_causal": {"num_trials": 1, "success_rate": 1.0, "avg_explore_steps": 0.0,
                       "avg_planning_quality": 0.8, "avg_replan_count": 0.0, "avg_execution_time": 0.5},
        "metadata_backed": {"num_trials": 1, "success_rate": 1.0, "avg_explore_steps": 1.0,
                            "avg_planning_quality": 0.85, "avg_replan_count": 0.0, "avg_execution_time": 0.6},
        "simulator_backed": {"num_trials": 1, "success_rate": 1.0, "avg_explore_steps": 2.0,
                             "avg_planning_quality": 0.9, "avg_replan_count": 0.0, "avg_execution_time": 0.8},
    },
}

ABLATION_JSON = {
    "config": {
        "tasks": ["task1"],
        "headless": True,
        "max_replans": 3,
        "num_conditions": 12,
        "dimensions": ["explore_strategy", "uncertainty", "recovery"],
    },
    "results": [
        {
            "condition_label": "causal_explore_U+_R+", "strategy": "causal_explore",
            "use_uncertainty": True, "use_recovery": True,
            "task": "task1", "success": True, "total_steps": 5,
            "explore_steps": 2, "uncertainty_score": 0.3,
            "replan_count": 0, "recovery_count": 0, "execution_time_seconds": 0.5,
        },
        {
            "condition_label": "random_U-_R-", "strategy": "random",
            "use_uncertainty": False, "use_recovery": False,
            "task": "task1", "success": False, "total_steps": 3,
            "explore_steps": 0, "uncertainty_score": 0.8,
            "replan_count": 0, "recovery_count": 0, "execution_time_seconds": 0.2,
            "error": "Step failed without recovery", "fallback_used": False,
        },
    ],
    "summary": {
        "causal_explore_U+_R+": {"num_trials": 1, "success_rate": 1.0, "avg_explore_steps": 2.0,
                                  "avg_uncertainty": 0.3, "avg_replan_count": 0.0,
                                  "avg_recovery_count": 0.0, "avg_execution_time": 0.5},
        "random_U-_R-": {"num_trials": 1, "success_rate": 0.0, "avg_explore_steps": 0.0,
                          "avg_uncertainty": 0.8, "avg_replan_count": 0.0,
                          "avg_recovery_count": 0.0, "avg_execution_time": 0.2},
    },
}


class TestExperimentReporter(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = pathlib.Path(self._tmp_dir.name)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def _write_json(self, name, data):
        p = self.tmp_path / name
        p.write_text(json.dumps(data))
        return p

    def test_init_with_comparative_data(self):
        input_file = self._write_json("test.json", COMPARATIVE_JSON)
        reporter = ExperimentReporter(str(input_file))
        self.assertEqual(reporter._detect_type(), "comparative")

    def test_init_with_ablation_data(self):
        input_file = self._write_json("test.json", ABLATION_JSON)
        reporter = ExperimentReporter(str(input_file))
        self.assertEqual(reporter._detect_type(), "ablation")

    def test_generate_comparative_markdown(self):
        input_file = self._write_json("test.json", COMPARATIVE_JSON)
        reporter = ExperimentReporter(str(input_file))
        md = reporter.generate_markdown()
        self.assertIn("对比实验报告", md)
        self.assertIn("no_causal", md)
        self.assertIn("metadata_backed", md)
        self.assertIn("simulator_backed", md)
        self.assertIn("汇总统计", md)

    def test_comparative_markdown_includes_error_and_fallback_columns(self):
        data = json.loads(json.dumps(COMPARATIVE_JSON))
        data["results"][2]["success"] = False
        data["results"][2]["error"] = "probe_failed_for_blue_block"
        data["results"][2]["fallback_used"] = True
        input_file = self._write_json("test.json", data)
        reporter = ExperimentReporter(str(input_file))
        md = reporter.generate_markdown()
        self.assertIn("Fallback", md)
        self.assertIn("Error", md)
        self.assertIn("probe_failed_for_blue_block", md)
        self.assertIn("yes", md)

    def test_generate_ablation_markdown(self):
        input_file = self._write_json("test.json", ABLATION_JSON)
        reporter = ExperimentReporter(str(input_file))
        md = reporter.generate_markdown()
        self.assertIn("消融实验报告", md)
        self.assertIn("causal_explore_U+_R+", md)
        self.assertIn("random_U-_R-", md)
        self.assertIn("消融分析", md)

    def test_ablation_markdown_includes_error_and_fallback_columns(self):
        data = json.loads(json.dumps(ABLATION_JSON))
        data["results"][1]["fallback_used"] = True
        input_file = self._write_json("test.json", data)
        reporter = ExperimentReporter(str(input_file))
        md = reporter.generate_markdown()
        self.assertIn("Fallback", md)
        self.assertIn("Error", md)
        self.assertIn("Step failed without recovery", md)
        self.assertIn("yes", md)

    def test_save_to(self):
        input_file = self._write_json("test.json", COMPARATIVE_JSON)
        reporter = ExperimentReporter(str(input_file))
        output_file = str(self.tmp_path / "report.md")
        path = reporter.save_to(output_file)
        self.assertTrue(os.path.exists(path))
        content = open(path).read()
        self.assertIn("对比实验报告", content)


class TestChartGenerator(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = pathlib.Path(self._tmp_dir.name)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def _write_json(self, name, data):
        p = self.tmp_path / name
        p.write_text(json.dumps(data))
        return p

    def test_init(self):
        input_file = self._write_json("test.json", COMPARATIVE_JSON)
        generator = ChartGenerator(str(input_file), output_dir=str(self.tmp_path))
        self.assertEqual(generator._detect_type(), "comparative")

    def test_generate_comparative_charts(self):
        input_file = self._write_json("test.json", COMPARATIVE_JSON)
        generator = ChartGenerator(str(input_file), output_dir=str(self.tmp_path))
        paths = generator.generate_all()
        self.assertGreaterEqual(len(paths), 3)
        for p in paths:
            self.assertTrue(os.path.exists(p))
            self.assertTrue(p.endswith(".png"))

    def test_generate_ablation_charts(self):
        input_file = self._write_json("test.json", ABLATION_JSON)
        generator = ChartGenerator(str(input_file), output_dir=str(self.tmp_path))
        paths = generator.generate_all()
        self.assertGreaterEqual(len(paths), 3)
        for p in paths:
            self.assertTrue(os.path.exists(p))
