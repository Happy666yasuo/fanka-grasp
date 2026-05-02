"""Tests for final demo module."""

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

from src.demo.final_demo import (
    DemoConfig,
    ScenarioResult,
    DemoSession,
    FinalDemo,
    SCENARIO_MULTI_BLOCK,
    SCENARIO_INTERACTIVE,
    SCENARIO_ALL,
    MULTI_BLOCK_TASKS,
    INTERACTIVE_TASKS,
    _mock_llm_call,
)


class TestDemoConfig(unittest.TestCase):
    def test_defaults(self):
        config = DemoConfig()
        self.assertTrue(config.headless)
        self.assertFalse(config.record)
        self.assertEqual(config.max_replans, 3)

    def test_custom(self):
        config = DemoConfig(headless=False, record=True, max_replans=5)
        self.assertFalse(config.headless)
        self.assertTrue(config.record)
        self.assertEqual(config.max_replans, 5)


class TestScenarioResult(unittest.TestCase):
    def test_to_dict(self):
        result = ScenarioResult(
            scenario_name="multi_block",
            task="把红色方块移到绿色区域",
            success=True,
            plan_steps=[{"skill": "pick"}],
            executed_actions=["observe", "pick", "place"],
            explore_steps=1,
            replan_count=0,
            execution_time_seconds=0.5,
            feedback_notes=["Plan executed successfully"],
        )
        d = result.to_dict()
        self.assertEqual(d["scenario"], "multi_block")
        self.assertTrue(d["success"])
        self.assertEqual(len(d["executed_actions"]), 3)

    def test_failed_result(self):
        result = ScenarioResult(
            scenario_name="multi_block",
            task="test",
            success=False,
            error="Test error",
            execution_time_seconds=0.1,
        )
        d = result.to_dict()
        self.assertEqual(d["error"], "Test error")


class TestDemoSession(unittest.TestCase):
    def test_to_dict(self):
        session = DemoSession()
        d = session.to_dict()
        self.assertIn("session_id", d)
        self.assertIn("results", d)
        self.assertIn("summary", d)

    def test_summary_empty(self):
        session = DemoSession()
        summary = session._build_summary()
        self.assertEqual(summary["total_scenarios"], 0)
        self.assertEqual(summary["success_rate"], 0.0)

    def test_summary_with_results(self):
        session = DemoSession()
        session.results = [
            ScenarioResult(
                scenario_name="multi_block", task="task1", success=True,
                execution_time_seconds=0.5,
            ),
            ScenarioResult(
                scenario_name="multi_block", task="task2", success=False,
                error="error", execution_time_seconds=0.3,
            ),
        ]
        summary = session._build_summary()
        self.assertEqual(summary["total_scenarios"], 2)
        self.assertEqual(summary["successful"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["success_rate"], 0.5)


class TestFinalDemo(unittest.TestCase):
    def test_init(self):
        demo = FinalDemo()
        self.assertTrue(demo.config.headless)

    def test_run_multi_block(self):
        demo = FinalDemo(DemoConfig(headless=True, record=False))
        session = demo.run(scenario=SCENARIO_MULTI_BLOCK)
        self.assertEqual(len(session.results), 3)
        for result in session.results:
            self.assertEqual(result.scenario_name, SCENARIO_MULTI_BLOCK)
            self.assertTrue(result.success)

    def test_run_interactive(self):
        demo = FinalDemo(DemoConfig(headless=True, record=False))
        session = demo.run(scenario=SCENARIO_INTERACTIVE)
        self.assertEqual(len(session.results), 1)
        self.assertEqual(session.results[0].scenario_name, SCENARIO_INTERACTIVE)
        self.assertTrue(session.results[0].success)

    def test_run_all(self):
        demo = FinalDemo(DemoConfig(headless=True, record=False))
        session = demo.run(scenario=SCENARIO_ALL)
        self.assertEqual(len(session.results), 4)  # 3 multi-block + 1 interactive
        self.assertTrue(all(r.success for r in session.results))

    def test_record_session(self):
        tmp_path = pathlib.Path(tempfile.mkdtemp())
        try:
            config = DemoConfig(headless=True, record=True, output_dir=str(tmp_path))
            demo = FinalDemo(config)
            session = demo.run(scenario=SCENARIO_MULTI_BLOCK)

            saved_files = list(tmp_path.glob("demo_session_*.json"))
            self.assertEqual(len(saved_files), 1)

            data = json.loads(saved_files[0].read_text())
            self.assertEqual(data["session_id"], session.session_id)
            self.assertEqual(len(data["results"]), 3)
        finally:
            import shutil
            shutil.rmtree(tmp_path)

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
