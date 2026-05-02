"""End-to-end integration tests for Phase 3.

Verifies the full pipeline: experiments -> reporting -> demo.
"""

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

from src.experiments.comparative_experiment import run_comparative_experiment
from src.experiments.ablation_experiment import run_ablation_experiment
from src.reporting.experiment_reporter import ExperimentReporter
from src.reporting.chart_generator import ChartGenerator
from src.demo.final_demo import FinalDemo, DemoConfig, SCENARIO_ALL


class TestIntegrationComparativeToReport(unittest.TestCase):
    """End-to-end: run comparative experiment -> generate report + charts."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = pathlib.Path(self._tmp_dir.name)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_full_comparative_pipeline(self):
        exp_dir = self.tmp_path / "experiments"
        report_dir = self.tmp_path / "reports"
        exp_dir.mkdir()
        report_dir.mkdir()

        # Step 1: Run experiment
        report, json_path, catalog_path = run_comparative_experiment(
            tasks=["把红色方块移到绿色区域"],
            output_dir=str(exp_dir),
            headless=True,
        )
        self.assertTrue(os.path.exists(json_path))
        self.assertEqual(len(report.results), 3)

        # Step 2: Generate markdown report
        reporter = ExperimentReporter(json_path)
        md_text = reporter.generate_markdown()
        md_path = report_dir / "comparative_report.md"
        md_path.write_text(md_text)
        self.assertIn("对比实验报告", md_text)
        self.assertTrue(md_path.exists())

        # Step 3: Generate charts
        chart_gen = ChartGenerator(json_path, output_dir=str(report_dir))
        chart_paths = chart_gen.generate_all()
        self.assertGreaterEqual(len(chart_paths), 3)
        for cp in chart_paths:
            self.assertTrue(os.path.exists(cp))

    def test_full_ablation_pipeline(self):
        exp_dir = self.tmp_path / "experiments"
        report_dir = self.tmp_path / "reports"
        exp_dir.mkdir()
        report_dir.mkdir()

        # Step 1: Run experiment
        report, json_path = run_ablation_experiment(
            tasks=["把红色方块移到绿色区域"],
            output_dir=str(exp_dir),
            headless=True,
        )
        self.assertTrue(os.path.exists(json_path))

        # Step 2: Generate markdown report
        reporter = ExperimentReporter(json_path)
        md_text = reporter.generate_markdown()
        self.assertIn("消融实验报告", md_text)

        # Step 3: Generate charts
        chart_gen = ChartGenerator(json_path, output_dir=str(report_dir))
        chart_paths = chart_gen.generate_all()
        self.assertGreaterEqual(len(chart_paths), 3)
        for cp in chart_paths:
            self.assertTrue(os.path.exists(cp))


class TestIntegrationDemo(unittest.TestCase):
    """End-to-end: run final demo -> verify outputs."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = pathlib.Path(self._tmp_dir.name)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_demo_all_scenarios(self):
        config = DemoConfig(
            headless=True,
            record=True,
            output_dir=str(self.tmp_path),
        )
        demo = FinalDemo(config)
        session = demo.run(scenario=SCENARIO_ALL)

        self.assertEqual(len(session.results), 4)  # 3 multi-block + 1 interactive
        self.assertEqual(session._build_summary()["success_rate"], 1.0)

        # Verify session was recorded
        saved_files = list(self.tmp_path.glob("demo_session_*.json"))
        self.assertEqual(len(saved_files), 1)


class TestIntegrationPhase1Phase2Compat(unittest.TestCase):
    """Verify that Phase 3 code is compatible with Phase 1/2 imports."""

    def test_phase1_imports(self):
        from causal_explore.probe_actions import PROBE_ACTION_REGISTRY, ProbeActionResult
        from causal_explore.probe_executor import ObjectManifest, ProbeExecutor
        from causal_explore.explore_strategies import (
            RandomStrategy, CuriosityDrivenStrategy, CausalExploreStrategy, STRATEGY_REGISTRY,
        )
        self.assertGreater(len(PROBE_ACTION_REGISTRY), 0)
        self.assertGreater(len(STRATEGY_REGISTRY), 0)

    def test_phase2_imports(self):
        from planner.llm_planner import LLMPlanner
        from planner.prompt_templates import build_system_prompt, build_task_prompt
        from planner.uncertainty_handler import UncertaintyHandler
        from planner.replan_handler import ReplanHandler
        self.assertIsNotNone(LLMPlanner)
        self.assertIsNotNone(UncertaintyHandler)
        self.assertIsNotNone(ReplanHandler)

    def test_embodied_agent_imports(self):
        from embodied_agent.contracts import (
            PlannerStep, CausalExploreOutput, ExecutorResult, PropertyBelief,
        )
        from embodied_agent.types import PlanStep, WorldState
        from embodied_agent.planning_bridge import ContractPlanningBridge
        from embodied_agent.planner_contracts import ContractPlannerAdapter
        self.assertIsNotNone(PlannerStep)
        self.assertIsNotNone(ContractPlanningBridge)


class TestIntegrationPaperMaterials(unittest.TestCase):
    """Verify paper materials exist and are well-formed."""

    def test_paper_draft_exists(self):
        paper_dir = os.path.join(
            os.path.dirname(__file__), "..", "outputs", "paper"
        )
        self.assertTrue(os.path.exists(os.path.join(paper_dir, "paper_draft.md")))
        self.assertTrue(os.path.exists(os.path.join(paper_dir, "presentation_outline.md")))
        self.assertTrue(os.path.exists(os.path.join(paper_dir, "architecture_diagram.md")))
        self.assertTrue(os.path.exists(os.path.join(paper_dir, "experiment_summary.md")))

    def test_paper_draft_content(self):
        paper_dir = os.path.join(
            os.path.dirname(__file__), "..", "outputs", "paper"
        )
        draft = open(os.path.join(paper_dir, "paper_draft.md")).read()
        self.assertIn("摘要", draft)
        self.assertIn("CausalExplore", draft)
        self.assertIn("LLM", draft)
        self.assertIn("实验", draft)

    def test_presentation_outline_content(self):
        paper_dir = os.path.join(
            os.path.dirname(__file__), "..", "outputs", "paper"
        )
        outline = open(os.path.join(paper_dir, "presentation_outline.md")).read()
        self.assertIn("Slide", outline)
        self.assertIn("答辩", outline)


class TestFullPhase3Acceptance(unittest.TestCase):
    """Acceptance test: run experiments, generate reports, run demo, verify outputs."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = pathlib.Path(self._tmp_dir.name)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_full_phase3_acceptance(self):
        exp_dir = self.tmp_path / "experiments"
        report_dir = self.tmp_path / "reports"
        demo_dir = self.tmp_path / "demo"
        for d in [exp_dir, report_dir, demo_dir]:
            d.mkdir()

        # 1. Comparative experiment
        comp_report, comp_json, _ = run_comparative_experiment(
            tasks=["把红色方块移到绿色区域"],
            output_dir=str(exp_dir),
            headless=True,
        )
        self.assertEqual(len(comp_report.results), 3)

        # 2. Ablation experiment
        abl_report, abl_json = run_ablation_experiment(
            tasks=["把红色方块移到绿色区域"],
            output_dir=str(exp_dir),
            headless=True,
        )
        self.assertEqual(len(abl_report.results), 12)

        # 3. Generate reports
        for json_path in [comp_json, abl_json]:
            reporter = ExperimentReporter(json_path)
            md = reporter.generate_markdown()
            self.assertGreater(len(md), 100)

            chart_gen = ChartGenerator(json_path, output_dir=str(report_dir))
            charts = chart_gen.generate_all()
            self.assertGreaterEqual(len(charts), 3)

        # 4. Run demo
        config = DemoConfig(headless=True, record=True, output_dir=str(demo_dir))
        demo = FinalDemo(config)
        session = demo.run(scenario=SCENARIO_ALL)
        self.assertEqual(session._build_summary()["success_rate"], 1.0)

        # 5. Verify all outputs exist
        self.assertTrue(os.path.exists(comp_json))
        self.assertTrue(os.path.exists(abl_json))
        self.assertGreaterEqual(len(list(report_dir.glob("*.png"))), 5)
        self.assertEqual(len(list(demo_dir.glob("demo_session_*.json"))), 1)
