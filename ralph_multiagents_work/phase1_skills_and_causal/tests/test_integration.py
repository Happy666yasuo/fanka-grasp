from __future__ import annotations

import sys
import os
import unittest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src')))

from embodied_agent.contracts import PlannerStep, ExecutorResult
from embodied_agent.simulator import create_pick_place_simulation

from src.skills.skill_registry import SkillRegistry
from src.skills.skill_executor import UnifiedSkillExecutor
from src.causal_explore.probe_actions import PROBE_ACTION_REGISTRY, ProbeActionResult
from src.causal_explore.probe_executor import ProbeExecutor, ObjectManifest
from src.causal_explore.explore_strategies import (
    RandomStrategy,
    CuriosityDrivenStrategy,
    CausalExploreStrategy,
    ExploreHistory,
    ExploreStep,
)
from src.causal_explore.eval_runner import (
    MultiStrategyEvalRunner,
    generate_comparison_report,
    ComparisonReport,
    StrategyEvalResult,
)


class TestSkillRegistrationToProbeExecution(unittest.TestCase):
    """End-to-end test: skill registration -> probe execution -> strategy evaluation."""

    def setUp(self):
        self.registry = SkillRegistry()

    def test_registry_contains_all_required_skills(self):
        required = ["pick", "place", "press", "push", "pull", "rotate"]
        for skill_name in required:
            self.assertIn(skill_name, self.registry)

    def test_all_probe_actions_registered(self):
        self.assertIn("lateral_push", PROBE_ACTION_REGISTRY)
        self.assertIn("top_press", PROBE_ACTION_REGISTRY)
        self.assertIn("side_pull", PROBE_ACTION_REGISTRY)
        self.assertIn("surface_tap", PROBE_ACTION_REGISTRY)
        self.assertIn("grasp_attempt", PROBE_ACTION_REGISTRY)

    def test_probe_executor_executes_all_probes(self):
        sim = create_pick_place_simulation(backend="mujoco", gui=False)
        try:
            executor = ProbeExecutor(sim, output_dir=tempfile.mkdtemp())
            results = executor.run_probe_sequence("red_block")
            self.assertEqual(len(results), 5)
            for r in results:
                self.assertIsInstance(r, ProbeActionResult)
                self.assertTrue(len(r.probe_name) > 0)
        finally:
            sim.shutdown()

    def test_probe_executor_builds_causal_output(self):
        sim = create_pick_place_simulation(backend="mujoco", gui=False)
        try:
            executor = ProbeExecutor(sim, output_dir=tempfile.mkdtemp())
            manifest = ObjectManifest(
                object_id="red_block",
                object_category="block",
                expected_properties=["movable", "pressable", "graspable"],
                candidate_affordances=["pushable", "graspable"],
            )
            results = executor.run_probe_sequence("red_block")
            output = executor.build_causal_output(manifest, results)
            self.assertEqual(output.object_id, "red_block")
            self.assertIn("movable", output.property_belief)
            self.assertGreater(len(output.affordance_candidates), 0)
        finally:
            sim.shutdown()

    def test_strategies_produce_valid_selections(self):
        sim = create_pick_place_simulation(backend="mujoco", gui=False)
        try:
            executor = ProbeExecutor(sim, output_dir=tempfile.mkdtemp())
            probes = list(PROBE_ACTION_REGISTRY.keys())
            objects = ["red_block"]
            history = ExploreHistory()

            random_strat = RandomStrategy()
            probe, obj = random_strat.select_next(history, probes, objects)
            self.assertIn(probe, probes)
            self.assertIn(obj, objects)

            curiosity_strat = CuriosityDrivenStrategy()
            probe, obj = curiosity_strat.select_next(history, probes, objects)
            self.assertIn(probe, probes)
            self.assertIn(obj, objects)

            causal_strat = CausalExploreStrategy(executor)
            probe, obj = causal_strat.select_next(history, probes, objects)
            self.assertIn(probe, probes)
            self.assertIn(obj, objects)
        finally:
            sim.shutdown()

    def test_eval_runner_generates_report(self):
        sim = create_pick_place_simulation(backend="mujoco", gui=False)
        try:
            manifests = [
                ObjectManifest(
                    object_id="red_block",
                    object_category="block",
                    expected_properties=["movable", "pressable"],
                    candidate_affordances=["pushable", "graspable"],
                ),
            ]
            runner = MultiStrategyEvalRunner(
                manifests=manifests,
                max_steps_per_object=4,
                random_seed=42,
            )
            report = runner.run_all()
            self.assertGreater(len(report.results), 0)

            json_path = runner.save_results(report)
            self.assertTrue(os.path.exists(json_path))

            md_text = generate_comparison_report(report, json_path=json_path)
            self.assertIn("CausalExplore", md_text)
            self.assertIn("策略对比报告", md_text)
        finally:
            sim.shutdown()


class TestExecutorIntegration(unittest.TestCase):
    """Test UnifiedSkillExecutor with PlannerStep for all skill types."""

    def setUp(self):
        self.registry = SkillRegistry()
        self.sim = create_pick_place_simulation(backend="mujoco", gui=False)

    def tearDown(self):
        self.sim.shutdown()

    def test_executor_resolves_pick_and_place(self):
        executor = UnifiedSkillExecutor(self.sim, self.registry)
        pick_step = PlannerStep(
            task_id="test_pick",
            step_index=0,
            selected_skill="pick",
            target_object="red_block",
        )
        result = executor.execute(pick_step)
        self.assertTrue(result.success)

        place_step = PlannerStep(
            task_id="test_place",
            step_index=1,
            selected_skill="place",
            target_object="red_block",
            skill_args={"target_zone": "green_zone"},
        )
        result = executor.execute(place_step)
        self.assertTrue(result.success)

    def test_executor_resolves_press_skill(self):
        executor = UnifiedSkillExecutor(self.sim, self.registry)
        step = PlannerStep(
            task_id="test_press",
            step_index=0,
            selected_skill="press",
            target_object="red_block",
            skill_args={"press_direction": "down", "force": 0.5},
        )
        result = executor.execute(step)
        self.assertIsInstance(result, ExecutorResult)

    def test_execute_all_stops_on_failure(self):
        executor = UnifiedSkillExecutor(self.sim, self.registry)
        steps = [
            PlannerStep(
                task_id="test_push", step_index=0,
                selected_skill="push", target_object="red_block",
                skill_args={"push_direction": "forward", "distance": 0.10},
            ),
            PlannerStep(
                task_id="test_pick_nonexistent", step_index=1,
                selected_skill="pick", target_object="nonexistent_object",
            ),
            PlannerStep(
                task_id="test_pull", step_index=2,
                selected_skill="pull", target_object="red_block",
            ),
        ]
        results = executor.execute_all(steps)
        self.assertEqual(len(results), 2)

    def test_headless_mode(self):
        executor = UnifiedSkillExecutor(self.sim, self.registry, headless=True)
        self.assertTrue(executor.headless)


class TestEndToEndPipeline(unittest.TestCase):
    """Full pipeline: skills -> executor -> probes -> strategies -> report."""

    def test_full_pipeline_with_causal_output(self):
        sim = create_pick_place_simulation(backend="mujoco", gui=False)
        try:
            # Step 1: Skills
            registry = SkillRegistry()
            self.assertGreater(len(registry), 0)

            # Step 2: Executor runs pick then place
            executor = UnifiedSkillExecutor(sim, registry)
            pick_result = executor.execute(PlannerStep(
                task_id="pipe", step_index=0,
                selected_skill="pick", target_object="red_block",
            ))
            self.assertTrue(pick_result.success)

            place_result = executor.execute(PlannerStep(
                task_id="pipe", step_index=1,
                selected_skill="place", target_object="red_block",
                skill_args={"target_zone": "green_zone"},
            ))
            self.assertTrue(place_result.success)

            # Step 3: CausalExplore probes
            probe_executor = ProbeExecutor(sim, output_dir=tempfile.mkdtemp())
            probe_results = probe_executor.run_probe_sequence("red_block")
            self.assertEqual(len(probe_results), 5)

            # Step 4: Build CausalExploreOutput from probe results
            manifest = ObjectManifest(
                object_id="red_block",
                object_category="block",
                expected_properties=["movable", "pressable", "graspable", "rigid"],
                candidate_affordances=["pushable", "pressable", "graspable", "tappable"],
            )
            causal_output = probe_executor.build_causal_output(manifest, probe_results)
            self.assertIsNotNone(causal_output.uncertainty_score)

            # Step 5: Verify artifact can be saved
            artifact_path = probe_executor.save_artifact(causal_output, probe_results)
            self.assertTrue(os.path.exists(artifact_path))
        finally:
            sim.shutdown()

    def test_comparison_report_produces_valid_markdown(self):
        report = ComparisonReport(
            results=[
                StrategyEvalResult(
                    strategy_name="random", object_id="red_block",
                    total_steps=4,
                    property_beliefs={"movable": 0.6, "graspable": 0.7},
                    affordance_candidates={"pushable": 0.5, "graspable": 0.8},
                    avg_displacement=0.015, unique_probes_used=3, wall_time_seconds=1.0,
                ),
                StrategyEvalResult(
                    strategy_name="curiosity", object_id="red_block",
                    total_steps=4,
                    property_beliefs={"movable": 0.7, "graspable": 0.8},
                    affordance_candidates={"pushable": 0.7, "graspable": 0.85},
                    avg_displacement=0.025, unique_probes_used=4, wall_time_seconds=1.5,
                ),
                StrategyEvalResult(
                    strategy_name="causal_explore", object_id="red_block",
                    total_steps=4,
                    property_beliefs={"movable": 0.8, "graspable": 0.85},
                    affordance_candidates={"pushable": 0.8, "graspable": 0.9},
                    avg_displacement=0.030, unique_probes_used=5, wall_time_seconds=2.0,
                ),
            ],
            manifests=[
                ObjectManifest(object_id="red_block", object_category="block",
                               expected_properties=["movable", "graspable"],
                               candidate_affordances=["pushable", "graspable"]),
            ],
        )
        md_text = generate_comparison_report(report)
        self.assertIn("Random", md_text)
        self.assertIn("Curiosity-Driven", md_text)
        self.assertIn("CausalExplore", md_text)
        self.assertIn("定量对比", md_text)
        self.assertIn("结论", md_text)


if __name__ == "__main__":
    unittest.main()
