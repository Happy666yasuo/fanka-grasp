from __future__ import annotations

import json
import math
import random
import sys
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../embodied_agent/src')))
from embodied_agent.mujoco_simulator import MujocoPickPlaceSimulation
from embodied_agent.simulator import create_pick_place_simulation

from .probe_actions import PROBE_ACTION_REGISTRY, ProbeActionResult
from .probe_executor import ObjectManifest, ProbeExecutor
from .explore_strategies import (
    BaseExploreStrategy,
    ExploreHistory,
    ExploreStep,
    RandomStrategy,
    CuriosityDrivenStrategy,
    CausalExploreStrategy,
)


@dataclass
class StrategyEvalResult:
    strategy_name: str
    object_id: str
    total_steps: int
    property_beliefs: dict[str, float]
    affordance_candidates: dict[str, float]
    avg_displacement: float
    unique_probes_used: int
    wall_time_seconds: float
    step_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ComparisonReport:
    results: list[StrategyEvalResult] = field(default_factory=list)
    manifests: list[ObjectManifest] = field(default_factory=list)

    def property_accuracy(self, result: StrategyEvalResult) -> float:
        beliefs = result.property_beliefs
        if not beliefs:
            return 0.0
        return sum(beliefs.values()) / len(beliefs)

    def affordance_accuracy(self, result: StrategyEvalResult) -> float:
        candidates = result.affordance_candidates
        if not candidates:
            return 0.0
        return sum(candidates.values()) / len(candidates)


DEFAULT_MANIFESTS = [
    ObjectManifest(
        object_id="red_block",
        object_category="block",
        expected_properties=["movable", "pressable", "graspable", "rigid"],
        candidate_affordances=["pushable", "pressable", "pullable", "graspable", "tappable"],
    ),
    ObjectManifest(
        object_id="blue_block",
        object_category="block",
        expected_properties=["movable", "pressable", "graspable", "rigid"],
        candidate_affordances=["pushable", "pressable", "pullable", "graspable", "tappable"],
    ),
]


PROBE_ORDER = ["lateral_push", "top_press", "side_pull", "surface_tap", "grasp_attempt"]


class MultiStrategyEvalRunner:
    """Run multiple exploration strategies on the same object manifests and compare."""

    def __init__(
        self,
        manifests: list[ObjectManifest] | None = None,
        max_steps_per_object: int = 8,
        output_dir: str | None = None,
        random_seed: int = 42,
    ) -> None:
        self.manifests = manifests or DEFAULT_MANIFESTS
        self.max_steps = max_steps_per_object
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(random_seed)

    def run_all(self) -> ComparisonReport:
        report = ComparisonReport(manifests=list(self.manifests))

        for manifest in self.manifests:
            sim = create_pick_place_simulation(backend="mujoco", gui=False)
            try:
                probe_executor = ProbeExecutor(sim, output_dir=str(self.output_dir))

                strategies: dict[str, BaseExploreStrategy] = {
                    "random": RandomStrategy(),
                    "curiosity": CuriosityDrivenStrategy(),
                    "causal_explore": CausalExploreStrategy(probe_executor),
                }

                for name, strategy in strategies.items():
                    result = self._run_strategy(
                        strategy_name=name,
                        strategy=strategy,
                        manifest=manifest,
                        simulation=sim,
                        probe_executor=probe_executor,
                    )
                    report.results.append(result)
            finally:
                sim.shutdown()

        return report

    def _run_strategy(
        self,
        strategy_name: str,
        strategy: BaseExploreStrategy,
        manifest: ObjectManifest,
        simulation: MujocoPickPlaceSimulation,
        probe_executor: ProbeExecutor,
    ) -> StrategyEvalResult:
        history = ExploreHistory()
        available_probes = list(PROBE_ACTION_REGISTRY.keys())
        available_objects = [manifest.object_id]

        start_time = time.monotonic()

        for step_idx in range(self.max_steps):
            probe_name, obj_name = strategy.select_next(
                history, available_probes, available_objects,
            )

            try:
                simulation.fast_reset()
                result = probe_executor.execute_probe(probe_name, obj_name)
            except Exception as exc:
                result = ProbeActionResult(
                    probe_name=probe_name,
                    object_name=obj_name,
                    success=False,
                    pre_position=(0, 0, 0),
                    post_position=(0, 0, 0),
                    displacement=(0, 0, 0),
                    displacement_magnitude=0.0,
                    contact_detected=False,
                    observations={"error": str(exc)},
                )

            step = ExploreStep(
                probe_name=probe_name,
                object_name=obj_name,
                result=result,
                step_index=step_idx,
            )
            history.record(step)

        wall_time = time.monotonic() - start_time

        all_results = [s.result for s in history.steps]

        causal_output = probe_executor.build_causal_output(manifest, all_results)

        property_beliefs = {
            label: belief.confidence
            for label, belief in causal_output.property_belief.items()
        }
        affordance_candidates = {
            c.name: c.confidence for c in causal_output.affordance_candidates
        }

        avg_disp = (
            sum(r.displacement_magnitude for r in all_results) / len(all_results)
            if all_results else 0.0
        )
        unique_probes = len({s.probe_name for s in history.steps})

        return StrategyEvalResult(
            strategy_name=strategy_name,
            object_id=manifest.object_id,
            total_steps=self.max_steps,
            property_beliefs=property_beliefs,
            affordance_candidates=affordance_candidates,
            avg_displacement=avg_disp,
            unique_probes_used=unique_probes,
            wall_time_seconds=wall_time,
            step_results=[s.result.to_dict() for s in history.steps],
        )

    def save_results(self, report: ComparisonReport) -> str:
        data = {
            "manifests": [m.to_dict() for m in self.manifests],
            "results": [
                {
                    "strategy_name": r.strategy_name,
                    "object_id": r.object_id,
                    "total_steps": r.total_steps,
                    "property_beliefs": r.property_beliefs,
                    "affordance_candidates": r.affordance_candidates,
                    "property_accuracy": report.property_accuracy(r),
                    "affordance_accuracy": report.affordance_accuracy(r),
                    "avg_displacement": r.avg_displacement,
                    "unique_probes_used": r.unique_probes_used,
                    "wall_time_seconds": r.wall_time_seconds,
                    "step_results": r.step_results,
                }
                for r in report.results
            ],
        }
        output_path = self.output_dir / "strategy_comparison.json"
        output_path.write_text(json.dumps(data, indent=2, default=str))
        return str(output_path)


def generate_comparison_report(report: ComparisonReport, json_path: str | None = None) -> str:
    """Generate a Markdown comparison report from evaluation results."""

    lines: list[str] = []
    lines.append("# CausalExplore 探索策略对比报告")
    lines.append("")
    lines.append("## 概述")
    lines.append("")
    lines.append(
        f"本报告对比了 **Random**、**Curiosity-Driven**、**CausalExplore** "
        f"三种探索策略在 {len(report.manifests)} 个物体上的表现。"
    )
    lines.append("")

    lines.append("## 物体清单")
    lines.append("")
    for m in report.manifests:
        lines.append(f"- **{m.object_id}** (类别: {m.object_category})")
        lines.append(f"  - 预期属性: {', '.join(m.expected_properties)}")
        lines.append(f"  - 候选用途: {', '.join(m.candidate_affordances)}")
    lines.append("")

    lines.append("## 策略描述")
    lines.append("")
    lines.append("| 策略 | 描述 |")
    lines.append("|------|------|")
    lines.append("| **Random** | 均匀随机选择 (探针, 物体) 对，无历史信息利用 |")
    lines.append("| **Curiosity-Driven** | 基于位移幅度的启发式好奇心，优先采样产生大位移的探针-物体对 |")
    lines.append("| **CausalExplore** | 利用不确定性估计驱动探索，优先降低 Property/Affordance 信念不确定性 |")
    lines.append("")

    lines.append("## 定量对比")
    lines.append("")

    lines.append("| 策略 | 物体 | 步数 | 属性推断准确率 | 用途推断准确率 | 平均位移(m) | 唯一探针数 | 耗时(s) |")
    lines.append("|------|------|------|----------------|----------------|-------------|------------|---------|")
    for r in report.results:
        p_acc = report.property_accuracy(r)
        a_acc = report.affordance_accuracy(r)
        lines.append(
            f"| {r.strategy_name} | {r.object_id} | {r.total_steps} "
            f"| {p_acc:.3f} | {a_acc:.3f} "
            f"| {r.avg_displacement:.4f} | {r.unique_probes_used} "
            f"| {r.wall_time_seconds:.2f} |"
        )
    lines.append("")

    strategy_names = sorted({r.strategy_name for r in report.results})
    lines.append("### 聚合指标（按策略平均）")
    lines.append("")
    lines.append("| 策略 | 属性推断准确率 | 用途推断准确率 | 平均位移(m) | 平均步数 |")
    lines.append("|------|----------------|----------------|-------------|----------|")
    for name in strategy_names:
        subset = [r for r in report.results if r.strategy_name == name]
        avg_p = sum(report.property_accuracy(r) for r in subset) / len(subset)
        avg_a = sum(report.affordance_accuracy(r) for r in subset) / len(subset)
        avg_d = sum(r.avg_displacement for r in subset) / len(subset)
        avg_s = sum(r.total_steps for r in subset) / len(subset)
        lines.append(f"| {name} | {avg_p:.3f} | {avg_a:.3f} | {avg_d:.4f} | {avg_s:.1f} |")
    lines.append("")

    lines.append("## 结论")
    lines.append("")

    causal_results = [r for r in report.results if r.strategy_name == "causal_explore"]
    curiosity_results = [r for r in report.results if r.strategy_name == "curiosity"]
    random_results = [r for r in report.results if r.strategy_name == "random"]

    if causal_results:
        avg_causal_p = sum(report.property_accuracy(r) for r in causal_results) / len(causal_results)
        avg_causal_a = sum(report.affordance_accuracy(r) for r in causal_results) / len(causal_results)
        lines.append(f"- **CausalExplore** 属性推断准确率: {avg_causal_p:.3f}, 用途推断准确率: {avg_causal_a:.3f}")

    if curiosity_results:
        avg_cur_p = sum(report.property_accuracy(r) for r in curiosity_results) / len(curiosity_results)
        avg_cur_a = sum(report.affordance_accuracy(r) for r in curiosity_results) / len(curiosity_results)
        lines.append(f"- **Curiosity-Driven** 属性推断准确率: {avg_cur_p:.3f}, 用途推断准确率: {avg_cur_a:.3f}")

    if random_results:
        avg_rand_p = sum(report.property_accuracy(r) for r in random_results) / len(random_results)
        avg_rand_a = sum(report.affordance_accuracy(r) for r in random_results) / len(random_results)
        lines.append(f"- **Random** 属性推断准确率: {avg_rand_p:.3f}, 用途推断准确率: {avg_rand_a:.3f}")

    lines.append("")

    if json_path:
        lines.append(f"详细结果已保存至: `{json_path}`")
        lines.append("")

    return "\n".join(lines)


def run_strategy_comparison(
    manifests: list[ObjectManifest] | None = None,
    max_steps: int = 8,
    output_dir: str | None = None,
    report_path: str | None = None,
    random_seed: int = 42,
) -> tuple[ComparisonReport, str]:
    """Convenience function: run comparison and return (report, markdown_text)."""
    runner = MultiStrategyEvalRunner(
        manifests=manifests,
        max_steps_per_object=max_steps,
        output_dir=output_dir,
        random_seed=random_seed,
    )
    report = runner.run_all()
    json_path = runner.save_results(report)
    md_text = generate_comparison_report(report, json_path=json_path)

    if report_path:
        Path(report_path).write_text(md_text)

    return report, md_text
