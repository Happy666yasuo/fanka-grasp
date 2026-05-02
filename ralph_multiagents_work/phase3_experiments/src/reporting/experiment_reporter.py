"""Experiment reporter: reads structured experiment JSON and produces Markdown reports.

Supports both comparative and ablation experiment output formats.

Usage:
    python -m src.reporting.experiment_reporter --input outputs/experiments/comparative_results.json
    python -m src.reporting.experiment_reporter --input outputs/experiments/ablation_results.json --output outputs/reports/report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ExperimentReporter:
    """Generate Markdown comparison tables from experiment JSON results."""

    def __init__(self, input_path: str) -> None:
        self.input_path = Path(input_path)
        self.data = json.loads(self.input_path.read_text(encoding="utf-8"))

    def generate_markdown(self) -> str:
        experiment_type = self._detect_type()
        if experiment_type == "comparative":
            return self._comparative_report()
        elif experiment_type == "ablation":
            return self._ablation_report()
        else:
            return self._generic_report()

    def _detect_type(self) -> str:
        config = self.data.get("config", {})
        if "conditions" in config and any("no_causal" in str(c) for c in config.get("conditions", [])):
            return "comparative"
        if "dimensions" in config:
            return "ablation"
        return "unknown"

    def _comparative_report(self) -> str:
        lines: list[str] = []
        lines.append("# 对比实验报告")
        lines.append("")
        lines.append("## 实验配置")
        lines.append("")
        config = self.data.get("config", {})
        lines.append(f"- 任务数量: {len(config.get('tasks', []))}")
        lines.append(f"- 实验条件: {', '.join(config.get('conditions', []))}")
        lines.append(f"- Max Replans: {config.get('max_replans', 'N/A')}")
        lines.append("")

        lines.append("## 实验条件说明")
        lines.append("")
        lines.append("| 条件 | 描述 |")
        lines.append("|------|------|")
        lines.append("| **no_causal** | 无 CausalExplore，纯视觉描述给 Planner |")
        lines.append("| **metadata_backed** | 离线预测，从 artifact catalog 加载 CausalExplore 输出 |")
        lines.append("| **simulator_backed** | 在线交互探索，MuJoCo 仿真实时执行探针动作 |")
        lines.append("")

        lines.append("## 详细结果")
        lines.append("")
        lines.append("| 条件 | 任务 | 成功 | 总步数 | 探索步数 | 规划质量 | 重规划次数 | 耗时(s) | Fallback | Error |")
        lines.append("|------|------|------|--------|----------|----------|------------|---------|----------|-------|")

        for r in self.data.get("results", []):
            task_short = r.get("task", "")[:30]
            error_text = str(r.get("error") or "-").replace("|", "\\|")
            fallback_text = "yes" if r.get("fallback_used", False) else "no"
            lines.append(
                f"| {r['condition']} | {task_short} "
                f"| {'✓' if r['success'] else '✗'} "
                f"| {r['total_steps']} "
                f"| {r['explore_steps']} "
                f"| {r['planning_quality']:.3f} "
                f"| {r['replan_count']} "
                f"| {r['execution_time_seconds']:.2f} "
                f"| {fallback_text} "
                f"| {error_text} |"
            )
        lines.append("")

        lines.append("## 汇总统计")
        lines.append("")
        summary = self.data.get("summary", {})
        if summary:
            lines.append("| 条件 | 试验数 | 成功率 | 平均探索步数 | 平均规划质量 | 平均重规划 | 平均耗时(s) |")
            lines.append("|------|--------|--------|--------------|--------------|------------|-------------|")
            for cond, stats in sorted(summary.items()):
                lines.append(
                    f"| {cond} "
                    f"| {stats.get('num_trials', 0)} "
                    f"| {stats.get('success_rate', 0):.2%} "
                    f"| {stats.get('avg_explore_steps', 0):.1f} "
                    f"| {stats.get('avg_planning_quality', 0):.3f} "
                    f"| {stats.get('avg_replan_count', 0):.1f} "
                    f"| {stats.get('avg_execution_time', 0):.2f} |"
                )

        lines.append("")
        lines.append("## 结论")
        lines.append("")
        lines.append(self._generate_comparative_conclusions(summary))

        return "\n".join(lines)

    def _ablation_report(self) -> str:
        lines: list[str] = []
        lines.append("# 消融实验报告")
        lines.append("")
        lines.append("## 实验配置")
        lines.append("")
        config = self.data.get("config", {})
        lines.append(f"- 消融维度: {', '.join(config.get('dimensions', []))}")
        lines.append(f"- 条件数: {config.get('num_conditions', 'N/A')}")
        lines.append(f"- 任务数: {len(config.get('tasks', []))}")
        lines.append("")

        lines.append("## 消融维度说明")
        lines.append("")
        lines.append("| 维度 | 取值 | 说明 |")
        lines.append("|------|------|------|")
        lines.append("| **探索策略** | random / curiosity / causal_explore | 不同探索策略对任务成功率的影响 |")
        lines.append("| **Uncertainty** | U+ / U- | Planner 是否使用 uncertainty handler 进行不确定性感知规划 |")
        lines.append("| **Recovery** | R+ / R- | Executor 是否启用失败恢复 (replan) 机制 |")
        lines.append("")

        lines.append("## 详细结果")
        lines.append("")
        lines.append("| 条件 | 策略 | U | R | 任务 | 成功 | 总步数 | 探索 | 不确定性 | 重规划 | 恢复 | 耗时(s) | Fallback | Error |")
        lines.append("|------|------|---|---|------|------|--------|------|----------|--------|------|---------|----------|-------|")

        for r in self.data.get("results", []):
            task_short = r.get("task", "")[:25]
            error_text = str(r.get("error") or "-").replace("|", "\\|")
            fallback_text = "yes" if r.get("fallback_used", False) else "no"
            lines.append(
                f"| {r['condition_label']} "
                f"| {r['strategy']} "
                f"| {'✓' if r['use_uncertainty'] else '✗'} "
                f"| {'✓' if r['use_recovery'] else '✗'} "
                f"| {task_short} "
                f"| {'✓' if r['success'] else '✗'} "
                f"| {r['total_steps']} "
                f"| {r['explore_steps']} "
                f"| {r['uncertainty_score']:.3f} "
                f"| {r['replan_count']} "
                f"| {r['recovery_count']} "
                f"| {r['execution_time_seconds']:.2f} "
                f"| {fallback_text} "
                f"| {error_text} |"
            )
        lines.append("")

        lines.append("## 汇总统计")
        lines.append("")
        summary = self.data.get("summary", {})
        if summary:
            lines.append("| 条件 | 试验数 | 成功率 | 平均探索 | 平均不确定性 | 平均重规划 | 平均恢复 |")
            lines.append("|------|--------|--------|----------|--------------|------------|----------|")
            for label, stats in sorted(summary.items()):
                lines.append(
                    f"| {label} "
                    f"| {stats.get('num_trials', 0)} "
                    f"| {stats.get('success_rate', 0):.2%} "
                    f"| {stats.get('avg_explore_steps', 0):.1f} "
                    f"| {stats.get('avg_uncertainty', 0):.3f} "
                    f"| {stats.get('avg_replan_count', 0):.1f} "
                    f"| {stats.get('avg_recovery_count', 0):.1f} |"
                )

        lines.append("")
        lines.append("## 消融分析")
        lines.append("")
        lines.append(self._generate_ablation_analysis(summary))

        return "\n".join(lines)

    def _generic_report(self) -> str:
        lines: list[str] = []
        lines.append("# 实验报告")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(self.data, indent=2, ensure_ascii=False))
        lines.append("```")
        return "\n".join(lines)

    @staticmethod
    def _generate_comparative_conclusions(summary: dict[str, Any]) -> str:
        parts: list[str] = []
        if "no_causal" in summary and "simulator_backed" in summary:
            nc_rate = summary["no_causal"]["success_rate"]
            sb_rate = summary["simulator_backed"]["success_rate"]
            delta = sb_rate - nc_rate
            parts.append(
                f"- **CausalExplore 对成功率的影响**: simulator-backed ({sb_rate:.1%}) vs "
                f"no-causal ({nc_rate:.1%}), 提升 {delta:.1%}"
            )

        if "no_causal" in summary and "metadata_backed" in summary:
            nc_quality = summary["no_causal"]["avg_planning_quality"]
            mb_quality = summary["metadata_backed"]["avg_planning_quality"]
            parts.append(
                f"- **规划质量对比**: metadata-backed ({mb_quality:.3f}) vs "
                f"no-causal ({nc_quality:.3f})"
            )

        if "simulator_backed" in summary:
            sb_explore = summary["simulator_backed"]["avg_explore_steps"]
            parts.append(f"- **在线探索代价**: simulator-backed 平均探索步数 = {sb_explore:.1f}")

        if not parts:
            parts.append("- 数据不足以生成结论。")
        return "\n".join(parts)

    @staticmethod
    def _generate_ablation_analysis(summary: dict[str, Any]) -> str:
        parts: list[str] = []

        causal_labels = [l for l in summary if "causal_explore" in l]
        random_labels = [l for l in summary if "random" in l]
        curiosity_labels = [l for l in summary if "curiosity" in l and "causal" not in l]

        if causal_labels and random_labels:
            causal_sr = max(summary[l]["success_rate"] for l in causal_labels)
            random_sr = max(summary[l]["success_rate"] for l in random_labels)
            parts.append(f"- **探索策略影响**: CausalExplore 最优成功率 {causal_sr:.1%} vs Random {random_sr:.1%}")

        u_plus = [l for l in summary if "U+" in l]
        u_minus = [l for l in summary if "U-" in l]
        if u_plus and u_minus:
            u_plus_sr = max(summary[l]["success_rate"] for l in u_plus)
            u_minus_sr = max(summary[l]["success_rate"] for l in u_minus)
            parts.append(f"- **Uncertainty 影响**: U+ 最优成功率 {u_plus_sr:.1%} vs U- {u_minus_sr:.1%}")

        r_plus = [l for l in summary if "R+" in l]
        r_minus = [l for l in summary if "R-" in l]
        if r_plus and r_minus:
            r_plus_sr = max(summary[l]["success_rate"] for l in r_plus)
            r_minus_sr = max(summary[l]["success_rate"] for l in r_minus)
            parts.append(f"- **Recovery 影响**: R+ 最优成功率 {r_plus_sr:.1%} vs R- {r_minus_sr:.1%}")

        if not parts:
            parts.append("- 数据不足以生成消融分析结论。")
        return "\n".join(parts)

    def save_to(self, output_path: str) -> str:
        md_text = self.generate_markdown()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md_text, encoding="utf-8")
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment Report Generator")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to experiment JSON results")
    parser.add_argument("--output", type=str, default=None,
                       help="Output Markdown path (prints to stdout if omitted)")
    args = parser.parse_args()

    reporter = ExperimentReporter(args.input)
    md_text = reporter.generate_markdown()

    if args.output:
        output_path = reporter.save_to(args.output)
        print(f"Report saved to: {output_path}")
    else:
        print(md_text)


if __name__ == "__main__":
    main()
