"""Chart generator: produces matplotlib charts from experiment JSON results.

Supports --headless mode (no GUI, saves to file).
Generates: success rate bar chart, explore steps distribution, uncertainty descent curves.

Usage:
    python -m src.reporting.chart_generator --input outputs/experiments/comparative_results.json
    python -m src.reporting.chart_generator --input outputs/experiments/ablation_results.json --headless
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless-safe backend

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


class ChartGenerator:
    """Generate matplotlib charts from experiment data."""

    def __init__(
        self,
        input_path: str,
        output_dir: str = "outputs/reports",
        headless: bool = True,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.data = json.loads(self.input_path.read_text(encoding="utf-8"))

        plt.rcParams.update({
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 150,
        })

    def generate_all(self) -> list[str]:
        experiment_type = self._detect_type()
        paths: list[str] = []

        paths.append(self._generate_success_rate_chart(experiment_type))
        paths.append(self._generate_explore_steps_chart(experiment_type))
        paths.append(self._generate_uncertainty_chart(experiment_type))

        if experiment_type == "comparative":
            paths.append(self._generate_planning_quality_chart())
        elif experiment_type == "ablation":
            paths.append(self._generate_replan_comparison_chart())

        return paths

    def _chart_path(self, experiment_type: str, stem: str) -> str:
        prefix = experiment_type if experiment_type in {"comparative", "ablation"} else "experiment"
        return str(self.output_dir / f"{prefix}_{stem}.png")

    def _detect_type(self) -> str:
        config = self.data.get("config", {})
        if "dimensions" in config:
            return "ablation"
        conditions = config.get("conditions", [])
        if any("no_causal" in str(c) for c in conditions):
            return "comparative"
        return "unknown"

    def _generate_success_rate_chart(self, experiment_type: str) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))

        if experiment_type == "comparative":
            self._bar_chart_comparative(ax, metric="success_rate")
            ax.set_title("Comparative Experiment: Success Rate by Condition")
        else:
            self._bar_chart_ablation(ax, metric="success_rate")
            ax.set_title("Ablation Experiment: Success Rate by Condition")

        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        path = self._chart_path(experiment_type, "success_rate_comparison")
        fig.savefig(path)
        plt.close(fig)
        return path

    def _generate_explore_steps_chart(self, experiment_type: str) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))

        if experiment_type == "comparative":
            self._bar_chart_comparative(ax, metric="avg_explore_steps")
            ax.set_title("Comparative Experiment: Average Explore Steps")
            ax.set_ylabel("Explore Steps")
        else:
            self._bar_chart_ablation(ax, metric="avg_explore_steps")
            ax.set_title("Ablation Experiment: Average Explore Steps")
            ax.set_ylabel("Explore Steps")

        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        path = self._chart_path(experiment_type, "explore_steps_distribution")
        fig.savefig(path)
        plt.close(fig)
        return path

    def _generate_uncertainty_chart(self, experiment_type: str) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))

        if experiment_type == "comparative":
            summary = self.data.get("summary", {})
            conditions = list(summary.keys())
            uncertainties = []
            for cond in conditions:
                cond_results = [r for r in self.data.get("results", []) if r["condition"] == cond]
                uncertainties.append(np.mean([r.get("planning_quality", 0) for r in cond_results]))

            colors = ["#e74c3c", "#3498db", "#2ecc71"]
            bars = ax.bar(conditions, uncertainties, color=colors[:len(conditions)])
            ax.set_title("Comparative Experiment: Planning Quality (Uncertainty Proxy)")
            ax.set_ylabel("Planning Quality Score")
            ax.set_ylim(0, 1.1)
            for bar, val in zip(bars, uncertainties):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        else:
            summary = self.data.get("summary", {})
            labels = sorted(summary.keys())
            values = [summary[l].get("avg_uncertainty", 0) for l in labels]

            n = len(labels)
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, n))
            bars = ax.bar(range(n), values, color=colors)
            ax.set_xticks(range(n))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_title("Ablation Experiment: Average Uncertainty Score")
            ax.set_ylabel("Uncertainty Score")
            ax.set_ylim(0, max(values) * 1.2 if values else 1.0)

        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        path = self._chart_path(experiment_type, "uncertainty_descent")
        fig.savefig(path)
        plt.close(fig)
        return path

    def _generate_planning_quality_chart(self) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))
        self._bar_chart_comparative(ax, metric="avg_planning_quality")
        ax.set_title("Comparative Experiment: Planning Quality by Condition")
        ax.set_ylabel("Planning Quality Score")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        path = self._chart_path("comparative", "planning_quality_comparison")
        fig.savefig(path)
        plt.close(fig)
        return path

    def _generate_replan_comparison_chart(self) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        summary = self.data.get("summary", {})
        labels = sorted(summary.keys())
        replans = [summary[l].get("avg_replan_count", 0) for l in labels]
        recoveries = [summary[l].get("avg_recovery_count", 0) for l in labels]

        n = len(labels)
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, n))

        axes[0].bar(range(n), replans, color=colors)
        axes[0].set_xticks(range(n))
        axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axes[0].set_title("Average Replan Count")
        axes[0].set_ylabel("Replans")
        axes[0].grid(axis="y", alpha=0.3)

        axes[1].bar(range(n), recoveries, color=colors)
        axes[1].set_xticks(range(n))
        axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axes[1].set_title("Average Recovery Count")
        axes[1].set_ylabel("Recoveries")
        axes[1].grid(axis="y", alpha=0.3)

        fig.suptitle("Ablation Experiment: Replan & Recovery Analysis")
        fig.tight_layout()

        path = self._chart_path("ablation", "replan_recovery_comparison")
        fig.savefig(path)
        plt.close(fig)
        return path

    def _bar_chart_comparative(self, ax: plt.Axes, metric: str) -> None:
        summary = self.data.get("summary", {})
        condition_order = ["no_causal", "metadata_backed", "simulator_backed"]
        labels = [c for c in condition_order if c in summary]
        values = [summary[l].get(metric, 0) for l in labels]
        colors = ["#e74c3c", "#f39c12", "#2ecc71"]

        bars = ax.bar(labels, values, color=colors[:len(labels)])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right")

    def _bar_chart_ablation(self, ax: plt.Axes, metric: str) -> None:
        summary = self.data.get("summary", {})
        labels = sorted(summary.keys())
        values = [summary[l].get(metric, 0) for l in labels]

        n = len(labels)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, n))
        bars = ax.bar(range(n), values, color=colors)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment Chart Generator")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to experiment JSON results")
    parser.add_argument("--output-dir", type=str, default="outputs/reports")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    generator = ChartGenerator(
        input_path=args.input,
        output_dir=args.output_dir,
        headless=args.headless,
    )

    print(f"Generating charts from: {args.input}")
    paths = generator.generate_all()
    for p in paths:
        print(f"  Saved: {p}")
    print(f"\nGenerated {len(paths)} charts.")


if __name__ == "__main__":
    main()
