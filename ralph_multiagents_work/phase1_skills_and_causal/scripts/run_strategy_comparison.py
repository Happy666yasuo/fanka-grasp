#!/usr/bin/env python3
"""Run CausalExplore strategy comparison and generate Markdown report."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.causal_explore.eval_runner import run_strategy_comparison

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    print("Running strategy comparison (8 steps, 2 objects)...")
    report, md_text = run_strategy_comparison(
        max_steps=8,
        output_dir="outputs",
        report_path="outputs/phase1_comparison_report.md",
        random_seed=42,
    )
    print(f"Generated report with {len(report.results)} results")
    print(f"Report saved to outputs/phase1_comparison_report.md")
    print()
    print(md_text)
