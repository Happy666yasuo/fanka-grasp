# Franka Cube Grasp — Experiment Result Plotter
# Conda env: beyondmimic (Python 3.10) or any env with matplotlib + tensorboard
"""
Generate comparison charts from TensorBoard logs of the 6-experiment matrix.

Reads TensorBoard event files and produces:
    1. Training reward curves (all 6 experiments)
    2. Success rate comparison (if logged)
    3. Episode length comparison
    4. Summary bar chart

Usage:
    conda activate beyondmimic
    cd ~/Desktop/beyondmimic/my_practice/franka_cube_grasp
    python scripts/plot_results.py --log_dir logs/experiments
    python scripts/plot_results.py --log_dir logs/experiments --output_dir plots/

Output:
    plots/reward_curves.png
    plots/success_rate.png
    plots/episode_length.png
    plots/summary_bar.png
"""
from __future__ import annotations

import argparse
import os
import glob
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend (no display needed)
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not found. Install with: pip install matplotlib")

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("[WARNING] tensorboard not found. Install with: pip install tensorboard")


# ============================================================================
# TensorBoard log reader
# ============================================================================

def read_tb_scalar(log_dir: str, tag: str) -> Tuple[List[int], List[float]]:
    """Read a scalar tag from TensorBoard event files.

    Args:
        log_dir: Directory containing event files (searches recursively).
        tag: The scalar tag name (e.g., 'rollout/ep_rew_mean').

    Returns:
        (steps, values) — lists of step numbers and corresponding values.
    """
    # Find event files recursively
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        return [], []

    # Use the directory containing the first event file
    event_dir = os.path.dirname(event_files[0])
    ea = EventAccumulator(event_dir)
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        return [], []

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def read_all_experiments(
    log_base: str,
    tags: List[str],
) -> Dict[str, Dict[str, Tuple[List[int], List[float]]]]:
    """Read scalar data from all experiment directories.

    Args:
        log_base: Base directory containing experiment subdirectories.
        tags: List of TensorBoard scalar tags to extract.

    Returns:
        {exp_name: {tag: (steps, values)}}
    """
    results = {}

    # Find experiment directories (e.g., exp-01_sac_sparse/)
    exp_dirs = sorted(glob.glob(os.path.join(log_base, "exp-*")))
    if not exp_dirs:
        # Try looking for any subdirectories
        exp_dirs = sorted([
            d for d in glob.glob(os.path.join(log_base, "*"))
            if os.path.isdir(d)
        ])

    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        results[exp_name] = {}
        for tag in tags:
            steps, values = read_tb_scalar(exp_dir, tag)
            if steps:
                results[exp_name][tag] = (steps, values)

    return results


# ============================================================================
# Smoothing utility
# ============================================================================

def smooth(values: List[float], window: int = 10) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(values) < window:
        return np.array(values)
    arr = np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


# ============================================================================
# Plot functions
# ============================================================================

# Color map for experiments
EXP_COLORS = {
    "exp-01": "#1f77b4",  # blue  — SAC sparse
    "exp-02": "#ff7f0e",  # orange — SAC shaped
    "exp-03": "#2ca02c",  # green — SAC curriculum
    "exp-04": "#d62728",  # red   — SAC PBRS
    "exp-05": "#9467bd",  # purple — SAC+HER sparse
    "exp-06": "#8c564b",  # brown — SAC+HER shaped
}

EXP_LABELS = {
    "exp-01": "SAC + Sparse",
    "exp-02": "SAC + Shaped",
    "exp-03": "SAC + Curriculum",
    "exp-04": "SAC + PBRS",
    "exp-05": "SAC+HER + Sparse",
    "exp-06": "SAC+HER + Shaped",
}


def get_color(exp_name: str) -> str:
    """Get color for an experiment by matching prefix."""
    for prefix, color in EXP_COLORS.items():
        if exp_name.startswith(prefix):
            return color
    return "#333333"


def get_label(exp_name: str) -> str:
    """Get readable label for an experiment."""
    for prefix, label in EXP_LABELS.items():
        if exp_name.startswith(prefix):
            return label
    return exp_name


def plot_scalar_comparison(
    results: Dict[str, Dict[str, Tuple[List[int], List[float]]]],
    tag: str,
    title: str,
    ylabel: str,
    output_path: str,
    smooth_window: int = 10,
) -> None:
    """Plot a comparison of a scalar across experiments.

    Args:
        results: {exp_name: {tag: (steps, values)}}
        tag: The TensorBoard tag to plot.
        title: Plot title.
        ylabel: Y-axis label.
        output_path: Save path for the figure.
        smooth_window: Moving average window for smoothing.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for exp_name, tags_data in sorted(results.items()):
        if tag not in tags_data:
            continue
        steps, values = tags_data[tag]
        color = get_color(exp_name)
        label = get_label(exp_name)

        # Plot raw (faint) + smoothed (solid)
        ax.plot(steps, values, alpha=0.15, color=color, linewidth=0.5)
        if len(values) > smooth_window:
            smoothed = smooth(values, smooth_window)
            # Align smoothed x-axis
            s_steps = steps[smooth_window - 1:][:len(smoothed)]
            ax.plot(s_steps, smoothed, color=color, label=label, linewidth=2)
        else:
            ax.plot(steps, values, color=color, label=label, linewidth=2)

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved: {output_path}")


def plot_summary_bar(
    results: Dict[str, Dict[str, Tuple[List[int], List[float]]]],
    tag: str,
    title: str,
    ylabel: str,
    output_path: str,
    use_last_n: int = 50,
) -> None:
    """Bar chart showing final performance (mean of last N values).

    Args:
        results: {exp_name: {tag: (steps, values)}}
        tag: The TensorBoard tag.
        title: Plot title.
        ylabel: Y-axis label.
        output_path: Save path.
        use_last_n: Number of final data points to average.
    """
    names = []
    means = []
    stds = []
    colors = []

    for exp_name in sorted(results.keys()):
        if tag not in results[exp_name]:
            continue
        _, values = results[exp_name][tag]
        if not values:
            continue
        last_vals = values[-use_last_n:]
        names.append(get_label(exp_name))
        means.append(np.mean(last_vals))
        stds.append(np.std(last_vals))
        colors.append(get_color(exp_name))

    if not names:
        print(f"[WARNING] No data for tag '{tag}', skipping bar chart.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    # Value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{mean:.2f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Franka Grasp — Plot experiment comparison charts"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/experiments",
        help="Base directory containing experiment subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=10,
        help="Moving average window for curve smoothing.",
    )
    args = parser.parse_args()

    if not HAS_MPL:
        print("[ERROR] matplotlib is required. Install: pip install matplotlib")
        return
    if not HAS_TB:
        print("[ERROR] tensorboard is required. Install: pip install tensorboard")
        return

    log_dir = os.path.abspath(args.log_dir)
    output_dir = os.path.abspath(args.output_dir)

    print("=" * 60)
    print("Franka Cube Grasp — Experiment Plotter")
    print("=" * 60)
    print(f"  log_dir    : {log_dir}")
    print(f"  output_dir : {output_dir}")
    print("=" * 60)

    # Tags to extract
    tags = [
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "rollout/success_rate",
        "train/ent_coef",
        "train/actor_loss",
        "train/critic_loss",
    ]

    # Read all data
    results = read_all_experiments(log_dir, tags)

    if not results:
        print(f"[ERROR] No experiment data found in: {log_dir}")
        print("  Expected subdirectories like: exp-01_sac_sparse/")
        print("  Run experiments first: bash scripts/run_experiments.sh")
        return

    print(f"\n[INFO] Found {len(results)} experiments:")
    for name in sorted(results.keys()):
        tags_found = list(results[name].keys())
        print(f"  {name}: {len(tags_found)} tags")

    # Generate plots
    print("\n[INFO] Generating plots...\n")

    # 1. Reward curves
    plot_scalar_comparison(
        results,
        tag="rollout/ep_rew_mean",
        title="Training Reward Curves — Franka Cube Grasp",
        ylabel="Mean Episode Reward",
        output_path=os.path.join(output_dir, "reward_curves.png"),
        smooth_window=args.smooth,
    )

    # 2. Episode length
    plot_scalar_comparison(
        results,
        tag="rollout/ep_len_mean",
        title="Episode Length — Franka Cube Grasp",
        ylabel="Mean Episode Length (steps)",
        output_path=os.path.join(output_dir, "episode_length.png"),
        smooth_window=args.smooth,
    )

    # 3. Success rate (if available)
    plot_scalar_comparison(
        results,
        tag="rollout/success_rate",
        title="Success Rate — Franka Cube Grasp",
        ylabel="Success Rate",
        output_path=os.path.join(output_dir, "success_rate.png"),
        smooth_window=args.smooth,
    )

    # 4. Summary bar chart — final reward
    plot_summary_bar(
        results,
        tag="rollout/ep_rew_mean",
        title="Final Performance Comparison (last 50 logs)",
        ylabel="Mean Episode Reward",
        output_path=os.path.join(output_dir, "summary_bar.png"),
    )

    # 5. Entropy coefficient (SAC temperature)
    plot_scalar_comparison(
        results,
        tag="train/ent_coef",
        title="SAC Entropy Coefficient (α) — Franka Cube Grasp",
        ylabel="α (ent_coef)",
        output_path=os.path.join(output_dir, "entropy_coef.png"),
        smooth_window=args.smooth,
    )

    print("\n[INFO] All plots saved to:", output_dir)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
