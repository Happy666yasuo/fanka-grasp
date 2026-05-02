from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")


def plot_success_rate_curve(evaluations_path: Path, output_path: Path, title: str) -> None:
    evaluation_data = np.load(evaluations_path, allow_pickle=True)
    timesteps = evaluation_data["timesteps"]
    if "successes" in evaluation_data:
        success_values = np.asarray(evaluation_data["successes"], dtype=np.float32)
        success_rate = success_values.mean(axis=1)
    else:
        results = np.asarray(evaluation_data["results"], dtype=np.float32)
        success_rate = np.zeros(results.shape[0], dtype=np.float32)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(timesteps, success_rate, linewidth=2.0, color="#1f77b4")
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Timesteps")
    axis.set_ylabel("Success rate")
    axis.set_title(title)
    axis.grid(alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)