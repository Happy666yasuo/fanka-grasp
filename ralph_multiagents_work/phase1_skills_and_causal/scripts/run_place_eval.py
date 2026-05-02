#!/usr/bin/env python3
"""Formal evaluation of place success rate using improved_place_object.

Goal: >= 0.20 on raw task-pool place (50 iterations).
"""
from __future__ import annotations

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../embodied_agent/src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embodied_agent.simulator import create_pick_place_simulation
from src.skills.place_improvements import improved_place_object


def run_place_eval(num_episodes: int = 50, tolerance: float = 0.08) -> dict:
    sim = create_pick_place_simulation(backend="mujoco", gui=False)
    results = {"success": 0, "failure": 0, "details": [], "total": num_episodes}
    failure_codes: dict[str, int] = {}

    start_time = time.monotonic()
    for i in range(num_episodes):
        sim.reset_task()
        try:
            sim.pick_object("red_block")
            result = improved_place_object(sim, "green_zone", tolerance=tolerance)
            if result.success:
                results["success"] += 1
            else:
                results["failure"] += 1
                code = result.error_code or "unknown"
                failure_codes[code] = failure_codes.get(code, 0) + 1
            results["details"].append({
                "episode": i,
                "success": result.success,
                "error_code": result.error_code,
                "reward": result.reward,
            })
        except Exception as exc:
            results["failure"] += 1
            code = f"exception: {exc}"
            failure_codes[code] = failure_codes.get(code, 0) + 1
            results["details"].append({
                "episode": i,
                "success": False,
                "error_code": str(exc),
                "reward": 0.0,
            })

    elapsed = time.monotonic() - start_time
    sim.shutdown()

    success_rate = results["success"] / num_episodes
    results["success_rate"] = success_rate
    results["failure_codes"] = failure_codes
    results["elapsed_seconds"] = elapsed
    results["passes_threshold"] = success_rate >= 0.20

    return results


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    print(f"Running place eval (50 episodes)...")
    results = run_place_eval(num_episodes=50)
    print(f"Success rate: {results['success_rate']:.2f} ({results['success']}/{results['total']})")
    print(f"Passes 0.20 threshold: {results['passes_threshold']}")
    print(f"Failure codes: {results['failure_codes']}")
    print(f"Elapsed: {results['elapsed_seconds']:.1f}s")

    output_path = Path("outputs") / "place_eval_results.json"
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Results saved to {output_path}")
