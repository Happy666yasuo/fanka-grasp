from __future__ import annotations

import argparse
import json
from typing import Sequence

from embodied_agent.executor import TaskExecutor
from embodied_agent.planner import RuleBasedPlanner
from embodied_agent.mujoco_simulator import MujocoPickPlaceSimulation
from embodied_agent.simulator import create_pick_place_simulation
from embodied_agent.skills import SkillLibrary


def build_executor(gui: bool) -> tuple[MujocoPickPlaceSimulation, TaskExecutor]:
    simulation = create_pick_place_simulation(backend="mujoco", gui=gui)
    skill_library = SkillLibrary(simulation)
    planner = RuleBasedPlanner()
    executor = TaskExecutor(planner=planner, skill_library=skill_library)
    return simulation, executor


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the embodied manipulation baseline.")
    parser.add_argument(
        "--instruction",
        default="把红色方块放到绿色区域",
        help="Natural-language instruction for the planner.",
    )
    parser.add_argument("--gui", action="store_true", help="Launch the simulator GUI when available.")
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Wait for Enter before closing the GUI window.",
    )
    args = parser.parse_args(argv)

    simulation, executor = build_executor(gui=args.gui)

    try:
        result = executor.run(args.instruction)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        if args.gui and args.keep_open:
            input("Press Enter to close the simulator...")
        return 0 if result.success else 1
    finally:
        simulation.shutdown()
