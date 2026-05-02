# MuJoCo v2 Physics Roadmap

## Goal

Preserve the current MuJoCo kinematic backend as the stable regression backend while preparing a more realistic MuJoCo robot backend that can be compared against the future IsaacLab training backend.

## Current Gap

The current backend validates task semantics and simulator contracts, but it is not yet a full Franka/Panda dynamics simulation.

Key gaps:

- Robot asset: no articulated Franka/Panda arm is active in the default MuJoCo backend.
- Controller: end-effector motion is kinematic, not produced by IK or joint-space control.
- Gripper dynamics: grasp/release is scripted and does not model finger contact force.
- Contact and friction: block interactions are sufficient for tests, not tuned for sim2real.
- Camera and perception: tests use privileged scene state, not image observations.

## v2 Acceptance Gate

Before replacing the current backend, the v2 backend must pass a parity benchmark on the same task set:

- `把红色方块移到绿色区域`
- `把蓝色方块移到黄色区域`
- `把黄色方块移到蓝色区域`

Each backend result must expose the same fields:

- backend name
- instruction
- object name
- zone name
- success
- error
- fallback used
- explore steps
- planning quality
- replan count

## Implementation Order

1. Add a separate `mujoco_robot` or `mujoco_v2` backend name without changing the default `mujoco` backend.
2. Load a Franka/Panda MJCF and table scene in a smoke test that only verifies reset and observation shape.
3. Add a bounded IK or controller layer for reaching above a named object and zone.
4. Add grasp/release tests with explicit failure reporting.
5. Run the parity benchmark against current MuJoCo kinematic output before interpreting success rates.

## Non-Goals For This Stage

- Do not remove the current kinematic backend.
- Do not require IsaacLab to be installed for MuJoCo v2 tests.
- Do not introduce vision-only observations before state-based parity is stable.
