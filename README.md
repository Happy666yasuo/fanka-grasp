# Fanka Grasp Current Mainline

This repository now tracks the current MuJoCo-backed embodied-agent and Ralph multi-agent pipeline.

The earlier Franka/IsaacLab SAC-only snapshot is retained in Git history, but the active project line is:

```text
MuJoCo validation backend
  -> CausalExplore evidence and contracts
  -> Ralph Phase1 skills/probes
  -> Ralph Phase2 planner/replan loop
  -> Ralph Phase3 comparative, ablation, and demo artifacts
```

## Current Status

- `embodied_agent`: MuJoCo kinematic pick/place backend, backend-neutral simulator protocol, legacy PyBullet fallback, IsaacLab adapter skeleton.
- `ralph_multiagents_work`: Phase1/2/3 research pipeline, refreshed against the MuJoCo backend.
- Current verification baseline: `embodied_agent 112 OK`, Ralph Phase1/2/3 `64/88/65 OK`.
- GitHub publication path: branch + PR, no force-push to `main`.

## Backend Roles

| Backend | Status | Role |
| --- | --- | --- |
| MuJoCo | default | Current simulator-backed tests, online probe execution, sim2sim/sim2real validation scaffold |
| PyBullet | legacy | Historical fallback only through explicit backend selection |
| IsaacLab | adapter skeleton | Future low-level skill training backend; not runnable yet |

## Verification

```bash
conda run -n beyondmimic python -m unittest discover embodied_agent/tests -v

cd ralph_multiagents_work/phase1_skills_and_causal
conda run -n beyondmimic python -m unittest discover tests -v

cd ../phase2_llm_planner
conda run -n beyondmimic python -m unittest discover tests -v

cd ../phase3_experiments
conda run -n beyondmimic python -m unittest discover tests -v
```

## Next Engineering Target

The next major implementation step is not more mock expansion. The priority is to make IsaacLab implement the same observation/action contract as the stable MuJoCo backend, then compare both backends on the same multi-object pick/place task set.
