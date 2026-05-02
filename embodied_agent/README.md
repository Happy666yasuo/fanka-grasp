# Embodied Agent Baseline

This project is the first implementation baseline for the university innovation project on language-driven embodied manipulation.

The current version provides a simulator-backed tabletop manipulation baseline:

- one tabletop MuJoCo pick/place scene with a kinematic end-effector
- red/blue/yellow blocks on the table
- green/blue/yellow target zones
- end-to-end instruction families for pick/place and multi-object task routing

## Architecture

- `planner.py`: converts a natural-language instruction into executable plan steps
- `mujoco_simulator.py`: builds the default MuJoCo tabletop scene and exposes robot skills
- `isaaclab_simulator.py`: registers the future IsaacLab backend name and raises a clear not-implemented error until the training adapter exists
- `simulator.py`: retains the legacy PyBullet backend and simulator factory/shared scene config
- `simulation_protocol.py`: defines the backend-neutral simulator contract for MuJoCo now and IsaacLab later
- `skills.py`: wraps simulator actions behind a skill library interface and can load learned skill policies
- `executor.py`: runs the planner and skill library as a closed loop
- `demo.py`: CLI entry point
- `rl_envs.py`: exposes `pick` and `place` as Gymnasium environments for RL training
- `training.py`: trains low-level skill policies with Stable-Baselines3
- `evaluation.py`: runs batch evaluation and writes experiment assets

## Scope of this baseline

- uses a rule-based planner as the first high-level planning module
- keeps a clean interface so the planner can later be replaced with an API-backed LLM and the scripted skills can be swapped with RL policies
- defaults simulator-backed tests to MuJoCo; PyBullet is retained only as a legacy backend
- supports `EMBODIED_SIM_BACKEND=mujoco|pybullet|isaaclab` when no explicit backend is passed

## Current Verification

As of 2026-05-02:

- `embodied_agent`: `112` unittest cases pass in `beyondmimic`.
- Ralph Phase1/2/3: `64/88/65` unittest cases pass.
- MuJoCo is the default simulator-backed validation backend.
- IsaacLab integration remains the next training-backend target.

## Reinforcement learning choice

The default low-level RL algorithm in this repository is `SAC`.

- the low-level action space is continuous: Cartesian end-effector deltas plus one gripper control scalar
- the current observation is low-dimensional state, not raw images, so off-policy continuous-control methods are a good fit
- SAC is more sample-efficient than PPO in simulator-backed continuous manipulation tasks, which matters for a small team with limited compute
- entropy regularization gives better exploration for sparse-success tasks such as grasping and placing
- Stable-Baselines3 has a mature SAC implementation with saving, loading, callbacks, and evaluation support

PPO remains a valid backup baseline and the training code supports it, but SAC is the recommended default for this project stage.

## Run the demo

From this directory:

```bash
python run_demo.py --instruction "把红色方块放到绿色区域"
python run_demo.py --instruction "put the red block in the green zone"
python run_demo.py --gui --keep-open
```

## Train learned skills

From this directory:

```bash
python train_skill.py --config configs/skills/pick_sac.yaml
python train_skill.py --config configs/skills/place_sac.yaml
python plot_training_curves.py --run-dir outputs/training/pick/<run_dir>
```

The training outputs will be written under `outputs/training/<skill>/...` and include:

- the saved final model
- the best model found during evaluation
- the best-model policy manifest
- the resolved config
- `monitor.csv` and `progress.csv`
- `success_rate_curve.png`
- a policy manifest for later batch evaluation

## Batch evaluation

```bash
python batch_evaluate.py --config configs/eval/batch_eval.yaml
```

The batch evaluation script writes per-episode results and summary JSON files under `outputs/evaluations/...`.

To summarize executor `failure_history` and recovery outcomes as Markdown tables:

```bash
conda activate beyondmimic
python analyze_recovery_failures.py --run-dir outputs/evaluations/20260419_213020_batch_eval_50_staged
```

The current multi-object benchmark rollout plan is documented in `MULTI_OBJECT_BENCHMARK_PLAN.md`.

## CausalExplore evaluation

The CausalExplore probe path exports simulator-style catalogs that can be consumed by the mock planning bridge:

```bash
conda activate beyondmimic
python run_causal_explore_eval.py --run-id causal_system_eval_v1_smoke --episodes 3 --objects red_block blue_block
python run_causal_explore_report.py --summary-path outputs/causal_explore/causal_system_eval_v1_smoke/evaluation_summary.json
python run_mock_system_demo.py --catalog-path outputs/causal_explore/causal_system_eval_v1_smoke/curiosity/catalog.json --scenario red_place_failure
```

Each strategy directory contains a planner-loadable root `catalog.json`, per-episode catalogs under `episodes/`, raw artifacts/evidence, `episode_metrics.jsonl`, and `system_metrics.jsonl`. The run root contains `evaluation_summary.json`, `planner_eval_summary.json`, `system_eval_summary.json`, and the generated Markdown report.

The evaluation now reports three layers:

- artifact-level metrics: displacement, uncertainty, artifact count, and probe recommendation rate from all episode artifacts
- planner-level metrics: whether the mock planning bridge inserts `probe` when planning from each episode catalog
- system-level metrics: fixed mock closed-loop scenarios over each episode catalog, including success rate, failure count, replan count, probe steps, and replanned probe count

## Run the planner test

```bash
python -m unittest discover -s tests
```

## Suggested next implementation steps

1. Execute `MULTI_OBJECT_BENCHMARK_PLAN.md`, starting with multi-object scene and evaluation plumbing.
2. Extend from the single red-block / green-zone task to a small multi-object, multi-target benchmark.
3. Replace the current rule-based planner with a more general instruction-to-plan module only after the new benchmark exists.
4. Replace perfect scene state with a perception layer after the closed-loop execution path is stable.
