# P2 Integration Status

> Purpose: define the current local demo baseline for the CausalExplore-to-planner-to-executor mock loop.
> Status: updated 2026-05-02. The P2 artifact/provider baseline is stable, and the current simulator-backed execution path has moved to MuJoCo. PyBullet remains only as a legacy backend.

## 1. What Is Done

The repository now has a stable artifact-backed mock integration path:

```text
registry fixture
  -> ArtifactRegistryCausalOutputProvider
  -> ContractPlanningBridge
  -> mock system runner
  -> structured executor-style result
```

Current local entrypoints:

- `embodied_agent/run_mock_planning_bridge.py`
- `embodied_agent/run_mock_system_demo.py`

Current fixed fixture:

- `embodied_agent/outputs/causal_explore/mock_registry_v1/registry.json`

The repository also has a simulator-style catalog fixture that exercises the real provider adapter shape:

```text
simulator-style catalog
  -> SimulatorArtifactCatalogCausalOutputProvider
  -> ContractPlanningBridge
  -> mock planning/system runners
```

Current simulator-style fixture:

- `embodied_agent/outputs/causal_explore/sim_catalog_v1/catalog.json`
- `embodied_agent/outputs/causal_explore/sim_catalog_v1/manifests/scene_sim_0001.json`
- `embodied_agent/outputs/causal_explore/sim_catalog_v1/artifacts/red_block.json`
- `embodied_agent/outputs/causal_explore/sim_catalog_v1/artifacts/blue_block.json`

The repository now has a MuJoCo-backed online probe path:

```text
MuJoCo probe execution
  -> evidence JSON
  -> CausalExploreOutput-compatible object artifacts
  -> simulator-style catalog
  -> SimulatorArtifactCatalogCausalOutputProvider
```

Current entrypoint:

- `embodied_agent/run_causal_explore_probe.py`

Related current implementation:

- `embodied_agent/src/embodied_agent/mujoco_simulator.py`
- `embodied_agent/src/embodied_agent/simulation_protocol.py`
- `embodied_agent/src/embodied_agent/simulator.py::create_pick_place_simulation`

## 2. What This Baseline Proves

This mock baseline proves that the following interfaces are already connected:

- `CausalExploreOutput` can be loaded from local artifact JSON.
- planner-side uncertainty can insert a structured `probe` step.
- executor-style output can return `success / reward / final_state / contact_state / error_code / rollout_path / failure_history`.
- planner replan output can be exported after a structured failure.
- simulator-style catalog artifacts can drive the same planner and system runner JSON shape through `--catalog-path`.
- catalog scene identity, manifest membership, object artifact loading, and artifact path propagation are covered by regression tests.
- MuJoCo-backed probes can generate evidence, object artifacts, a scene manifest, and a catalog consumable by the existing provider path.
- Ralph Phase1/Phase3 simulator-backed paths now use MuJoCo by default.

This baseline is intended for:

- local smoke verification
- demo scripting
- contract stabilization before IsaacLab integration and full robot dynamics

## 3. What Is Not Done Yet

The following work is still outside the current baseline:

- IsaacLab training adapter is not yet connected to the simulator protocol.
- MuJoCo backend is currently kinematic, not full Franka/Panda dynamics.
- static Phase3 experiment JSON/report artifacts should be refreshed before paper submission.

In short:

- current status = `artifact-backed mock integration + MuJoCo simulator-backed v1`
- current adapter status = `simulator-style catalog provider smoke is wired`
- current online status = `MuJoCo probe execution is wired`
- not yet done = `IsaacLab training adapter and full robot dynamics`

## 4. Standard Demo Commands

Planning-only uncertain path:

```bash
conda activate beyondmimic
cd ~/Desktop/beyondmimic/my_new_practice/大创/projects/embodied_agent
python run_mock_planning_bridge.py --scenario uncertain_red_to_green
```

Planning-only confident path:

```bash
conda activate beyondmimic
cd ~/Desktop/beyondmimic/my_new_practice/大创/projects/embodied_agent
python run_mock_planning_bridge.py --scenario confident_blue_to_yellow
```

System success path:

```bash
conda activate beyondmimic
cd ~/Desktop/beyondmimic/my_new_practice/大创/projects/embodied_agent
python run_mock_system_demo.py --scenario blue_success
```

System failure and replan path:

```bash
conda activate beyondmimic
cd ~/Desktop/beyondmimic/my_new_practice/大创/projects/embodied_agent
python run_mock_system_demo.py --scenario red_place_failure
```

Simulator-style catalog planning path:

```bash
conda activate beyondmimic
cd ~/Desktop/beyondmimic/my_new_practice/大创/projects/embodied_agent
python run_mock_planning_bridge.py \
  --catalog-path outputs/causal_explore/sim_catalog_v1/catalog.json \
  --scenario uncertain_red_to_green
```

Simulator-style catalog system failure path:

```bash
conda activate beyondmimic
cd ~/Desktop/beyondmimic/my_new_practice/大创/projects/embodied_agent
python run_mock_system_demo.py \
  --catalog-path outputs/causal_explore/sim_catalog_v1/catalog.json \
  --scenario red_place_failure
```

MuJoCo-backed probe generation path:

```bash
conda activate beyondmimic
cd ~/Desktop/beyondmimic/my_new_practice/大创/projects/embodied_agent
python run_causal_explore_probe.py \
  --run-id mujoco_probe_v1_smoke \
  --objects red_block blue_block
```

Use a generated probe catalog with the existing planner bridge:

```bash
conda activate beyondmimic
cd ~/Desktop/beyondmimic/my_new_practice/大创/projects/embodied_agent
python run_mock_planning_bridge.py \
  --catalog-path outputs/causal_explore/mujoco_probe_v1_smoke/catalog.json \
  --scenario uncertain_red_to_green
```

## 5. Next Recommended Step

The next step should extend the MuJoCo-backed probe runner into a refreshed formal evaluation harness while preserving:

- `CausalExploreOutput` schema
- planner contract output shape
- executor-style result shape
- failure and replan JSON layout

That keeps both fixture paths useful as regression harnesses while refreshing quantitative comparison runs for the current MuJoCo backend.

## 6. Real-Provider Adapter Status

The repository now also has a first adapter for future simulator-backed artifacts:

- planner path: `SimulatorArtifactCatalogCausalOutputProvider`
- runner path: both `run_mock_planning_bridge.py` and `run_mock_system_demo.py` can accept `--catalog-path`

Current intent of this adapter:

- read a simulator-style catalog JSON
- validate object identity against an object manifest when provided
- load object-level artifact JSON
- convert the artifact into the existing `CausalExploreOutput` schema

What it still does **not** prove:

- IsaacLab execution has not been coupled to the provider end-to-end.
- Full robot dynamics have not been validated; current MuJoCo backend is kinematic.
- Static Phase3 output artifacts had not been regenerated in the earlier 2026-05-02 snapshot; 2026-05-03 refreshed comparative, ablation, reports, charts, and final demo outputs for the current MuJoCo backend.

So the current truth is:

- adapter path exists
- repo-level simulator-style fixture exists and is tested
- MuJoCo-backed probe execution exists and is tested
- multi-method simulator-backed code path exists in Ralph Phase3; formal output refresh was completed on 2026-05-03 for the current MuJoCo backend
