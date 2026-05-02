# P1 Execution Record

> Date: 2026-04-26
> Scope: first narrow P1 step after P0 baseline/interface freeze.

## 1. P1 Entry Decision

P1 has two possible near-term directions from `NewPlan.md`:

1. Continue raw task-pool place improvement.
2. Prepare broader skill and integration scaffolding for CausalExplore/Planner/Executor.

The first executed P1 step is the second option: add a tested schema contract module for system integration.

Reason:

- Raw task-pool place improvement requires training/evaluation cycles and should not be started without a dedicated config and runtime budget.
- The system integration path needs a concrete machine-readable contract before CausalExplore mock output or structured planner output can be wired in.
- A schema-only module does not change current planner/executor behavior, so it is low risk and easy to verify.

## 2. Implemented Step

Added:

- `embodied_agent/src/embodied_agent/contracts.py`
- `embodied_agent/tests/test_contracts.py`

The module defines:

- `PropertyBelief`
- `AffordanceCandidate`
- `CausalExploreOutput`
- `PlannerStep`
- `ContactState`
- `FailureRecord`
- `ExecutorResult`

Behavior covered by tests:

- High CausalExplore uncertainty triggers `requires_probe()`.
- Planner contract rejects continuous-control arguments such as `joint_positions`.
- Executor result serializes `failure_history` in the shared interface shape.

## 3. Verification

Command:

```bash
conda activate beyondmimic
python -m unittest tests.test_contracts -v
```

Observed result:

```text
Ran 3 tests in 0.000s

OK
```

## 4. Next P1 Step

## 4. Second Implemented Step

Extended:

- `embodied_agent/src/embodied_agent/contracts.py`
- `embodied_agent/tests/test_contracts.py`

Added:

- `build_planner_step_from_causal_output(...)`

Behavior covered by tests:

- High `uncertainty_score` becomes a structured `probe` planner step.
- Low uncertainty emits the requested skill, such as `pick`.
- Confident affordance candidates become planner preconditions such as `affordance.graspable.confidence >= 0.70`.

Verification:

```bash
conda activate beyondmimic
python -m unittest tests.test_contracts -v
```

Observed result:

```text
Ran 5 tests in 0.000s

OK
```

## 5. Third Implemented Step

Extended:

- `embodied_agent/src/embodied_agent/contracts.py`
- `embodied_agent/tests/test_contracts.py`

Added:

- `simulate_contract_step(...)`

Behavior covered by tests:

- Uncertain CausalExplore output produces a mock executor result whose `executed_skill` is `probe`.
- Confident CausalExplore output produces a mock executor result whose `executed_skill` is the requested skill, such as `pick`.
- The mock result uses the same `ExecutorResult` serialization path as the shared contract.

Verification:

```bash
conda activate beyondmimic
python -m unittest tests.test_contracts -v
```

Observed result:

```text
Ran 7 tests in 0.001s

OK
```

## 21. Eighteenth Implemented Step

Updated:

- `embodied_agent/tests/test_planning_bridge.py`
- `embodied_agent/outputs/causal_explore/mock_registry_v1/registry.json`
- `embodied_agent/outputs/causal_explore/mock_registry_v1/artifacts/red_block.json`
- `embodied_agent/outputs/causal_explore/mock_registry_v1/artifacts/blue_block.json`

Purpose:

- Promote the artifact-backed provider path from a temporary unit-test fixture to a fixed repository asset.
- Give the P2 mock integration path a stable local registry and sample artifacts that later planner or executor demos can reuse directly.

Current scope:

- repository-backed registry fixture exists under `outputs/causal_explore/mock_registry_v1`
- fixture covers one uncertain object (`red_block`) and one more confident object (`blue_block`)
- `test_planning_bridge.py` now verifies loading through the repository fixture path, not only a temporary directory

Verification:

```bash
conda activate beyondmimic
python -m unittest tests.test_planning_bridge -v
```

Observed result:

```text
Ran 8 tests in 0.001s

OK
```

## 22. Nineteenth Implemented Step

Added:

- `embodied_agent/src/embodied_agent/mock_planning_runner.py`
- `embodied_agent/run_mock_planning_bridge.py`
- `embodied_agent/tests/test_mock_planning_runner.py`

Purpose:

- Move the P2 mock integration path from unit-test-only assets to a hand-runnable local demo entrypoint.
- Keep the runner narrow: read the repository-backed registry fixture, build a default world state, run the contract planning bridge, and emit structured planner steps as JSON.

Current scope:

- default registry points to `outputs/causal_explore/mock_registry_v1/registry.json`
- supports explicit `--registry-path` override
- outputs JSON with `task_id`, `instruction`, `registry_path`, and `planner_steps`
- does not execute simulator, skill rollout, or training

Verification:

```bash
conda activate beyondmimic
python -m unittest tests.test_mock_planning_runner -v
```

Observed result:

```text
Ran 3 tests in 0.001s

OK
```

## 23. Twentieth Implemented Step

Added:

- `embodied_agent/src/embodied_agent/mock_system_runner.py`
- `embodied_agent/run_mock_system_demo.py`
- `embodied_agent/tests/test_mock_system_runner.py`

Purpose:

- Extend the repository-backed mock integration path from planning-only output to a minimal system loop with executor-style results.
- Keep the scope narrow: use the existing contract planner steps, synthesize a structured `ExecutorResult`, and emit replanned steps after an injected `place` failure.

Current scope:

- success path returns `causal_outputs`, `planner_steps`, and `executor_result`
- failure path injects `released_outside_zone` on `place`
- failure path appends structured `failure_history`
- failure path calls the existing contract replan path and exports `replanned_steps`
- no simulator, no learned policy rollout, no training side effects

Verification:

```bash
conda activate beyondmimic
python -m unittest tests.test_mock_system_runner -v
```

Observed result:

```text
Ran 3 tests in 0.002s

OK
```

## 24. Twenty-First Implemented Step

Updated:

- `embodied_agent/src/embodied_agent/mock_planning_runner.py`
- `embodied_agent/run_mock_planning_bridge.py`
- `embodied_agent/src/embodied_agent/mock_system_runner.py`
- `embodied_agent/run_mock_system_demo.py`
- `embodied_agent/tests/test_mock_planning_runner.py`
- `embodied_agent/tests/test_mock_system_runner.py`
- `P2_MOCK_INTEGRATION_STATUS.md`

Purpose:

- Turn the current mock integration into a stable named-demo baseline instead of relying only on free-form CLI arguments.
- Make the then-current boundary explicit: artifact-backed mock integration was done; simulator-backed CausalExplore was still pending at this checkpoint.
- 2026-05-02 update: simulator-backed CausalExplore has since been wired through the MuJoCo backend; PyBullet is legacy only.

Current scope:

- adds named planning scenarios:
  - `uncertain_red_to_green`
  - `confident_blue_to_yellow`
- adds named system scenarios:
  - `blue_success`
  - `red_place_failure`
- includes the selected scenario in JSON output
- documents the standard local demo commands and the then-current mock-vs-simulator boundary

Verification:

```bash
conda activate beyondmimic
python -m unittest tests.test_mock_planning_runner tests.test_mock_system_runner -v
```

Observed result:

```text
Ran 8 tests in 0.004s

OK
```

## 25. Twenty-Second Implemented Step

Updated:

- `embodied_agent/src/embodied_agent/planning_bridge.py`
- `embodied_agent/src/embodied_agent/mock_planning_runner.py`
- `embodied_agent/src/embodied_agent/mock_system_runner.py`
- `embodied_agent/run_mock_planning_bridge.py`
- `embodied_agent/run_mock_system_demo.py`
- `embodied_agent/tests/test_simulator_catalog_provider.py`
- `P2_MOCK_INTEGRATION_STATUS.md`

Purpose:

- Add the first non-mock provider adapter boundary for future simulator-backed CausalExplore artifacts.
- Keep the existing planner and executor-style JSON shape unchanged while allowing the runners to read a simulator-style artifact catalog.

Current scope:

- adds `SimulatorArtifactCatalogCausalOutputProvider`
- validates object membership against an optional object manifest
- validates `scene_id` consistency between catalog and artifact payload
- supports `catalog_path` on both planning and system mock runners
- leaves the default repository-backed `mock_registry_v1` path unchanged

Verification:

```bash
conda activate beyondmimic
python -m unittest tests.test_simulator_catalog_provider -v
```

Observed result:

```text
Ran 5 tests in 0.003s

OK
```

## 6. Fourth Implemented Step

Added:

- `P1_RAW_PLACE_DIAGNOSTIC_PLAN.md`

Purpose:

- Define the raw task-pool place diagnostic gate before any new training run.
- Keep the next action focused on artifact review and failure slicing instead of blind retraining.
- Preserve the current staged-chain baseline while isolating the raw-place risk.

The plan records:

- current raw/staged task-pool place baseline
- canonical eval and training configs
- runtime-reset dataset and audit paths
- commands for dataset audit, formal eval, and recovery analysis
- manual artifact-review questions
- training gate hypotheses
- acceptance criteria for promoting a new raw place baseline

No training was started in this step.

## 7. Fifth Implemented Step

Added:

- `P1_RAW_PLACE_ARTIFACT_REVIEW.md`

Purpose:

- Review existing raw/staged task-pool place artifacts without running training or formal re-evaluation.
- Convert the current raw-place risk into a concrete technical hypothesis.

Findings recorded:

- Staged formal result: `0.48`
- Raw formal result: `0.08`
- Raw per-task result: `red_to_green = 3/17`, `blue_to_yellow = 1/17`, `yellow_to_blue = 0/16`
- Raw recovery success rate: `0.00`
- Runtime-reset curated dataset: `18/175` records kept
- Kept dataset counts: `red_to_green = 7`, `blue_to_yellow = 5`, `yellow_to_blue = 6`

Current hypothesis:

- The next raw-place iteration should target zone-centered transport under real post-pick/raw entry states.
- Planner-level recovery and open-gripper timing alone are not the right first target.

No training or formal re-evaluation was run in this step.

## 8. Sixth Implemented Step

Added:

- `P1_RAW_PLACE_SUPPORT_COMPARISON.md`

Purpose:

- Compare failed raw task-pool place entry states against the curated runtime-reset dataset support.
- Decide whether the next raw-place intervention should broaden data support or change objective/observations.

Findings recorded:

- Curated runtime-reset records: `18`
- Failed raw policy-entry records: `92`
- Failed raw entries by task: `red_to_green = 28`, `blue_to_yellow = 32`, `yellow_to_blue = 32`
- Out-of-curated-range counts:
  - `object_zone_distance_xy = 68/92`
  - `ee_object_distance = 85/92`
  - `object_height = 84/92`
  - `lift_progress = 84/92`
  - `held_local_z = 87/92`

Decision:

- The next raw-place iteration should first test broader audited post-pick/raw-entry support.
- Objective or observation changes should come after checking whether broader support closes the raw gap.

No training, formal re-evaluation, or artifact rewriting was run in this step.

## 9. Seventh Implemented Step

Added:

- `P1_RAW_PLACE_BROADER_DATASET_SPEC.md`

Purpose:

- Specify a broader audited dataset variant before generating any data.
- Make the next training branch explicit and reviewable.

Proposed variant:

- name: `task_pool_post_pick_broadened_v1`
- output JSONL: `outputs/datasets/task_pool_post_pick_broadened_v1.jsonl`
- output audit: `outputs/datasets/task_pool_post_pick_broadened_v1_audit.json`

Proposed threshold changes:

- `min_object_height`: `0.705 -> 0.64`
- `lift_progress_range`: `[0.065, 0.09] -> [0.0, 0.20]`
- `object_zone_distance_xy_range`: `[0.35, 0.60] -> [0.05, 0.60]`
- `max_ee_object_distance`: `0.16 -> 0.50`
- `min_held_local_z`: `0.08 -> -0.20`

Guardrail:

- The existing strict dataset behavior should remain default if implemented later.
- The broadened variant must produce an audit before training.

No dataset was generated and no training was run in this step.

## 10. Eighth Implemented Step

Modified:

- `embodied_agent/build_place_runtime_reset_dataset.py`

Added:

- `embodied_agent/tests/test_place_runtime_dataset_builder.py`

Purpose:

- Add `--preset broadened_v1` support while preserving strict behavior as the default.
- Make the broader dataset candidate reproducible before any dataset generation or training.

Verified behavior:

- `_resolve_thresholds("strict")` returns the previous strict thresholds.
- `_resolve_thresholds("broadened_v1")` returns the proposed broadened v1 thresholds.
- Unknown presets are rejected.

Verification:

```bash
conda activate beyondmimic
python -m unittest tests.test_place_runtime_dataset_builder -v
```

Observed result:

```text
Ran 3 tests in 0.005s

OK
```

No dataset was generated and no training was run in this step.

## 11. Ninth Implemented Step

Added:

- `P1_RAW_PLACE_BROADENED_V1_TEMP_AUDIT.md`

Command run:

```bash
conda activate beyondmimic
python build_place_runtime_reset_dataset.py \
  --preset broadened_v1 \
  --output-jsonl /tmp/task_pool_post_pick_broadened_v1.jsonl \
  --output-audit /tmp/task_pool_post_pick_broadened_v1_audit.json
```

Observed output:

- kept records: `86`
- task counts: `red_to_green = 26`, `blue_to_yellow = 34`, `yellow_to_blue = 26`
- output paths were under `/tmp`, so the project strict v2 dataset was not overwritten.

Temporary support range:

- `object_zone_distance_xy`: `0.1575` to `0.5886`
- `ee_object_distance`: `0.0790` to `0.4871`
- `object_height`: `0.6414` to `0.7529`
- `lift_progress`: `0.0000` to `0.1079`
- `held_local_z`: `-0.1824` to `0.2449`

Decision:

- `broadened_v1` passes the basic temporary audit gate.
- It is suitable for generating a named project dataset next, but it is not enough to update any active manifest.

## 12. Tenth Implemented Step

Generated:

- `embodied_agent/outputs/datasets/task_pool_post_pick_broadened_v1.jsonl`
- `embodied_agent/outputs/datasets/task_pool_post_pick_broadened_v1_audit.json`

Command run:

```bash
conda activate beyondmimic
python build_place_runtime_reset_dataset.py \
  --preset broadened_v1 \
  --output-jsonl outputs/datasets/task_pool_post_pick_broadened_v1.jsonl \
  --output-audit outputs/datasets/task_pool_post_pick_broadened_v1_audit.json
```

Observed output:

- JSONL line count: `86`
- audit preset: `broadened_v1`
- kept records: `86`
- task counts: `red_to_green = 26`, `blue_to_yellow = 34`, `yellow_to_blue = 26`
- rejection counts: `object_height = 65`, `lift_progress = 18`, `object_zone_distance_xy = 6`

Guardrail:

- This generated a candidate dataset only.
- It does not update the active manifest.
- It does not prove training improvement until a controlled training/eval run is performed.

## 13. Eleventh Implemented Step

Added:

- `embodied_agent/configs/skills/place_sac_runtime_aligned_bcinit_task_pool_broadened_v1.yaml`

Purpose:

- Prepare a controlled training branch that uses `task_pool_post_pick_broadened_v1.jsonl`.
- Keep all other training parameters aligned with `place_sac_runtime_aligned_bcinit_task_pool_v2.yaml`.

Only changed dataset path:

```yaml
post_pick_states_jsonl: ../../outputs/datasets/task_pool_post_pick_broadened_v1.jsonl
```

Guardrail:

- This is a config-only step.
- No training was run.
- No active manifest was updated.

## 14. Twelfth Implemented Step

Modified:

- `embodied_agent/src/embodied_agent/contracts.py`
- `embodied_agent/tests/test_contracts.py`

Added:

- `build_planner_step_from_plan_step(...)`

Purpose:

- Export existing runtime `PlanStep` objects into the shared `PlannerStep` contract format.
- Prepare for future planner-facing structured output without replacing the current rule-based planner.

Verified behavior:

- Runtime `pick` step exports as contract `selected_skill = pick`, `target_object = red_block`, `expected_effect = holding(red_block)`.
- Runtime `place` step exports as contract `selected_skill = place`, `target_object = red_block`, `skill_args.target_zone = green_zone`, `expected_effect = placed(red_block,green_zone)`.

Verification:

```bash
conda activate beyondmimic
python -m unittest tests.test_contracts -v
```

Observed result:

```text
Ran 9 tests in 0.000s

OK
```

## 15. Thirteenth Implemented Step

Added:

- `embodied_agent/src/embodied_agent/planner_contracts.py`
- `embodied_agent/tests/test_planner_contracts.py`

Purpose:

- Provide a real planner-facing adapter module instead of relying only on helper functions in `contracts.py`.
- Keep the current `RuleBasedPlanner` unchanged while exporting its plans as contract `PlannerStep` lists.

Exposed behavior:

- `plan_contract(...)`
- `replan_contract(...)`

Verification:

```bash
conda activate beyondmimic
python -m unittest tests.test_planner_contracts -v
```

Observed result:

```text
Ran 2 tests in 0.000s

OK
```

## 18. Fifteenth Implemented Step

Updated:

- `embodied_agent/src/embodied_agent/planning_bridge.py`
- `embodied_agent/tests/test_planning_bridge.py`

Purpose:

- Extend the P2 mock integration bridge from plan-only insertion to both plan and replan insertion.
- Preserve the current `RuleBasedPlanner` output shape while ensuring uncertainty on the target object can trigger a structured `probe` step before the first target action after replanning.

Current scope:

- `plan_contract(...)` injects `probe` on uncertain targets
- `replan_contract(...)` injects `probe` on uncertain targets
- confident and missing-causal-output paths stay unchanged

Verification:

```bash
conda activate beyondmimic
python -m unittest tests.test_planning_bridge -v
```

Observed result:

```text
Ran 4 tests in 0.000s

OK
```

## 19. Sixteenth Implemented Step

Updated:

- `embodied_agent/src/embodied_agent/planning_bridge.py`
- `embodied_agent/tests/test_planning_bridge.py`

Purpose:

- Reduce manual wiring in the P2 mock integration path by letting the planning bridge pull `CausalExploreOutput` objects from an injected provider.
- Keep the bridge contract narrow: explicit `causal_outputs` still works, but provider lookup now covers the no-manual-dict path.

Current scope:

- supports optional `causal_output_provider`
- fetches outputs by target object id when `causal_outputs` is not passed
- explicit `causal_outputs` overrides provider lookup
- uncertainty-triggered `probe` insertion behavior remains unchanged

Verification:

```bash
conda activate beyondmimic
python -m unittest tests.test_planning_bridge -v
```

Observed result:

```text
Ran 6 tests in 0.000s

OK
```

## 20. Seventeenth Implemented Step

Updated:

- `embodied_agent/src/embodied_agent/contracts.py`
- `embodied_agent/src/embodied_agent/planning_bridge.py`
- `embodied_agent/tests/test_planning_bridge.py`

Purpose:

- Move the P2 mock integration path from an in-memory stub provider to a local artifact-backed registry provider.
- Keep the data path simple: registry resolves `object_id -> artifact json`, provider reads the artifact, bridge injects `probe` when uncertainty remains high.

Current scope:

- adds `causal_output_from_dict(...)` for artifact deserialization
- adds `ArtifactRegistryCausalOutputProvider`
- registry supports relative artifact paths resolved from the registry file location
- bridge behavior stays unchanged after loading outputs: explicit `causal_outputs` still overrides provider lookup

Verification:

```bash
conda activate beyondmimic
python -m unittest tests.test_planning_bridge -v
```

Observed result:

```text
Ran 7 tests in 0.001s

OK
```

## 16. Next P1 Step

The next narrow P1 step should be one of:

1. Run a short training only if compute use is deliberate and the expected run name is recorded first.
2. Reproduce the formal raw/staged place eval before promoting any new model.
3. Start P2 CausalExplore mock integration against `SYSTEM_INTERFACES.md`.
4. Keep expanding planner-side structured export only if a concrete consumer needs more fields.

Recommended next step: start P2 mock integration if staying low-compute; otherwise stop before training unless compute use is explicitly desired.

## 17. Fourteenth Implemented Step

Added:

- `embodied_agent/src/embodied_agent/planning_bridge.py`
- `embodied_agent/tests/test_planning_bridge.py`

Purpose:

- Start the P2 mock integration path with a real planning bridge.
- Keep the current `RuleBasedPlanner` for task decomposition, but insert a contract `probe` step before the first target action when `CausalExploreOutput` reports high uncertainty.

Current scope:

- implements `plan_contract(...)`
- keeps the confident path unchanged
- inserts one `probe` step before the first target skill on the uncertain path

Verification:

```bash
conda activate beyondmimic
python -m unittest tests.test_planning_bridge -v
```

Observed result:

```text
Ran 2 tests in 0.000s

OK
```
