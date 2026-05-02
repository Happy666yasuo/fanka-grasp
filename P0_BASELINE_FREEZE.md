# P0 Baseline Freeze

> Date: 2026-04-26
> Scope: freeze the current execution baseline for the three-track embodied-agent plan.
> Source of truth: `embodied_agent/CHECKPOINTS.md`, `embodied_agent/EVAL_STANDARD.md`, `status_snapshot_20260419/01_current_status.md`.

## 1. Baseline Decision

The current project baseline is no longer the 2026-04-19 multi-object smoke bottleneck alone.

Use the latest `embodied_agent` checkpoints as the active baseline:

- Staged chain execution is accepted for the present phase.
- Recovery telemetry is validated in real batch outputs.
- Raw task-pool place remains the main execution-risk area for broader task coverage and future CausalExplore integration.

## 2. Historical Bottleneck

The 2026-04-19 snapshot is still useful as diagnosis history:

| Metric | Historical result |
| --- | --- |
| `scripted_multi_object_smoke` | `1.00` |
| `learned_pick_multi_object_smoke` | `0.50` |
| `learned_place_multi_object_smoke` | `0.50` |
| `learned_pick_place_multi_object_smoke` | `0.25` |
| `bc_pick_candidate_grid_multi_object_smoke` | `0.25` |
| `bc_pick_place_multi_object_smoke` | `0.083` |
| `conservative_pick_multi_object_smoke` | `0.50` |

Interpretation:

- The multi-object benchmark plumbing was valid because scripted execution stayed stable.
- Learned pick and learned place failed for different reasons and should not be treated as one generic skill problem.
- This snapshot should not be cited as the latest system capability after the staged-chain checkpoints.

## 3. Active Chain Baseline

Use the staged chain results from `embodied_agent/EVAL_STANDARD.md` and `embodied_agent/CHECKPOINTS.md` as the active chain baseline.

| Gate | Active accepted result |
| --- | --- |
| 20-episode staged regression | `scripted_baseline_20 = 1.00` |
| 20-episode staged regression | `learned_pick_hybrid_20 = 1.00` |
| 20-episode staged regression | `learned_place_hybrid_20 = 0.95` |
| 20-episode staged regression | `learned_pick_place_full_20 = 1.00` |
| 50-episode staged confirmation | `scripted_baseline_50 = 1.00` |
| 50-episode staged confirmation | `learned_pick_hybrid_50 = 1.00` |
| 50-episode staged confirmation | `learned_place_hybrid_50 = 0.96` |
| 50-episode staged confirmation | `learned_pick_place_full_50 = 1.00` |

Supporting artifacts:

- 20-episode regression: `outputs/evaluations/20260419_202924_batch_eval_20_staged`
- 50-episode confirmation: `outputs/evaluations/20260419_203004_batch_eval_50_staged`
- Pick manifest: `outputs/training/pick/20260419_164442_smoke_pick_mainline_regression_retry/best_policy_manifest.json`
- Place manifest: `outputs/training/place/20260419_190903_place_release_regression_v1/best_policy_manifest.json`

## 4. Recovery Baseline

Recovery telemetry is accepted as real-output validated behavior.

| Gate | Result |
| --- | --- |
| 20-episode recovery validation | `episodes_with_replan = 1`, `successful_recovery_episodes = 1`, `recovery_success_rate = 1.00` |
| 50-episode recovery confirmation | `episodes_with_replan = 2`, `successful_recovery_episodes = 2`, `recovery_success_rate = 1.00` |

Required fields in future chain evaluations:

- `replan_count`
- `failure_history`
- `mean_replan_count`
- `max_replan_count`
- `episodes_with_replan`
- `successful_recovery_episodes`
- `recovery_success_rate`
- `failure_source_counts`
- `failure_action_counts`
- `recovery_policy_counts`

Operational rule:

- Use summary-level recovery blocks for dashboards.
- Use episode-level `failure_history` as the source of truth when debugging concrete failures.

## 5. Task-Pool Place Baseline

Task-pool runtime-aligned place is the active risk area.

| Gate | Result |
| --- | --- |
| Smoke staged/raw | `0.50 / 0.0833` |
| Formal staged/raw | `0.48 / 0.08` |
| Staged per-task | `red_to_green = 8/17`, `blue_to_yellow = 7/17`, `yellow_to_blue = 9/16` |
| Raw per-task | `red_to_green = 3/17`, `blue_to_yellow = 1/17`, `yellow_to_blue = 0/16` |

Canonical artifacts:

- Training run: `outputs/training/place/20260420_153811_formal_task_pool_v2_run1`
- Active manifest: `outputs/training/place/task_pool_v2_active_best_policy_manifest.json`
- Smoke check: `outputs/evaluations/20260420_154523_batch_eval_place_runtime_aligned_task_pool_v2_smoke`
- Formal check: `outputs/evaluations/20260420_154545_batch_eval_place_runtime_aligned_task_pool_v2_formal`

Raw failure slicing for weakest tasks:

- `blue_to_yellow`: 5 `transport_failed` + 11 `released_outside_zone`
- `yellow_to_blue`: 5 `transport_failed` + 11 `released_outside_zone`
- No `release_missing` cases were observed.

Interpretation:

- The staged task-pool policy is usable as a selection baseline.
- Raw place is not solved.
- Do not relax curated runtime-reset thresholds until raw failure slicing changes qualitatively.

## 6. Evaluation Rules

- Pick candidates must use `configs/eval/pick_formal_eval.yaml`: 50 fixed-layout episodes + 50 randomized-layout episodes.
- Place candidates must use `configs/eval/place_formal_eval.yaml`: 50 fixed-layout episodes + 50 randomized-layout episodes.
- Chain-level regression must use 20-episode staged regression and 50-episode staged confirmation.
- `configs/eval/batch_eval.yaml` remains a short smoke test only.
- Do not promote a new baseline on 5-episode evidence.
- If future work moves to IsaacSim or IsaacLab, every training/eval/smoke command must include `--headless`; smoke runs must keep `num_envs <= 4`.

## 7. Next Baseline Movement

The next baseline update should happen only when one of these changes is validated:

1. Raw task-pool place improves with failure slicing that changes qualitatively.
2. New skills such as `press`, `push`, `pull`, or `rotate` gain independent smoke and formal gates.
3. CausalExplore produces simulator-backed outputs that planner mock integration can consume without manual field editing.
4. Structured planner uses uncertainty and `failure_history` in a real closed-loop demo.
