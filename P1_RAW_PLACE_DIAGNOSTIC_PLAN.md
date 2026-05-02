# P1 Raw Task-Pool Place Diagnostic Plan

> Date: 2026-04-26
> Scope: diagnostic checklist before any new raw task-pool place training run.
> Rule: do not start training until this checklist identifies which failure slice the next run is meant to change.

## 1. Current Risk

The active staged chain is accepted for the current phase, but raw task-pool place remains weak:

| Gate | Current result |
| --- | --- |
| task-pool formal staged | `0.48` |
| task-pool formal raw | `0.08` |
| raw `red_to_green` | `3/17` |
| raw `blue_to_yellow` | `1/17` |
| raw `yellow_to_blue` | `0/16` |

Known raw failure split:

- `blue_to_yellow`: 5 `transport_failed` + 11 `released_outside_zone`
- `yellow_to_blue`: 5 `transport_failed` + 11 `released_outside_zone`
- No `release_missing` cases were observed.

Interpretation:

- The raw bottleneck is not primarily “arrive near zone and refuse to open.”
- The next run must target zone-centered transport and release position, not just open-gripper timing.
- Do not relax the `0.08` zone tolerance to make metrics look better.

## 2. Canonical Inputs

Evaluation config:

```text
my_new_practice/大创/projects/embodied_agent/configs/eval/batch_eval_place_runtime_aligned_task_pool_v2_formal.yaml
```

Training config:

```text
my_new_practice/大创/projects/embodied_agent/configs/skills/place_sac_runtime_aligned_bcinit_task_pool_v2.yaml
```

Runtime-reset dataset:

```text
my_new_practice/大创/projects/embodied_agent/outputs/datasets/task_pool_post_pick_curated_v2.jsonl
```

Dataset audit:

```text
my_new_practice/大创/projects/embodied_agent/outputs/datasets/task_pool_post_pick_curated_v2_audit.json
```

Current active manifest:

```text
my_new_practice/大创/projects/embodied_agent/outputs/training/place/task_pool_v2_active_best_policy_manifest.json
```

## 3. Commands

All commands are from:

```bash
conda activate beyondmimic
cd /home/happywindman/Desktop/beyondmimic/my_new_practice/大创/projects/embodied_agent
```

### 3.1 Rebuild or Audit Runtime-Reset Dataset

Run only when the post-pick source files are present and the dataset needs to be refreshed:

```bash
python build_place_runtime_reset_dataset.py
```

Expected output:

- JSON summary containing `output_jsonl`
- JSON summary containing `output_audit`
- nonzero `kept_records`
- task counts for `red_to_green`, `blue_to_yellow`, and `yellow_to_blue`

Do not proceed to training if:

- any task has zero kept records
- rejection counts are dominated by one threshold without being understood
- source files are missing

### 3.2 Re-run Formal Task-Pool Place Eval

Use this to reproduce the current baseline before any training change:

```bash
python batch_evaluate.py --config configs/eval/batch_eval_place_runtime_aligned_task_pool_v2_formal.yaml
```

Expected artifacts:

- a new directory under `outputs/evaluations/`
- `evaluation_summary.json`
- `*_episodes.jsonl` files for staged and raw experiments

Pass/fail role:

- This is not expected to solve raw place.
- It verifies the current baseline can still be reproduced.
- It provides fresh episode records for failure slicing.

### 3.3 Analyze Recovery and Failure History

Replace `<run_dir>` with the run produced by the formal eval:

```bash
python analyze_recovery_failures.py --run-dir <run_dir>
```

Expected output:

- table of failure source counts
- table of failure action counts
- recovery policy counts when replans occur

Use episode-level records as the source of truth when classifying a concrete failure.

## 4. Manual Artifact Review

Before training, inspect the latest raw eval artifacts and answer:

1. Are `transport_failed` cases still ending with the object held and lifted?
2. Are `released_outside_zone` cases mostly far from the `0.08` tolerance, or near misses?
3. Does one task dominate failures, or are `blue_to_yellow` and `yellow_to_blue` symmetric?
4. Are failures correlated with `object_zone_distance_xy`, `ee_object_distance`, or `held_local_z` from the runtime-reset dataset?
5. Does staged performance remain higher than raw performance under the same manifest?

If the answer to question 5 is no, stop and inspect evaluation plumbing before training.

## 5. Training Gate

Do not start a new `place_sac_runtime_aligned_*` run until one of these hypotheses is chosen:

| Hypothesis | What should change |
| --- | --- |
| Transport is the main issue | Lower `transport_failed`; object reaches zone-centered pre-release pose more often |
| Release position is the main issue | Lower `released_outside_zone`; final XY distance moves toward `< 0.08` |
| Dataset filter is too narrow | More balanced task counts without admitting low-quality post-pick states |
| Observation lacks needed geometry | Better raw result without staged-only shortcuts |

Each training run must name its intended failure-slice target in the run note or config copy.

## 6. Acceptance Criteria for Next Baseline

A new raw task-pool place baseline can be promoted only if:

- formal raw score improves over `0.08`
- staged score does not regress materially from `0.48`
- failure slicing changes in the intended direction
- the new run has both `evaluation_summary.json` and episode-level JSONL artifacts
- the active manifest is updated only after formal evidence, not smoke-only evidence

## 7. Non-Goals

- Do not switch simulator platforms in this step.
- Do not relax success tolerance.
- Do not replace the planner.
- Do not start IsaacSim or IsaacLab GUI.
- Do not run training with unbounded or undocumented compute settings.
