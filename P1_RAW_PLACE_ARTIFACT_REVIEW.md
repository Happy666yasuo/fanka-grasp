# P1 Raw Place Artifact Review

> Date: 2026-04-26
> Scope: artifact-only review of the current task-pool runtime-aligned place baseline.
> No training or formal re-evaluation was run for this review.

## 1. Reviewed Artifacts

Run directory:

```text
my_new_practice/大创/projects/embodied_agent/outputs/evaluations/20260420_154545_batch_eval_place_runtime_aligned_task_pool_v2_formal
```

Files reviewed:

- `evaluation_summary.json`
- `task_pool_v2_formal_raw_50_summary.json`
- `task_pool_v2_formal_staged_50_summary.json`
- `task_pool_v2_formal_raw_50_episodes.jsonl`
- `task_pool_v2_formal_staged_50_episodes.jsonl`
- `outputs/datasets/task_pool_post_pick_curated_v2_audit.json`

## 2. Current Formal Results

| Mode | Episodes | Success rate | Mean goal distance XY | `use_staging` |
| --- | ---: | ---: | ---: | --- |
| staged | 50 | `0.48` | `0.2618` | `true` |
| raw | 50 | `0.08` | `0.2162` | `false` |

Per-task success counts:

| Mode | `red_to_green` | `blue_to_yellow` | `yellow_to_blue` |
| --- | ---: | ---: | ---: |
| staged | `8/17` | `7/17` | `9/16` |
| raw | `3/17` | `1/17` | `0/16` |

Interpretation:

- Raw place remains the bottleneck.
- The weakness is task-pool wide, not isolated to one object-zone pair.
- `yellow_to_blue` is currently the hardest raw task.

## 3. Recovery Summary

| Mode | Episodes with replan | Total step failures | Recovery success rate | Recovery policies |
| --- | ---: | ---: | ---: | --- |
| staged | `26/50` | `52` | `0.00` | `repick_then_place = 26` |
| raw | `46/50` | `92` | `0.00` | `repick_then_place = 33`, `retry_place_only = 13` |

Interpretation:

- Raw mode triggers replanning in almost every episode.
- Existing recovery policies do not rescue task-pool place failures in this formal run.
- This supports treating raw place as a skill/state-alignment problem before expecting planner-level recovery to fix it.

## 4. Runtime-Reset Dataset Audit

Dataset audit:

```text
outputs/datasets/task_pool_post_pick_curated_v2_audit.json
```

Counts:

| Metric | Count |
| --- | ---: |
| total records | `175` |
| kept records | `18` |
| kept from raw-entry source | `7` |
| kept from staged source | `11` |

Kept task counts:

| Task | Kept records |
| --- | ---: |
| `red_to_green` | `7` |
| `blue_to_yellow` | `5` |
| `yellow_to_blue` | `6` |

Rejection counts:

| Rejection reason | Count |
| --- | ---: |
| `object_height` | `109` |
| `lift_progress` | `28` |
| `ee_object_distance` | `11` |
| `object_zone_distance_xy` | `9` |

Interpretation:

- The curated dataset is balanced enough to include all three task-pool entries, but it is small.
- Most source states are rejected because the object is not high enough.
- The strict filter likely protects sample quality, but the small kept set may limit raw generalization.

## 5. Answers to Diagnostic Questions

1. Are `transport_failed` cases still ending with the object held and lifted?
   - Prior checkpoint says yes for the 10 transport failures in `blue_to_yellow` and `yellow_to_blue`; they ended still holding lifted objects.
2. Are `released_outside_zone` cases mostly far from tolerance?
   - Prior checkpoint says yes. Mean miss distances were about `0.1836` and `0.2021`; only one `yellow_to_blue` failure was near `0.08`.
3. Does one task dominate failures?
   - No single task fully explains the issue. Raw success is weak for all tasks, with `yellow_to_blue = 0/16`.
4. Are failures correlated with runtime-reset dataset geometry?
   - The audit suggests the curated set is very small and dominated by object-height filtering. This should be checked before new training.
5. Does staged remain higher than raw under the same manifest?
   - Yes. Staged `0.48` vs raw `0.08`.

## 6. Next Technical Hypothesis

The next raw-place iteration should target:

```text
zone-centered transport under real post-pick/raw entry states
```

Do not spend the next run only on:

- increasing open-gripper probability
- relaxing success tolerance
- planner-level recovery policy changes
- more episodes without changing the training distribution or objective

## 7. Recommended Next Action

Before training:

1. Compare the 18 kept curated states against the failed raw `place_entry_states`.
2. Check whether failed raw entries are outside the curated training support in:
   - `object_zone_distance_xy`
   - `ee_object_distance`
   - object height
   - held local z
3. If failed raw entries are outside support, build a broader but still audited dataset variant.
4. If failed raw entries are inside support, adjust objective or observations toward zone-centered transport and release accuracy.

Only after choosing one of those branches should a new `place_sac_runtime_aligned_*` training run start.
