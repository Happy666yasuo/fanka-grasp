# P1 Raw Place Support Comparison

> Date: 2026-04-26
> Scope: compare failed raw task-pool place entry states against the curated runtime-reset dataset support.
> No training, formal re-evaluation, or artifact rewriting was run.

## 1. Inputs

Curated training support:

```text
embodied_agent/outputs/datasets/task_pool_post_pick_curated_v2.jsonl
```

Failed raw place-entry states:

```text
embodied_agent/outputs/evaluations/20260420_154545_batch_eval_place_runtime_aligned_task_pool_v2_formal/task_pool_v2_formal_raw_50_place_entry_states.jsonl
```

Only failed `pre_place_policy_entry` and `pre_place_runtime_raw` records were compared.

## 2. Counts

| Set | Count |
| --- | ---: |
| curated runtime-reset records | `18` |
| failed raw `pre_place_policy_entry` records | `92` |
| failed raw `pre_place_runtime_raw` records | `92` |

Failed policy-entry records by task:

| Task | Count |
| --- | ---: |
| `red_to_green` | `28` |
| `blue_to_yellow` | `32` |
| `yellow_to_blue` | `32` |

## 3. Curated Support Ranges

| Metric | Min | Mean | Max |
| --- | ---: | ---: | ---: |
| `object_zone_distance_xy` | `0.4015` | `0.5065` | `0.5886` |
| `ee_object_distance` | `0.0825` | `0.0979` | `0.1226` |
| `object_height` | `0.7157` | `0.7263` | `0.7348` |
| `lift_progress` | `0.0707` | `0.0813` | `0.0898` |
| `held_local_z` | `0.0819` | `0.0867` | `0.0911` |

This support is narrow by design. It mostly represents clean, close, lifted, post-pick states.

## 4. Failed Raw Policy-Entry Ranges

| Metric | Min | Mean | Max |
| --- | ---: | ---: | ---: |
| `object_zone_distance_xy` | `0.0462` | `0.2866` | `0.5463` |
| `ee_object_distance` | `0.0897` | `0.3917` | `1.1781` |
| `object_height` | `0.0223` | `0.6821` | `0.9326` |
| `lift_progress` | `0.0000` | `0.0613` | `0.2876` |
| `held_local_z` | `-0.1824` | `0.0218` | `0.8079` |

The failed raw policy-entry distribution is much broader and often far from the curated support.

## 5. Out-of-Support Counts

Out-of-curated-range counts among the 92 failed raw policy-entry records:

| Metric | Out-of-range count |
| --- | ---: |
| `object_zone_distance_xy` | `68/92` |
| `ee_object_distance` | `85/92` |
| `object_height` | `84/92` |
| `lift_progress` | `84/92` |
| `held_local_z` | `87/92` |

The same counts were observed for failed raw runtime-entry records.

## 6. Interpretation

The current raw place failures are not simply failures inside the curated training support.

The dominant signal is support mismatch:

- failed entries often start with much larger `ee_object_distance`
- many failed entries have object height or lift progress below the curated window
- `held_local_z` is frequently far outside the curated narrow range
- many entries are already closer to the target zone than curated samples, which means the policy may not have enough examples for near-zone release and correction behavior

This supports a data-support diagnosis before an objective-only diagnosis.

## 7. Next Training Hypothesis

The next raw-place training iteration should first test:

```text
broader audited post-pick/raw-entry support
```

The next dataset variant should consider admitting controlled examples with:

- lower object heights and lower lift progress
- larger `ee_object_distance`
- wider `held_local_z`
- near-zone states where `object_zone_distance_xy < 0.35`

Guardrails:

- Keep task counts balanced across `red_to_green`, `blue_to_yellow`, and `yellow_to_blue`.
- Keep an audit JSON with rejection counts and support ranges.
- Do not admit failed states blindly; every broadened threshold needs a reason.
- Compare staged and raw formal results after training; do not promote on smoke-only evidence.

## 8. Decision

Before changing reward/objective, build or simulate a broader audited dataset candidate and inspect its support.

If the broader dataset still does not cover failed raw entries, adjust reset generation.
If it covers failed raw entries and training still fails, then move to observation/objective changes for zone-centered transport and release accuracy.
