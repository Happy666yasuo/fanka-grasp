# P1 Raw Place Broader Dataset Variant Spec

> Date: 2026-04-26
> Scope: proposed dataset variant for the next raw task-pool place iteration.
> This is a specification only. No dataset was generated and no training was run.

## 1. Why This Variant Exists

The current curated runtime-reset dataset is high quality but narrow:

| Metric | Current curated range |
| --- | --- |
| `object_zone_distance_xy` | `0.4015` to `0.5886` |
| `ee_object_distance` | `0.0825` to `0.1226` |
| `object_height` | `0.7157` to `0.7348` |
| `lift_progress` | `0.0707` to `0.0898` |
| `held_local_z` | `0.0819` to `0.0911` |

Failed raw place entries are much broader:

| Metric | Failed raw range |
| --- | --- |
| `object_zone_distance_xy` | `0.0462` to `0.5463` |
| `ee_object_distance` | `0.0897` to `1.1781` |
| `object_height` | `0.0223` to `0.9326` |
| `lift_progress` | `0.0000` to `0.2876` |
| `held_local_z` | `-0.1824` to `0.8079` |

The next dataset should broaden support, but not admit every failed or physically poor state.

## 2. Variant Name

Proposed name:

```text
task_pool_post_pick_broadened_v1
```

Expected output paths if implemented later:

```text
outputs/datasets/task_pool_post_pick_broadened_v1.jsonl
outputs/datasets/task_pool_post_pick_broadened_v1_audit.json
```

## 3. Proposed Thresholds

Current strict thresholds from `build_place_runtime_reset_dataset.py`:

| Field | Current strict value |
| --- | --- |
| `min_object_height` | `0.705` |
| `lift_progress_range` | `[0.065, 0.09]` |
| `object_zone_distance_xy_range` | `[0.35, 0.60]` |
| `max_ee_object_distance` | `0.16` |
| `min_held_local_z` | `0.08` |

Proposed broadened v1 thresholds:

| Field | Proposed value | Reason |
| --- | --- | --- |
| `min_object_height` | `0.64` | Include valid held states that start lower than strict support but are still near table-top pick height or above. |
| `lift_progress_range` | `[0.0, 0.20]` | Cover raw entries with low lift progress and moderately over-lifted transport states. |
| `object_zone_distance_xy_range` | `[0.05, 0.60]` | Include near-zone correction/release states that strict support excluded. |
| `max_ee_object_distance` | `0.50` | Cover many raw states with larger EE-object offsets while still excluding extreme `> 1.0` outliers. |
| `min_held_local_z` | `-0.20` | Include alternate grasp/carry poses observed in raw failures. |

These thresholds are intentionally broad but still bounded. They should be treated as a candidate for audit, not as final training truth.

## 4. Required Audit Fields

The broadened audit JSON must include:

- source paths
- thresholds
- total record count
- kept record count
- rejection counts
- kept counts by task
- kept support range for:
  - `object_zone_distance_xy`
  - `ee_object_distance`
  - `object_height`
  - `lift_progress`
  - `held_local_z`
- duplicate or repeated-state counts if available

## 5. Promotion Gate

Do not train on the broadened dataset unless the audit satisfies all conditions:

| Condition | Required result |
| --- | --- |
| all tasks represented | each of `red_to_green`, `blue_to_yellow`, `yellow_to_blue` has nonzero kept records |
| no single task dominates | largest task count is no more than `2x` smallest task count, or imbalance is explicitly documented |
| support broadens strict dataset | at least three of five tracked metrics have wider support than strict v2 |
| extreme outliers controlled | `ee_object_distance > 0.50` and object dropped states remain excluded |
| audit exists | JSON audit is saved beside the JSONL |

## 6. Expected Evaluation After Training

If a model is trained from this variant later, evaluate with:

```bash
conda activate beyondmimic
cd /home/happywindman/Desktop/beyondmimic/my_new_practice/大创/projects/embodied_agent
python batch_evaluate.py --config configs/eval/batch_eval_place_runtime_aligned_task_pool_v2_formal.yaml
```

Promotion criteria:

- formal raw score improves over `0.08`
- formal staged score does not materially regress from `0.48`
- failure slicing moves in the intended direction
- recovery success does not remain `0.00` if planner recovery is expected to help
- active manifest is not updated from smoke-only evidence

## 7. Implementation Note

If implemented, prefer adding parameters to the dataset builder instead of hard-coding a second script.

Suggested minimal extension:

```text
build_place_runtime_reset_dataset.py --preset broadened_v1
```

The existing strict behavior should remain the default.

## 8. Non-Goals

- Do not use this spec to loosen evaluation success tolerance.
- Do not mix this dataset with un-audited failed states.
- Do not promote a model before formal raw/staged evaluation.
- Do not run IsaacSim or IsaacLab.
