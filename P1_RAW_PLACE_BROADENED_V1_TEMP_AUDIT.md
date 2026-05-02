# P1 Raw Place Broadened v1 Temporary Audit

> Date: 2026-04-26
> Scope: temporary audit of `--preset broadened_v1`.
> Output was written to `/tmp` only. The existing strict project dataset was not overwritten.

## 1. Command

Run from:

```bash
conda activate beyondmimic
cd /home/happywindman/Desktop/beyondmimic/my_new_practice/大创/projects/embodied_agent
```

Command:

```bash
python build_place_runtime_reset_dataset.py \
  --preset broadened_v1 \
  --output-jsonl /tmp/task_pool_post_pick_broadened_v1.jsonl \
  --output-audit /tmp/task_pool_post_pick_broadened_v1_audit.json
```

Observed output:

```json
{
  "output_jsonl": "/tmp/task_pool_post_pick_broadened_v1.jsonl",
  "output_audit": "/tmp/task_pool_post_pick_broadened_v1_audit.json",
  "kept_records": 86,
  "task_counts": {
    "red_to_green": 26,
    "blue_to_yellow": 34,
    "yellow_to_blue": 26
  }
}
```

## 2. Counts

| Metric | Count |
| --- | ---: |
| total records | `175` |
| kept records | `86` |
| kept from raw-entry source | `40` |
| kept from staged source | `46` |

Task counts:

| Task | Kept records |
| --- | ---: |
| `red_to_green` | `26` |
| `blue_to_yellow` | `34` |
| `yellow_to_blue` | `26` |

Rejection counts:

| Rejection reason | Count |
| --- | ---: |
| `object_height` | `65` |
| `lift_progress` | `18` |
| `object_zone_distance_xy` | `6` |

## 3. Broadened Support Range

| Metric | Min | Mean | Max |
| --- | ---: | ---: | ---: |
| `object_zone_distance_xy` | `0.1575` | `0.3944` | `0.5886` |
| `ee_object_distance` | `0.0790` | `0.2905` | `0.4871` |
| `object_height` | `0.6414` | `0.6904` | `0.7529` |
| `lift_progress` | `0.0000` | `0.0460` | `0.1079` |
| `held_local_z` | `-0.1824` | `-0.0562` | `0.2449` |

## 4. Comparison Against Strict v2

| Metric | Strict v2 kept | Broadened v1 kept |
| --- | ---: | ---: |
| total kept records | `18` | `86` |
| `red_to_green` | `7` | `26` |
| `blue_to_yellow` | `5` | `34` |
| `yellow_to_blue` | `6` | `26` |

The broadened v1 candidate passes the basic representation gate:

- all tasks have nonzero kept records
- largest task count is less than `2x` the smallest task count
- support is materially wider than strict v2 in `ee_object_distance`, `object_height`, `lift_progress`, and `held_local_z`

## 5. Decision

This candidate is suitable for the next controlled dataset-generation step in the project tree, but only after deciding the final output paths.

Recommended next action:

1. Generate `outputs/datasets/task_pool_post_pick_broadened_v1.jsonl`.
2. Generate `outputs/datasets/task_pool_post_pick_broadened_v1_audit.json`.
3. Review the saved audit once more.
4. Only then decide whether to launch a short training run.

Do not update the active place manifest from this audit alone.
