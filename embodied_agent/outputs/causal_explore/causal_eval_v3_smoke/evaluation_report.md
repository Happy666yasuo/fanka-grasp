# CausalExplore Evaluation Report: causal_eval_v3_smoke

## Strategy Summary

| Strategy | Mean displacement XY | Mean uncertainty | Artifact count | Requires probe rate | Planner eval cases | Planner probe count | Planner probe rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 0.2004 | 0.0617 | 6 | 0.0000 | 6 | 0 | 0.0000 |
| curiosity | 0.1635 | 0.1333 | 6 | 0.1667 | 6 | 1 | 0.1667 |
| causal | 0.1651 | 0.0500 | 6 | 0.0000 | 6 | 0 | 0.0000 |

## Planner Scenarios

### random

| Catalog episode | Scenario | Object | Zone | Skills | Probe inserted |
|---|---|---|---|---|---|
| episode_000 | red_to_green | red_block | green_zone | observe -> pick -> place | false |
| episode_000 | blue_to_yellow | blue_block | yellow_zone | observe -> pick -> place | false |
| episode_001 | red_to_green | red_block | green_zone | observe -> pick -> place | false |
| episode_001 | blue_to_yellow | blue_block | yellow_zone | observe -> pick -> place | false |
| episode_002 | red_to_green | red_block | green_zone | observe -> pick -> place | false |
| episode_002 | blue_to_yellow | blue_block | yellow_zone | observe -> pick -> place | false |

### curiosity

| Catalog episode | Scenario | Object | Zone | Skills | Probe inserted |
|---|---|---|---|---|---|
| episode_000 | red_to_green | red_block | green_zone | observe -> pick -> place | false |
| episode_000 | blue_to_yellow | blue_block | yellow_zone | observe -> pick -> place | false |
| episode_001 | red_to_green | red_block | green_zone | observe -> probe -> pick -> place | true |
| episode_001 | blue_to_yellow | blue_block | yellow_zone | observe -> pick -> place | false |
| episode_002 | red_to_green | red_block | green_zone | observe -> pick -> place | false |
| episode_002 | blue_to_yellow | blue_block | yellow_zone | observe -> pick -> place | false |

### causal

| Catalog episode | Scenario | Object | Zone | Skills | Probe inserted |
|---|---|---|---|---|---|
| episode_000 | red_to_green | red_block | green_zone | observe -> pick -> place | false |
| episode_000 | blue_to_yellow | blue_block | yellow_zone | observe -> pick -> place | false |
| episode_001 | red_to_green | red_block | green_zone | observe -> pick -> place | false |
| episode_001 | blue_to_yellow | blue_block | yellow_zone | observe -> pick -> place | false |
| episode_002 | red_to_green | red_block | green_zone | observe -> pick -> place | false |
| episode_002 | blue_to_yellow | blue_block | yellow_zone | observe -> pick -> place | false |
