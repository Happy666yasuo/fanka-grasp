# CausalExplore Evaluation Report: causal_eval_v2_smoke

## Strategy Summary

| Strategy | Mean displacement XY | Mean uncertainty | Artifact count | Requires probe rate | Planner probe count |
|---|---:|---:|---:|---:|---:|
| random | 0.2004 | 0.0617 | 6 | 0.0000 | 0 |
| curiosity | 0.1635 | 0.1333 | 6 | 0.1667 | 0 |
| causal | 0.1651 | 0.0500 | 6 | 0.0000 | 0 |

## Planner Scenarios

### random

| Scenario | Object | Zone | Skills | Probe inserted |
|---|---|---|---|---|
| red_to_green | red_block | green_zone | observe -> pick -> place | false |
| blue_to_yellow | blue_block | yellow_zone | observe -> pick -> place | false |

### curiosity

| Scenario | Object | Zone | Skills | Probe inserted |
|---|---|---|---|---|
| red_to_green | red_block | green_zone | observe -> pick -> place | false |
| blue_to_yellow | blue_block | yellow_zone | observe -> pick -> place | false |

### causal

| Scenario | Object | Zone | Skills | Probe inserted |
|---|---|---|---|---|
| red_to_green | red_block | green_zone | observe -> pick -> place | false |
| blue_to_yellow | blue_block | yellow_zone | observe -> pick -> place | false |
