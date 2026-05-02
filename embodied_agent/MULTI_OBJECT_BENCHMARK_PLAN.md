# Multi-Object Benchmark Implementation Plan

## Goal

Extend the current single-task embodied agent baseline from one red block and one green zone to a small multi-object, multi-target benchmark without regressing the validated single-task mainline.

The first implementation target is intentionally narrow:

- keep the current observe -> pick -> place task shape
- allow multiple named blocks and multiple named goal zones
- keep one instruction per episode
- preserve existing executor recovery telemetry and failure analysis outputs
- do not retrain pick/place as part of the benchmark plumbing work

## Scope

Included in this plan:

- multi-object scene creation and layout sampling
- multi-object instruction parsing in the rule-based planner
- experiment-side task pools for benchmark evaluation
- episode-level logging of task identity and target selection
- regression tests for ambiguity handling and target stability during recovery

Explicitly excluded from this plan:

- LLM planner integration
- perception / vision input
- reward redesign
- pick/place retraining
- long-horizon multi-step tasks

## Implementation Checklist

### Phase 1 - Documentation and Baseline Freeze

1. Keep the current single-task staged chain regression as the protected baseline.
2. Update README to point to this plan and stop treating executor recovery as the primary next step.
3. Keep EVAL_STANDARD as the source of truth for the current single-task 20/50 episode regression targets.

### Phase 2 - Scene and Simulator Generalization

Files:

- src/embodied_agent/simulator.py

Required changes:

1. Add a supported catalog of named objects and named zones with default positions and colors.
2. Make BulletPickPlaceSimulation configurable with active object names and active zone names so the default single-task run still creates only red_block and green_zone.
3. Generalize _create_scene_objects to create all active bodies instead of hardcoding one red block and one green zone.
4. Generalize fast_reset and reset_task so they can reset either:
   - the current single-task pair via object_xy / zone_xy, or
   - a full object_layout / zone_layout mapping for multi-object experiments.
5. Add a scene-level layout sampler that returns positions for all active objects and zones while enforcing minimum pairwise separation.
6. Keep the existing sample_task_layout helper for backward compatibility by making it a thin single-object wrapper.

### Phase 3 - Planner Generalization

Files:

- src/embodied_agent/planner.py

Required changes:

1. Extend OBJECT_ALIASES and ZONE_ALIASES to multiple named blocks and zones.
2. Remove or restrict generic aliases that become ambiguous once more than one object or zone exists.
3. Update _resolve_target so ambiguity becomes an explicit error instead of silently choosing a fallback target.
4. Carry the target zone through the pick step parameters so learned pick execution no longer depends on an implicit green_zone default.
5. Preserve current replan behavior, but ensure replan uses the originally targeted object and zone under multi-object state.

### Phase 4 - Executor and Skill Plumbing

Files:

- src/embodied_agent/skills.py
- src/embodied_agent/executor.py

Required changes:

1. Remove the implicit red_block default from place execution when multiple objects exist.
2. Remove the implicit green_zone default from learned pick execution and require explicit zone context.
3. Keep backward compatibility by allowing a fallback only when the scene truly contains a single object or a single zone.
4. Ensure executor helper methods do not silently use next(iter(...)) when the scene contains multiple candidates.
5. Treat unresolved multi-object target extraction as a real planning error.

### Phase 5 - Evaluation Benchmark Support

Files:

- src/embodied_agent/evaluation.py
- configs/eval/

Required changes:

1. Add task_pool support so an experiment can evaluate multiple instruction/object/zone combinations.
2. Derive the active scene entity set from the experiment task pool.
3. Record task identity in each episode log:
   - task_id
   - object_name
   - zone_name
4. Preserve existing recovery summary metrics and episode-level failure_history output.
5. Add at least one new multi-object smoke config before adding a larger regression config.

### Phase 6 - Testing

Files:

- tests/test_planner.py
- tests/test_executor.py
- tests/test_skills.py
- tests/test_evaluation.py
- tests/test_rl_envs.py
- tests/test_simulator.py (new)

Required test coverage:

1. Planner resolves multiple color-coded objects and zones in both English and Chinese.
2. Planner raises explicit ambiguity errors when the instruction is underspecified.
3. Replan preserves the original object_name and zone_name under multi-object state.
4. SkillLibrary learned pick execution receives the actual target zone, not a hidden green_zone constant.
5. Evaluation task pools emit the correct task metadata into episode records.
6. Simulator layout sampling returns complete object_positions and zone_positions with valid separation.
7. Existing single-skill RL env smoke tests remain green.

## First Execution Batch

The first implementation batch should be limited to:

1. add this document and reference it from README
2. generalize simulator scene creation and reset/layout plumbing
3. add planner target-zone propagation for pick
4. add evaluation task_pool support and one smoke config
5. add targeted planner / evaluation / simulator tests

The first batch is successful only if the current single-task staged baseline remains intact.

## Acceptance Criteria

1. The current single-task 50-episode staged regression must remain at the validated level after the benchmark plumbing changes.
2. A scripted multi-object smoke benchmark must reach full success before any learned benchmark interpretation is trusted.
3. Multi-object episode logs must keep recovery fields while also recording task_id, object_name, and zone_name.
4. Ambiguous instructions must fail explicitly rather than defaulting to the first object or first zone.

## Immediate Next Step

Start with simulator and evaluation plumbing, because planner generalization is only trustworthy after the scene can actually represent multiple named objects and zones.