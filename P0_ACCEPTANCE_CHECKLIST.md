# P0 Acceptance Checklist

> Scope: checklist for completing `NewPlan.md` P0.

## 1. Files

- [x] `my_new_practice/大创/projects/NewPlan.md` exists and is the main team plan entry.
- [x] `my_new_practice/大创/projects/P0_BASELINE_FREEZE.md` freezes the active baseline.
- [x] `my_new_practice/大创/projects/SYSTEM_INTERFACES.md` defines CausalExplore, Planner, and Executor schemas.
- [x] This checklist records P0 acceptance criteria.
- [x] No new P0 document is written under old `my_practice`.

## 2. Baseline Consistency

- [x] 2026-04-19 smoke bottlenecks are preserved as historical diagnosis, not current capability.
- [x] 2026-04-19 staged chain handoff is used as the active chain baseline.
- [x] 2026-04-19 recovery telemetry is treated as validated real-output behavior.
- [x] 2026-04-20 task-pool runtime-aligned place is recorded as the current raw-place risk baseline.
- [x] Five-episode smoke is not promoted as a primary regression metric.

## 3. Interface Consistency

- [x] CausalExplore output includes object identity, property belief, affordance candidates, uncertainty, recommended probe, contact region, skill constraints, and artifact path.
- [x] Planner output includes task identity, step index, skill selection, target object, skill args, preconditions, expected effect, and fallback action.
- [x] Executor output includes success, reward, final state, contact state, error code, rollout path, and failure history.
- [x] `failure_history` is defined as the source of truth for concrete recovery debugging.
- [x] Planner is explicitly forbidden from outputting continuous control.

## 4. Safety and Project Rules

- [x] All shell commands used during P0 were prefixed with `conda activate beyondmimic`.
- [x] No `pip uninstall` or `conda remove` commands were used.
- [x] No `rm -rf` command was used.
- [x] No IsaacSim or IsaacLab GUI command was run.
- [x] IsaacSim/IsaacLab future command rule is documented: use `--headless`, smoke `num_envs <= 4`.

## 5. P0 Exit Criteria

P0 is complete when:

- [x] The active baseline is clear enough that future team members stop citing old smoke bottlenecks as current capability.
- [x] The three-module interface is explicit enough to start mock integration.
- [x] New documents are limited to `my_new_practice`.
- [x] Verification confirms the expected files exist and contain the required sections.
- [x] Scoped status review confirms the P0-created files are limited to `NewPlan.md`, `P0_BASELINE_FREEZE.md`, `SYSTEM_INTERFACES.md`, and this checklist.

Note: the repository already had unrelated modified and untracked files before this P0 step. Those pre-existing files are intentionally ignored here rather than reverted or cleaned.

## 6. Next Step After P0

Start P1 only after verification. The first P1 implementation target should be narrow:

1. Add a mock or adapter plan for consuming `SYSTEM_INTERFACES.md`.
2. Choose whether P1 begins with raw task-pool place improvement or new skill smoke scaffolding.
3. Preserve current staged chain acceptance gates while expanding functionality.
