# Embodied Agent Checkpoints

## CP-2026-05-02-mujoco-backend-v1

- Date: 2026-05-02
- Scope:
  - added MuJoCo kinematic pick/place backend
  - added simulator factory and backend-neutral simulator protocol
  - kept PyBullet as explicit legacy backend
  - switched default simulator tests and Ralph simulator-backed paths to MuJoCo
  - added non-default object regression coverage for simulator-backed Ralph Phase3 tasks
- Validated commands:
  - `conda run -n beyondmimic python -m unittest discover tests -v` in `embodied_agent`
  - `conda run -n beyondmimic python -m unittest discover tests -v` in Ralph Phase1/2/3
- Validated results:
  - `embodied_agent`: 108 tests OK
  - Ralph Phase1: 63 tests OK
  - Ralph Phase2: 88 tests OK
  - Ralph Phase3: 63 tests OK
- Current conclusion:
  - MuJoCo is now the default simulator-backed validation backend
  - PyBullet is retained only as legacy fallback/comparison
  - IsaacLab remains the next training-backend integration target

## CP-2026-05-03-backend-selection-and-phase3-refresh

- Scope:
  - added `EMBODIED_SIM_BACKEND` default backend selection for `create_pick_place_simulation`
  - kept explicit `backend=` arguments higher priority than the environment variable
  - registered an IsaacLab placeholder backend with a clear not-implemented runtime error
  - added MuJoCo regressions for non-default object probe and multi-object reset/capture/restore/pick/place
  - refreshed Ralph Phase3 comparative, ablation, reports, charts, and final demo outputs
- Current conclusion:
  - MuJoCo remains the default simulator-backed validation backend
  - PyBullet remains legacy only
  - IsaacLab is now a registered backend name but not a runnable training environment
  - Phase3 formal outputs now match the MuJoCo backend migration state
- Validated results:
  - `embodied_agent`: 112 tests OK
  - Ralph Phase1: 64 tests OK
  - Ralph Phase2: 88 tests OK
  - Ralph Phase3: 65 tests OK

## CP-2026-04-27-causal-explore-system-eval-v1

- Date: 2026-04-27
- Branch: `franka-grasp`
- Scope:
  - added mock closed-loop CausalExplore system evaluation across every strategy episode catalog
  - exported `system_eval_summary.json` and per-strategy `system_metrics.jsonl`
  - merged system-level counts into `evaluation_summary.json`
  - extended Markdown reports with System Summary and System Scenarios tables
- Canonical smoke command:
  - `python run_causal_explore_eval.py --run-id causal_system_eval_v1_smoke --episodes 3 --objects red_block blue_block`
  - `python run_causal_explore_report.py --summary-path outputs/causal_explore/causal_system_eval_v1_smoke/evaluation_summary.json`
  - `python run_mock_system_demo.py --catalog-path outputs/causal_explore/causal_system_eval_v1_smoke/curiosity/catalog.json --scenario red_place_failure`
- Expected artifacts:
  - `outputs/causal_explore/<run_id>/evaluation_summary.json`
  - `outputs/causal_explore/<run_id>/planner_eval_summary.json`
  - `outputs/causal_explore/<run_id>/system_eval_summary.json`
  - `outputs/causal_explore/<run_id>/evaluation_report.md`
  - `outputs/causal_explore/<run_id>/<strategy>/episode_metrics.jsonl`
  - `outputs/causal_explore/<run_id>/<strategy>/system_metrics.jsonl`
- Validation gate:
  - focused: `python -m unittest tests.test_causal_explore_eval tests.test_causal_explore_report tests.test_causal_explore_system_eval tests.test_causal_explore_runner tests.test_simulator_catalog_provider -v`
  - full: `python -m unittest discover tests -v`

## CP-2026-04-27-causal-explore-planner-eval-v3

- Date: 2026-04-27
- Branch: `franka-grasp`
- Scope:
  - changed planner-facing CausalExplore metrics from root-catalog-only to cross-episode catalog aggregation
  - added `planner_eval_case_count` and cross-episode `catalog_evaluations` to planner eval output
  - updated Markdown reports to show planner eval cases, planner probe rate, and per-episode planner rows
- Canonical smoke command:
  - `python run_causal_explore_eval.py --run-id causal_eval_v3_smoke --episodes 3 --objects red_block blue_block`
  - `python run_causal_explore_report.py --summary-path outputs/causal_explore/causal_eval_v3_smoke/evaluation_summary.json`
- Expected artifacts:
  - `outputs/causal_explore/<run_id>/evaluation_summary.json`
  - `outputs/causal_explore/<run_id>/planner_eval_summary.json`
  - `outputs/causal_explore/<run_id>/evaluation_report.md`
  - `outputs/causal_explore/<run_id>/<strategy>/episode_metrics.jsonl`
- Validation gate:
  - focused: `python -m unittest tests.test_causal_explore_eval tests.test_causal_explore_report tests.test_causal_explore_runner tests.test_simulator_catalog_provider -v`
  - full: `python -m unittest discover tests -v`

## CP-2026-04-27-causal-explore-eval-v2

- Date: 2026-04-27
- Branch: `franka-grasp`
- Scope:
  - extended CausalExplore eval outputs with per-object episode metrics
  - added planner-facing strategy evaluation using generated catalogs
  - added Markdown report generation for strategy comparison tables
- Canonical smoke command:
  - `python run_causal_explore_eval.py --run-id causal_eval_v2_smoke --episodes 3 --objects red_block blue_block`
  - `python run_causal_explore_report.py --summary-path outputs/causal_explore/causal_eval_v2_smoke/evaluation_summary.json`
- Expected artifacts:
  - `outputs/causal_explore/<run_id>/evaluation_summary.json`
  - `outputs/causal_explore/<run_id>/planner_eval_summary.json`
  - `outputs/causal_explore/<run_id>/evaluation_report.md`
  - `outputs/causal_explore/<run_id>/<strategy>/episode_metrics.jsonl`
- Validation gate:
  - focused: `python -m unittest tests.test_causal_explore_eval tests.test_causal_explore_report tests.test_causal_explore_runner tests.test_simulator_catalog_provider -v`
  - full: `python -m unittest discover tests -v`

## CP-2026-04-20-task-pool-place-runtime-aligned-baseline

- Date: 2026-04-20
- Branch: `franka-grasp`
- Base commit: `72f9fc0`
- Scope:
  - promoted the runtime-aligned multi-task place run as the current task-pool baseline
  - pinned the active place manifest to the formal run best model
  - summarized the raw failure split for `blue_to_yellow` and `yellow_to_blue`
- Canonical training artifacts:
  - run dir: `outputs/training/place/20260420_153811_formal_task_pool_v2_run1`
  - active manifest: `outputs/training/place/task_pool_v2_active_best_policy_manifest.json`
  - best manifest: `outputs/training/place/20260420_153811_formal_task_pool_v2_run1/best_policy_manifest.json`
- Canonical evaluation artifacts:
  - smoke check: `outputs/evaluations/20260420_154523_batch_eval_place_runtime_aligned_task_pool_v2_smoke`
  - formal check: `outputs/evaluations/20260420_154545_batch_eval_place_runtime_aligned_task_pool_v2_formal`
- Validated results:
  - smoke staged/raw: `0.50 / 0.0833`
  - formal staged/raw: `0.48 / 0.08`
  - staged per-task:
    - `red_to_green = 8/17`
    - `blue_to_yellow = 7/17`
    - `yellow_to_blue = 9/16`
  - raw per-task:
    - `red_to_green = 3/17`
    - `blue_to_yellow = 1/17`
    - `yellow_to_blue = 0/16`
- Raw failure slicing:
  - `blue_to_yellow`: 16 failures = 5 `transport_failed` + 11 `released_outside_zone`
  - `yellow_to_blue`: 16 failures = 5 `transport_failed` + 11 `released_outside_zone`
  - there were no `release_missing` cases in either task, so the raw bottleneck is not “arrive near zone then refuse to open”
  - all 10 `transport_failed` cases ended still holding the object with nearly closed gripper (`gripper_open_ratio` about `0.0002` to `0.023`), and the object remained lifted (`object_height` about `0.77` to `0.85`)
  - released failures were usually not borderline misses: `blue_to_yellow` released cases had `min_object_zone_distance_xy` mean `0.1836`, and `yellow_to_blue` had mean `0.2021`; only one `yellow_to_blue` failure came close to the `0.08` zone tolerance (`0.0844`)
  - the only raw success among these two tasks was `blue_to_yellow` episode 37, which released at `min_zone = 0.0640`
- Current conclusion:
  - staged task-pool place behavior is good enough to serve as the current selection baseline
  - raw place failure is split between missing release decisions on a subset and off-target release on the majority
  - do not relax curated thresholds yet; next iteration should focus on release timing and zone-centered transport under raw post-pick states

## CP-2026-04-19-staged-chain-handoff

- Date: 2026-04-19
- Branch: `franka-grasp`
- Base commit: `72f9fc0`
- Scope:
  - finalized scripted-pick -> learned-place handoff canonicalization
  - finalized staged chain regression as the canonical integration gate
- Canonical evaluation artifacts:
  - 20-episode regression: `outputs/evaluations/20260419_202924_batch_eval_20_staged`
  - 50-episode confirmation: `outputs/evaluations/20260419_203004_batch_eval_50_staged`
- Validated results:
  - `scripted_baseline_20 = 1.00`
  - `learned_pick_hybrid_20 = 1.00`
  - `learned_place_hybrid_20 = 0.95`
  - `learned_pick_place_full_20 = 1.00`
  - `scripted_baseline_50 = 1.00`
  - `learned_pick_hybrid_50 = 1.00`
  - `learned_place_hybrid_50 = 0.96`
  - `learned_pick_place_full_50 = 1.00`
- Supporting manifests:
  - pick: `outputs/training/pick/20260419_164442_smoke_pick_mainline_regression_retry/best_policy_manifest.json`
  - place: `outputs/training/place/20260419_190903_place_release_regression_v1/best_policy_manifest.json`
- Current conclusion:
  - P0 full learned chain stability is closed
  - P1 hybrid compatibility is effectively closed for current project scope
  - remaining hybrid failures are 2/50 tail cases, both `released_outside_zone`
- Recommended next focus:
  - stop retraining pick/place as the default response
  - move the project center of gravity from skill repair to execution robustness and broader task coverage

## CP-2026-04-19-executor-recovery-telemetry

- Date: 2026-04-19
- Branch: `franka-grasp`
- Base commit: `72f9fc0`
- Scope:
  - added executor post-condition failure detection and bounded replanning
  - refined recovery policy by failure type instead of always replaying a fixed observe -> pick -> place chain
  - propagated `replan_count` and `failure_history` into batch evaluation outputs
  - fixed the executor `holding` post-condition after a real staged batch-eval regression
- Validation artifacts:
  - regression-exposing run: `outputs/evaluations/20260419_212251_batch_eval_20_staged`
  - repaired validation run: `outputs/evaluations/20260419_212513_batch_eval_20_staged`
  - 50-episode recovery confirmation: `outputs/evaluations/20260419_213020_batch_eval_50_staged`
- Validated results:
  - repaired 20-episode staged run returned:
    - `scripted_baseline_20 = 1.00`
    - `learned_pick_hybrid_20 = 1.00`
    - `learned_place_hybrid_20 = 1.00`
    - `learned_pick_place_full_20 = 1.00`
  - `learned_place_hybrid_20` produced one successful real recovery:
    - `episodes_with_replan = 1`
    - `successful_recovery_episodes = 1`
    - `recovery_success_rate = 1.00`
    - `recovery_policy_counts = {"repick_then_place": 1}`
  - 50-episode staged confirmation returned:
    - `scripted_baseline_50 = 1.00`
    - `learned_pick_hybrid_50 = 1.00`
    - `learned_place_hybrid_50 = 1.00`
    - `learned_pick_place_full_50 = 1.00`
  - `learned_place_hybrid_50` produced two successful real recoveries:
    - `episodes_with_replan = 2`
    - `successful_recovery_episodes = 2`
    - `recovery_success_rate = 1.00`
    - `recovery_policy_counts = {"repick_then_place": 2}`
- Important implementation note:
  - executor `holding` means the object is currently attached/held
  - do not reuse isolated `is_pick_success()` lift-threshold semantics as a runtime post-condition predicate
- Current conclusion:
  - recovery telemetry is now validated in real batch outputs, not only in unit tests
  - real-output validation is required whenever post-condition semantics or recovery logic changes
