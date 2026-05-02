# Formal Evaluation Standard

This project uses skill-isolated formal evaluation as the default selection protocol.

## Pick

- Canonical config: [configs/eval/pick_formal_eval.yaml](configs/eval/pick_formal_eval.yaml)
- Default protocol: 50 fixed-layout episodes + 50 randomized-layout episodes per candidate manifest
- Use this protocol for every new pick candidate before comparing full-task chain behavior

## Place

- Canonical config: [configs/eval/place_formal_eval.yaml](configs/eval/place_formal_eval.yaml)
- Default protocol: 50 fixed-layout episodes + 50 randomized-layout episodes per candidate manifest
- Use this protocol for every new place candidate before comparing full-task chain behavior

## Task-Pool Runtime-Aligned Place

- Canonical smoke config: [configs/eval/batch_eval_place_runtime_aligned_task_pool_v2_smoke.yaml](configs/eval/batch_eval_place_runtime_aligned_task_pool_v2_smoke.yaml)
- Canonical formal config: [configs/eval/batch_eval_place_runtime_aligned_task_pool_v2_formal.yaml](configs/eval/batch_eval_place_runtime_aligned_task_pool_v2_formal.yaml)
- Shared model selector: `outputs/training/place/task_pool_v2_active_best_policy_manifest.json`
- Current baseline training run: `outputs/training/place/20260420_153811_formal_task_pool_v2_run1`
- Current baseline validation runs:
	- smoke: `outputs/evaluations/20260420_154523_batch_eval_place_runtime_aligned_task_pool_v2_smoke`
	- formal: `outputs/evaluations/20260420_154545_batch_eval_place_runtime_aligned_task_pool_v2_formal`
- Current baseline result:
	- smoke staged/raw: `0.50 / 0.0833`
	- formal staged/raw: `0.48 / 0.08`
	- staged per-task: `red_to_green = 8/17`, `blue_to_yellow = 7/17`, `yellow_to_blue = 9/16`
	- raw per-task: `red_to_green = 3/17`, `blue_to_yellow = 1/17`, `yellow_to_blue = 0/16`
- Selection rule:
	- use the staged 50-episode result as the current task-pool place selection metric
	- always keep the raw 50-episode result beside the staged score; do not promote a new baseline on smoke-only evidence
	- do not relax the strict curated runtime-reset thresholds until raw failure slicing changes qualitatively
- Current raw failure diagnosis for the two weakest tasks:
	- `blue_to_yellow`: 5 `transport_failed` + 11 `released_outside_zone`
	- `yellow_to_blue`: 5 `transport_failed` + 11 `released_outside_zone`
	- no `release_missing` cases were observed, so raw regressions are dominated by missing release execution or release at the wrong lateral position
	- only one `yellow_to_blue` failure was close to the zone tolerance (`0.0844` vs success threshold `0.08`); most released failures were materially outside the zone

## Chain-Level Eval

- Canonical staged chain-regression config: [configs/eval/batch_eval_20_staged.yaml](configs/eval/batch_eval_20_staged.yaml)
- Default chain-regression protocol: 20 randomized episodes with runtime staging enabled for learned skills
- Stability confirmation config: [configs/eval/batch_eval_50_staged.yaml](configs/eval/batch_eval_50_staged.yaml)
- [configs/eval/batch_eval.yaml](configs/eval/batch_eval.yaml) remains a short smoke test only
- Do not use 5-episode batch eval as the primary metric for pick, place, or chain-level regression decisions

## Current Accepted Chain Baseline

- 20-episode regression run: `outputs/evaluations/20260419_202924_batch_eval_20_staged`
- Current 20-episode acceptance target:
	- `scripted_baseline_20 = 1.00`
	- `learned_pick_hybrid_20 = 1.00`
	- `learned_place_hybrid_20 = 0.95`
	- `learned_pick_place_full_20 = 1.00`
- 50-episode confirmation run: `outputs/evaluations/20260419_203004_batch_eval_50_staged`
- Current 50-episode confirmation target:
	- `scripted_baseline_50 = 1.00`
	- `learned_pick_hybrid_50 = 1.00`
	- `learned_place_hybrid_50 = 0.96`
	- `learned_pick_place_full_50 = 1.00`
- Current mainline decision:
	- full learned chain is stable enough to treat as solved for the present phase
	- scripted-pick -> learned-place compatibility is no longer a blocker; remaining failures are tail edge cases
- Residual hybrid-place gap in the 50-episode confirmation run: 2/50 failures, both classified as `released_outside_zone`

## Recovery Telemetry

- Every chain batch-eval episode record now persists executor recovery fields in `*_episodes.jsonl`:
	- `replan_count`
	- `failure_history`
- Each `failure_history` item records:
	- failed step
	- failure source
	- reason
	- replan attempt index
	- selected recovery policy
- Every experiment entry in `evaluation_summary.json` now includes a `recovery` block with:
	- `mean_replan_count`
	- `max_replan_count`
	- `episodes_with_replan`
	- `successful_recovery_episodes`
	- `recovery_success_rate`
	- `episodes_with_step_failures`
	- `total_step_failures`
	- `failure_source_counts`
	- `failure_action_counts`
	- `recovery_policy_counts`
- Latest real-output validation run: `outputs/evaluations/20260419_212513_batch_eval_20_staged`
- Validation result from that run:
	- `scripted_baseline_20 = 1.00`
	- `learned_pick_hybrid_20 = 1.00`
	- `learned_place_hybrid_20 = 1.00`
	- `learned_pick_place_full_20 = 1.00`
	- `learned_place_hybrid_20` includes one real recovered failure with:
		- `episodes_with_replan = 1`
		- `successful_recovery_episodes = 1`
		- `recovery_success_rate = 1.00`
		- `failure_source_counts = {"execution_error": 1}`
		- `failure_action_counts = {"place": 1}`
		- `recovery_policy_counts = {"repick_then_place": 1}`
- 50-episode recovery-confirmation run: `outputs/evaluations/20260419_213020_batch_eval_50_staged`
- Validation result from that run:
	- `scripted_baseline_50 = 1.00`
	- `learned_pick_hybrid_50 = 1.00`
	- `learned_place_hybrid_50 = 1.00`
	- `learned_pick_place_full_50 = 1.00`
	- `learned_place_hybrid_50` includes two real recovered failures with:
		- `episodes_with_replan = 2`
		- `successful_recovery_episodes = 2`
		- `recovery_success_rate = 1.00`
		- `failure_source_counts = {"execution_error": 2}`
		- `failure_action_counts = {"place": 2}`
		- `recovery_policy_counts = {"repick_then_place": 2}`
- Interpretation rule:
	- use the summary-level `recovery` block for regression dashboards and pass/fail trend monitoring
	- use the episode-level `failure_history` field as the source of truth for debugging a concrete recovery sequence