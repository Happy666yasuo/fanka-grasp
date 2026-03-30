#!/usr/bin/env bash
# Franka Cube Grasp — Batch Experiment Runner
# Conda env: beyondmimic (Python 3.10)
#
# Runs the full 6-experiment matrix sequentially:
#   exp-01: SAC + sparse
#   exp-02: SAC + shaped
#   exp-03: SAC + shaped + curriculum (uses curriculum reward)
#   exp-04: SAC + PBRS
#   exp-05: SAC+HER + sparse
#   exp-06: SAC+HER + shaped
#
# Usage:
#   conda activate beyondmimic
#   cd ~/Desktop/beyondmimic/my_practice/franka_cube_grasp
#   bash scripts/run_experiments.sh
#
# ⚠️ ALL runs use --headless. num_envs=64. buffer_size=100_000 (from sac_cfg.py).
# ⚠️ Total estimated time: ~6 x (500K steps @ 64 envs) — may take many hours.

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_DIR}"

NUM_ENVS=64
TOTAL_STEPS=500000
SEED=42
LOG_BASE="logs/experiments"

echo "============================================================"
echo "Franka Cube Grasp — Batch Experiments"
echo "============================================================"
echo "  Project dir : ${PROJ_DIR}"
echo "  num_envs    : ${NUM_ENVS}"
echo "  total_steps : ${TOTAL_STEPS}"
echo "  seed        : ${SEED}"
echo "  log_base    : ${LOG_BASE}"
echo "============================================================"
echo ""

# ============================================================================
# Experiment matrix
# ============================================================================

run_experiment() {
    local exp_id="$1"
    local algo="$2"
    local reward_type="$3"
    local log_dir="${LOG_BASE}/${exp_id}"

    echo "------------------------------------------------------------"
    echo "[${exp_id}] algo=${algo}, reward=${reward_type}"
    echo "  log_dir: ${log_dir}"
    echo "  Starting at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------"

    python scripts/train.py \
        --headless \
        --num_envs "${NUM_ENVS}" \
        --total_timesteps "${TOTAL_STEPS}" \
        --algo "${algo}" \
        --reward_type "${reward_type}" \
        --seed "${SEED}" \
        --log_dir "${log_dir}"

    echo "[${exp_id}] Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
}

# exp-01: SAC + sparse (baseline)
run_experiment "exp-01_sac_sparse" "sac" "sparse"

# exp-02: SAC + shaped (multi-stage dense reward)
run_experiment "exp-02_sac_shaped" "sac" "shaped"

# exp-03: SAC + curriculum (adaptive difficulty)
run_experiment "exp-03_sac_curriculum" "sac" "curriculum"

# exp-04: SAC + PBRS (potential-based reward shaping)
run_experiment "exp-04_sac_pbrs" "sac" "pbrs"

# exp-05: SAC+HER + sparse (hindsight experience replay)
run_experiment "exp-05_sac-her_sparse" "sac_her" "sparse"

# exp-06: SAC+HER + shaped (HER + dense reward)
run_experiment "exp-06_sac-her_shaped" "sac_her" "shaped"

echo "============================================================"
echo "All 6 experiments complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. View TensorBoard:"
echo "     tensorboard --logdir ${LOG_BASE}"
echo "  2. Generate comparison plots:"
echo "     python scripts/plot_results.py --log_dir ${LOG_BASE}"
echo "  3. Export best model to ONNX:"
echo "     python sim2sim/export_onnx.py --checkpoint ${LOG_BASE}/<best>/checkpoints/latest.zip"
