#!/usr/bin/env bash
# Server GPU 0 chain (3090, slower). dyn_c overnight.
# Runs 3 sparse-point-robot variants (light env, fits well on 3090).
# Total ETA: ~6-8h.
# IMPORTANT: server doesn't accept --gpu flag; uses CUDA_VISIBLE_DEVICES.

set +e
cd "$(dirname "$0")/.."
LOG=log
mkdir -p "$LOG"

export CUDA_VISIBLE_DEVICES=0

echo "[$(date)] === SERVER GPU0 chain start (CUDA_VISIBLE_DEVICES=0) ==="

# 1) sparse-point-robot gauss FIX (baseline)
echo "[$(date)] S1: sparse-pointR vanCFM+optA gauss FIX (baseline)"
python launch_experiment.py configs/dyn_c/sparse-point-robot-vanCFM-optA-gauss-fix.json \
    --wandb-run-name "sparseptR_vanCFM_optA_gauss_fix_s1" \
    > "$LOG/dynC_S1_sparse_gauss_fix_s1.log" 2>&1
echo "[$(date)] S1 done"

# 2) sparse-point-robot gauss + dynC
echo "[$(date)] S2: sparse-pointR vanCFM+optA gauss + dynC"
python launch_experiment.py configs/dyn_c/sparse-point-robot-vanCFM-optA-gauss-dynC.json \
    --wandb-run-name "sparseptR_vanCFM_optA_gauss_dynC1_s1" \
    > "$LOG/dynC_S2_sparse_gauss_dynC_s1.log" 2>&1
echo "[$(date)] S2 done"

# 3) sparse-point-robot prior FIX (S0 — fills out 2x2 ablation)
echo "[$(date)] S0: sparse-pointR vanCFM+optA prior FIX (baseline)"
python launch_experiment.py configs/dyn_c/sparse-point-robot-vanCFM-optA-prior-fix.json \
    --wandb-run-name "sparseptR_vanCFM_optA_prior_fix_s1" \
    > "$LOG/dynC_S0_sparse_prior_fix_s1.log" 2>&1
echo "[$(date)] S0 done"

echo "[$(date)] === SERVER GPU0 chain complete ==="
