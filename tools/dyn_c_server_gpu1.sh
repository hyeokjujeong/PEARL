#!/usr/bin/env bash
# Server GPU 1 chain (faster GPU). dyn_c overnight.
# Runs sequentially: 2 cheetah (dynC gauss s2, dynC prior) + 1 sparse-pointR.
# Total ETA: ~12h.
# IMPORTANT: server doesn't accept --gpu flag; uses CUDA_VISIBLE_DEVICES.

set +e
cd "$(dirname "$0")/.."
LOG=log
mkdir -p "$LOG"

export CUDA_VISIBLE_DEVICES=1

echo "[$(date)] === SERVER GPU1 chain start (CUDA_VISIBLE_DEVICES=1) ==="

# 1) cheetah dyn-c gauss, seed 2 (~5h)
echo "[$(date)] C2: cheetah vanCFM+optA gauss + dynC, seed 2"
python launch_experiment.py configs/dyn_c/cheetah-dir-vanCFM-optA-gauss-dynC.json \
    --wandb-run-name "cheetah_vanCFM_optA_gauss_dynC1_s2" \
    > "$LOG/dynC_C2_cheetah_gauss_s2.log" 2>&1
echo "[$(date)] C2 done"

# 2) cheetah dyn-c prior, seed 1 (~5h)
echo "[$(date)] C3: cheetah vanCFM+optA prior + dynC, seed 1"
python launch_experiment.py configs/dyn_c/cheetah-dir-vanCFM-optA-prior-dynC.json \
    --wandb-run-name "cheetah_vanCFM_optA_prior_dynC1_s1" \
    > "$LOG/dynC_C3_cheetah_prior_s1.log" 2>&1
echo "[$(date)] C3 done"

# 3) sparse-point-robot prior + dynC (~1-2h)
echo "[$(date)] S3: sparse-pointR vanCFM+optA prior + dynC"
python launch_experiment.py configs/dyn_c/sparse-point-robot-vanCFM-optA-prior-dynC.json \
    --wandb-run-name "sparseptR_vanCFM_optA_prior_dynC1_s1" \
    > "$LOG/dynC_S3_sparse_prior_dynC_s1.log" 2>&1
echo "[$(date)] S3 done"

echo "[$(date)] === SERVER GPU1 chain complete ==="
