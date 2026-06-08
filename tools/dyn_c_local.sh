#!/usr/bin/env bash
# Local GPU chain for dynamic-c-inference overnight experiments.
# Runs sequentially: 1 cheetah dynC (gauss, seed 1) + 4 tmaze variants.
# Total ETA: ~11h.

set +e
cd "$(dirname "$0")/.."
PY=~/anaconda3/envs/pearl/bin/python
LOG=log
mkdir -p "$LOG"

PROJ=pearl_dyn_c

echo "[$(date)] === LOCAL chain start ==="

# 1) cheetah dyn-c gauss, seed 1 (~5h)
echo "[$(date)] C1: cheetah vanCFM+optA gauss + dynC, seed 1"
${PY} launch_experiment.py configs/dyn_c/cheetah-dir-vanCFM-optA-gauss-dynC.json \
    --wandb-run-name "cheetah_vanCFM_optA_gauss_dynC1_s1" \
    > "$LOG/dynC_C1_cheetah_gauss_s1.log" 2>&1
echo "[$(date)] C1 done"

# 2) T-maze gauss fixed (baseline) (~1.5h)
echo "[$(date)] T1: tmaze vanCFM+optA gauss FIX (baseline)"
${PY} launch_experiment.py configs/dyn_c/tmaze-passive-vanCFM-optA-gauss-fix.json \
    --wandb-run-name "tmaze_vanCFM_optA_gauss_fix_s1" \
    > "$LOG/dynC_T1_tmaze_gauss_fix_s1.log" 2>&1
echo "[$(date)] T1 done"

# 3) T-maze gauss dynC (~1.5h)
echo "[$(date)] T2: tmaze vanCFM+optA gauss + dynC"
${PY} launch_experiment.py configs/dyn_c/tmaze-passive-vanCFM-optA-gauss-dynC.json \
    --wandb-run-name "tmaze_vanCFM_optA_gauss_dynC1_s1" \
    > "$LOG/dynC_T2_tmaze_gauss_dynC_s1.log" 2>&1
echo "[$(date)] T2 done"

# 4) T-maze prior dynC (~1.5h)
echo "[$(date)] T3: tmaze vanCFM+optA prior + dynC"
${PY} launch_experiment.py configs/dyn_c/tmaze-passive-vanCFM-optA-prior-dynC.json \
    --wandb-run-name "tmaze_vanCFM_optA_prior_dynC1_s1" \
    > "$LOG/dynC_T3_tmaze_prior_dynC_s1.log" 2>&1
echo "[$(date)] T3 done"

# 5) T-maze B (decoderCFM + dynC) bonus (~1.5h)
echo "[$(date)] T4 (bonus): tmaze B decoderCFM + Option A + dynC"
${PY} launch_experiment.py configs/dyn_c/tmaze-passive-B-decoderCFM-dynC.json \
    --wandb-run-name "tmaze_B_decoderCFM_optA_prior_dynC1_s1" \
    > "$LOG/dynC_T4_tmaze_B_decoderCFM_dynC_s1.log" 2>&1
echo "[$(date)] T4 done"

echo "[$(date)] === LOCAL chain complete ==="
