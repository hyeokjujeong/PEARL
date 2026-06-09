#!/usr/bin/env bash
# v5 local chain — ~10h. Runs sequentially on local GPU:
#   L1: V5-A prior version (sparse-pR, ~1.5h) — fill in prior dimension of v5
#   L2: T-maze B+prior+fix-c (~1.5h) — T4 sanity (dynC was the culprit?)
#   L3: cheetah dynC 1500 epoch (~7h) — converge check
#
# Run:
#   nohup ./tools/v5_local.sh > log/v5_local_chain.log 2>&1 & disown

set +e
cd "$(dirname "$0")/.."
PY=~/anaconda3/envs/pearl/bin/python
LOG=log
mkdir -p "$LOG"

echo "[$(date)] === v5 local chain start ==="

# 1) L1: V5-A prior (sparse-pR, train fix-c + eval dual, ~1.5h)
echo "[$(date)] L1: spR_v5_prior_trainFix_evalDual"
${PY} launch_experiment.py configs/v5/spR_v5_prior_trainFix_evalDual.json \
    --wandb-run-name "spR_v5_prior_trainFix_evalDual_s1" \
    > "$LOG/v5_L1_spR_prior_trainFix.log" 2>&1
echo "[$(date)] L1 done"

# 2) L2: T-maze B+prior+fix-c sanity (T4 failure debug, ~1.5h)
echo "[$(date)] L2: tmaze_B_decoderCFM_optA_prior_fix"
${PY} launch_experiment.py configs/v5/tmaze-B-decoderCFM-prior-fix.json \
    --wandb-run-name "tmaze_B_decoderCFM_optA_prior_fix_s1" \
    > "$LOG/v5_L2_tmaze_B_fix.log" 2>&1
echo "[$(date)] L2 done"

# 3) L3: cheetah dynC 1500 epoch (converge check, ~7h)
echo "[$(date)] L3: cheetah_vanCFM_optA_gauss_dynC_1500ep"
${PY} launch_experiment.py configs/v5/cheetah-vanCFM-optA-gauss-dynC-1500ep.json \
    --wandb-run-name "cheetah_vanCFM_optA_gauss_dynC1_1500ep_s1" \
    > "$LOG/v5_L3_cheetah_1500ep.log" 2>&1
echo "[$(date)] L3 done"

echo "[$(date)] === v5 local chain complete ==="
