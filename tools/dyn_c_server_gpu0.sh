#!/usr/bin/env bash
# Server GPU 0 chain (3080 Ti, slower). dyn_c overnight.
# Uses --gpu 0 flag inline (works on fix-flow-modes / dynamic-c-inference
# branch — these have --gpu CLI flag). PCI_BUS_ID order forced so GPU 0
# matches nvidia-smi numbering.

set +e
cd "$(dirname "$0")/.."
LOG=log
mkdir -p "$LOG"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES  # let --gpu flag pick the device

echo "[$(date)] === SERVER GPU0 chain start (--gpu 0, PCI order) ==="

# 1) sparse-point-robot gauss FIX (baseline)
echo "[$(date)] S1: sparse-pointR vanCFM+optA gauss FIX (baseline)"
python launch_experiment.py configs/dyn_c/sparse-point-robot-vanCFM-optA-gauss-fix.json \
    --gpu 0 \
    --wandb-run-name "sparseptR_vanCFM_optA_gauss_fix_s1" \
    > "$LOG/dynC_S1_sparse_gauss_fix_s1.log" 2>&1
echo "[$(date)] S1 done"

# 2) sparse-point-robot gauss + dynC
echo "[$(date)] S2: sparse-pointR vanCFM+optA gauss + dynC"
python launch_experiment.py configs/dyn_c/sparse-point-robot-vanCFM-optA-gauss-dynC.json \
    --gpu 0 \
    --wandb-run-name "sparseptR_vanCFM_optA_gauss_dynC1_s1" \
    > "$LOG/dynC_S2_sparse_gauss_dynC_s1.log" 2>&1
echo "[$(date)] S2 done"

# 3) sparse-point-robot prior FIX (S0 — fills out 2x2 ablation)
echo "[$(date)] S0: sparse-pointR vanCFM+optA prior FIX (baseline)"
python launch_experiment.py configs/dyn_c/sparse-point-robot-vanCFM-optA-prior-fix.json \
    --gpu 0 \
    --wandb-run-name "sparseptR_vanCFM_optA_prior_fix_s1" \
    > "$LOG/dynC_S0_sparse_prior_fix_s1.log" 2>&1
echo "[$(date)] S0 done"

echo "[$(date)] === SERVER GPU0 chain complete ==="
