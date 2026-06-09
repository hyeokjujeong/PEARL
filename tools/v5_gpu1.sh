#!/usr/bin/env bash
# v5 ablation, GPU1 (4090). Watcher chain:
#   1. wait until S3 (sparse-point-robot prior+dynC) process is gone
#   2. then launch V5-A (sparse-pR vanCFM+optA gauss, train fix-c + eval dual)
#
# This avoids GPU1 OOM when S3 is still running. Safe to nohup right now —
# the script will sleep-poll for S3 and start V5-A automatically.
#
# Run server-side:
#   nohup ./tools/v5_gpu1.sh > log/v5_gpu1_chain.log 2>&1 & disown

set +e
cd "$(dirname "$0")/.."
LOG=log
mkdir -p "$LOG"

S3_PATTERN="sparse-point-robot-vanCFM-optA-prior-dynC"

echo "[$(date)] === v5 GPU1 watcher start ==="
echo "[$(date)] waiting for S3 ($S3_PATTERN) to finish ..."

# poll every 60s; exit loop when no matching process
while pgrep -f "$S3_PATTERN" > /dev/null; do
    sleep 60
done

echo "[$(date)] S3 process gone — starting V5-A"

# V5-A: sparse-pR vanCFM+optA gauss, train fix-c + eval dual (fix + dyn)
python launch_experiment.py configs/v5/spR_v5_gauss_trainFix_evalDual.json \
    --gpu 1 \
    --wandb-run-name "spR_v5_gauss_trainFix_evalDual_s1" \
    > "$LOG/v5_A_spR_gauss_trainFix.log" 2>&1
echo "[$(date)] V5-A done"

echo "[$(date)] === v5 GPU1 watcher complete ==="
