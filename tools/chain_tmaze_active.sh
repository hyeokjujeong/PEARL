#!/usr/bin/env bash
set +e
PY=~/anaconda3/envs/pearl/bin/python
WAIT_PID=317018   # B-config launcher; run active tasks only after it finishes

echo "[$(date)] waiting for B launcher (PID ${WAIT_PID}) to finish..."
while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
done
echo "[$(date)] B launcher done -> starting tmaze-active chain"

echo "[$(date)] === active flow B: Option A + latent 2, 1000ep ==="
${PY} launch_experiment.py configs/tmaze-active-flow-B-optionA-latent2.json \
    --num-iterations 1000 \
    --wandb-project pearl-v2 \
    --wandb-run-name "tmactive_B_optA_latent2_1000ep_s1" > tmactive_B_optA_latent2_s1.log 2>&1
echo "[$(date)] active flow B done"

echo "[$(date)] === active baseline (PEARL), 1000ep ==="
${PY} launch_experiment.py configs/tmaze-active.json \
    --num-iterations 1000 \
    --wandb-project pearl-v2 \
    --wandb-run-name "tmactive_baseline_1000ep_s1" > tmactive_baseline_s1.log 2>&1
echo "[$(date)] active baseline done"

echo "[$(date)] === tmaze-active chain complete ==="
