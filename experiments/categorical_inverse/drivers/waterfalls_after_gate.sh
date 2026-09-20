#!/usr/bin/env bash
# The categorical-IM waterfalls for the four parents, after the previews and the gate (one GPU consumer at a time
# from this clone). Writes only experiments/categorical_inverse/outputs/.
set -u
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$PWD; export PYTHONPATH=$ROOT
PY=/home/sevan/research/PIM/physically-implicit-modeling/.pim/bin/python
LOGS=$ROOT/logs/categorical_inverse
while systemctl --user is-active --quiet catinv_preview || systemctl --user is-active --quiet catinv_gate; do sleep 60; done
echo "=== [$(date '+%F %T')] STAGE waterfalls" | tee -a "$LOGS/driver.log"
for RUN in ray_ablation/L-dw-8ray-20m ray_ablation/L-dw-5ray-20m ray_ablation/L-dw-16ray-20m ray_ablation/L-dw-128ray-20m; do
  [ -f "experiments/categorical_inverse/scores/preview_$(basename $RUN)_appearance-fac.json" ] || { echo "no preview for $RUN — skipped" | tee -a "$LOGS/driver.log"; continue; }
  "$PY" -u experiments/categorical_inverse/scripts/waterfall.py "$RUN" >> "$LOGS/waterfall.log" 2>&1 && tail -n 1 "$LOGS/waterfall.log" | tee -a "$LOGS/driver.log" || echo "FAILED: waterfall $RUN" | tee -a "$LOGS/driver.log"
done
echo "=== [$(date '+%F %T')] STAGE waterfalls complete" | tee -a "$LOGS/driver.log"
