#!/usr/bin/env bash
# Categorical inverse map — PREVIEW on the four ray-family parents at the production recipe (2026-09-20 night).
# Runs from the STAGING CLONE; writes only under this clone (experiments/categorical_inverse/{probes,scores}, logs/).
#   systemd-run --user --unit=catinv_preview -p MemoryMax=30G --collect --working-directory=$PWD \
#     /usr/bin/bash -c 'bash experiments/categorical_inverse/drivers/preview_all.sh > logs/categorical_inverse/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$PWD
export PYTHONPATH=$ROOT
PY=/home/sevan/research/PIM/physically-implicit-modeling/.pim/bin/python
LOGS=$ROOT/logs/categorical_inverse
NT=https://ntfy.sh/swirling-tornado-ai691k
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
for RUN in ray_ablation/L-dw-5ray-20m ray_ablation/L-dw-8ray-20m ray_ablation/L-dw-16ray-20m ray_ablation/L-dw-128ray-20m; do
  N=$(basename "$RUN")
  if [ -f "$ROOT/experiments/categorical_inverse/scores/preview_${N}_appearance-fac.json" ]; then stage "$N already done — skipping"; continue; fi
  stage "preview $N"
  "$PY" -u experiments/categorical_inverse/scripts/preview.py --run "$RUN" > "$LOGS/preview_$N.log" 2>&1 \
    || { echo "FAILED: $N" | tee -a "$LOGS/driver.log"; curl -sS --max-time 20 -H "Title: PIM categorical-IM preview FAILED: $N" -H "Priority: low" -d "$(tail -n 12 "$LOGS/preview_$N.log" | cut -c1-300)" "$NT" >/dev/null 2>&1; continue; }
  tail -n 4 "$LOGS/preview_$N.log" | tee -a "$LOGS/driver.log"
done
stage "chain complete"
curl -sS --max-time 20 -H "Title: PIM categorical-IM preview: done" -H "Priority: low" -H "Tags: memo" \
  -d "$(grep -h 'NEW categorical-state IM' "$LOGS"/preview_*.log | cut -c1-200)" "$NT" >/dev/null 2>&1 || true
