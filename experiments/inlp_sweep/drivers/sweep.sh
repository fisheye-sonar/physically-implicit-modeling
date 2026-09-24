#!/usr/bin/env bash
# inlp_sweep — per-variable INLP cascades + K-copy shrink writes, six discworld runs in Sevan's order (2026-09-14 evening)
#   systemd-run --user --unit=inlp_sweep -p MemoryMax=40G --collect --working-directory=$PWD /usr/bin/bash experiments/inlp_sweep/drivers/sweep.sh
set -u; cd "$(dirname "$0")/../../.." || exit 1; export PYTHONPATH=$PWD
L=logs/inlp_sweep; NT=https://ntfy.sh/swirling-tornado-ai691k; mkdir -p $L
ping() { curl -sS --max-time 20 -H "Title: $1" -d "$2" "$NT" > /dev/null 2>&1 || true; }
while read -r R TGT; do
  N=$(basename $R)$( [ "$TGT" = "appearance-fac" ] && echo _fac ); S0=$(date +%s)
  echo "=== [$(date +%T)] START $R $TGT ===" >> $L/driver.log
  if .pim/bin/python -u experiments/inlp_sweep/scripts/inlp_dw_sweep.py --run $R --target $TGT > $L/inlp_$N.log 2>&1
  then echo "=== [$(date +%T)] DONE $R $TGT ($(( ($(date +%s) - S0) / 60 )) min) ===" >> $L/driver.log; .pim/bin/python experiments/inlp_sweep/scripts/plot.py >> $L/plot.log 2>&1; ping "PIM inlp_sweep: $N done" "$(( ($(date +%s) - S0) / 60 )) min"
  else echo "=== FAILED $R $TGT ===" >> $L/driver.log; ping "PIM inlp_sweep FAILED" "$R $TGT"; fi
done <<LIST
noise_ablation/L-dw-noiseless-20m full
ray_ablation/L-dw-8ray-20m full
smooth_ablation/L-dw-smooth-20m full
interface_ablation/L-dw-8ray-tok-20m full
noise_ablation/L-dw-noiseless-20m appearance-fac
ray_ablation/L-dw-8ray-20m appearance-fac
LIST
echo "=== chain complete ===" >> $L/driver.log; ping "PIM inlp_sweep chain complete" "6 runs"
