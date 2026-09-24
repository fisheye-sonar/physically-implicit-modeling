#!/usr/bin/env bash
# inlp_sweep — the 5-ray additions (Sevan, 16:40): full state, then appearance-fac; queued behind unit inlp_sweep
set -u; cd "$(dirname "$0")/../../.." || exit 1; export PYTHONPATH=$PWD
L=logs/inlp_sweep; NT=https://ntfy.sh/swirling-tornado-ai691k
ping() { curl -sS --max-time 20 -H "Title: $1" -d "$2" "$NT" > /dev/null 2>&1 || true; }
for TGT in full appearance-fac; do
  R=ray_ablation/L-dw-5ray-20m; N=L-dw-5ray-20m$( [ "$TGT" = "appearance-fac" ] && echo _fac ); S0=$(date +%s)
  echo "=== [$(date +%T)] START $R $TGT ===" >> $L/driver.log
  if .pim/bin/python -u experiments/inlp_sweep/scripts/inlp_dw_sweep.py --run $R --target $TGT > $L/inlp_$N.log 2>&1
  then echo "=== [$(date +%T)] DONE $R $TGT ($(( ($(date +%s) - S0) / 60 )) min) ===" >> $L/driver.log; .pim/bin/python experiments/inlp_sweep/scripts/plot.py >> $L/plot.log 2>&1; ping "PIM inlp_sweep: $N done" "$(( ($(date +%s) - S0) / 60 )) min"
  else echo "=== FAILED $R $TGT ===" >> $L/driver.log; ping "PIM inlp_sweep FAILED" "$R $TGT"; fi
done
echo "=== 5ray additions complete ===" >> $L/driver.log; ping "PIM inlp_sweep: 5-ray additions complete" "full + appearance-fac"
