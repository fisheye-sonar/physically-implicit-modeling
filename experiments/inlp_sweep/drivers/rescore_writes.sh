#!/usr/bin/env bash
# inlp_sweep — rescore the K-copy writes of the runs that ran under the independent-sum write (joint weighted-ridge solve; cascades unchanged)
set -u; cd "$(dirname "$0")/../../.." || exit 1; export PYTHONPATH=$PWD
L=logs/inlp_sweep; NT=https://ntfy.sh/swirling-tornado-ai691k
for R in noise_ablation/L-dw-noiseless-20m ray_ablation/L-dw-8ray-20m smooth_ablation/L-dw-smooth-20m; do
  N=$(basename $R); S0=$(date +%s); echo "=== [$(date +%T)] START rescore $R ===" >> $L/rescore.log
  if .pim/bin/python -u experiments/inlp_sweep/scripts/inlp_dw_sweep.py --run $R --writes-only > $L/rescore_$N.log 2>&1
  then echo "=== [$(date +%T)] DONE rescore $R ($(( ($(date +%s) - S0) / 60 )) min) ===" >> $L/rescore.log
  else echo "=== FAILED rescore $R ===" >> $L/rescore.log; curl -sS --max-time 20 -H "Title: PIM inlp_sweep rescore FAILED" -d "$R" $NT > /dev/null 2>&1 || true; fi
done
echo "=== rescore complete ===" >> $L/rescore.log; curl -sS --max-time 20 -H "Title: PIM inlp_sweep: writes rescored" -d "noiseless, 8ray, smooth" $NT > /dev/null 2>&1 || true
