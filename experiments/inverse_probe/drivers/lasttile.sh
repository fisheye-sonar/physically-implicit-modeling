#!/usr/bin/env bash
# inverse_probe — Sevan's last-tile test + the reconstruction control (2026-09-14 evening)
set -u; cd "$(dirname "$0")/../../.." || exit 1; export PYTHONPATH=$PWD
L=logs/inverse_probe; NT=https://ntfy.sh/swirling-tornado-ai691k
ping() { curl -sS --max-time 20 -H "Title: $1" -d "$2" "$NT" > /dev/null 2>&1 || true; }
run() { # <run> <extra args...> <logname>
  R=$1; shift; LOG=$1; shift
  echo "=== [$(date +%T)] START $R $* ===" >> $L/lasttile_driver.log
  if .pim/bin/python -u experiments/inverse_probe/scripts/othello_inverse.py --run $R --hidden 128 --epochs 200 --tag mirror128 "$@" > $L/$LOG.log 2>&1
  then echo "=== [$(date +%T)] DONE $R ===" >> $L/lasttile_driver.log
  else echo "=== FAILED $R ===" >> $L/lasttile_driver.log; ping "PIM inverse_probe FAILED" "$R $*"; fi
}
run adjacency_ablation/L-oth-adjacent-20m     mirror128_L-oth-adjacent-20m_lasttile --cases last-tile
run initial_othello_comparison/L-oth-20m      mirror128_L-oth-20m_lasttile          --cases last-tile
run adjacency_ablation/L-oth-adjacent-20m     mirror128_L-oth-adjacent-20m_recon    --recon-only
run initial_othello_comparison/L-oth-20m      mirror128_L-oth-20m_recon             --recon-only
echo "=== chain complete ===" >> $L/lasttile_driver.log
ping "PIM inverse_probe last-tile chain complete" "4 runs done; scores in experiments/inverse_probe/scores/*_lasttile.json, *_recon.json"
