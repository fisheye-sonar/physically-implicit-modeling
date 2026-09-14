#!/usr/bin/env bash
# inverse_probe — dw-blink by subset: reappearance vs visible (2026-09-14 evening, Sevan)
set -u; cd "$(dirname "$0")/../../.." || exit 1; export PYTHONPATH=$PWD
L=logs/inverse_probe; NT=https://ntfy.sh/swirling-tornado-ai691k
for SEL in reappearance visible; do
  echo "=== [$(date +%T)] START blink $SEL ===" >> $L/blink_subsets_driver.log
  if .pim/bin/python -u experiments/inverse_probe/scripts/discworld_inverse.py --run blink_ablation/L-dw-blink-20m --hidden 128 --epochs 200 --tag mirror128 --select $SEL --canonical-on-subset > $L/mirror128_L-dw-blink-20m_sel-$SEL.log 2>&1
  then echo "=== [$(date +%T)] DONE blink $SEL ===" >> $L/blink_subsets_driver.log
  else echo "=== FAILED blink $SEL ===" >> $L/blink_subsets_driver.log; curl -sS --max-time 20 -H "Title: PIM inverse_probe blink subset FAILED" -d "$SEL" $NT > /dev/null 2>&1 || true; fi
done
echo "=== chain complete ===" >> $L/blink_subsets_driver.log
curl -sS --max-time 20 -H "Title: PIM inverse_probe blink subsets done" -d "reappearance + visible" $NT > /dev/null 2>&1 || true
