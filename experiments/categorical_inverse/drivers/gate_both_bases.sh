#!/usr/bin/env bash
# The parity gate again, WITH the queue's bases (PIM_DW_BASES=frustum,cartesian) so the cartesian block is compared too.
# Runs after the waterfalls. Nothing under runs/ is written.
set -u
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$PWD; export PYTHONPATH=$ROOT PIM_DW_BASES=frustum,cartesian PIM_SKIP_TOPICS=training_curve
PY=/home/sevan/research/PIM/physically-implicit-modeling/.pim/bin/python
LOGS=$ROOT/logs/categorical_inverse
while systemctl --user is-active --quiet catinv_waterfall; do sleep 30; done
echo "=== [$(date '+%F %T')] STAGE parity gate, both bases (L-dw-8ray-20m__seed1)" | tee -a "$LOGS/driver.log"
"$PY" -u experiments/master_eval_refactor/scripts/parity_gate.py ray_ablation/L-dw-8ray-20m__seed1 > "$LOGS/gate_both_bases.log" 2>&1
echo "gate (both bases) rc=$?" | tee -a "$LOGS/driver.log"
F=experiments/master_eval_refactor/scores/ray_ablation__L-dw-8ray-20m__seed1.diff.txt
[ -f "$F" ] && mv "$F" experiments/categorical_inverse/scores/gate_L-dw-8ray-20m__seed1_both_bases.diff.txt
grep -E "leaves compared|MISSING|DIFF /" experiments/categorical_inverse/scores/gate_L-dw-8ray-20m__seed1_both_bases.diff.txt | cut -c1-150 | tee -a "$LOGS/driver.log"
echo "=== [$(date '+%F %T')] STAGE gate (both bases) complete" | tee -a "$LOGS/driver.log"
