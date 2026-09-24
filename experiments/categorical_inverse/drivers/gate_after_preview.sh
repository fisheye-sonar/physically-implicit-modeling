#!/usr/bin/env bash
# Parity gate for the categorical-inverse branch, run AFTER the previews (one GPU consumer at a time from this clone).
# Rescores ray_ablation/L-dw-8ray-20m__seed1 INTO SCRATCH through this branch's pim.scoring with the SETTINGS cell
# alone (no dw_cat_im → the categorical map is out of scope) and diffs every leaf against the scores.json on disk.
# Expected: the appearance-fac block loses its (continuous-state) IM / IM-NN arms, best.IM*, inverse_map — and
# NOTHING else differs. Nothing under runs/ is written.
set -u
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$PWD; export PYTHONPATH=$ROOT
PY=/home/sevan/research/PIM/physically-implicit-modeling/.pim/bin/python
LOGS=$ROOT/logs/categorical_inverse
while systemctl --user is-active --quiet catinv_preview; do sleep 60; done
echo "=== [$(date '+%F %T')] STAGE parity gate (L-dw-8ray-20m__seed1)" | tee -a "$LOGS/driver.log"
"$PY" -u experiments/master_eval_refactor/scripts/parity_gate.py ray_ablation/L-dw-8ray-20m__seed1 > "$LOGS/gate.log" 2>&1
echo "gate rc=$?" | tee -a "$LOGS/driver.log"
F=experiments/master_eval_refactor/scores/ray_ablation__L-dw-8ray-20m__seed1.diff.txt
[ -f "$F" ] && mv "$F" experiments/categorical_inverse/scores/gate_L-dw-8ray-20m__seed1.diff.txt
echo "=== [$(date '+%F %T')] STAGE gate complete" | tee -a "$LOGS/driver.log"
