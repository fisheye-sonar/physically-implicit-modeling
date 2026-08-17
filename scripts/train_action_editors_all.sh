#!/usr/bin/env bash
# All 12 trained-editor arms for the `trained_editors_actions` thread (2026-08-14).
#
#   3 world models  x  2 trained editors  x  2 losses
#     models : XG_A_H256 (actions in) · XG_C_H256 (observer) · CTRL_H256 (no actions/teleports)
#     editors: finetune (world model adapts to a FIXED frozen pseudoinverse write, WITH the
#              prediction-retention term) · mlp (frozen world model, E(h, start, target) -> dh)
#     losses : --edit-k 1 (next-step RMSE at the edit frame) · --edit-k 8 (8-step rollout RMSE)
#
# Everything else is held fixed: 3000 steps, batch 64, lr 1e-4, retention 1.0, seed 0.
set -euo pipefail
cd /home/sevan/research/physically-implicit-modeling
PY=.pim/bin/python
mkdir -p runs/action_editors logs

for M in XG_A_H256 XG_C_H256 CTRL_H256; do
  for E in finetune mlp; do
    for KK in 1 8; do
      NAME="${M}__${E}__k${KK}"
      $PY scripts/train_action_editors.py --model "$M" --editor "$E" --edit-k "$KK" \
          --steps 3000 --seed 0 > "logs/${NAME}.log" 2>&1
      echo "done ${NAME}"
    done
  done
done
echo "ALL DONE"
