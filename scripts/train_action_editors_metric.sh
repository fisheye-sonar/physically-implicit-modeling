#!/usr/bin/env bash
# The metric-corrected (un-whitened) fine-tuning variant — Sevan, 2026-08-14.
# Identical to the `finetune` arms except the FIXED write the model must learn to honour is
# Δ = Σ^1 Wᵀ(W Σ^1 Wᵀ + εI)⁻¹δ instead of the Euclidean pseudoinverse. Same readout target,
# different metric — so this isolates the metric as a variable in the trained setting.
set -euo pipefail
cd /home/sevan/research/physically-implicit-modeling
PY=.pim/bin/python
for M in XG_A_H256 XG_C_H256 CTRL_H256; do
  for KK in 1 8; do
    NAME="${M}__finetune__metric__k${KK}"
    $PY scripts/train_action_editors.py --model "$M" --editor finetune --write metric \
        --edit-k "$KK" --steps 3000 --seed 0 > "logs/${NAME}.log" 2>&1
    echo "done ${NAME}"
  done
done
echo "ALL DONE METRIC"
