#!/usr/bin/env bash
# Hidden-size ablation on ACTION models (branch `rogerio_controls`, 2026-08-13).
#
# Mirrors the plain-GRU hidden-size sweep (`runs/controls/H{8,32,128,256,512}`) on two action
# model families, so the four findings of that sweep can be checked for replication:
#   F1 prediction saturates by H~128 · F2 linear readability rises monotonically ·
#   F3 canonicality moves the OTHER way · F4 grabbability does not move at all.
#
# Hidden size is the ONLY variable within each family. Everything else — architecture depth,
# objective, iterations/epochs, seed, world settings, observation noise — is held fixed.
set -euo pipefail
cd /home/sevan/research/physically-implicit-modeling
PY=.pim/bin/python
SIZES="8 32 128 256 512"
mkdir -p runs/action_sweep logs

# ── 0. held-out eval split for the exogenous family ──────────────────────────
# datasets/7_cont_teleport used seeds 0..89999 for training, so the eval split is generated
# at a disjoint base seed with the SAME generator defaults (move_scale 4.0, p_action 0.30).
if [ ! -f datasets/13_cont_teleport_eval/eval.h5 ]; then
  mkdir -p datasets/13_cont_teleport_eval
  $PY scripts/gen_continuous_dataset.py --mode teleport \
      --out datasets/13_cont_teleport_eval/eval.h5 \
      --n-samples 4000 --base-seed 200000 --n-workers 16 \
      > logs/gen_eval_split.log 2>&1
fi

# ── 1. exogenous teleport family: actions GIVEN, and the actions-withheld control ─────
# XG_A_* : action-conditioned  (the model is told the teleport it must render)
# XG_C_* : identical data and recipe, actions withheld — isolates action-knowledge from capacity
for H in $SIZES; do
  $PY scripts/train_action_gru_continuous.py \
      --dataset datasets/7_cont_teleport/train.h5 --run-dir runs/action_sweep \
      --run-name "XG_A_H${H}" --use-actions --hidden-size "$H" \
      --n-epochs 400 --seed 0 > "logs/XG_A_H${H}.log" 2>&1
  echo "done XG_A_H${H}"
done
for H in $SIZES; do
  $PY scripts/train_action_gru_continuous.py \
      --dataset datasets/7_cont_teleport/train.h5 --run-dir runs/action_sweep \
      --run-name "XG_C_H${H}" --hidden-size "$H" \
      --n-epochs 400 --seed 0 > "logs/XG_C_H${H}.log" 2>&1
  echo "done XG_C_H${H}"
done

# ── 2. endogenous interactive family (level 3: force dynamics + death + REINFORCE survival) ──
# The registry's "weak" architecture (single Linear encoder/decoder, no multistep loss) at 6000
# iterations, so hidden size is the only variable. NOTE: obs-noise is left at the script default
# 0.2 — the REPO STANDARD — so these runs do NOT carry the 0.05 deviation of the older
# runs/endogenous/L* runs, and are therefore not bit-comparable to them.
for H in $SIZES; do
  $PY scripts/train_endogenous.py --level 3 --out "runs/action_sweep/EN_H${H}" \
      --iters 6000 --hidden "$H" --batched-sim --seed 0 \
      > "logs/EN_H${H}.log" 2>&1
  echo "done EN_H${H}"
done

echo "ALL DONE"
