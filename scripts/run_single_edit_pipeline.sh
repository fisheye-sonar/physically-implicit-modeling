#!/usr/bin/env bash
set -euo pipefail
cd /home/sevan/research/physically-implicit-modeling
exec > logs/single_edit_pipeline.log 2>&1
bash scripts/train_action_editors_all.sh
bash scripts/train_action_editors_metric.sh
.pim/bin/python scripts/eval_action_sweep.py --family exogenous \
  --runs XG_A_H8 XG_A_H32 XG_A_H128 XG_A_H256 XG_A_H512 \
         XG_C_H8 XG_C_H32 XG_C_H128 XG_C_H256 XG_C_H512 --n-edits 128 --n-probe 600
.pim/bin/python scripts/eval_action_sweep.py --family endogenous \
  --runs EN_H8 EN_H32 EN_H128 EN_H256 EN_H512 --n-edits 96
.pim/bin/python scripts/eval_action_sweep.py --family passive --runs H8 H32 H128 H256 H512 --n-probe 800
.pim/bin/python scripts/eval_action_editors.py --models XG_A_H256 XG_C_H256 CTRL_H256 --n-edits 128
echo PIPELINE_DONE
