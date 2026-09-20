#!/usr/bin/env bash
# The frames-as-tokens 8-ray model at the production recipe, after the both-bases gate. Nothing under runs/ is written.
set -u
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$PWD; export PYTHONPATH=$ROOT
PY=/home/sevan/research/PIM/physically-implicit-modeling/.pim/bin/python
LOGS=$ROOT/logs/categorical_inverse
while systemctl --user is-active --quiet catinv_gate2; do sleep 30; done
echo "=== [$(date '+%F %T')] STAGE token-model preview (L-dw-8ray-tok-20m)" | tee -a "$LOGS/driver.log"
"$PY" -u experiments/categorical_inverse/scripts/preview_tokens.py --run interface_ablation/L-dw-8ray-tok-20m > "$LOGS/preview_L-dw-8ray-tok-20m.log" 2>&1
rc=$?
[ $rc -ne 0 ] && echo "token preview FAILED rc=$rc" | tee -a "$LOGS/driver.log"
grep -E "NEW categorical-state IM|OLD continuous|g R² held-out" "$LOGS/preview_L-dw-8ray-tok-20m.log" | cut -c1-260 | tee -a "$LOGS/driver.log"
echo "=== [$(date '+%F %T')] STAGE token-model preview complete (rc=$rc)" | tee -a "$LOGS/driver.log"
