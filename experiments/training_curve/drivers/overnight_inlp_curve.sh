#!/usr/bin/env bash
# ── 2026-09-01 overnight: INLP on L-dw-20m, THEN the training curve ─────────────────
#
# Sequential on purpose. The curve collects 21.6 GB of residuals per checkpoint and the
# cascade holds float64 copies of its own residuals; overlapping the two on a 59 GB box
# is the OOM class that took the desktop down twice today. One unit, one MemoryMax,
# one job at a time. INLP first (short; its failure is NON-fatal so a bug in tonight's
# new script cannot cost the curve), then experiments/training_curve/drivers/training_curve.sh.
set -u
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
NT=https://ntfy.sh/swirling-tornado-ai691k
LOGS=$ROOT/logs/inlp/overnight_inlp_curve
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" \
         -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }

ping "PIM overnight: started" "INLP (L-dw-20m, frustum, 9 points at n_seq 30k, ~45 min) then the training curve (~6 h)."

stage "I  INLP L-dw-20m frustum"
if PYTHONPATH=$ROOT "$PY" -u experiments/inlp/scripts/inlp_dw.py runs/initial_othello_comparison/L-dw-20m frustum 30000 \
     > "$LOGS/i_inlp.log" 2>&1; then
  ping "PIM INLP: DONE" "$(grep -E '^point|best EI|wiring|unedited' "$LOGS/i_inlp.log" | tail -24)" white_check_mark
else
  ping "PIM INLP: FAILED (curve continues)" "$(tail -15 "$LOGS/i_inlp.log")" warning
  echo "INLP FAILED — continuing to the curve" | tee -a "$LOGS/driver.log"
fi

stage "II training curve"
bash "$ROOT/experiments/training_curve/drivers/training_curve.sh"
stage "chain complete"
