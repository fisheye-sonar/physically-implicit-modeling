#!/usr/bin/env bash
# ── L-oth-adjacent-drop07-390k: Transformer-L on oth-adjacent with DROPOUT 0.7 (2026-09-13, evening) ──
#
# Fourth arm of the dropout ablation, Sevan's ask after the 0.3 result. Dropout 0 -> 0.1 -> 0.3 was monotone:
# fewer orthogonal colour copies (283 / 232 / 185 at point 1), better probe skill, and the canonical ND edit
# going from inert to guarded-positive (+0.165 / fid 0.77). 0.7 pushes the read-channel noise far past the
# usual range to see whether the compression continues, whether editability keeps improving, and where the
# model stops reaching the Bayes floor. Identical to adjacency_ablation/L-oth-adjacent-20m except --dropout 0.7
# and 390k steps (resumable, so extend later with --resume --steps 780000).
#
# Stages (each gated on the previous one; a failed stage pings and stops):
#   W  wait for the GPU: unit oth_adjacent_drop03 must have exited (already inactive at launch)
#   C  GPU  train 390k steps, dropout 0.3, resumable                          (~13 h on the 4090)
#   D  GPU  master_eval (scores) + build_full_table
# Launch (on wsl-sevan):
#   systemd-run --user --unit=oth_adjacent_drop07 -p MemoryMax=40G --collect --working-directory=$PWD \
#       /usr/bin/bash -c 'bash experiments/dropout_ablation/drivers/oth_adjacent_drop07.sh >> logs/dropout_ablation/L-oth-adjacent-drop07-390k/unit.log 2>&1'
# Relaunch after an interruption: the same command — training continues from ckpt/latest.pt.
set -u
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
export PYTHONPATH=$ROOT
NT=https://ntfy.sh/swirling-tornado-ai691k
TOPIC=dropout_ablation
NAME=L-oth-adjacent-drop07-390k
INST=oth-adjacent
STEPS=390000
DROPOUT=0.7
WAIT_UNIT=oth_adjacent_drop03
LOGS=$ROOT/logs/$TOPIC/$NAME
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" \
         -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM $NAME FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

[ -f "$ROOT/datasets/othello/$INST/train/train_20000000.npz" ] || fail "preflight" "train_20000000.npz missing"
[ -f "$ROOT/datasets/othello/$INST/edits/v1/cases_1001.pkl" ] || fail "preflight" "cases_1001.pkl missing"

stage "W wait for unit $WAIT_UNIT to exit (GPU handoff)"
for i in $(seq 1 720); do                      # up to 12 h
  systemctl --user is-active --quiet "$WAIT_UNIT" || break
  sleep 60
done
systemctl --user is-active --quiet "$WAIT_UNIT" && fail "W wait" "$WAIT_UNIT still active after 12 h"
sleep 30                                       # let its CUDA context release
ping "PIM $NAME: chain started" "oth-adjacent, dropout $DROPOUT, $STEPS steps (resumable) -> score -> tables. ~13 h on the 4090."

stage "C train (GPU) $TOPIC/$NAME  dropout $DROPOUT  steps $STEPS"
"$PY" -u scripts/train.py --env othello --instance "$INST" --arch transformer_l --dropout "$DROPOUT" \
    --topic "$TOPIC" --run-name "$NAME" --steps "$STEPS" --resume \
    >> "$LOGS/c_train.log" 2>&1 || fail "C training" "$(tail -20 "$LOGS/c_train.log")"
ping "PIM $NAME: training DONE" "$(grep -E '^done|best' "$LOGS/c_train.log" | tail -2)
Scoring next." checkered_flag

stage "D score + tables (GPU)"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace master_eval.ipynb --ExecutePreprocessor.timeout=21600 ) \
  > "$LOGS/d_score.log" 2>&1 || fail "D scoring" "$(tail -20 "$LOGS/d_score.log")"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace build_full_tables.ipynb ) \
  > "$LOGS/d_table.log" 2>&1 || fail "D table" "$(tail -20 "$LOGS/d_table.log")"

SCORES=$ROOT/runs/$TOPIC/$NAME/scores.json
ping "PIM $NAME: ALL DONE" \
"$("$PY" - "$SCORES" <<'PYEOF'
import json, sys
try:
    s = json.load(open(sys.argv[1])); g = s["gates"]; b = s["best"]
    print(f"val {s['val_loss']:.4f} | CE {g['ce']:.3f} (bayes {g['bayes_ce']:.3f}) legal_mass {g['legal_mass']:.3f}")
    print(f"skill LIN {max(s['probe_skill']['mine|linear|sequence']):+.3f} MLP {max(s['probe_skill']['mine|mlp|sequence']):+.3f} | "
          f"unedited {s['unedited']['edit_index_union']:+.3f} | " +
          " ".join(f"{k} {v['edit_index_union']:+.3f}/{v['fidelity_ratio']:.2f}" for k, v in b.items() if v))
except Exception as e:
    print(f"(summary failed: {e})")
PYEOF
)" white_check_mark
stage "chain complete"
