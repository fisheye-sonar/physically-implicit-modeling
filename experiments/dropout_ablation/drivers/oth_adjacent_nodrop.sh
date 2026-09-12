#!/usr/bin/env bash
# ── L-oth-adjacent-nodrop-390k: Transformer-L on oth-adjacent with DROPOUT 0 (2026-09-11) ──
#
# The dropout ablation for the materialisation theory: minGPT's dropout 0.1 on the embedding,
# attention and residual paths rewards writing a variable in many directions, so part of the
# 85–90 orthogonal colour copies on oth-adjacent (and the K=64 rescue) may be a regulariser
# effect rather than the adjacency rule's fused computation. Identical to
# adjacency_ablation/L-oth-adjacent-20m except: --dropout 0, and 390k steps (half; Othello
# editability is at ~95% of its 780k value by then — findings/training-curve.md — and the run
# is RESUMABLE, so it can be extended to 780k later with --resume --steps 780000).
#
# Stages (each gated on the previous one; a failed stage pings and stops):
#   W  wait for the oth-adjacent data build (corpus + cases + labels) to complete
#   C  GPU  train 390k steps, dropout 0, resumable                            (~12.5 h on the 4090)
#   D  GPU  master_eval (scores + oth-adjacent floors) + build_full_table + curves
# Launch (on wsl-sevan):
#   systemd-run --user --unit=oth_adjacent_nodrop -p MemoryMax=40G --collect --working-directory=$PWD \
#       /usr/bin/bash -c 'bash experiments/dropout_ablation/drivers/oth_adjacent_nodrop.sh > logs/dropout_ablation/L-oth-adjacent-nodrop-390k/unit.log 2>&1'
# Relaunch after an interruption: the same command — training continues from ckpt/latest.pt.
set -u
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
export PYTHONPATH=$ROOT
NT=https://ntfy.sh/swirling-tornado-ai691k
TOPIC=dropout_ablation
NAME=L-oth-adjacent-nodrop-390k
INST=oth-adjacent
STEPS=390000
LOGS=$ROOT/logs/$TOPIC/$NAME
BUILD_LOG=$ROOT/logs/dropout_ablation/build_oth_adjacent.log
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" \
         -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM $NAME FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM $NAME: chain started" "oth-adjacent, dropout 0, $STEPS steps (resumable) -> score -> tables. ~13 h on the 4090."

stage "W wait for the oth-adjacent data build"
for i in $(seq 1 240); do
  grep -q "data build complete" "$BUILD_LOG" 2>/dev/null && break
  grep -q -E "Traceback|Error" "$BUILD_LOG" 2>/dev/null && fail "W data build" "$(tail -5 "$BUILD_LOG")"
  sleep 60
done
grep -q "data build complete" "$BUILD_LOG" || fail "W data build" "timed out waiting 4 h for $BUILD_LOG"
[ -f "$ROOT/datasets/othello/$INST/train/train_20000000.npz" ] || fail "W data build" "train_20000000.npz missing"
[ -f "$ROOT/datasets/othello/$INST/edits/v1/cases_1001.pkl" ] || fail "W data build" "cases_1001.pkl missing"

stage "C train (GPU) $TOPIC/$NAME  dropout 0  steps $STEPS"
"$PY" -u scripts/train.py --env othello --instance "$INST" --arch transformer_l --dropout 0 \
    --topic "$TOPIC" --run-name "$NAME" --steps "$STEPS" --resume \
    >> "$LOGS/c_train.log" 2>&1 || fail "C training" "$(tail -20 "$LOGS/c_train.log")"
ping "PIM $NAME: training DONE" "$(grep -E '^done|best' "$LOGS/c_train.log" | tail -2)
Scoring next." checkered_flag

stage "D score + tables (GPU)"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace master_eval.ipynb --ExecutePreprocessor.timeout=21600 ) \
  > "$LOGS/d_score.log" 2>&1 || fail "D scoring" "$(tail -20 "$LOGS/d_score.log")"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace build_full_table.ipynb ) \
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
