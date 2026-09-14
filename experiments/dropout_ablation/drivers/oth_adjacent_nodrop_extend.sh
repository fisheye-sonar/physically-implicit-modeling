#!/usr/bin/env bash
# ── Extend L-oth-adjacent-nodrop-390k to 780k steps as a NEW run, L-oth-adjacent-nodrop-20m (2026-09-12) ──
#
# The 390k run stays as a scored snapshot. Its resumable state (ckpt/latest.pt at step 390k) is
# copied into a new run dir WITHOUT its scores.json / probes (those describe the 390k model), and
# training continues there with --resume to 780k: same optimizer state, RNG and batch stream, so
# the result is what an uninterrupted 780k no-dropout run would have produced. Then master_eval
# scores the new run fresh (oth-adjacent floors already exist) and the tables are rebuilt.
#   systemd-run --user --unit=oth_adjacent_nodrop_ext -p MemoryMax=40G --collect --working-directory=$PWD \
#       /usr/bin/bash -c 'bash experiments/dropout_ablation/drivers/oth_adjacent_nodrop_extend.sh > logs/dropout_ablation/L-oth-adjacent-nodrop-20m/unit.log 2>&1'
# Relaunch after an interruption: the same command (resume continues from the latest state).
set -u
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$PWD; PY=$ROOT/.pim/bin/python; export PYTHONPATH=$ROOT
NT=https://ntfy.sh/swirling-tornado-ai691k
TOPIC=dropout_ablation; SRC=L-oth-adjacent-nodrop-390k; NAME=L-oth-adjacent-nodrop-20m; INST=oth-adjacent; STEPS=780000
LOGS=$ROOT/logs/$TOPIC/$NAME; mkdir -p "$LOGS"; echo $$ > "$LOGS/driver.pid"
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM $NAME FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM $NAME: chain started" "extending the no-dropout oth-adjacent run 390k -> 780k (resume; ~12.5 h on the 4090) -> score -> tables."

stage "X copy the 390k resumable state into $NAME"
if [ ! -d "$ROOT/runs/$TOPIC/$NAME" ]; then
  [ -f "$ROOT/runs/$TOPIC/$SRC/ckpt/latest.pt" ] || fail "X copy" "no ckpt/latest.pt in $SRC"
  rsync -a --exclude scores.json --exclude probes/ --exclude figures/ "$ROOT/runs/$TOPIC/$SRC/" "$ROOT/runs/$TOPIC/$NAME/" || fail "X copy" "rsync failed"
  printf "\n## %s\n- copy \`runs/%s/%s/\` (checkpoints, config, metrics; NOT scores.json / probes) → \`runs/%s/%s/\` — the no-dropout run continued to 780k under a new name; the 390k dir stays as the scored snapshot\n" "$(date +%F)" "$TOPIC" "$SRC" "$TOPIC" "$NAME" >> "$ROOT/runs/MOVES.md"
else
  echo "  $NAME exists — resuming it" | tee -a "$LOGS/driver.log"
fi

stage "C train (GPU) $TOPIC/$NAME  --resume to $STEPS  dropout 0"
"$PY" -u scripts/train.py --env othello --instance "$INST" --arch transformer_l --dropout 0 \
    --topic "$TOPIC" --run-name "$NAME" --steps "$STEPS" --resume \
    >> "$LOGS/c_train.log" 2>&1 || fail "C training" "$(tail -20 "$LOGS/c_train.log")"
ping "PIM $NAME: training DONE" "$(grep -E '^done|best' "$LOGS/c_train.log" | tail -2)
Scoring next." checkered_flag

stage "D score + tables (GPU)"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook --execute --inplace master_eval.ipynb --ExecutePreprocessor.timeout=21600 ) \
  > "$LOGS/d_score.log" 2>&1 || fail "D scoring" "$(tail -20 "$LOGS/d_score.log")"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook --execute --inplace build_full_table.ipynb ) \
  > "$LOGS/d_table.log" 2>&1 || fail "D table" "$(tail -20 "$LOGS/d_table.log")"
SCORES=$ROOT/runs/$TOPIC/$NAME/scores.json
ping "PIM $NAME: ALL DONE" "$("$PY" - "$SCORES" <<'PYEOF'
import json, sys
try:
    s = json.load(open(sys.argv[1])); g = s["gates"]; b = s["best"]
    print(f"val {s['val_loss']:.4f} | CE {g['ce']:.3f} (bayes {g['bayes_ce']:.3f}) legal_mass {g['legal_mass']:.3f}")
    print(f"skill LIN {max(s['probe_skill']['mine|linear|sequence']):+.3f} MLP {max(s['probe_skill']['mine|mlp|sequence']):+.3f} | unedited {s['unedited']['edit_index_union']:+.3f} | " + " ".join(f"{k} {v['edit_index_union']:+.3f}/{v['fidelity_ratio']:.2f}" for k, v in b.items() if v))
except Exception as e:
    print(f"(summary failed: {e})")
PYEOF
)" white_check_mark
stage "chain complete"
