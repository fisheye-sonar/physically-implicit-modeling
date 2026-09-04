#!/usr/bin/env bash
# ── Recurrent-L on a discworld instance: train, score, tables ────────────────────────
#   usage: recurrent.sh <instance> <run-name>      e.g.  recurrent.sh dw-noiseless R-dw-noiseless-20m
#
# The parameterised form of recurrent_dw.sh (the first run's driver, kept as its record).
# Matched recipe: 780k steps, batch 256, lr 1e-3 constant, wd 1e-4, clip 1, seed 0 —
# TrainConfig's defaults. Then master_eval scores the run (fresh probes; a new
# (instance, arch) pair triggers both floors automatically) and the tables rebuild.
set -u
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
NT=https://ntfy.sh/swirling-tornado-ai691k
INST=${1:?instance}
NAME=${2:?run-name}
TOPIC=architecture_gate
LOGS=$ROOT/logs/recurrent/$NAME
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" \
         -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM $NAME FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM $NAME: started" "Recurrent-L (4x1024 GRU) on $INST. Train 780k (~5 h), then score + tables."

stage "C train (GPU) $INST -> $TOPIC/$NAME"
"$PY" scripts/train.py --env discworld --arch recurrent_l --instance "$INST" \
    --topic "$TOPIC" --run-name "$NAME" --steps 780000 \
    > "$LOGS/c_train.log" 2>&1 || fail "C training" "$(tail -20 "$LOGS/c_train.log")"
ping "PIM $NAME: training DONE" "$(grep -E '^done|best' "$LOGS/c_train.log" | tail -2)
Scoring now (~1 h)." checkered_flag

stage "D score (GPU)"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace master_eval.ipynb --ExecutePreprocessor.timeout=14400 ) \
  > "$LOGS/d_score.log" 2>&1 || fail "D scoring" "$(tail -20 "$LOGS/d_score.log")"

stage "E tables"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace build_full_table.ipynb ) \
  > "$LOGS/e_table.log" 2>&1 || fail "E table" "$(tail -20 "$LOGS/e_table.log")"

ping "PIM $NAME: ALL DONE" "$("$PY" - "$ROOT/runs/$TOPIC/$NAME/scores.json" "$INST" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"val {s['val_loss']:.5f}")
for basis, T in s["bases"].items():
    b = T["best"]
    print(f"{basis}: lin {max(T['probe_skill_linear']):+.3f} mlp {max(T['probe_skill_mlp']):+.3f} | "
          f"unedited {T['unedited']['edit_index']:+.3f} | PI {b['PI']['edit_index']:+.3f}/{b['PI']['fidelity_ratio']:.2f} "
          f"({b['PI']['dims']} pt{b['PI']['point']}) GS {b['GS']['edit_index']:+.3f}/{b['GS']['fidelity_ratio']:.2f}")
try:
    A = json.load(open(f"runs/_baselines/{sys.argv[2]}/baselines.json"))["archs"]["recurrent_l"]
    for basis, blk in A["bases"].items():
        print(f"floors {basis}: obs {blk['observation']['mlp']['skill']:+.3f}  random-init {blk['random_init']['mlp']['skill']:+.3f} (MLP)")
except Exception as e:
    print("floors: n/a", e)
EOF
)" white_check_mark
stage "chain complete"
