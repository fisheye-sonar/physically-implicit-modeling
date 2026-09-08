#!/usr/bin/env bash
# ── Recurrent-L on dw-pn04 (the canonical 20M, noisy instance): train, score, tables ──
#
# The recomputation test (pim/models/recurrent.py docstring). Matched recipe: 780k steps,
# batch 256, lr 1e-3 constant, wd 1e-4, clip 1, seed 0 — TrainConfig's defaults, the same
# ones every canonical run used. Then master_eval scores it (fresh probes, its own
# random-init and observation floors — a new arch triggers both automatically), and the
# tables rebuild. Stages gate on exit codes; a failed stage pings and stops.
set -u
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
NT=https://ntfy.sh/swirling-tornado-ai691k
LOGS=$ROOT/logs/recurrent/R-dw-20m
TOPIC=architecture_gate
NAME=R-dw-20m
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" \
         -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM Recurrent-L FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM Recurrent-L: started" "4x1024 GRU, 25.4M params, dw-pn04 20M. Train 780k steps, then score + tables."

stage "C train (GPU)"
"$PY" scripts/train.py --env discworld --arch recurrent_l --topic "$TOPIC" --run-name "$NAME" \
    --steps 780000 > "$LOGS/c_train.log" 2>&1 || fail "C training" "$(tail -20 "$LOGS/c_train.log")"
ping "PIM Recurrent-L: training DONE" "$(grep -E '^done|best' "$LOGS/c_train.log" | tail -2)
Scoring now: probes, both floors, all editors (~45 min)." checkered_flag

stage "D score (GPU)"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace master_eval.ipynb --ExecutePreprocessor.timeout=14400 ) \
  > "$LOGS/d_score.log" 2>&1 || fail "D scoring" "$(tail -20 "$LOGS/d_score.log")"

stage "E tables"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace build_full_table.ipynb ) \
  > "$LOGS/e_table.log" 2>&1 || fail "E table" "$(tail -20 "$LOGS/e_table.log")"

ping "PIM Recurrent-L: ALL DONE" "$("$PY" - "$ROOT/runs/$TOPIC/$NAME/scores.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"val {s['val_loss']:.5f}")
for basis, T in s["bases"].items():
    b = T["best"]
    print(f"{basis}: lin {max(T['probe_skill_linear']):+.3f} mlp {max(T['probe_skill_mlp']):+.3f} | "
          f"unedited {T['unedited']['edit_index']:+.3f} | PI {b['PI']['edit_index']:+.3f}/{b['PI']['fidelity_ratio']:.2f} "
          f"({b['PI']['dims']} pt{b['PI']['point']}) GS {b['GS']['edit_index']:+.3f}/{b['GS']['fidelity_ratio']:.2f}")
try:
    A = json.load(open("runs/_baselines/dw-pn04/baselines.json"))["archs"]["recurrent_l"]
    for basis, blk in A["bases"].items():
        print(f"floors {basis}: obs {blk['observation']['mlp']['skill']:+.3f}  random-init {blk['random_init']['mlp']['skill']:+.3f} (MLP)")
except Exception as e:
    print("floors: n/a", e)
EOF
)" white_check_mark
stage "chain complete"
