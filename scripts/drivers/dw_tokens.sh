#!/usr/bin/env bash
# ── L-dw-8ray-tok-20m: discworld dw-8ray as TOKENS through the Othello model ─────────────
#
# The interface ablation (2026-09-05). Same instance (dw-8ray), same 20M sequences, same
# 8 x 512 Transformer-L stack and recipe as ray_ablation/L-dw-8ray-20m — but every frame is
# ONE token of the instance's frame vocabulary (datasets/discworld/dw-8ray/tokens/), the
# input is an embedding table instead of Linear(8, 512), and the output is a softmax over
# frames trained with cross-entropy: exactly the Othello setup with a bigger vocabulary.
#
# Stages (each gated on the previous one; a failed stage pings and stops):
#   C  GPU  train 780k steps                                            (~8-13 h)
#   W       wait for the scorer-ready marker (the token scorer is built while C runs)
#   D  GPU  master_eval + build_full_table + training_curves
# Launch:
#   systemd-run --user --unit=dw_tok -p MemoryMax=40G --collect --working-directory=$PWD \
#       /usr/bin/bash -c 'bash scripts/drivers/dw_tokens.sh > logs/interface_ablation/L-dw-8ray-tok-20m/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
export PYTHONPATH=$ROOT
NT=https://ntfy.sh/swirling-tornado-ai691k
TOPIC=interface_ablation
NAME=L-dw-8ray-tok-20m
LOGS=$ROOT/logs/$TOPIC/$NAME
READY=$ROOT/experiments/dw_tokens/scorer_ready      # touched when master_eval can score token runs
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" \
         -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM $NAME FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM $NAME: chain started" "dw-8ray frames as tokens through the Othello Transformer-L (CE over the frame vocabulary). Train 780k -> wait for scorer -> score -> tables."

stage "C train (GPU) $TOPIC/$NAME"
"$PY" scripts/train.py --env discworld --repr tokens --instance dw-8ray --arch transformer_l \
    --topic "$TOPIC" --run-name "$NAME" --steps 780000 \
    > "$LOGS/c_train.log" 2>&1 || fail "C training" "$(tail -20 "$LOGS/c_train.log")"
ping "PIM $NAME: training DONE" "$(grep -E '^done|best' "$LOGS/c_train.log" | tail -2)" checkered_flag

stage "W wait for scorer-ready marker $READY"
if [ ! -f "$READY" ]; then
  ping "PIM $NAME: waiting for the token scorer" "Training is done; master_eval cannot score token runs yet. Waiting (up to 12 h) for $READY."
fi
for _ in $(seq 1 720); do [ -f "$READY" ] && break; sleep 60; done
[ -f "$READY" ] || fail "W scorer never became ready (12 h)" "touch $READY once master_eval scores token runs, then run stage D by hand"

stage "D score + tables (GPU)"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace master_eval.ipynb --ExecutePreprocessor.timeout=14400 ) \
  > "$LOGS/d_score.log" 2>&1 || fail "D scoring" "$(tail -20 "$LOGS/d_score.log")"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace build_full_table.ipynb ) \
  > "$LOGS/d_table.log" 2>&1 || fail "D table" "$(tail -20 "$LOGS/d_table.log")"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace training_curves.ipynb ) > "$LOGS/d_curves.log" 2>&1 || true

SCORES=$ROOT/runs/$TOPIC/$NAME/scores.json
ping "PIM $NAME: ALL DONE" \
"$("$PY" - "$SCORES" <<'PYEOF'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
    out = [f"val {s.get('val_loss', float('nan')):.5f}"]
    ps = s.get("probe_skill", {})
    out += [f"skill {k} {max(v):+.3f}" for k, v in list(ps.items())[:4] if isinstance(v, list) and v]
    u = s.get("unedited", {})
    if isinstance(u, dict):
        out += [f"unedited {u.get(k):+.3f}" for k in ("edit_index_union", "edit_index") if isinstance(u.get(k), (int, float))]
    for ed, b in (s.get("best") or {}).items():
        if b:
            ei = b.get("edit_index_union", b.get("edit_index", float("nan")))
            out.append(f"{ed} {ei:+.3f}/{b.get('fidelity_ratio', float('nan')):.2f}")
    print(" | ".join(out))
except Exception as e:  # never let the summary kill the ping
    print(f"(summary failed: {e})")
PYEOF
)" white_check_mark
stage "chain complete"
