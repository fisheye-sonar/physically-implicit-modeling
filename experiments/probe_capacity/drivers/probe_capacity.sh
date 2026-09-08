#!/usr/bin/env bash
# ── probe-capacity sweep: discworld then Othello, one GPU job at a time ─────────────
set -u
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
NT=https://ntfy.sh/swirling-tornado-ai691k
LOGS=$ROOT/logs/probe_capacity
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM capacity sweep FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

test -f datasets/discworld/dw-pn04/probe_250k/test.h5 || fail "corpus" "discworld probe_250k missing"
ping "PIM capacity sweep: started" "widths LIN,16,64,128,512,1024,2048 x trained/random-init/observation; discworld (250k seq) then Othello (170k games). ~4 h."
for env in discworld othello; do
  stage "$env"
  PYTHONPATH=$ROOT "$PY" -u experiments/probe_capacity/scripts/probe_capacity.py "$env" \
    > "$LOGS/$env.log" 2>&1 || fail "$env" "$(tail -15 "$LOGS/$env.log")"
done
ping "PIM capacity sweep: ALL DONE" "$("$PY" - <<'EOF'
import json
for env in ("discworld", "othello"):
    try:
        S = json.load(open(f"experiments/probe_capacity/scores/probe_capacity_{env}.json"))
        for src, cells in S["cells"].items():
            print(env, src, " ".join(f"{w}={c['skill']:+.3f}" for w, c in cells.items()))
    except Exception as e:
        print(env, "n/a", e)
EOF
)" white_check_mark
stage "chain complete"
