#!/usr/bin/env bash
# Score the token run with the discworld analysis through the adapter, then draw waterfalls.
#   systemd-run --user --unit=dw_tok_bridge -p MemoryMax=24G --collect --working-directory=$PWD \
#       bash experiments/dw_tokens/bridge/drivers/bridge.sh
set -u
cd "$(dirname "$0")/../../../.." || exit 1
ROOT=$PWD; export PYTHONPATH=$ROOT
NT=https://ntfy.sh/swirling-tornado-ai691k
LOGS=$ROOT/logs/dw_tokens/bridge; mkdir -p "$LOGS"
PY=$ROOT/.pim/bin/python
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM dw_tokens_bridge FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }
stage "A score (bridge, expected frame + argmax feedback)"
"$PY" -u experiments/dw_tokens/bridge/scripts/score_bridge.py > "$LOGS/score.log" 2>&1 || fail "score" "$(tail -15 "$LOGS/score.log")"
stage "B waterfalls"
"$PY" -u experiments/dw_tokens/bridge/scripts/waterfalls.py > "$LOGS/waterfalls.log" 2>&1 || fail "waterfalls" "$(tail -15 "$LOGS/waterfalls.log")"
stage "chain complete"
