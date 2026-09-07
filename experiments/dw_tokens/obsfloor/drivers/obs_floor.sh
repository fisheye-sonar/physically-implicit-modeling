#!/usr/bin/env bash
# One-hot observation floors for tokenised dw-8ray. Launch under a capped unit:
#   systemd-run --user --unit=dw_tok_obsfloor -p MemoryMax=24G --collect --working-directory=$PWD \
#       bash experiments/dw_tokens/obsfloor/drivers/obs_floor.sh
set -u
cd "$(dirname "$0")/../../../.." || exit 1
ROOT=$PWD; export PYTHONPATH=$ROOT
NT=https://ntfy.sh/swirling-tornado-ai691k
LOGS=$ROOT/logs/dw_tokens/obsfloor; mkdir -p "$LOGS"
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
echo "=== [$(date '+%F %T')] STAGE obs_floor ===" | tee -a "$LOGS/driver.log"
"$ROOT/.pim/bin/python" -u experiments/dw_tokens/obsfloor/scripts/obs_floor.py > "$LOGS/obs_floor.log" 2>&1 \
  || { ping "PIM dw_tokens_obsfloor FAILED" "$(tail -15 "$LOGS/obs_floor.log")" warning; echo "FAILED" >> "$LOGS/driver.log"; exit 1; }
echo "=== [$(date '+%F %T')] STAGE chain complete ===" | tee -a "$LOGS/driver.log"
