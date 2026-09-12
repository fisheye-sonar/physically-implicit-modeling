#!/usr/bin/env bash
# ── score whatever master_eval finds unscored, then rebuild the tables ──────────────────
#   Generic: master_eval adds the probe-target blocks the SETTINGS ask for and skips every
#   run that is current; build_full_table redraws; the headline script prints one line per
#   extra-target block. Log dir = logs/<name>/ (arg 1, default score_pending).
# Under a unit:
#   systemd-run --user --unit=<name> -p MemoryMax=45G --collect --working-directory=$PWD \
#     /usr/bin/bash -c 'bash scripts/drivers/score_pending.sh <name> > logs/<name>/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
export PYTHONPATH=$ROOT
PY=$ROOT/.pim/bin/python
NAME=${1:-score_pending}
LOGS=$ROOT/logs/$NAME
NT=https://ntfy.sh/swirling-tornado-ai691k
mkdir -p "$LOGS"

ping()  { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail()  { ping "PIM $NAME FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }
nb()    { "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout=-1 "notebooks/$1.ipynb" > "$LOGS/$1.log" 2>&1; }

stage "master_eval"
nb master_eval || fail "master_eval" "$(tail -25 "$LOGS/master_eval.log")"
stage "build_full_table"
nb build_full_table || fail "build_full_table" "$(tail -25 "$LOGS/build_full_table.log")"
"$PY" scripts/drivers/probe_targets_headline.py > "$LOGS/headline.txt" 2>&1 || true
ping "PIM $NAME: DONE" "$(grep -E "pos@|snapped" "$LOGS/headline.txt" | head -8; grep -E "wrote|adding blocks|SKIPPED" "$LOGS/master_eval.log" | head -6)" white_check_mark
stage "chain complete"
