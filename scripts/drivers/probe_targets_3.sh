#!/usr/bin/env bash
# ── probe-target sweep, chain 3: redo grid-32x16 (queued behind chain 2) ─────────────────
#   Chain 1 was OOM-killed at its 45 GB cap on grid-32x16 (2026-09-10 04:47): the streamed
#   fitter materialised the train split's predictions and labels as int64 (rows × 512 tiles
#   ≈ 2 × 25 GB) for its in-sample statistics. Fixed in pim/probes/baselines.py (streamed
#   error counts, identical numbers). This chain refits that one variant on both 8-ray
#   models and scores it, after chain 2 (which fits grid-64x32 with the fixed code) is done.
# Under a unit:
#   systemd-run --user --unit=probe_targets_3 -p MemoryMax=45G --collect --working-directory=$PWD \
#     /usr/bin/bash -c 'bash scripts/drivers/probe_targets_3.sh > logs/probe_targets_3/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
export PYTHONPATH=$ROOT
PY=$ROOT/.pim/bin/python
LOGS=$ROOT/logs/probe_targets_3
PREV=$ROOT/logs/probe_targets_2/driver.log
NT=https://ntfy.sh/swirling-tornado-ai691k
FRAME=ray_ablation/L-dw-8ray-20m
TOK=interface_ablation/L-dw-8ray-tok-20m
mkdir -p "$LOGS"

ping()  { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail()  { ping "PIM probe-targets-3 FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }
nb()    { "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout=-1 "notebooks/$1.ipynb" > "$LOGS/$2.log" 2>&1; }
fit()   { local tag; tag="fit_$(basename "$1")_$2"
          "$PY" -u scripts/fit_probes.py --run "$1" --target "$2" > "$LOGS/$tag.log" 2>&1 \
            || fail "$tag" "$(tail -15 "$LOGS/$tag.log")"; }
score() { nb master_eval "master_eval_$1" || fail "master_eval ($1)" "$(tail -25 "$LOGS/master_eval_$1.log")"
          nb build_full_table "build_full_table_$1" || fail "build_full_table ($1)" "$(tail -25 "$LOGS/build_full_table_$1.log")"
          "$PY" scripts/drivers/probe_targets_headline.py > "$LOGS/headline_$1.txt" 2>&1 || true; }

stage "0 waiting for chain 2 (unit probe_targets_2) to complete"
waited=0
until grep -q "chain complete" "$PREV" 2>/dev/null || [ "$(systemctl --user is-active probe_targets_2 2>/dev/null)" != "active" ]; do
  sleep 120; waited=$((waited + 120))
  if [ $waited -ge 43200 ]; then fail "wait" "chain 2 still running after 12 h"; fi
done
ping "PIM probe-targets-3: chain started" "redoing grid-32x16 on both 8-ray models with the streamed-stats fitter (~90 min)"
for v in grid-32x16; do
  stage "variant $v"
  fit "$FRAME" "$v"
  fit "$TOK" "$v"
  score "$v"
  ping "PIM probe-targets-3: $v scored" "$(grep -E "$v|appearance " "$LOGS/headline_$v.txt" | head -8)"
done
ping "PIM probe-targets-3: ALL DONE — the whole resolution sweep is in" "$(grep -v mine_signed "$LOGS/headline_grid-32x16.txt" | head -30)" white_check_mark
stage "chain complete"
