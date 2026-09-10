#!/usr/bin/env bash
# ── probe-target resolution sweep, chain 2 (queued behind chain 1) ────────────────────────
#   Sevan's additions (2026-09-09 23:30 PT): the MISALIGNED 30-cell controls — product grids
#   with the appearance partition's cell count but a structure that is NOT the runs of rays
#   (grid-6x5, grid-10x3) — and the two ends of the resolution gradient (grid-4x2 coarser
#   than anything, grid-64x32 far finer than the frame). One probe set per model, no floors,
#   each scored into the two 8-ray runs' scores.json and the tables as it lands.
#   Starts when chain 1 (scripts/drivers/probe_targets.sh, unit probe_targets) writes its
#   "chain complete" marker — or has stopped without it — so the GPU is never shared.
#   Nothing here depends on an agent session: a systemd unit runs it, ntfy carries the pings.
# Under a unit:
#   systemd-run --user --unit=probe_targets_2 -p MemoryMax=45G --collect --working-directory=$PWD \
#     /usr/bin/bash -c 'bash scripts/drivers/probe_targets_2.sh > logs/probe_targets_2/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
export PYTHONPATH=$ROOT
PY=$ROOT/.pim/bin/python
LOGS=$ROOT/logs/probe_targets_2
PREV=$ROOT/logs/probe_targets/driver.log
NT=https://ntfy.sh/swirling-tornado-ai691k
FRAME=ray_ablation/L-dw-8ray-20m
TOK=interface_ablation/L-dw-8ray-tok-20m
mkdir -p "$LOGS"

ping()  { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail()  { ping "PIM probe-targets-2 FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }
nb()    { "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout=-1 "notebooks/$1.ipynb" > "$LOGS/$2.log" 2>&1; }
fit()   { local tag; tag="fit_$(basename "$1")_$2"
          "$PY" -u scripts/fit_probes.py --run "$1" --target "$2" > "$LOGS/$tag.log" 2>&1 \
            || fail "$tag" "$(tail -15 "$LOGS/$tag.log")"; }
score() { nb master_eval "master_eval_$1" || fail "master_eval ($1)" "$(tail -25 "$LOGS/master_eval_$1.log")"
          nb build_full_table "build_full_table_$1" || fail "build_full_table ($1)" "$(tail -25 "$LOGS/build_full_table_$1.log")"
          "$PY" scripts/drivers/probe_targets_headline.py > "$LOGS/headline_$1.txt" 2>&1 || true; }

stage "0 waiting for chain 1 (unit probe_targets) to complete"
waited=0
until grep -q "chain complete" "$PREV" 2>/dev/null || [ "$(systemctl --user is-active probe_targets 2>/dev/null)" != "active" ]; do
  sleep 120; waited=$((waited + 120))
  if [ $waited -ge 43200 ]; then fail "wait" "chain 1 still running after 12 h — not starting chain 2 on a shared GPU"; fi
done
if ! grep -q "chain complete" "$PREV" 2>/dev/null; then
  ping "PIM probe-targets-2: chain 1 ended WITHOUT its completion marker" "$(tail -3 "$PREV")  — starting chain 2 anyway" warning
fi
ping "PIM probe-targets-2: chain started" "misaligned 30-cell controls grid-6x5, grid-10x3; then the gradient ends grid-4x2, grid-64x32 (~75 min each, both 8-ray models)"

for v in grid-6x5 grid-10x3 grid-4x2 grid-64x32; do
  stage "variant $v"
  fit "$FRAME" "$v"
  fit "$TOK" "$v"
  score "$v"
  ping "PIM probe-targets-2: $v scored" "$(grep -E "$v|appearance " "$LOGS/headline_$v.txt" | head -8)"
done

ping "PIM probe-targets-2: ALL DONE" "$(cat "$LOGS/headline_grid-64x32.txt" | grep -v mine_signed | head -30)" white_check_mark
stage "chain complete"
