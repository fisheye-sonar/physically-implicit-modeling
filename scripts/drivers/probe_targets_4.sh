#!/usr/bin/env bash
# ── probe-target resolution sweep, chain 4: the NOISELESS run (queued behind chain 3) ─────
#   Sevan (2026-09-10 05:30 PT): does L-dw-noiseless-20m (128 rays, radius 0.5) show the same
#   resolution trend as dw-8ray, and does it prefer a finer grid? Around its existing 16x8
#   control: grid-8x4 (32, coarser), appearance-lat (233 — the frame's lateral partition,
#   depth dropped), grid-32x16 (512), grid-64x32 (2048, last so a memory failure loses
#   nothing else). One probe set per variant, no floors. The full appearance partition
#   (2,889 cells) is left out: its label tensor alone is 22 GB on the GPU.
# Under a unit:
#   systemd-run --user --unit=probe_targets_4 -p MemoryMax=45G --collect --working-directory=$PWD \
#     /usr/bin/bash -c 'bash scripts/drivers/probe_targets_4.sh > logs/probe_targets_4/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
export PYTHONPATH=$ROOT
PY=$ROOT/.pim/bin/python
LOGS=$ROOT/logs/probe_targets_4
PREV=$ROOT/logs/probe_targets_3/driver.log
NT=https://ntfy.sh/swirling-tornado-ai691k
RUN=noise_ablation/L-dw-noiseless-20m
mkdir -p "$LOGS"

ping()  { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail()  { ping "PIM probe-targets-4 FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }
nb()    { "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout=-1 "notebooks/$1.ipynb" > "$LOGS/$2.log" 2>&1; }
fit()   { local tag; tag="fit_$(basename "$1")_$2"
          "$PY" -u scripts/fit_probes.py --run "$1" --target "$2" > "$LOGS/$tag.log" 2>&1 \
            || fail "$tag" "$(tail -15 "$LOGS/$tag.log")"; }
score() { nb master_eval "master_eval_$1" || fail "master_eval ($1)" "$(tail -25 "$LOGS/master_eval_$1.log")"
          nb build_full_table "build_full_table_$1" || fail "build_full_table ($1)" "$(tail -25 "$LOGS/build_full_table_$1.log")"
          "$PY" scripts/drivers/probe_targets_headline.py > "$LOGS/headline_$1.txt" 2>&1 || true; }

stage "0 waiting for chain 3 (unit probe_targets_3) to complete"
waited=0
until grep -q "chain complete" "$PREV" 2>/dev/null || [ "$(systemctl --user is-active probe_targets_3 2>/dev/null)" != "active" ]; do
  sleep 120; waited=$((waited + 120))
  if [ $waited -ge 57600 ]; then fail "wait" "chain 3 still running after 16 h"; fi
done
ping "PIM probe-targets-4: chain started" "noiseless resolution sweep: grid-8x4, appearance-lat, grid-32x16, grid-64x32 on L-dw-noiseless-20m (~40 min each)"
for v in grid-8x4 appearance-lat grid-32x16 grid-64x32; do
  stage "variant $v"
  fit "$RUN" "$v"
  score "$v"
  ping "PIM probe-targets-4: $v scored" "$(grep -E "noiseless" "$LOGS/headline_$v.txt" | head -8)"
done
ping "PIM probe-targets-4: ALL DONE — noiseless sweep in" "$(grep noiseless "$LOGS/headline_grid-64x32.txt" | head -12)" white_check_mark
stage "chain complete"
