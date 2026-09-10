#!/usr/bin/env bash
# ── probe-target resolution sweep, chain 5: the NOISELESS run, remaining variants ──────────
#   Chain 4 (probe_targets_4) scored grid-8x4 and was oomd-killed 50 s into appearance-lat:
#   AppearanceTarget.cell_of ran the ray–disc test on all 15.6 M probe positions at once
#   (~60 GB at 128 rays). Fixed by chunking (grid_target.AppearanceTarget.CHUNK); this chain
#   runs the three variants chain 4 did not reach.
#   Sevan (2026-09-10 05:30 PT): does L-dw-noiseless-20m (128 rays, radius 0.5) show the same
#   resolution trend as dw-8ray, and does it prefer a finer grid? Around its existing 16x8
#   control: grid-8x4 (32, coarser), appearance-lat (233 — the frame's lateral partition,
#   depth dropped), grid-32x16 (512), grid-64x32 (2048, last so a memory failure loses
#   nothing else). One probe set per variant, no floors. The full appearance partition
#   (2,889 cells) is left out: its label tensor alone is 22 GB on the GPU.
# Under a unit:
#   systemd-run --user --unit=probe_targets_5 -p MemoryMax=45G --collect --working-directory=$PWD \
#     /usr/bin/bash -c 'bash scripts/drivers/probe_targets_5.sh > logs/probe_targets_5/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
export PYTHONPATH=$ROOT
PY=$ROOT/.pim/bin/python
LOGS=$ROOT/logs/probe_targets_5
NT=https://ntfy.sh/swirling-tornado-ai691k
RUN=noise_ablation/L-dw-noiseless-20m
mkdir -p "$LOGS"

ping()  { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail()  { ping "PIM probe-targets-5 FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }
nb()    { "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout=-1 "notebooks/$1.ipynb" > "$LOGS/$2.log" 2>&1; }
fit()   { local tag; tag="fit_$(basename "$1")_$2"
          "$PY" -u scripts/fit_probes.py --run "$1" --target "$2" > "$LOGS/$tag.log" 2>&1 \
            || fail "$tag" "$(tail -15 "$LOGS/$tag.log")"; }
score() { nb master_eval "master_eval_$1" || fail "master_eval ($1)" "$(tail -25 "$LOGS/master_eval_$1.log")"
          nb build_full_table "build_full_table_$1" || fail "build_full_table ($1)" "$(tail -25 "$LOGS/build_full_table_$1.log")"
          "$PY" scripts/drivers/probe_targets_headline.py > "$LOGS/headline_$1.txt" 2>&1 || true; }

ping "PIM probe-targets-5: chain started" "noiseless sweep, the variants chain 4 did not reach (appearance-lat OOM fixed by chunked labelling): appearance-lat, grid-32x16, grid-64x32"
for v in appearance-lat grid-32x16 grid-64x32; do
  stage "variant $v"
  fit "$RUN" "$v"
  score "$v"
  ping "PIM probe-targets-5: $v scored" "$(grep -E "noiseless" "$LOGS/headline_$v.txt" | head -8)"
done
ping "PIM probe-targets-5: ALL DONE — noiseless sweep in" "$(grep noiseless "$LOGS/headline_grid-64x32.txt" | head -12)" white_check_mark
stage "chain complete"
