#!/usr/bin/env bash
# ── probe-target controls, overnight 2026-09-09 → 10 ─────────────────────────────────────
#   Othello: the signed mine/theirs REGRESSION probe target on all three Othello runs
#            (+ floors) — fitted INLINE by master_eval (minutes per run). Runs FIRST, so the
#            notebook path is exercised within minutes of launch.
#   dw-8ray: the APPEARANCE target (30 cells = the runs of lit rays, the observation-exact
#            partition) on L-dw-8ray-20m and L-dw-8ray-tok-20m (+ floors), then a sweep of
#            target resolutions (depth-split ×2, ×3, the 16×8 grid; if before 06:00 PT:
#            appearance-lat, grid-8x4, grid-32x16), one probe per residual point, no floors.
#   Every fit persists into runs/<run>/probes/ or runs/_baselines/dw-8ray/probes/ BEFORE
#   master_eval reads it; master_eval only ADDS the blocks whose probes exist.
# Under a unit:
#   systemd-run --user --unit=probe_targets -p MemoryMax=45G --collect --working-directory=$PWD \
#     /usr/bin/bash -c 'bash scripts/drivers/probe_targets.sh > logs/probe_targets/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
export PYTHONPATH=$ROOT
PY=$ROOT/.pim/bin/python
LOGS=$ROOT/logs/probe_targets
NT=https://ntfy.sh/swirling-tornado-ai691k
FRAME=ray_ablation/L-dw-8ray-20m
TOK=interface_ablation/L-dw-8ray-tok-20m
DEADLINE=$(TZ=America/Los_Angeles date -d "tomorrow 06:00" +%s)
mkdir -p "$LOGS"

ping()  { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail()  { ping "PIM probe-targets FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }
nb()    { "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout=-1 "notebooks/$1.ipynb" > "$LOGS/$2.log" 2>&1; }
fit()   { # fit <run> <target> [--random-init|--observation]
          local tag; tag="fit_$(basename "$1")_$2${3:+_${3#--}}"
          "$PY" -u scripts/fit_probes.py --run "$1" --target "$2" ${3:-} > "$LOGS/$tag.log" 2>&1 \
            || fail "$tag" "$(tail -15 "$LOGS/$tag.log")"; }
score() { # score <label>: master_eval (adds every block whose probes exist) + the tables
          nb master_eval "master_eval_$1" || fail "master_eval ($1)" "$(tail -25 "$LOGS/master_eval_$1.log")"
          nb build_full_table "build_full_table_$1" || fail "build_full_table ($1)" "$(tail -25 "$LOGS/build_full_table_$1.log")"
          "$PY" scripts/drivers/probe_targets_headline.py > "$LOGS/headline_$1.txt" 2>&1 || true; }

ping "PIM probe-targets: chain started" "1 master_eval: Othello mine_signed x3 + floors (~1.5 h) → 2 dw-8ray appearance probes, 2 models (~40 min each) → 3 floors (~1.5 h) → 4 master_eval + tables → sweep d2, d3, grid-16x8 (~70 min each) → if before 06:00 PT: appearance-lat, grid-8x4, grid-32x16. ETA main chain ~03:00 PT."

stage "1 master_eval: Othello mine_signed (3 runs, inline fits) + Othello floors"
score othello
ping "PIM probe-targets: Othello mine_signed DONE" "$(grep mine_signed "$LOGS/headline_othello.txt" | head -6)" white_check_mark

stage "2 appearance probes (trained models)"
fit "$FRAME" appearance
fit "$TOK" appearance
ping "PIM probe-targets: appearance probes DONE" "$(grep -h 'best point' "$LOGS"/fit_*appearance.log | head -4)"

stage "3 appearance floors"
fit "$FRAME" appearance --random-init
fit "$TOK" appearance --random-init
fit "$FRAME" appearance --observation
ping "PIM probe-targets: floors DONE" "$(grep -h 'skill' "$LOGS"/fit_*appearance_*.log | tail -6)"

stage "4 master_eval: dw appearance blocks + floors + tables"
score main
ping "PIM probe-targets: MAIN RESULTS" "$(cat "$LOGS/headline_main.txt" | head -20)" white_check_mark

for v in appearance-d2 appearance-d3 grid-16x8; do
  stage "5 variant $v"
  fit "$FRAME" "$v"
  fit "$TOK" "$v"
  score "$v"
  ping "PIM probe-targets: variant $v scored" "$(grep "$v" "$LOGS/headline_$v.txt" | head -6)"
done

if [ "$(date +%s)" -lt "$DEADLINE" ]; then
  for v in appearance-lat grid-8x4 grid-32x16; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then stage "deadline reached before $v — stopping the sweep"; break; fi
    stage "6 extra variant $v"
    fit "$FRAME" "$v"
    fit "$TOK" "$v"
    score "$v"
    ping "PIM probe-targets: extra variant $v scored" "$(grep "$v" "$LOGS/headline_$v.txt" | head -6)"
  done
else
  stage "past 06:00 PT after the first three variants — no extra variants"
fi

LAST=$(ls -t "$LOGS"/headline_*.txt | head -1)
ping "PIM probe-targets: ALL DONE" "$(cat "$LAST" | head -30)" white_check_mark
stage "chain complete"
