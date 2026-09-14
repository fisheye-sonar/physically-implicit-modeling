#!/usr/bin/env bash
# dw-smooth GENERATION (2026-09-12 night): the anti-aliasing instance — dw-noiseless geometry
# (128 rays, radius 0.5, no noise) with the smooth power-dome disc profile (1-(perp/r)^2)^2 baked
# into the renderer (SimConfig.soft_shading="power", soft_profile_power=2.0; profile H of
# experiments/antialias_pilot, Sevan's pick). CPU only; runs as its own unit so the GPU chain
# (scripts/drivers/overnight_2026-09-12.sh) can start the moment it finishes.
#   systemd-run --user --unit=dw_smooth_gen -p MemoryMax=40G --collect --working-directory=$PWD \
#       /usr/bin/bash -c 'bash scripts/drivers/dw_smooth_gen.sh > logs/smooth_ablation/dw_smooth_gen/unit.log 2>&1'
# Stages: eval 10k -> edits 10k (EF=20, always in frustum) -> probe 120k -> probe 250k -> 20M corpus
# (40 shards, ~410 GB memmap, ~1 h) -> edit selection (first 1000 cases with >= 2 differing rays).
# Seeds (bigcorpus._SMOOTH_RANGES): train 200e9, eval 225.2e9, edits 225.3e9, probe 1040e9+200,
# probe_large 1050e9+200 — all forbidden to every other instance and vice versa.
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
export PYTHONPATH=$ROOT
NT=https://ntfy.sh/swirling-tornado-ai691k
INST=dw-smooth
LOGS=$ROOT/logs/smooth_ablation/dw_smooth_gen
mkdir -p "$LOGS"
ping()  { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail()  { ping "PIM dw-smooth gen FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

INST_DIR=$ROOT/datasets/discworld/$INST
SIM_FLAGS=(--n-objects 2 --frames 40 --obs-res 128 --boundary open
           --position-noise 0.0 --obs-noise-std 0.0 --fixed-reflectivities --always-in-frustum
           --soft-shading power --soft-profile-power 2.0)

stage "G1 eval split"
if [ ! -f "$INST_DIR/eval/test.h5" ]; then
  "$PY" scripts/generate_dataset.py --role eval --instance "$INST" --n 10000 \
      "${SIM_FLAGS[@]}" --seed 225200000000 --n-workers 16 --compression-level 4 \
      > "$LOGS/g1_eval.log" 2>&1 || fail "G1 eval split" "$(tail -15 "$LOGS/g1_eval.log")"
else echo "  eval split already present — skipping" | tee -a "$LOGS/driver.log"; fi

stage "G2 edit bench"
if [ ! -f "$INST_DIR/edits/v1/edits.h5" ]; then
  "$PY" scripts/generate_dataset.py --role edits --instance "$INST" --n 10000 \
      "${SIM_FLAGS[@]}" --edit-frame 20 --edit-always-in-frustum \
      --seed 225300000000 --n-workers 16 --compression-level 4 \
      > "$LOGS/g2_edits.log" 2>&1 || fail "G2 edit bench" "$(tail -15 "$LOGS/g2_edits.log")"
else echo "  edit bench already present — skipping" | tee -a "$LOGS/driver.log"; fi

stage "G3 probe corpora"
if [ ! -f "$INST_DIR/probe/probe_120k.h5" ]; then
  "$PY" scripts/generate_dataset.py --role probe --size 120k --instance "$INST" --n 120000 \
      "${SIM_FLAGS[@]}" --edit-frame 20 --seed 1040000000200 --n-workers 16 --compression-level 4 \
      > "$LOGS/g3_probe_120k.log" 2>&1 || fail "G3 probe 120k" "$(tail -15 "$LOGS/g3_probe_120k.log")"
else echo "  probe 120k already present — skipping" | tee -a "$LOGS/driver.log"; fi
if [ ! -f "$INST_DIR/probe/probe_250k.h5" ]; then
  "$PY" scripts/generate_dataset.py --role probe --size 250k --instance "$INST" --n 250000 \
      "${SIM_FLAGS[@]}" --edit-frame 20 --seed 1050000000200 --n-workers 16 --compression-level 4 \
      > "$LOGS/g3_probe_250k.log" 2>&1 || fail "G3 probe 250k" "$(tail -15 "$LOGS/g3_probe_250k.log")"
else echo "  probe 250k already present — skipping" | tee -a "$LOGS/driver.log"; fi
ping "PIM dw-smooth gen: splits DONE" "eval / edits / probe 120k / probe 250k written. Starting the 20M corpus (~1 h)."

stage "G4 20M corpus"
"$PY" -u -m pim.environments.discworld.bigcorpus "$INST" > "$LOGS/g4_corpus.log" 2>&1 \
    || fail "G4 20M corpus" "$(tail -20 "$LOGS/g4_corpus.log")"

stage "G5 edit selection"
"$PY" scripts/make_edit_selection.py --instance "$INST" --n 1000 --pool 4000 --min-rays 2 \
    > "$LOGS/g5_selection.log" 2>&1 || fail "G5 edit selection" "$(tail -15 "$LOGS/g5_selection.log")"

ping "PIM dw-smooth gen: DONE" "$(grep -E 'VERIFIED|corpus complete' "$LOGS/g4_corpus.log" | tail -2)
$(tail -3 "$LOGS/g5_selection.log")" white_check_mark
stage "chain complete"
