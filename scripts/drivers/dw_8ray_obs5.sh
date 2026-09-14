#!/usr/bin/env bash
# dw-8ray-obs5 (2026-09-13 evening, Sevan): dw-8ray seen by FIVE observers on a ring, discs in the
# circular arena (pim/environments/discworld/observers.py). Same recipe as dw-8ray otherwise.
#   systemd-run --user --unit=dw_8ray_obs5 -p MemoryMax=45G --collect --working-directory=$PWD \
#       /usr/bin/bash -c 'bash scripts/drivers/dw_8ray_obs5.sh > logs/observer_ablation/dw_8ray_obs5/unit.log 2>&1'
# Stages:
#   B  generate (CPU): eval 10k -> edits 10k (EF 20, stay-inside) -> probe 120k -> probe 250k -> 20M corpus
#      (40 shards, 128 GB memmap) -> edit selection (first 1000 with >= 2 differing entries)
#   B' smoke-train 200 steps into runs/_pipeline_smoke (the corpus loads)
#   C  train Transformer-L 780k (GPU, ~8 h) -> runs/observer_ablation/L-dw-8ray-obs5-20m
#   W  wait for the scorer ready-marker (the Cartesian basis override + the per-observer
#      appearance-fac target are built while B/C run; harness/OVERNIGHT.md §1 ready-marker pattern)
#   D  appearance-fac probes + random-init + observation floors, master_eval, test loss, tables
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
export PYTHONPATH=$ROOT
NT=https://ntfy.sh/swirling-tornado-ai691k
INST=dw-8ray-obs5
RUN_TOPIC=observer_ablation
RUN_NAME=L-dw-8ray-obs5-20m
LOGS=$ROOT/logs/$RUN_TOPIC/dw_8ray_obs5
READY=$ROOT/experiments/multi_observer/SCORER_READY
mkdir -p "$LOGS"
ping()  { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail()  { ping "PIM obs5 FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }
nb()    { ( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook --execute --inplace "$1.ipynb" \
            --ExecutePreprocessor.timeout=57600 ) > "$LOGS/$2.log" 2>&1 || fail "$2" "$(tail -25 "$LOGS/$2.log")"; }

ping "PIM obs5: chain started" "B generate dw-8ray-obs5 (~3 h) -> smoke-train -> C train 780k (~8 h) -> W wait scorer marker -> D fac probes + score + tables. ETA ~05:30." rocket

INST_DIR=$ROOT/datasets/discworld/$INST
SIM_FLAGS=(--n-objects 2 --frames 40 --obs-res 10 --drop-edge-rays --radius 1.0 --max-edit-attempts 2000
           --boundary open --position-noise 0.0 --obs-noise-std 0.0 --fixed-reflectivities --always-in-frustum
           --n-observers 5 --region circle)

stage "B generate instance (CPU)"
if [ ! -f "$INST_DIR/eval/test.h5" ]; then
  "$PY" scripts/generate_dataset.py --role eval --instance "$INST" --n 10000 "${SIM_FLAGS[@]}" \
      --seed 255200000000 --n-workers 16 --compression-level 4 > "$LOGS/b1_eval.log" 2>&1 || fail "B1 eval" "$(tail -15 "$LOGS/b1_eval.log")"
fi
if [ ! -f "$INST_DIR/edits/v1/edits.h5" ]; then
  "$PY" scripts/generate_dataset.py --role edits --instance "$INST" --n 10000 "${SIM_FLAGS[@]}" --edit-frame 20 --edit-always-in-frustum \
      --seed 255300000000 --n-workers 16 --compression-level 4 > "$LOGS/b1b_edits.log" 2>&1 || fail "B1b edits" "$(tail -15 "$LOGS/b1b_edits.log")"
fi
if [ ! -f "$INST_DIR/probe/probe_120k.h5" ]; then
  "$PY" scripts/generate_dataset.py --role probe --size 120k --instance "$INST" --n 120000 "${SIM_FLAGS[@]}" --edit-frame 20 \
      --seed 1060000000200 --n-workers 16 --compression-level 4 > "$LOGS/b2_probe.log" 2>&1 || fail "B2 probe 120k" "$(tail -15 "$LOGS/b2_probe.log")"
fi
if [ ! -f "$INST_DIR/probe/probe_250k.h5" ]; then
  "$PY" scripts/generate_dataset.py --role probe --size 250k --instance "$INST" --n 250000 "${SIM_FLAGS[@]}" --edit-frame 20 \
      --seed 1070000000200 --n-workers 16 --compression-level 4 > "$LOGS/b2b_probe_250k.log" 2>&1 || fail "B2b probe 250k" "$(tail -15 "$LOGS/b2b_probe_250k.log")"
fi
"$PY" -u -m pim.environments.discworld.bigcorpus "$INST" > "$LOGS/b3_corpus.log" 2>&1 || fail "B3 20M corpus" "$(tail -20 "$LOGS/b3_corpus.log")"
"$PY" scripts/make_edit_selection.py --instance "$INST" --n 1000 --pool 4000 --min-rays 2 > "$LOGS/b4_selection.log" 2>&1 \
    || fail "B4 edit selection" "$(tail -15 "$LOGS/b4_selection.log")"
ping "PIM obs5: generation DONE" "$(grep -E 'VERIFIED|corpus complete' "$LOGS/b3_corpus.log" | tail -2)
$(tail -1 "$LOGS/b4_selection.log")
Smoke-train, then training (~8 h)." rocket

stage "B' smoke-train"
"$PY" -u scripts/train.py --env discworld --arch transformer_l --instance "$INST" --topic _pipeline_smoke \
    --run-name dw-8ray-obs5-smoke --steps 200 --smoke > "$LOGS/smoke_train.log" 2>&1 || fail "smoke-train" "$(tail -20 "$LOGS/smoke_train.log")"

stage "C train (GPU, 780k)"
"$PY" scripts/train.py --env discworld --arch transformer_l --instance "$INST" --topic "$RUN_TOPIC" --run-name "$RUN_NAME" \
    --steps 780000 > "$LOGS/c_train.log" 2>&1 || fail "C training" "$(tail -20 "$LOGS/c_train.log")"
ping "PIM obs5: training DONE" "$(grep '^done' "$LOGS/c_train.log" | tail -1)" checkered_flag

stage "W wait for scorer marker"
WAITED=0
while [ ! -f "$READY" ]; do
  sleep 60; WAITED=$((WAITED + 1))
  if [ $((WAITED % 60)) -eq 0 ]; then ping "PIM obs5: waiting for scorer" "trained; $READY absent after $((WAITED / 60)) h" warning; fi
  if [ $WAITED -ge 720 ]; then fail "W wait" "scorer marker absent after 12 h"; fi
done
echo "  marker present: $(cat "$READY")" | tee -a "$LOGS/driver.log"

stage "D appearance-fac probes + floors, score, tables (GPU)"
fit() { local tag=$1; shift
        "$PY" -u scripts/fit_probes.py --run "$RUN_TOPIC/$RUN_NAME" --target appearance-fac --basis cartesian "$@" \
            > "$LOGS/fit_fac_$tag.log" 2>&1 || fail "D fit $tag" "$(tail -15 "$LOGS/fit_fac_$tag.log")"; }
fit model
fit random_init --random-init
fit observation --observation
nb master_eval d_master_eval
"$PY" -W ignore experiments/bayes_floor/scripts/test_loss.py > "$LOGS/d_test_loss.log" 2>&1 || true
nb build_full_tables d_full_tables
SCORES=$ROOT/runs/$RUN_TOPIC/$RUN_NAME/scores.json
ping "PIM obs5: ALL DONE" "$("$PY" - "$SCORES" <<'EOF2'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"L-dw-8ray-obs5-20m  val {s['val_loss']:.5f}")
for basis, T in s["bases"].items():
    b = T["best"]
    print(f"{basis}: skill LIN {max(T['probe_skill_linear']):+.3f} MLP {max(T['probe_skill_mlp']):+.3f} | unedited {T['unedited']['edit_index']:+.3f} | "
          + " ".join(f"{k} {v['edit_index']:+.3f}/{v['fidelity_ratio']:.2f}" for k, v in b.items() if v))
EOF2
)" white_check_mark
stage "chain complete"
