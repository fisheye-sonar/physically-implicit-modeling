#!/usr/bin/env bash
# ── dw-noiseless: the whole programme, unattended ────────────────────────────
#
# dw-pn04 with ALL noise off (obs_noise_std 0.0 AND position_noise_std 0.0), at the
# identical mass scale: 20M sequences, Transformer-L, the matched canonical recipe.
# Everything else — 2 objects, 40 frames, 128 rays, fixed reflectivities,
# always-in-frustum, open boundary, split sizes, training hyperparameters — is
# byte-for-byte dw-pn04's, so noise is the ONLY variable.
#
# ⛔ SEEDS ARE FRESH, NOT PAIRED. Reusing dw-pn04's seeds does NOT give the same worlds
#    with jitter removed: `always_in_frustum` accepts initial conditions by simulating
#    forward, and the noise draws are consumed inside that acceptance loop, so with
#    noise off the RNG stream diverges from the first rejection (measured 2026-08-31:
#    0/5 seeds matched, ~5-unit trajectory divergence). Fresh block, base 30e9, stride
#    500M — verified disjoint from every other range in bigcorpus.INSTANCES.
#
# Stages (each gated on the previous one's exit code; a failed stage pings and stops):
#   A  GPU  rescore both canonical runs at EVAL_VERSION 2026-08-31.2 (adds GS-mine)
#   B  CPU  generate the dw-noiseless instance: eval suite, probe split, 20M corpus
#           (runs CONCURRENTLY with A — A is GPU-bound, B is CPU/disk-bound)
#   C  GPU  train Transformer-L, 780k steps, matched recipe            (~8 h)
#   D  GPU  master_eval scores the new run + build_full_table          (~20 min)
#
# Resumable: every stage is idempotent (generation skips finished shards; master_eval
# skips runs already at EVAL_VERSION; training refuses to silently clobber a run dir).
# Re-running this script after a crash continues from where it stopped.
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
NT=https://ntfy.sh/swirling-tornado-ai691k
INST=dw-noiseless
LOGS=$ROOT/runs/dw_noiseless/logs
RUN_TOPIC=noise_ablation
RUN_NAME=L-dw-noiseless-20m
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"

ping() {  # ping <title> <body> [tag]
  curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" \
       -d "$2" "$NT" > /dev/null 2>&1 || true
}
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM dw-noiseless FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM dw-noiseless: chain started" \
"Stage A (rescore + GS-mine) and Stage B (20M corpus generation) run concurrently.
Then train 780k steps (~8h), then score. Next ping at the A/B boundary."

# ── Stage A — GPU: rescore at EVAL_VERSION .2, adding the GS-mine arm ────────
stage "A rescore (GPU)"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace master_eval.ipynb --ExecutePreprocessor.timeout=14400 ) \
  > "$LOGS/a_rescore.log" 2>&1 &
A_PID=$!

# ── Stage B — CPU: build the dw-noiseless instance ───────────────────────────
# B1: eval suite (val/test/edits) + B2: probe split, both via the shared generator,
# at seeds far from the training block; then B3: the 20M streaming corpus.
stage "B generate instance (CPU)"
INST_DIR=$ROOT/datasets/discworld/$INST
SIM_FLAGS=(--n-objects 2 --frames 40 --obs-res 128 --boundary open
           --position-noise 0.0 --obs-noise-std 0.0
           --fixed-reflectivities --always-in-frustum)

if [ ! -f "$INST_DIR/eval/dataset.json" ]; then
  "$PY" scripts/generate_dataset.py "$INST_DIR/eval" \
      --n-train 100 --n-val 10000 --n-test 10000 --n-edits 10000 \
      "${SIM_FLAGS[@]}" --edit-frame 20 --edit-always-in-frustum \
      --seed 52000000000 --seed-val 52100000000 --seed-test 52200000000 \
      --seed-edits 52300000000 --n-workers 16 --compression-level 4 \
      > "$LOGS/b1_eval.log" 2>&1 || fail "B1 eval suite" "$(tail -15 "$LOGS/b1_eval.log")"
else
  echo "  eval suite already present — skipping" | tee -a "$LOGS/driver.log"
fi

if [ ! -f "$INST_DIR/probe/dataset.json" ]; then
  "$PY" scripts/generate_dataset.py "$INST_DIR/probe" \
      --n-train 100 --n-val 100 --n-test 120000 --n-edits 100 \
      "${SIM_FLAGS[@]}" --edit-frame 20 \
      --seed 950000000000 --seed-val 950000000100 --seed-test 950000000200 \
      --seed-edits 950000120200 --n-workers 16 --compression-level 4 \
      > "$LOGS/b2_probe.log" 2>&1 || fail "B2 probe split" "$(tail -15 "$LOGS/b2_probe.log")"
else
  echo "  probe split already present — skipping" | tee -a "$LOGS/driver.log"
fi

# B3: the 20M corpus (410 GB). Idempotent per shard via _done_NNN markers.
"$PY" -m pim.environments.discworld.bigcorpus "$INST" \
    > "$LOGS/b3_corpus.log" 2>&1 || fail "B3 20M corpus" "$(tail -20 "$LOGS/b3_corpus.log")"

wait $A_PID; A_RC=$?
[ $A_RC -eq 0 ] || fail "A rescore" "$(tail -20 "$LOGS/a_rescore.log")"

ping "PIM dw-noiseless: generation + rescore DONE" \
"$(grep -E 'VERIFIED|corpus complete' "$LOGS/b3_corpus.log" | tail -2)
GS-mine rescore: $(grep -c 'wrote runs' "$LOGS/a_rescore.log") run(s) rescored.
Starting the 780k-step training (~8h)." rocket

# ── Stage C — GPU: train Transformer-L on the noiseless instance ─────────────
stage "C train (GPU)"
"$PY" scripts/train.py --env discworld --arch transformer_l --instance "$INST" \
    --topic "$RUN_TOPIC" --run-name "$RUN_NAME" --steps 780000 \
    > "$LOGS/c_train.log" 2>&1 || fail "C training" "$(tail -20 "$LOGS/c_train.log")"

ping "PIM dw-noiseless: training DONE" \
"$(grep '^done' "$LOGS/c_train.log" | tail -1)
Scoring now (probes + all editors, ~20 min)." checkered_flag

# ── Stage D — GPU: score the new run, rebuild the master tables ──────────────
stage "D score + tables (GPU)"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace master_eval.ipynb --ExecutePreprocessor.timeout=14400 ) \
  > "$LOGS/d_score.log" 2>&1 || fail "D scoring" "$(tail -20 "$LOGS/d_score.log")"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace build_full_table.ipynb ) \
  > "$LOGS/d_table.log" 2>&1 || fail "D table" "$(tail -20 "$LOGS/d_table.log")"

SCORES=$ROOT/runs/$RUN_TOPIC/$RUN_NAME/scores.json
ping "PIM dw-noiseless: ALL DONE" \
"$("$PY" - "$SCORES" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
T = s["targets"]["pos"]
b = T["best"]
print(f"val {s['val_loss']:.5f} | unedited EI {T['unedited']['edit_index']:+.3f} | "
      f"skill lin {max(T['probe_skill_linear']):+.3f}")
print(" ".join(f"{k} {v['edit_index']:+.3f}" for k, v in b.items() if v))
EOF
)" white_check_mark
stage "chain complete"
