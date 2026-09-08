#!/usr/bin/env bash
# ── dw-8ray: the whole programme, unattended ─────────────────────────────────
#
# dw-noiseless with the disc radius DOUBLED (1.0) and the observation cut to 8 usable
# rays (10 cast, the two frustum-wall rays dropped), at the identical mass scale: 20M
# sequences, Transformer-L (same 8 x 512 architecture; only the input/output
# projections are 8-wide), the matched canonical recipe. Seeds are a fresh block
# (train base 60e9; eval 85e9; probe 980e9; probe_large 990e9), verified disjoint from
# every other range by bigcorpus.verify().
#
# Stages (each gated on the previous one's exit code; a failed stage pings and stops):
#   B  CPU  generate the dw-8ray instance: eval suite, probe split, probe_250k, 20M corpus
#   C  GPU  train Transformer-L, 780k steps, matched recipe                       (~8 h)
#   D  GPU  master_eval (probes cached in the run, baselines, all editors) + tables
#
# Resumable: every stage is idempotent (generation skips finished splits/shards;
# master_eval skips runs already at EVAL_VERSION; training refuses to clobber a run dir).
# Run it under a transient unit so it survives the launching session:
#   systemd-run --user --unit=dw_8ray -p MemoryMax=48G --collect \
#       --working-directory=$PWD bash scripts/drivers/dw_8ray.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
NT=https://ntfy.sh/swirling-tornado-ai691k
INST=dw-8ray
RUN_TOPIC=ray_ablation
RUN_NAME=L-dw-8ray-20m
LOGS=$ROOT/logs/$RUN_TOPIC/dw_8ray
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"

ping() {  # ping <title> <body> [tag]
  curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" \
       -d "$2" "$NT" > /dev/null 2>&1 || true
}
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM dw-8ray FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM dw-8ray: chain started" "generate instance (eval, probe, probe_250k, 20M corpus) -> train 780k (~8h) -> score."

# ── Stage B — CPU: build the dw-8ray instance ────────────────────────────────
stage "B generate instance (CPU)"
INST_DIR=$ROOT/datasets/discworld/$INST
# --max-edit-attempts 2000: with radius-1.0 discs a collision-free, in-frustum teleport
# target is rarer, and the default 50 attempts fails ~1 case in 100 (smoke 2026-09-03).
# Cases that succeed within 50 attempts are unchanged (same RNG stream).
SIM_FLAGS=(--n-objects 2 --frames 40 --obs-res 10 --drop-edge-rays --radius 1.0
           --boundary open --position-noise 0.0 --obs-noise-std 0.0
           --fixed-reflectivities --always-in-frustum --max-edit-attempts 2000)

if [ ! -f "$INST_DIR/eval/dataset.json" ]; then
  "$PY" scripts/generate_dataset.py "$INST_DIR/eval" \
      --n-train 100 --n-val 10000 --n-test 10000 --n-edits 10000 \
      "${SIM_FLAGS[@]}" --edit-frame 20 --edit-always-in-frustum \
      --seed 85000000000 --seed-val 85100000000 --seed-test 85200000000 \
      --seed-edits 85300000000 --n-workers 16 --compression-level 4 \
      > "$LOGS/b1_eval.log" 2>&1 || fail "B1 eval suite" "$(tail -15 "$LOGS/b1_eval.log")"
else
  echo "  eval suite already present — skipping" | tee -a "$LOGS/driver.log"
fi

if [ ! -f "$INST_DIR/probe/dataset.json" ]; then
  "$PY" scripts/generate_dataset.py "$INST_DIR/probe" \
      --n-train 100 --n-val 100 --n-test 120000 --n-edits 100 \
      "${SIM_FLAGS[@]}" --edit-frame 20 \
      --seed 980000000000 --seed-val 980000000100 --seed-test 980000000200 \
      --seed-edits 980000120200 --n-workers 16 --compression-level 4 \
      > "$LOGS/b2_probe.log" 2>&1 || fail "B2 probe split" "$(tail -15 "$LOGS/b2_probe.log")"
else
  echo "  probe split already present — skipping" | tee -a "$LOGS/driver.log"
fi

if [ ! -f "$INST_DIR/probe_250k/dataset.json" ]; then
  "$PY" scripts/generate_dataset.py "$INST_DIR/probe_250k" \
      --n-train 100 --n-val 100 --n-test 250000 --n-edits 100 \
      "${SIM_FLAGS[@]}" --edit-frame 20 \
      --seed 990000000000 --seed-val 990000000100 --seed-test 990000000200 \
      --seed-edits 990001000000 --n-workers 16 --compression-level 4 \
      > "$LOGS/b2b_probe_250k.log" 2>&1 || fail "B2b probe_250k" "$(tail -15 "$LOGS/b2b_probe_250k.log")"
else
  echo "  probe_250k already present — skipping" | tee -a "$LOGS/driver.log"
fi

# B3: the 20M corpus (25.6 GB at 8 rays). Idempotent per shard via _done_NNN markers.
"$PY" -m pim.environments.discworld.bigcorpus "$INST" \
    > "$LOGS/b3_corpus.log" 2>&1 || fail "B3 20M corpus" "$(tail -20 "$LOGS/b3_corpus.log")"

ping "PIM dw-8ray: generation DONE" \
"$(grep -E 'VERIFIED|corpus complete' "$LOGS/b3_corpus.log" | tail -2)
Starting the 780k-step training (~8h)." rocket

# ── Stage C — GPU: train Transformer-L on the 8-ray instance ─────────────────
stage "C train (GPU)"
"$PY" scripts/train.py --env discworld --arch transformer_l --instance "$INST" \
    --topic "$RUN_TOPIC" --run-name "$RUN_NAME" --steps 780000 \
    > "$LOGS/c_train.log" 2>&1 || fail "C training" "$(tail -20 "$LOGS/c_train.log")"

ping "PIM dw-8ray: training DONE" \
"$(grep '^done' "$LOGS/c_train.log" | tail -1)
Scoring now (probes + baselines + all editors, ~30 min)." checkered_flag

# ── Stage D — GPU: score the new run (+ its baselines), rebuild the master tables ──
stage "D score + tables (GPU)"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace master_eval.ipynb --ExecutePreprocessor.timeout=14400 ) \
  > "$LOGS/d_score.log" 2>&1 || fail "D scoring" "$(tail -20 "$LOGS/d_score.log")"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace build_full_table.ipynb ) \
  > "$LOGS/d_table.log" 2>&1 || fail "D table" "$(tail -20 "$LOGS/d_table.log")"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace training_curves.ipynb ) > "$LOGS/d_curves.log" 2>&1 || true

SCORES=$ROOT/runs/$RUN_TOPIC/$RUN_NAME/scores.json
ping "PIM dw-8ray: ALL DONE" \
"$("$PY" - "$SCORES" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"val {s['val_loss']:.5f}")
for basis, T in s["bases"].items():
    b = T["best"]
    print(f"{basis}: skill LIN {max(T['probe_skill_linear']):+.3f} MLP {max(T['probe_skill_mlp']):+.3f} | "
          f"unedited {T['unedited']['edit_index']:+.3f} | " +
          " ".join(f"{k} {v['edit_index']:+.3f}/{v['fidelity_ratio']:.2f}" for k, v in b.items() if v))
EOF
)" white_check_mark
stage "chain complete"
