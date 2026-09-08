#!/usr/bin/env bash
# ── dw-blink: the whole programme, unattended (2026-09-07) ───────────────────
#
# dw-noiseless (128 rays, no noise, radius 0.5) plus BLACKOUTS: an object drops out of
# the observation for Geometric(1/7) frames (realised mean ~5.3 after the cap 12 and sequence end, prob 0.05/object/frame, never before
# frame 3, never both at once) while its physics continues, with a 0.5 marker on its edge
# ray the frame before and the last hidden frame (pim/environments/discworld/blink.py).
# Position is thereby a guaranteed member of the causal state: the reappearance frame
# cannot be predicted from the current observation. Same mass scale and recipe as every
# other instance: 20M sequences, Transformer-L, 780k steps. Seeds are a fresh block (train
# base 110e9; eval 135e9; probe 1000e9; probe_large 1010e9), verified disjoint from every
# other range by bigcorpus.verify().
#
# Stages (each gated on the previous one's exit code; a failed stage pings and stops):
#   B  CPU  generate the dw-blink instance: eval suite (20k edits), probe, probe_250k, 20M corpus
#   C  GPU  train Transformer-L, 780k steps, matched recipe                        (~8 h)
#   D  GPU  master_eval (probes cached in the run, baselines, all editors) + tables
#   E  GPU  the blink subset analysis (experiments/blink_ablation/scripts/subset_editability.py)
#
# Resumable: every stage is idempotent (generation skips finished splits/shards;
# master_eval skips runs already at EVAL_VERSION; training refuses to clobber a run dir).
# Run it under a transient unit so it survives the launching session:
#   systemd-run --user --unit=dw_blink -p MemoryMax=40G --collect \
#       --working-directory=$PWD /usr/bin/bash -c 'bash scripts/drivers/dw_blink.sh > logs/blink_ablation/dw_blink/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
NT=https://ntfy.sh/swirling-tornado-ai691k
INST=dw-blink
RUN_TOPIC=blink_ablation
RUN_NAME=L-dw-blink-20m
LOGS=$ROOT/logs/$RUN_TOPIC/dw_blink
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"

ping() {  # ping <title> <body> [tag]
  curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" \
       -d "$2" "$NT" > /dev/null 2>&1 || true
}
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM dw-blink FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM dw-blink: chain started" "generate instance (eval, probe, probe_250k, 20M corpus, ~1.5h) -> train 780k (~8h) -> score + subset analysis."

# ── Stage B — CPU: build the dw-blink instance ───────────────────────────────
stage "B generate instance (CPU)"
INST_DIR=$ROOT/datasets/discworld/$INST
SIM_FLAGS=(--n-objects 2 --frames 40 --obs-res 128 --radius 0.5
           --boundary open --position-noise 0.0 --obs-noise-std 0.0
           --fixed-reflectivities --always-in-frustum
           --blink-prob 0.05 --blink-mean 7 --blink-max 12 --blink-warmup 3)

# 20,000 edits (twice the usual 10k): the reappearance subset (edited object hidden up to
# frame 19, visible at 20) is ~3% of cases, so 20k gives ~600 of them (pilot 2026-09-07).
if [ ! -f "$INST_DIR/eval/dataset.json" ]; then
  "$PY" scripts/generate_dataset.py "$INST_DIR/eval" \
      --n-train 100 --n-val 10000 --n-test 10000 --n-edits 20000 \
      "${SIM_FLAGS[@]}" --edit-frame 20 --edit-always-in-frustum \
      --seed 135000000000 --seed-val 135100000000 --seed-test 135200000000 \
      --seed-edits 135300000000 --n-workers 16 --compression-level 4 \
      > "$LOGS/b1_eval.log" 2>&1 || fail "B1 eval suite" "$(tail -15 "$LOGS/b1_eval.log")"
else
  echo "  eval suite already present — skipping" | tee -a "$LOGS/driver.log"
fi

if [ ! -f "$INST_DIR/probe/dataset.json" ]; then
  "$PY" scripts/generate_dataset.py "$INST_DIR/probe" \
      --n-train 100 --n-val 100 --n-test 120000 --n-edits 100 \
      "${SIM_FLAGS[@]}" --edit-frame 20 \
      --seed 1000000000000 --seed-val 1000000000100 --seed-test 1000000000200 \
      --seed-edits 1000000120200 --n-workers 16 --compression-level 4 \
      > "$LOGS/b2_probe.log" 2>&1 || fail "B2 probe split" "$(tail -15 "$LOGS/b2_probe.log")"
else
  echo "  probe split already present — skipping" | tee -a "$LOGS/driver.log"
fi

if [ ! -f "$INST_DIR/probe_250k/dataset.json" ]; then
  "$PY" scripts/generate_dataset.py "$INST_DIR/probe_250k" \
      --n-train 100 --n-val 100 --n-test 250000 --n-edits 100 \
      "${SIM_FLAGS[@]}" --edit-frame 20 \
      --seed 1010000000000 --seed-val 1010000000100 --seed-test 1010000000200 \
      --seed-edits 1010001000000 --n-workers 16 --compression-level 4 \
      > "$LOGS/b2b_probe_250k.log" 2>&1 || fail "B2b probe_250k" "$(tail -15 "$LOGS/b2b_probe_250k.log")"
else
  echo "  probe_250k already present — skipping" | tee -a "$LOGS/driver.log"
fi

# B3: the 20M corpus (410 GB at 128 rays). Idempotent per shard via _done_NNN markers.
"$PY" -m pim.environments.discworld.bigcorpus "$INST" \
    > "$LOGS/b3_corpus.log" 2>&1 || fail "B3 20M corpus" "$(tail -20 "$LOGS/b3_corpus.log")"

ping "PIM dw-blink: generation DONE" \
"$(grep -E 'VERIFIED|corpus complete' "$LOGS/b3_corpus.log" | tail -2)
Starting the 780k-step training (~8h)." rocket

# ── Stage C — GPU: train Transformer-L on the blink instance ─────────────────
stage "C train (GPU)"
"$PY" -u scripts/train.py --env discworld --arch transformer_l --instance "$INST" \
    --topic "$RUN_TOPIC" --run-name "$RUN_NAME" --steps 780000 \
    > "$LOGS/c_train.log" 2>&1 || fail "C training" "$(tail -20 "$LOGS/c_train.log")"

ping "PIM dw-blink: training DONE" \
"$(grep '^done' "$LOGS/c_train.log" | tail -1)
Scoring now (probes + baselines + all editors, ~40 min)." checkered_flag

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
ping "PIM dw-blink: canonical scoring DONE" \
"$("$PY" - "$SCORES" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"val {s['val_loss']:.5f}")
for basis, T in s["bases"].items():
    b = T["best"]
    print(f"{basis}: skill LIN {max(T['probe_skill_linear']):+.3f} MLP {max(T['probe_skill_mlp']):+.3f} | "
          f"unedited {T['unedited']['edit_index']:+.3f} | " +
          " ".join(f"{k} {v['edit_index']:+.3f}/{v['fidelity_ratio']:.2f}" for k, v in b.items() if v))
PYEOF
)" white_check_mark

# ── Stage E — GPU: the blink subset analysis (reappearance / mid-blackout / visible) ──
stage "E subset analysis (GPU)"
"$PY" -u experiments/blink_ablation/scripts/subset_editability.py \
    --run "runs/$RUN_TOPIC/$RUN_NAME" > "$LOGS/e_subsets.log" 2>&1 \
  || fail "E subset analysis" "$(tail -20 "$LOGS/e_subsets.log")"
ping "PIM dw-blink: ALL DONE" "$(tail -25 "$LOGS/e_subsets.log")" white_check_mark
stage "chain complete"
