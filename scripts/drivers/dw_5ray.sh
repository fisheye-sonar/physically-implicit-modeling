#!/usr/bin/env bash
# ── dw-5ray: the whole programme, unattended (2026-09-10 night) ─────────────────────────
#
# dw-8ray with the observation cut further, to 5 usable rays (7 cast, the two frustum-wall
# rays dropped), radius 1.0 kept (the floor for 5 rays: at 4 a reachable disc can light no
# ray). Everything else identical to dw-8ray: 20M sequences, Transformer-L, matched recipe.
# Seeds are a fresh block (train base 160e9; eval 185e9; probe 1020e9; probe_large 1030e9),
# registered in bigcorpus.INSTANCES and verified disjoint by bigcorpus.verify().
# Question (Sevan): dw-8ray is the best-edited discworld run under the factorised categorical
# target (all three editors land); does pushing quantisation further raise it?
#
# Stages (each gated on the previous one's exit code; a failed stage pings and stops):
#   A  wait for the running unit appearance_fac_tok to finish (one GPU job at a time)
#   B  CPU  generate the dw-5ray instance (layout v2): eval, edit bench, probe 120k/250k, 20M corpus (~2 h)
#   C  GPU  train Transformer-L, 780k steps, matched recipe                                  (~8 h)
#   D  GPU  master_eval (canonical probes, baselines, all editors) + tables                    (~30 min)
#   E  GPU  appearance-fac probes + floors on the new run, then score + tables                 (~1 h)
#
# Resumable: every stage is idempotent. Run it under a transient unit so it survives the
# launching session:
#   systemd-run --user --unit=dw_5ray -p MemoryMax=48G --collect \
#       --working-directory=$PWD bash scripts/drivers/dw_5ray.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
export PYTHONPATH=$ROOT
NT=https://ntfy.sh/swirling-tornado-ai691k
INST=dw-5ray
RUN_TOPIC=ray_ablation
RUN_NAME=L-dw-5ray-20m
LOGS=$ROOT/logs/$RUN_TOPIC/dw_5ray
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"

ping() {  # ping <title> <body> [tag]
  curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" \
       -d "$2" "$NT" > /dev/null 2>&1 || true
}
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM dw-5ray FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM dw-5ray: chain queued" "waits for appearance_fac_tok, then: generate instance (~2h) -> train 780k (~8h) -> score (~30 min) -> appearance-fac probes + score (~1h). ETA ~11:30 PT."

# ── Stage A — wait for the GPU (the token-model factorised evaluation) ─────────────────
stage "A wait for unit appearance_fac_tok"
WAITED=0
while st=$(systemctl --user is-active appearance_fac_tok 2>/dev/null); [ "$st" = "active" ] || [ "$st" = "activating" ]; do
  sleep 60; WAITED=$((WAITED + 1))
  if [ $WAITED -ge 240 ]; then fail "A wait" "appearance_fac_tok still active after 4 h — not starting dw-5ray"; fi
done
ping "PIM dw-5ray: chain started" "appearance_fac_tok is $st. Generating the instance now (~2h), then training (~8h)."

# ── Stage B — CPU: build the dw-8ray instance ────────────────────────────────
stage "B generate instance (CPU)"
INST_DIR=$ROOT/datasets/discworld/$INST
# --max-edit-attempts 2000: with radius-1.0 discs a collision-free, in-frustum teleport
# target is rarer, and the default 50 attempts fails ~1 case in 100 (smoke 2026-09-03).
# Cases that succeed within 50 attempts are unchanged (same RNG stream).
SIM_FLAGS=(--n-objects 2 --frames 40 --obs-res 7 --drop-edge-rays --radius 1.0
           --boundary open --position-noise 0.0 --obs-noise-std 0.0
           --fixed-reflectivities --always-in-frustum --max-edit-attempts 2000)

# Layout v2 (2026-09-10, research/specs/DATASET_LAYOUT_SPEC.md §4f): one role per call,
# only the file the role needs, straight into the instance's role directory. Seeds are the
# ones the v1 suite used for the SAME split, so the data is identical to what a v1 suite
# would have produced for that split.
if [ ! -f "$INST_DIR/eval/test.h5" ]; then
  "$PY" scripts/generate_dataset.py --role eval --instance "$INST" --n 10000 \
      "${SIM_FLAGS[@]}" --seed 185200000000 --n-workers 16 --compression-level 4 \
      > "$LOGS/b1_eval.log" 2>&1 || fail "B1 eval split" "$(tail -15 "$LOGS/b1_eval.log")"
else
  echo "  eval split already present — skipping" | tee -a "$LOGS/driver.log"
fi

if [ ! -f "$INST_DIR/edits/v1/edits.h5" ]; then
  "$PY" scripts/generate_dataset.py --role edits --instance "$INST" --n 10000 \
      "${SIM_FLAGS[@]}" --edit-frame 20 --edit-always-in-frustum \
      --seed 185300000000 --n-workers 16 --compression-level 4 \
      > "$LOGS/b1b_edits.log" 2>&1 || fail "B1b edit bench" "$(tail -15 "$LOGS/b1b_edits.log")"
else
  echo "  edit bench already present — skipping" | tee -a "$LOGS/driver.log"
fi

if [ ! -f "$INST_DIR/probe/probe_120k.h5" ]; then
  "$PY" scripts/generate_dataset.py --role probe --size 120k --instance "$INST" --n 120000 \
      "${SIM_FLAGS[@]}" --edit-frame 20 \
      --seed 1020000000200 --n-workers 16 --compression-level 4 \
      > "$LOGS/b2_probe.log" 2>&1 || fail "B2 probe corpus 120k" "$(tail -15 "$LOGS/b2_probe.log")"
else
  echo "  probe corpus 120k already present — skipping" | tee -a "$LOGS/driver.log"
fi

if [ ! -f "$INST_DIR/probe/probe_250k.h5" ]; then
  "$PY" scripts/generate_dataset.py --role probe --size 250k --instance "$INST" --n 250000 \
      "${SIM_FLAGS[@]}" --edit-frame 20 \
      --seed 1030000000200 --n-workers 16 --compression-level 4 \
      > "$LOGS/b2b_probe_250k.log" 2>&1 || fail "B2b probe corpus 250k" "$(tail -15 "$LOGS/b2b_probe_250k.log")"
else
  echo "  probe corpus 250k already present — skipping" | tee -a "$LOGS/driver.log"
fi

# B3: the 20M corpus (25.6 GB at 8 rays). Idempotent per shard via _done_NNN markers.
"$PY" -m pim.environments.discworld.bigcorpus "$INST" \
    > "$LOGS/b3_corpus.log" 2>&1 || fail "B3 20M corpus" "$(tail -20 "$LOGS/b3_corpus.log")"

ping "PIM dw-5ray: generation DONE" \
"$(grep -E 'VERIFIED|corpus complete' "$LOGS/b3_corpus.log" | tail -2)
Starting the 780k-step training (~8h)." rocket

# ── Stage C — GPU: train Transformer-L on the 8-ray instance ─────────────────
stage "C train (GPU)"
"$PY" scripts/train.py --env discworld --arch transformer_l --instance "$INST" \
    --topic "$RUN_TOPIC" --run-name "$RUN_NAME" --steps 780000 \
    > "$LOGS/c_train.log" 2>&1 || fail "C training" "$(tail -20 "$LOGS/c_train.log")"

ping "PIM dw-5ray: training DONE" \
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
ping "PIM dw-5ray: canonical scoring DONE" \
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
# ── Stage E — GPU: the factorised categorical target on the new run (+ floors), then score ──
stage "E appearance-fac probes + floors + score (GPU)"
bash scripts/drivers/probe_target_fit.sh "$RUN_TOPIC/$RUN_NAME" appearance-fac dw_5ray_fac \
    > "$LOGS/e_fac.log" 2>&1 || fail "E appearance-fac" "$(tail -20 "$LOGS/e_fac.log")"
ping "PIM dw-5ray: ALL DONE" "$(grep -E "5ray" "$ROOT/logs/dw_5ray_fac/headline.txt" 2>/dev/null | head -6)" white_check_mark
stage "chain complete"
