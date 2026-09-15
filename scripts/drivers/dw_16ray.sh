#!/usr/bin/env bash
# ── dw-16ray: the whole programme, unattended, on the WSL remote (2026-09-14, Sevan) ─────────
#
# dw-8ray with the observation raised to 16 usable rays (18 cast, the two frustum-wall rays
# dropped), radius 1.0 kept — the ray-count axis upward: dw-5ray (5) -> dw-8ray (8) -> dw-16ray
# (16) at one radius; dw-noiseless is 128 rays at radius 0.5. Everything else identical to
# dw-8ray: 20M sequences, Transformer-L, matched recipe. Seeds are a fresh block (train base
# 270e9; eval 295e9; probe 1080e9; probe_large 1090e9), registered in bigcorpus.INSTANCES.
#
# Stages (each gated on the previous one's exit code; a failed stage pings and stops):
#   B  CPU  generate the dw-16ray instance (layout v2): eval, edit bench, probe 120k/250k, 20M corpus (51 GB memmap)
#   C  GPU  train Transformer-L, 780k steps, matched recipe                                  (~10.5 h on the 4090)
#   D  GPU  master_eval (canonical probes, baselines, all editors) + tables                    (~30 min)
#   E  GPU  appearance-fac probes + floors on the new run, then score + tables                 (~1 h)
#
# Resumable: every stage is idempotent (generation skips finished splits / shards; training uses --resume, a
# no-op once 780k is reached; scoring skips runs already at the eval version). ⚠ master_eval scans EVERY run dir
# on the machine — park runs that were scored elsewhere under runs/_synced_to_lab/ (ledgered) or the rescore of a
# stale run whose instance is not built here fails the stage (2026-09-15 01:46). Under a unit:
#   systemd-run --user --unit=dw_16ray -p MemoryMax=40G --collect --working-directory=$PWD \
#       /usr/bin/bash -c 'bash scripts/drivers/dw_16ray.sh > logs/ray_ablation/dw_16ray/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
export PYTHONPATH=$ROOT
NT=https://ntfy.sh/swirling-tornado-ai691k
INST=dw-16ray
RUN_TOPIC=ray_ablation
RUN_NAME=L-dw-16ray-20m
LOGS=$ROOT/logs/$RUN_TOPIC/dw_16ray
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"

ping() {  # ping <title> <body> [tag]
  curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" \
       -d "$2" "$NT" > /dev/null 2>&1 || true
}
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM dw-16ray FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM dw-16ray: chain started" "generate dw-16ray on wsl-sevan (~3-5 h) -> train 780k (~10.5 h on the 4090) -> score (~30 min) -> appearance-fac probes + score (~1 h). ETA ~05:00-07:00 PT 09-15." rocket

# ── Stage B — CPU: build the dw-16ray instance ────────────────────────────────
stage "B generate instance (CPU)"
INST_DIR=$ROOT/datasets/discworld/$INST
# --max-edit-attempts 2000: with radius-1.0 discs a collision-free, in-frustum teleport
# target is rarer, and the default 50 attempts fails ~1 case in 100 (smoke 2026-09-03).
# Cases that succeed within 50 attempts are unchanged (same RNG stream).
SIM_FLAGS=(--n-objects 2 --frames 40 --obs-res 18 --drop-edge-rays --radius 1.0
           --boundary open --position-noise 0.0 --obs-noise-std 0.0
           --fixed-reflectivities --always-in-frustum --max-edit-attempts 2000)

# Layout v2 (2026-09-10, research/specs/DATASET_LAYOUT_SPEC.md §4f): one role per call,
# only the file the role needs, straight into the instance's role directory. Seeds are the
# ones the v1 suite used for the SAME split, so the data is identical to what a v1 suite
# would have produced for that split.
if [ ! -f "$INST_DIR/eval/test.h5" ]; then
  "$PY" scripts/generate_dataset.py --role eval --instance "$INST" --n 10000 \
      "${SIM_FLAGS[@]}" --seed 295200000000 --n-workers 16 --compression-level 4 \
      > "$LOGS/b1_eval.log" 2>&1 || fail "B1 eval split" "$(tail -15 "$LOGS/b1_eval.log")"
else
  echo "  eval split already present — skipping" | tee -a "$LOGS/driver.log"
fi

if [ ! -f "$INST_DIR/edits/v1/edits.h5" ]; then
  "$PY" scripts/generate_dataset.py --role edits --instance "$INST" --n 10000 \
      "${SIM_FLAGS[@]}" --edit-frame 20 --edit-always-in-frustum \
      --seed 295300000000 --n-workers 16 --compression-level 4 \
      > "$LOGS/b1b_edits.log" 2>&1 || fail "B1b edit bench" "$(tail -15 "$LOGS/b1b_edits.log")"
else
  echo "  edit bench already present — skipping" | tee -a "$LOGS/driver.log"
fi

if [ ! -f "$INST_DIR/probe/probe_120k.h5" ]; then
  "$PY" scripts/generate_dataset.py --role probe --size 120k --instance "$INST" --n 120000 \
      "${SIM_FLAGS[@]}" --edit-frame 20 \
      --seed 1080000000200 --n-workers 16 --compression-level 4 \
      > "$LOGS/b2_probe.log" 2>&1 || fail "B2 probe corpus 120k" "$(tail -15 "$LOGS/b2_probe.log")"
else
  echo "  probe corpus 120k already present — skipping" | tee -a "$LOGS/driver.log"
fi

if [ ! -f "$INST_DIR/probe/probe_250k.h5" ]; then
  "$PY" scripts/generate_dataset.py --role probe --size 250k --instance "$INST" --n 250000 \
      "${SIM_FLAGS[@]}" --edit-frame 20 \
      --seed 1090000000200 --n-workers 16 --compression-level 4 \
      > "$LOGS/b2b_probe_250k.log" 2>&1 || fail "B2b probe corpus 250k" "$(tail -15 "$LOGS/b2b_probe_250k.log")"
else
  echo "  probe corpus 250k already present — skipping" | tee -a "$LOGS/driver.log"
fi

# B3: the 20M corpus (51 GB at 16 rays). Idempotent per shard via _done_NNN markers.
"$PY" -m pim.environments.discworld.bigcorpus "$INST" \
    > "$LOGS/b3_corpus.log" 2>&1 || fail "B3 20M corpus" "$(tail -20 "$LOGS/b3_corpus.log")"

# B4: the canonical edit-case SELECTION (2026-09-12 rule, every instance): the first 1000 cases whose two
# clean renders at the edit frame differ on >= 2 rays -> edits/v1/selection.json (the bench every scorer uses).
# (Added 2026-09-14 14:10 after launch — the 5-ray template predates the rule; on the live run this step was
#  executed by hand at 14:08 while stage C trained, so the file was in place long before stage D.)
if [ ! -f "$INST_DIR/edits/v1/selection.json" ]; then
  "$PY" scripts/make_edit_selection.py --instance "$INST" --n 1000 --pool 4000 --min-rays 2 \
      > "$LOGS/b4_selection.log" 2>&1 || fail "B4 edit selection" "$(tail -15 "$LOGS/b4_selection.log")"
else
  echo "  edit selection already present — skipping" | tee -a "$LOGS/driver.log"
fi

ping "PIM dw-16ray: generation DONE" \
"$(grep -E 'VERIFIED|corpus complete' "$LOGS/b3_corpus.log" | tail -2)
Starting the 780k-step training (~10.5 h on the 4090)." rocket

# ── Stage C — GPU: train Transformer-L on the 16-ray instance ─────────────────
stage "C train (GPU)"
"$PY" scripts/train.py --env discworld --arch transformer_l --instance "$INST" \
    --topic "$RUN_TOPIC" --run-name "$RUN_NAME" --steps 780000 --resume \
    > "$LOGS/c_train.log" 2>&1 || fail "C training" "$(tail -20 "$LOGS/c_train.log")"

ping "PIM dw-16ray: training DONE" \
"$(grep '^done' "$LOGS/c_train.log" | tail -1)
Scoring now (probes + baselines + all editors, ~30 min)." checkered_flag

# ── Stage D — GPU: score the new run (+ its baselines), rebuild the master tables ──
stage "D score + tables (GPU)"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace master_eval.ipynb --ExecutePreprocessor.timeout=14400 ) \
  > "$LOGS/d_score.log" 2>&1 || fail "D scoring" "$(tail -20 "$LOGS/d_score.log")"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace build_full_tables.ipynb ) \
  > "$LOGS/d_table.log" 2>&1 || fail "D table" "$(tail -20 "$LOGS/d_table.log")"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace training_curves.ipynb ) > "$LOGS/d_curves.log" 2>&1 || true

SCORES=$ROOT/runs/$RUN_TOPIC/$RUN_NAME/scores.json
ping "PIM dw-16ray: canonical scoring DONE" \
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
bash scripts/drivers/probe_target_fit.sh "$RUN_TOPIC/$RUN_NAME" appearance-fac dw_16ray_fac \
    > "$LOGS/e_fac.log" 2>&1 || fail "E appearance-fac" "$(tail -20 "$LOGS/e_fac.log")"
ping "PIM dw-16ray: ALL DONE" "$(grep -E "16ray" "$ROOT/logs/dw_16ray_fac/headline.txt" 2>/dev/null | head -6)" white_check_mark
stage "chain complete"
