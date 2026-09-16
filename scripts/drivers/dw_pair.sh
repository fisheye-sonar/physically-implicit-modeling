#!/usr/bin/env bash
# ── dw-pair: the whole programme, unattended, on the WSL remote (2026-09-15, Sevan) ───────────────
#
# dw-noiseless (128 rays, radius 0.5, no noise, 2 objects, 40 frames, open boundary, always in frustum,
# 20M sequences, Transformer-L matched recipe) with ONE change: the two discs are a RIGID PAIR at centre
# distance 2.0 with the same velocity (sim.py pair_separation; Monte-Carlo acceptance 0.23 vs 0.21 for
# independent discs). The state has 2 free position dims + an orientation instead of 4. The single-object
# teleport bench is deliberately KEPT (Sevan: the edits are expected to fail — that is the point), the
# targets are unchanged, and the regression block is scored in BOTH bases (frustum + cartesian) with both
# floor sets (master_eval dw_bases_by_instance); the categorical read-out is appearance-fac only.
# Seeds: train 300e9, eval 325.2e9, edits 325.3e9, probe 1100e9+200, probe_large 1110e9+200.
#
# Stages (each gated on the previous one's exit code; a failed stage pings and stops):
#   B  CPU  generate the instance (layout v2): eval, edit bench, probe 120k/250k, 20M corpus (410 GB memmap), selection
#   C  GPU  train Transformer-L, 780k steps, matched recipe, --resume (idempotent)                (~12 h on the 4090)
#   D  GPU  master_eval (both bases + floors, all editors) + tables                                (~30 min)
#   E  GPU  appearance-fac probes + floors on the new run, then score + tables                    (~3-4 h at 128 rays)
#
# ⚠ master_eval scans EVERY run dir on the machine: every synced run must already be at the eval version.
# Under a unit (44G: the 40G cap killed a scorer kernel once on 2026-09-15):
#   systemd-run --user --unit=dw_pair -p MemoryMax=44G --collect --working-directory=$PWD \
#       /usr/bin/bash -c 'bash scripts/drivers/dw_pair.sh > logs/pair_ablation/dw_pair/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
export PYTHONPATH=$ROOT
NT=https://ntfy.sh/swirling-tornado-ai691k
INST=dw-pair
RUN_TOPIC=pair_ablation
RUN_NAME=L-dw-pair-20m
LOGS=$ROOT/logs/$RUN_TOPIC/dw_pair
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"

ping() {  # ping <title> <body> [tag]
  curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" \
       -d "$2" "$NT" > /dev/null 2>&1 || true
}
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM dw-pair FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM dw-pair: chain started" "generate dw-pair on wsl-sevan (~2 h, 410 GB) -> train 780k (~12 h on the 4090) -> score both bases (~30 min) -> appearance-fac probes + score (~3-4 h). ETA ~2026-09-16 afternoon." rocket

# ── Stage B — CPU: build the dw-pair instance ────────────────────────────────
stage "B generate instance (CPU)"
INST_DIR=$ROOT/datasets/discworld/$INST
# --max-edit-attempts 2000: with radius-1.0 discs a collision-free, in-frustum teleport
# target is rarer, and the default 50 attempts fails ~1 case in 100 (smoke 2026-09-03).
# Cases that succeed within 50 attempts are unchanged (same RNG stream).
SIM_FLAGS=(--n-objects 2 --frames 40 --obs-res 128 --radius 0.5
           --boundary open --position-noise 0.0 --obs-noise-std 0.0
           --fixed-reflectivities --always-in-frustum --pair-separation 2.0)

# Layout v2 (2026-09-10, research/specs/DATASET_LAYOUT_SPEC.md §4f): one role per call,
# only the file the role needs, straight into the instance's role directory. Seeds are the
# ones the v1 suite used for the SAME split, so the data is identical to what a v1 suite
# would have produced for that split.
if [ ! -f "$INST_DIR/eval/test.h5" ]; then
  "$PY" scripts/generate_dataset.py --role eval --instance "$INST" --n 10000 \
      "${SIM_FLAGS[@]}" --seed 325200000000 --n-workers 16 --compression-level 4 \
      > "$LOGS/b1_eval.log" 2>&1 || fail "B1 eval split" "$(tail -15 "$LOGS/b1_eval.log")"
else
  echo "  eval split already present — skipping" | tee -a "$LOGS/driver.log"
fi

if [ ! -f "$INST_DIR/edits/v1/edits.h5" ]; then
  "$PY" scripts/generate_dataset.py --role edits --instance "$INST" --n 10000 \
      "${SIM_FLAGS[@]}" --edit-frame 20 --edit-always-in-frustum \
      --seed 325300000000 --n-workers 16 --compression-level 4 \
      > "$LOGS/b1b_edits.log" 2>&1 || fail "B1b edit bench" "$(tail -15 "$LOGS/b1b_edits.log")"
else
  echo "  edit bench already present — skipping" | tee -a "$LOGS/driver.log"
fi

if [ ! -f "$INST_DIR/probe/probe_120k.h5" ]; then
  "$PY" scripts/generate_dataset.py --role probe --size 120k --instance "$INST" --n 120000 \
      "${SIM_FLAGS[@]}" --edit-frame 20 \
      --seed 1100000000200 --n-workers 16 --compression-level 4 \
      > "$LOGS/b2_probe.log" 2>&1 || fail "B2 probe corpus 120k" "$(tail -15 "$LOGS/b2_probe.log")"
else
  echo "  probe corpus 120k already present — skipping" | tee -a "$LOGS/driver.log"
fi

if [ ! -f "$INST_DIR/probe/probe_250k.h5" ]; then
  "$PY" scripts/generate_dataset.py --role probe --size 250k --instance "$INST" --n 250000 \
      "${SIM_FLAGS[@]}" --edit-frame 20 \
      --seed 1110000000200 --n-workers 16 --compression-level 4 \
      > "$LOGS/b2b_probe_250k.log" 2>&1 || fail "B2b probe corpus 250k" "$(tail -15 "$LOGS/b2b_probe_250k.log")"
else
  echo "  probe corpus 250k already present — skipping" | tee -a "$LOGS/driver.log"
fi

# B3: the 20M corpus (410 GB at 128 rays). Idempotent per shard via _done_NNN markers.
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

ping "PIM dw-pair: generation DONE" \
"$(grep -E 'VERIFIED|corpus complete' "$LOGS/b3_corpus.log" | tail -2)
Starting the 780k-step training (~12 h on the 4090)." rocket

# ── Stage C — GPU: train Transformer-L on the rigid-pair instance ─────────────────
stage "C train (GPU)"
"$PY" scripts/train.py --env discworld --arch transformer_l --instance "$INST" \
    --topic "$RUN_TOPIC" --run-name "$RUN_NAME" --steps 780000 --resume \
    > "$LOGS/c_train.log" 2>&1 || fail "C training" "$(tail -20 "$LOGS/c_train.log")"

ping "PIM dw-pair: training DONE" \
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
ping "PIM dw-pair: canonical scoring DONE" \
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
bash scripts/drivers/probe_target_fit.sh "$RUN_TOPIC/$RUN_NAME" appearance-fac dw_pair_fac \
    > "$LOGS/e_fac.log" 2>&1 || fail "E appearance-fac" "$(tail -20 "$LOGS/e_fac.log")"
ping "PIM dw-pair: ALL DONE" "$(grep -E "pair" "$ROOT/logs/dw_pair_fac/headline.txt" 2>/dev/null | head -6)" white_check_mark
stage "chain complete"
