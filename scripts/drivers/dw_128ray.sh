#!/usr/bin/env bash
# ── dw-128ray: the whole programme, unattended (2026-09-15 night) ───────────────────────
#
# The ray-count axis closed at the top. dw-8ray geometry in every field (disc radius 1.0,
# frustum-wall rays dropped, --max-edit-attempts 2000, 2 objects, 40 frames, no noise, fixed
# reflectivities, always-in-frustum, open boundary, 20M sequences, Transformer-L, matched
# recipe) with 130 rays CAST -> 128 kept. This is NOT dw-noiseless (radius 0.5, 128 cast, no
# drop): with dw-5ray / dw-8ray / dw-16ray it is one axis at one radius, and against
# dw-noiseless it separates the ray count from the disc size the two had confounded.
# Seeds: a fresh block (train base 300e9; eval 325.2e9; edits 325.3e9; probe 1100e9+200;
# probe_large 1110e9+200), registered in bigcorpus.INSTANCES and verified disjoint by
# bigcorpus.verify().
# Question (Sevan, 2026-09-15): is the editability loss at 128 rays a ray-count effect or a
# disc-size effect?
#
# Stages (each gated on the previous one's exit code; a failed stage pings and stops):
#   B  CPU  generate the dw-128ray instance (layout v2): eval, edit bench, probe 120k/250k,
#           20M corpus (410 GB memmap, ~2 h; dw-smooth measured 2 h 07)
#   C  GPU  train Transformer-L, 780k steps, matched recipe                       (~8 h on the 5090)
#   D  GPU  master_eval in BOTH bases (canonical probes, baselines, PI/ND/GS/IM/IM-NN) + tables (~40 min)
#   E  GPU  appearance-fac probes + floors on the new run, then score + tables        (~1 h)
#
# Resumable: every stage is idempotent (splits skip when present, shards via _done_NNN,
# train.py --resume from ckpt/latest.pt). Run under a transient unit:
#   systemd-run --user --unit=dw_128ray -p MemoryMax=48G --collect --working-directory=$PWD \
#       /usr/bin/bash -c 'bash scripts/drivers/dw_128ray.sh > logs/ray_ablation/dw_128ray/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
export PYTHONPATH=$ROOT
export PIM_DW_BASES=frustum,cartesian     # the scorer fits/scores both bases (tables' BASIS knob switches)
NT=https://ntfy.sh/swirling-tornado-ai691k
INST=dw-128ray
RUN_TOPIC=ray_ablation
RUN_NAME=L-dw-128ray-20m
LOGS=$ROOT/logs/$RUN_TOPIC/dw_128ray
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"

ping() {  # ping <title> <body> [tag]
  curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" \
       -d "$2" "$NT" > /dev/null 2>&1 || true
}
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM dw-128ray FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM dw-128ray: chain started" "generate instance (~2h, 410 GB) -> train 780k (~8h) -> score both bases (~40 min) -> appearance-fac (~1h). ETA ~12 h."

# ── Stage B — CPU: build the dw-128ray instance ──────────────────────────────
stage "B generate instance (CPU)"
INST_DIR=$ROOT/datasets/discworld/$INST
# Flags are bigcorpus._COMMON_FLAGS + _RAYS_128R + no-noise, spelled out so the splits and the
# corpus are one recipe. --max-edit-attempts 2000: radius-1.0 teleports need more tries.
SIM_FLAGS=(--n-objects 2 --frames 40 --obs-res 130 --drop-edge-rays --radius 1.0
           --boundary open --position-noise 0.0 --obs-noise-std 0.0
           --fixed-reflectivities --always-in-frustum --max-edit-attempts 2000)

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
"$PY" -u -m pim.environments.discworld.bigcorpus "$INST" \
    > "$LOGS/b3_corpus.log" 2>&1 || fail "B3 20M corpus" "$(tail -20 "$LOGS/b3_corpus.log")"

# B4: edit selection (the first 1000 cases with >= 2 differing rays), as every instance since v2.
if [ ! -f "$INST_DIR/edits/v1/selection.json" ]; then
  "$PY" scripts/make_edit_selection.py --instance "$INST" --n 1000 --pool 4000 --min-rays 2 \
      > "$LOGS/b4_selection.log" 2>&1 || fail "B4 edit selection" "$(tail -15 "$LOGS/b4_selection.log")"
else
  echo "  edit selection already present — skipping" | tee -a "$LOGS/driver.log"
fi

ping "PIM dw-128ray: generation DONE" \
"$(grep -E 'VERIFIED|corpus complete' "$LOGS/b3_corpus.log" | tail -2)
$(tail -2 "$LOGS/b4_selection.log" 2>/dev/null)
Starting the 780k-step training (~8h)." rocket

# ── Stage C — GPU: train Transformer-L on the 128-ray instance ───────────────
stage "C train (GPU)"
RESUME=()
[ -f "$ROOT/runs/$RUN_TOPIC/$RUN_NAME/ckpt/latest.pt" ] && RESUME=(--resume)
"$PY" scripts/train.py --env discworld --arch transformer_l --instance "$INST" \
    --topic "$RUN_TOPIC" --run-name "$RUN_NAME" --steps 780000 "${RESUME[@]}" \
    > "$LOGS/c_train.log" 2>&1 || fail "C training" "$(tail -20 "$LOGS/c_train.log")"

ping "PIM dw-128ray: training DONE" \
"$(grep '^done' "$LOGS/c_train.log" | tail -1)
Scoring now (probes + baselines + all editors, both bases, ~40 min)." checkered_flag

# ── Stage D — GPU: score the new run (+ its baselines) in both bases, rebuild the tables ──
stage "D score + tables (GPU)"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace master_eval.ipynb --ExecutePreprocessor.timeout=14400 ) \
  > "$LOGS/d_score.log" 2>&1 || fail "D scoring" "$(tail -20 "$LOGS/d_score.log")"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace build_full_tables.ipynb ) \
  > "$LOGS/d_table.log" 2>&1 || fail "D table" "$(tail -20 "$LOGS/d_table.log")"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace build_paper_tables_and_figs.ipynb ) \
  > "$LOGS/d_paper_table.log" 2>&1 || fail "D paper table" "$(tail -20 "$LOGS/d_paper_table.log")"

SCORES=$ROOT/runs/$RUN_TOPIC/$RUN_NAME/scores.json
ping "PIM dw-128ray: canonical scoring DONE" \
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

# ── Stage E — GPU: the factorised categorical target on the new run (+ floors), then score ──
stage "E appearance-fac probes + floors + score (GPU)"
bash scripts/drivers/probe_target_fit.sh "$RUN_TOPIC/$RUN_NAME" appearance-fac dw_128ray_fac \
    > "$LOGS/e_fac.log" 2>&1 || fail "E appearance-fac" "$(tail -20 "$LOGS/e_fac.log")"
ping "PIM dw-128ray: ALL DONE" "$(grep -E "128ray" "$ROOT/logs/dw_128ray_fac/headline.txt" 2>/dev/null | head -6)" white_check_mark
stage "chain complete"
