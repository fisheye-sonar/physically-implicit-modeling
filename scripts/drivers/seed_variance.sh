#!/usr/bin/env bash
# ── seed-variance pilot on L-dw-noiseless-20m (2026-09-11 night, Sevan) ────────────────────
#   A  GPU  train seed 1, 390k steps (half the canonical budget)                      (~4 h)
#   B  GPU  train seed 2, 390k steps                                                    (~4 h)
#   C  CPU  lay out the seed-0 run's step-421,875 checkpoint as a replicate run dir
#   D  GPU  master_eval (canonical frustum rows for the 3 replicate runs) + appearance-fac
#           probes / floors / scores for each (floors are cache hits)                 (~1.5 h)
#   E  GPU  probe-seed replicates: 20 regression + 6 factorised linear seeds on the canonical
#           run and the two trained replicates → runs/<run>/variance.json               (~4 h)
#   F  CPU  build_full_table (replicate ± columns, Table 4)
# Replicate runs are named <parent>__seed<k>, carry a `replicate` block in config.json, and
# are folded into the parent's ± column by build_full_table (never rows).
#   systemd-run --user --unit=seed_variance -p MemoryMax=45G --collect --working-directory=$PWD \
#     /usr/bin/bash -c 'bash scripts/drivers/seed_variance.sh > logs/seed_variance/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
export PYTHONPATH=$ROOT
PY=$ROOT/.pim/bin/python
LOGS=$ROOT/logs/seed_variance
NT=https://ntfy.sh/swirling-tornado-ai691k
PARENT=noise_ablation/L-dw-noiseless-20m
INST=dw-noiseless
STEPS=390000
mkdir -p "$LOGS"
ping()  { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail()  { ping "PIM seed-variance FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }
nb()    { "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout=-1 "notebooks/$1.ipynb" > "$LOGS/$2.log" 2>&1; }

ping "PIM seed-variance: chain started" "L-dw-noiseless-20m: train seed 1 + seed 2 at 390k (~4 h each) -> seed-0 ckpt 422k -> score + appearance-fac -> 20+6 probe seeds x 3 runs -> tables. ETA ~10:30 PT."
for SEED in 1 2; do
  RUN=L-dw-noiseless-20m__seed$SEED
  stage "A/B train seed $SEED ($RUN, $STEPS steps)"
  if [ -f "$ROOT/runs/$PARENT"__seed$SEED/best_model.pt ] && grep -q '"steps": '$STEPS "$ROOT/runs/$PARENT"__seed$SEED/config.json 2>/dev/null && [ -f "$ROOT/runs/$PARENT"__seed$SEED/metrics.jsonl ] && tail -n 1 "$ROOT/runs/$PARENT"__seed$SEED/metrics.jsonl | grep -q '"step": '$STEPS; then
    echo "  $RUN already trained — skipping" | tee -a "$LOGS/driver.log"
  else
    "$PY" scripts/train.py --env discworld --arch transformer_l --instance $INST \
        --topic noise_ablation --run-name "$RUN" --steps $STEPS --seed $SEED --replicate-of $PARENT \
        > "$LOGS/train_seed$SEED.log" 2>&1 || fail "train seed $SEED" "$(tail -20 "$LOGS/train_seed$SEED.log")"
    ping "PIM seed-variance: seed $SEED trained" "$(grep '^done' "$LOGS/train_seed$SEED.log" | tail -1)"
  fi
done

stage "C lay out seed-0 checkpoint 421,875 as a replicate run"
"$PY" experiments/seed_variance/scripts/layout_checkpoint_replicate.py $PARENT 421875 > "$LOGS/c_layout.log" 2>&1 \
  || fail "C layout" "$(tail -10 "$LOGS/c_layout.log")"

stage "D score the replicates (canonical + appearance-fac)"
for R in "$PARENT"__seed1 "$PARENT"__seed2 "$PARENT"__seed0_s421875; do
  bash scripts/drivers/probe_target_fit.sh "$R" appearance-fac seed_variance_fac_$(basename "$R") \
      > "$LOGS/d_fac_$(basename "$R").log" 2>&1 || fail "D appearance-fac $(basename "$R")" "$(tail -20 "$LOGS/d_fac_$(basename "$R").log")"
done
ping "PIM seed-variance: replicates scored" "$(grep -E "noiseless" "$LOGS/../seed_variance_fac_L-dw-noiseless-20m__seed2/headline.txt" 2>/dev/null | head -8)"

stage "E probe-seed replicates (20 regression + 6 factorised linear seeds) on 3 runs"
for R in "$PARENT" "$PARENT"__seed1 "$PARENT"__seed2; do
  "$PY" -u experiments/seed_variance/scripts/probe_seeds.py --run "$R" --targets full appearance-fac --seeds 20 6 \
      > "$LOGS/e_probe_seeds_$(basename "$R").log" 2>&1 || fail "E probe seeds $(basename "$R")" "$(tail -20 "$LOGS/e_probe_seeds_$(basename "$R").log")"
  ping "PIM seed-variance: probe seeds done on $(basename "$R")" "$(grep -E "^full|^appearance-fac" "$LOGS/e_probe_seeds_$(basename "$R").log" | tail -2)"
done

stage "F tables"
nb build_full_table f_table || fail "F build_full_table" "$(tail -20 "$LOGS/f_table.log")"
ping "PIM seed-variance: ALL DONE" "replicates + probe seeds scored; variance.json in each run dir; tables rebuilt." white_check_mark
stage "chain complete"
