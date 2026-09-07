#!/usr/bin/env bash
# ── L-oth-20m-mse: Othello Transformer-L with MSE on the one-hot next move ───────────
#
# Identical to initial_othello_comparison/L-oth-20m in architecture (their GPT, 8 x 512,
# 61-way head), data (oth-uniform 20M), recipe (780k steps, batch 256, lr 1e-3 constant,
# wd 1e-4, clip 1, seed 0) — the ONLY change is the objective: `mse_next_move_onehot`
# instead of cross-entropy, so the head's outputs are read as probability estimates
# (`output_kind="raw"`: no softmax, no clipping, no renormalisation) by every scorer.
#
# Stages (each gated on the previous one; a failed stage pings and stops):
#   C  GPU  train 780k steps                                            (~19.5 h)
#   D  GPU  master_eval scores the run (fresh probes cached in the run; the (oth-uniform,
#           transformer_l_tokens) floors are cache hits) + build_full_table + curves
#   E  GPU  experiments/othello_mse_head: is the head a distribution? raw vs clipnorm
# Run it under a transient unit so it survives the launching session:
#   systemd-run --user --unit=oth_mse -p MemoryMax=40G --collect \
#       --working-directory=$PWD bash scripts/drivers/oth_mse.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
export PYTHONPATH=$ROOT
NT=https://ntfy.sh/swirling-tornado-ai691k
TOPIC=objective_ablation
NAME=L-oth-20m-mse
LOGS=$ROOT/logs/$TOPIC/$NAME
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" \
         -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM $NAME FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM $NAME: chain started" "Othello Transformer-L, MSE on the one-hot next move (raw head). Train 780k (~19.5 h) -> score -> tables -> distribution check."

stage "C train (GPU) $TOPIC/$NAME"
"$PY" scripts/train.py --env othello --arch transformer_l --objective mse_onehot \
    --topic "$TOPIC" --run-name "$NAME" --steps 780000 \
    > "$LOGS/c_train.log" 2>&1 || fail "C training" "$(tail -20 "$LOGS/c_train.log")"
ping "PIM $NAME: training DONE" "$(grep -E '^done|best' "$LOGS/c_train.log" | tail -2)
Scoring now (~30 min)." checkered_flag

stage "D score + tables (GPU)"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace master_eval.ipynb --ExecutePreprocessor.timeout=14400 ) \
  > "$LOGS/d_score.log" 2>&1 || fail "D scoring" "$(tail -20 "$LOGS/d_score.log")"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace build_full_table.ipynb ) \
  > "$LOGS/d_table.log" 2>&1 || fail "D table" "$(tail -20 "$LOGS/d_table.log")"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace training_curves.ipynb ) > "$LOGS/d_curves.log" 2>&1 || true

stage "E distribution check (GPU)"
"$PY" experiments/othello_mse_head/scripts/distribution_check.py --run "runs/$TOPIC/$NAME" \
    > "$LOGS/e_distribution.log" 2>&1 || fail "E distribution check" "$(tail -20 "$LOGS/e_distribution.log")"

SCORES=$ROOT/runs/$TOPIC/$NAME/scores.json
ping "PIM $NAME: ALL DONE" \
"$("$PY" - "$SCORES" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
g = s["gates"]; b = s["best"]
print(f"val {s['val_loss']:.5f} | gates[{g.get('output_kind','?')}] legal_mass {g['legal_mass']:.3f} top1_legal {g['top1_legal']:.3f} "
      f"out_sum {g.get('out_sum_mean', float('nan')):.3f} neg {g.get('out_neg_mass_mean', float('nan')):.3f}")
print(f"skill LIN {max(s['probe_skill']['mine|linear|sequence']):+.3f} MLP {max(s['probe_skill']['mine|mlp|sequence']):+.3f} | "
      f"unedited {s['unedited']['edit_index_union']:+.3f} | " +
      " ".join(f"{k} {v['edit_index_union']:+.3f}/{v['fidelity_ratio']:.2f}" for k, v in b.items() if v))
EOF
)
$(tail -6 "$LOGS/e_distribution.log" | head -5)" white_check_mark
stage "chain complete"
