#!/usr/bin/env bash
# ── L-oth-noflip-20m: Othello Transformer-L on the NO-FLIP variant (the flip ablation) ──────
#
# Identical to initial_othello_comparison/L-oth-20m (their GPT, 8 x 512, 61-way head, 20M
# games, 780k steps, batch 256, lr 1e-3 constant, wd 1e-4, clip 1, seed 0) — the ONLY change is
# the instance: oth-noflip, whose placed discs never recolour the discs they enclose. Same
# legality, passes, game end and index law (datasets/othello/oth-noflip/instance.json).
#
# Stages (each gated on the previous one; a failed stage pings and stops):
#   A  CPU  corpus: train 20M + test 10k + probe 20k + probe_large 170k (~75 min)
#   B  CPU  the 1001 intervention cases (Li's recipe + prefix-length mix) + probe labels
#   C  GPU  train 780k steps                                            (~20 h)
#   W       wait for the scorer-ready marker (master_eval made instance-aware while C runs)
#   D  GPU  master_eval (new baselines for oth-noflip) + build_full_table + training_curves
# Launch:
#   systemd-run --user --unit=oth_noflip -p MemoryMax=40G --collect --working-directory=$PWD \
#       /usr/bin/bash -c 'bash scripts/drivers/oth_noflip.sh > logs/flip_ablation/L-oth-noflip-20m/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
export PYTHONPATH=$ROOT
NT=https://ntfy.sh/swirling-tornado-ai691k
TOPIC=flip_ablation
NAME=L-oth-noflip-20m
INST=oth-noflip
LOGS=$ROOT/logs/$TOPIC/$NAME
READY=$ROOT/experiments/flip_ablation/scorer_ready
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" \
         -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM $NAME FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM $NAME: chain started" "oth-noflip: corpus 20M (~75 min) -> cases -> train 780k (~20 h) -> score -> tables."

stage "A corpus (CPU) $INST"
"$PY" -u -m pim.environments.othello.corpus 20000000 "" "$INST" > "$LOGS/a_corpus.log" 2>&1 \
  || fail "A corpus" "$(tail -15 "$LOGS/a_corpus.log")"
ping "PIM $NAME: corpus DONE" "$(grep -E 'games in|disjoint' "$LOGS/a_corpus.log" | tail -5)"

stage "B intervention cases + probe labels (CPU)"
"$PY" -u scripts/make_othello_edits.py --instance "$INST" > "$LOGS/b_cases.log" 2>&1 \
  || fail "B cases" "$(tail -15 "$LOGS/b_cases.log")"
"$PY" -u -c "
from pim.environments.othello import corpus as oc
p = oc.build(only=('probe', 'probe_large'), instance='$INST', log=print)
for k in ('probe', 'probe_large'):
    d = oc.probe_data(p[k], flip=oc.flip_of('$INST')); print(k, d.tokens.shape, 'labelled')
" >> "$LOGS/b_cases.log" 2>&1 || fail "B probe labels" "$(tail -15 "$LOGS/b_cases.log")"

stage "C train (GPU) $TOPIC/$NAME"
"$PY" scripts/train.py --env othello --instance "$INST" --arch transformer_l \
    --topic "$TOPIC" --run-name "$NAME" --steps 780000 \
    > "$LOGS/c_train.log" 2>&1 || fail "C training" "$(tail -20 "$LOGS/c_train.log")"
ping "PIM $NAME: training DONE" "$(grep -E '^done|best' "$LOGS/c_train.log" | tail -2)
Scoring next." checkered_flag

stage "W wait for scorer-ready marker $READY"
[ -f "$READY" ] || ping "PIM $NAME: waiting for the scorer" "Training done; waiting (up to 12 h) for $READY."
for _ in $(seq 1 720); do [ -f "$READY" ] && break; sleep 60; done
[ -f "$READY" ] || fail "W scorer never became ready (12 h)" "touch $READY once master_eval is instance-aware, then run stage D by hand"

stage "D score + tables (GPU)"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace master_eval.ipynb --ExecutePreprocessor.timeout=21600 ) \
  > "$LOGS/d_score.log" 2>&1 || fail "D scoring" "$(tail -20 "$LOGS/d_score.log")"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace build_full_table.ipynb ) \
  > "$LOGS/d_table.log" 2>&1 || fail "D table" "$(tail -20 "$LOGS/d_table.log")"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace training_curves.ipynb ) > "$LOGS/d_curves.log" 2>&1 || true

SCORES=$ROOT/runs/$TOPIC/$NAME/scores.json
ping "PIM $NAME: ALL DONE" \
"$("$PY" - "$SCORES" <<'PYEOF'
import json, sys
try:
    s = json.load(open(sys.argv[1])); g = s["gates"]; b = s["best"]
    print(f"val {s['val_loss']:.4f} | gates legal_mass {g['legal_mass']:.3f} top1_legal {g['top1_legal']:.3f} CE {g['ce']:.3f} (bayes {g['bayes_ce']:.3f})")
    print(f"skill LIN {max(s['probe_skill']['mine|linear|sequence']):+.3f} MLP {max(s['probe_skill']['mine|mlp|sequence']):+.3f} | "
          f"unedited {s['unedited']['edit_index_union']:+.3f} | " +
          " ".join(f"{k} {v['edit_index_union']:+.3f}/{v['fidelity_ratio']:.2f}" for k, v in b.items() if v))
except Exception as e:
    print(f"(summary failed: {e})")
PYEOF
)" white_check_mark
stage "chain complete"
