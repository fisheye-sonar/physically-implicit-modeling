#!/usr/bin/env bash
# ── training curve: editability vs TRAINING STEP, on the canonical runs' own checkpoints
#
# Both canonical L runs saved log-spaced checkpoints. This lays 8 of each out as run dirs
# (runs/training_curve/<run>_s<step>/, built by experiments/training_curve/scripts/make_training_curve.py) and
# scores every one through the unchanged master_eval — probes, both floors are per
# (instance, arch) so they are reused, and all three editors. ~25 min per discworld
# point (fresh probes at each checkpoint), ~20 per Othello point: ~6 h for 16.
#
# Resumable: master_eval skips any curve dir already scored at EVAL_VERSION.
set -u
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
NT=https://ntfy.sh/swirling-tornado-ai691k
LOGS=$ROOT/logs/training_curve
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"

ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" \
         -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM training-curve FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM training-curve: started" \
"16 checkpoints (steps 1k..780k, both envs) -> master_eval (~6h) -> tables."

stage "A build curve dirs"
"$PY" experiments/training_curve/scripts/make_training_curve.py > "$LOGS/a_build.log" 2>&1 \
  || fail "A build" "$(tail -20 "$LOGS/a_build.log")"

stage "B score (GPU)"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace master_eval.ipynb --ExecutePreprocessor.timeout=36000 ) \
  > "$LOGS/b_score.log" 2>&1 || fail "B score" "$(tail -20 "$LOGS/b_score.log")"

stage "C tables"
( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook \
    --execute --inplace build_full_table.ipynb ) \
  > "$LOGS/c_table.log" 2>&1 || fail "C table" "$(tail -20 "$LOGS/c_table.log")"

ping "PIM training-curve: ALL DONE" "$("$PY" - <<'EOF'
import json, glob
for f in sorted(glob.glob("runs/training_curve/*/scores.json")):
    s = json.load(open(f)); r = s["run"].split("/")[-1]
    if s["env"] == "discworld":
        T = s["bases"]["frustum"]; b = T["best"]
        print(f"{r}: lin {max(T['probe_skill_linear']):+.3f} "
              f"PI {b['PI']['edit_index']:+.3f}/{b['PI']['fidelity_ratio']:.2f} "
              f"GS {b['GS']['edit_index']:+.3f}")
    else:
        b = s["best"]
        print(f"{r}: lin {max(s['probe_skill']['mine|linear|sequence']):+.3f} "
              + " ".join(f"{k} {v['edit_index_union']:+.3f}" for k, v in b.items() if v))
EOF
)" white_check_mark
stage "chain complete"
