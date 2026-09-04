#!/usr/bin/env bash
# Othello editability by move number, end to end, as one capped transient unit:
#   synthesise cases (skipped if present) -> score synth (PI, ND) -> score Li's 1001
#   (PI, ND) -> figures + validation table.   Add `GS` to EDITORS for the MLP replication.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs/othello_edit_by_step
EDITORS="${EDITORS:-PI ND}"
CASES="${CASES:-experiments/othello_edit_by_step/cases/synth_seed0_n256.pkl}"
systemd-run --user --unit=oth_edit_step -p MemoryMax=16G --collect \
  --working-directory="$PWD" --setenv=EDITORS="$EDITORS" --setenv=CASES="$CASES" \
  /usr/bin/bash -c '
    S=experiments/othello_edit_by_step/scripts; L=logs/othello_edit_by_step/edit_by_step.log
    {
      [ -f "$CASES" ] || .pim/bin/python $S/synth_cases.py --per-step 256 --seed 0
      .pim/bin/python $S/edit_by_step.py --cases "$CASES" --label synth --editors $EDITORS
      .pim/bin/python $S/edit_by_step.py --cases li --label li --editors $EDITORS
      .pim/bin/python $S/plot_by_step.py --label synth --compare li
      echo "done  edit_by_step pipeline"
    } > $L 2>&1; rc=$?
    curl -s -o /dev/null -H "Title: PIM edit_by_step" -d "othello edit_by_step exit $rc" \
      https://ntfy.sh/swirling-tornado-ai691k; exit $rc'
