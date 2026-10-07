#!/bin/bash
# Two-token edits at n = 1000 paired cases per Othello variant (2026-09-24, Sevan's request).
# Cases come from each instance's extended pool (scripts/make_othello_edits.py --n <pool>; its first 1000
# cases are the bench). Each variant retries up to 4 times: the pair search resumes from its partial log.
# Launched as a memory-capped systemd user unit:
#   systemd-run --user --unit=two-flip-n1000 --collect -p MemoryMax=42G -p OOMScoreAdjust=1000 \
#       --working-directory=<repo> bash experiments/two_flip_full/run_n1000.sh
cd "$(dirname "$0")/../.."
LOG=logs/two_flip_full
for spec in "adjacency_ablation/L-oth-adjacent-20m::3000" "flip_ablation/L-oth-noflip-20m:--no-legal:1000" \
            "initial_othello_comparison/L-oth-20m::3000" "adjacent_flip_ablation/L-oth-adjacent-flip-20m::8000"; do
  IFS=: read -r run flag pool <<< "$spec"; name=$(basename "$run")
  for attempt in 1 2 3 4; do
    echo "=== n1000 $run attempt $attempt $(date '+%F %T')" >> $LOG/driver.log
    .pim/bin/python scripts/two_flip_editability.py --run "$run" $flag --n 1000 --pool "$pool" --workers 24 \
        --out two_flip_editability_n1000.json >> "$LOG/${name}_n1000.log" 2>&1
    rc=$?
    echo "=== n1000 $run attempt $attempt exit $rc $(date '+%F %T')" >> $LOG/driver.log
    [ $rc -eq 0 ] && break
  done
done
echo "=== n1000 ALL DONE $(date '+%F %T')" >> $LOG/driver.log
