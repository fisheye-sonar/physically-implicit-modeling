#!/usr/bin/env bash
# Causal-state experiment end to end, as one capped transient unit:
#   relevance flags for N test games -> probes by relevance (all points) -> qualitative boards
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs/othello_causal_state
N="${N:-5000}"
TAG="${TAG:-}"     # e.g. TAG=_v2 after a labelling change: a new flags file, nothing overwritten
systemd-run --user --unit=oth_causal -p MemoryMax=24G --collect \
  --working-directory="$PWD" --setenv=N="$N" --setenv=TAG="$TAG" /usr/bin/bash -c '
    S=experiments/othello_causal_state/scripts; L=logs/othello_causal_state/causal_state$TAG.log
    {
      [ -f experiments/othello_causal_state/scores/relevance_test$N$TAG.npz ] || \
        .pim/bin/python $S/label_relevance.py --n-games $N --workers 24 --name test$N$TAG
      .pim/bin/python $S/probe_by_relevance.py --relevance scores/relevance_test$N$TAG.npz
      .pim/bin/python $S/qualitative.py --relevance scores/relevance_test$N$TAG.npz
      echo "done  causal_state pipeline"
    } > $L 2>&1; rc=$?
    curl -s -o /dev/null -H "Title: PIM causal_state" -d "othello causal_state exit $rc" \
      https://ntfy.sh/swirling-tornado-ai691k; exit $rc'
