#!/usr/bin/env bash
# Othello decodability by move number, as a capped transient unit (the script refuses to
# refit on a probe-cache miss, so the cap is belt-and-braces).
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs/othello_by_step/decode
systemd-run --user --unit=oth_decode_step -p MemoryMax=16G --collect \
  --working-directory="$PWD" /usr/bin/bash -c \
  '.pim/bin/python experiments/othello_by_step/decode/scripts/decode_by_step.py "$@" \
     > logs/othello_by_step/decode/decode_by_step.log 2>&1; rc=$?; \
   curl -s -o /dev/null -H "Title: PIM decode_by_step" -d "othello decode_by_step exit $rc" \
     https://ntfy.sh/swirling-tornado-ai691k; exit $rc' _ "$@"
