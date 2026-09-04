#!/usr/bin/env bash
# GS read-out landing pilot, as a capped transient unit (a probe-cache miss would be a
# 20 GB refit; the cap turns that into a unit OOM instead of a machine OOM).
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs/gs_readout_pilot
systemd-run --user --unit=gs_pilot -p MemoryMax=24G --collect \
  --working-directory="$PWD" /usr/bin/bash -c \
  '.pim/bin/python experiments/gs_readout_pilot/scripts/gs_readout_pilot.py "$@" \
     > logs/gs_readout_pilot/gs_readout_pilot.log 2>&1; rc=$?; \
   curl -s -o /dev/null -H "Title: PIM gs_pilot" -d "gs_readout_pilot exit $rc" \
     https://ntfy.sh/swirling-tornado-ai691k; exit $rc' _ "$@"
