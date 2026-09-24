#!/usr/bin/env bash
# One categorical-IM catch-up job of the paper_ci queue (cpu lane: it SHARES the GPU with the gpu-lane job).
# scripts/drivers/score_pending.sh holds no lock and master_eval is executed IN PLACE, so this never starts
# while another master_eval execution runs on this host, nor when the host's gpu-lane training job is within
# 2.5 h of its own scoring stage (a catch-up job lasts ~2 h). It prints while it waits, so the queue's stall
# check sees a live log. Env (PIM_ADD_CAT_IM, PIM_ONLY_RUNS, PIM_DW_BASES, ...) comes from the queue job.
set -u
cd "$(dirname "$0")/../../.." || exit 1
NAME=${1:?usage: catchup_job.sh <name>}
PY=.pim/bin/python
busy() { ps -eo comm,args | awk '$1 ~ /^python/ && /nbconvert/ && /master_eval/' | grep -q .; }
near_scoring() {
  "$PY" - <<'PYEOF'
import json, socket, sys
try:
    s = json.load(open("experiments/paper_ci/dashboard/state.json"))
    left = [(j.get("progress") or {}).get("train_left_h") for j in s.get("jobs", [])
            if j.get("status") == "running" and j.get("host") == "lab" and j.get("lane", "gpu") == "gpu"]
    sys.exit(0 if any(x is not None and x < 2.5 for x in left) else 1)
except Exception:
    sys.exit(1)
PYEOF
}
while busy || near_scoring; do
  echo "[$(date '+%F %T')] $NAME waiting: a master_eval execution is running here, or the gpu-lane job is within 2.5 h of scoring"
  sleep 300
done
exec bash scripts/drivers/score_pending.sh "$NAME"
