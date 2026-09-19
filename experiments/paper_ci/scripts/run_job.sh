#!/usr/bin/env bash
# ── run_job.sh — the per-job wrapper the dispatcher launches inside a systemd-run unit ──────
#   usage (cwd = repo root): bash experiments/paper_ci/scripts/run_job.sh <job id> <attempt>
#
# Runs on WHICHEVER host the dispatcher chose. It records the attempt in
# experiments/paper_ci/state/<id>.run.json (started, pid, unit; then ended, rc, duration) so
# the dispatcher can fold the outcome in even after a reboot, runs the job's command with the
# job's environment, and pings the alert channel itself on FAILURE — that ping does not
# depend on the lab box being up. Success pings come from the dispatcher, with the numbers.
set -u
ROOT=$PWD
Q=$ROOT/experiments/paper_ci
ID=${1:?job id}
ATTEMPT=${2:-1}
JOB=$Q/queue/$ID.json
REC=$Q/state/$ID.run.json
LOGS=$ROOT/logs/paper_ci/$ID
mkdir -p "$LOGS" "$Q/state"
PY=$ROOT/.pim/bin/python
export PYTHONPATH=$ROOT
HOST=${PIM_CI_HOST:-$(hostname)}
NT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("ntfy",{}).get("alerts","https://ntfy.sh/swirling-tornado-ai691k"))' "$Q/config.json" 2>/dev/null || echo https://ntfy.sh/swirling-tornado-ai691k)

[ -f "$JOB" ] || { echo "no such job: $JOB"; exit 2; }
CMD=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cmd"])' "$JOB")
# the job's environment (a dict in the job file) — exported for the command
while IFS='=' read -r k v; do [ -n "$k" ] && export "$k=$v"; done < <(
  "$PY" -c 'import json,sys; [print(f"{k}={v}") for k, v in json.load(open(sys.argv[1])).get("env", {}).items()]' "$JOB")

write_rec() {  # merge fields into the run record atomically
  "$PY" - "$REC" "$@" <<'PYEOF'
import json, os, sys, tempfile
p, kv = sys.argv[1], sys.argv[2:]
d = json.load(open(p)) if os.path.exists(p) else {}
for x in kv:
    k, v = x.split("=", 1)
    try:
        v = json.loads(v)
    except Exception:
        pass
    d[k] = v
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p), prefix=".tmp")
with os.fdopen(fd, "w") as f:
    json.dump(d, f, indent=1)
os.replace(tmp, p)
PYEOF
}

T0=$(date +%s)
write_rec "id=\"$ID\"" "host=\"$HOST\"" "attempt=$ATTEMPT" "pid=$$" "unit=\"${PIM_CI_UNIT:-pimci-$ID}\"" \
          "started=\"$(date -Is)\"" "started_ts=$T0" "ended=null" "rc=null" "cmd=$(printf '%s' "$CMD" | "$PY" -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
echo "=== [$(date '+%F %T')] $ID attempt $ATTEMPT on $HOST: $CMD" | tee -a "$LOGS/driver.log"
bash -c "$CMD" > "$LOGS/attempt_$ATTEMPT.log" 2>&1
RC=$?
T1=$(date +%s)
write_rec "ended=\"$(date -Is)\"" "ended_ts=$T1" "rc=$RC" "duration_s=$((T1 - T0))"
echo "=== [$(date '+%F %T')] $ID attempt $ATTEMPT rc=$RC after $(( (T1 - T0) / 60 )) min" | tee -a "$LOGS/driver.log"
if [ "$RC" != "0" ]; then
  curl -sS --max-time 20 -H "Title: PIM CI FAILED: $ID on $HOST (attempt $ATTEMPT, rc $RC)" -H "Tags: rotating_light" \
       -H "Priority: high" -d "$(tail -n 15 "$LOGS/attempt_$ATTEMPT.log" | cut -c1-300)" "$NT" > /dev/null 2>&1 || true
fi
exit $RC
