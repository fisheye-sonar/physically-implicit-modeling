#!/usr/bin/env bash
# stage watcher + 30-min heartbeat for unit score_cartesian (runs as its own unit; exits when the job ends)
set -u; cd "$(dirname "$0")/../.." || exit 1; ROOT=$PWD; PY=$ROOT/.pim/bin/python; LOGS=$ROOT/logs/score_cartesian
NT=https://ntfy.sh/swirling-tornado-ai691k
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
hb() {
  n=$("$PY" - <<'PY'
import json, glob, os, time
have=[]; 
for p in glob.glob("runs/*/*/scores.json"):
    if p.split("/")[1].startswith("_") or p.split("/")[1]=="archive": continue
    try: d=json.load(open(p))
    except Exception: continue
    b=d.get("bases")
    if isinstance(b, dict) and "cartesian" in b and "frustum" in b: have.append(p.split("/")[2])
newest=max((os.path.getmtime(p) for p in glob.glob("runs/*/*/scores.json")), default=0)
print(f"{len(have)} runs with both blocks; newest scores.json {int((time.time()-newest)/60)} min ago; latest: {sorted(have)[-1] if have else '-'}")
PY
)
  echo "[$(date '+%F %T')] HB unit=$(systemctl --user is-active score_cartesian) | $n | unit mem $(systemctl --user show score_cartesian -p MemoryCurrent --value 2>/dev/null | awk '{printf "%.1fG", $1/1e9}') | disk free $(df -h "$ROOT" | awk 'NR==2{print $4}')" >> "$LOGS/heartbeat.log"
}
t=0
while true; do
  st=$(systemctl --user is-active score_cartesian)
  if grep -q "chain complete" "$LOGS/driver.log" 2>/dev/null; then echo "[$(date '+%F %T')] WATCH: chain complete" >> "$LOGS/heartbeat.log"; hb; exit 0; fi
  if grep -q "^FAILED" "$LOGS/driver.log" 2>/dev/null; then echo "[$(date '+%F %T')] WATCH: FAILED marker" >> "$LOGS/heartbeat.log"; exit 1; fi
  if [ "$st" != "active" ] && [ "$st" != "activating" ]; then
    echo "[$(date '+%F %T')] WATCH: unit $st WITHOUT completion marker — failure" >> "$LOGS/heartbeat.log"
    ping "PIM score_cartesian: unit $st without completion marker" "$(tail -15 "$LOGS/master_eval.log" 2>/dev/null)" warning; exit 1
  fi
  if [ $((t % 1800)) -eq 0 ]; then hb; fi
  sleep 60; t=$((t+60))
done
