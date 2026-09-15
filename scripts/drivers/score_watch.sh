#!/usr/bin/env bash
# ── score_watch.sh <unit> — stage watcher + 30-min heartbeat for a scoring unit (runs as its own unit) ──
#   Reads logs/<unit>/driver.log; exits on `chain complete` / `FAILED`; pings if the unit dies without
#   the marker. Heartbeat: runs whose every block carries IM, newest scores.json age, unit memory, disk.
set -u; UNIT=${1:?unit}; cd "$(dirname "$0")/../.." || exit 1; ROOT=$PWD; PY=$ROOT/.pim/bin/python; LOGS=$ROOT/logs/$UNIT
NT=https://ntfy.sh/swirling-tornado-ai691k
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
hb() {
  n=$("$PY" - <<'PY'
import json, glob, os, time
done=[]; total=0
for p in glob.glob("runs/*/*/scores.json"):
    top=p.split("/")[1]
    if top.startswith("_") or top=="archive": continue
    try: d=json.load(open(p))
    except Exception: continue
    blocks=list(d.get("bases",{}).values()) + ([d] if "arms" in d else [])
    if not blocks: continue
    total+=1
    if all(any(a.get("editor")=="IM" for a in b.get("arms",[])) for b in blocks): done.append(p.split("/")[2])
newest=max((os.path.getmtime(p) for p in glob.glob("runs/*/*/scores.json")), default=0)
print(f"{len(done)}/{total} runs with IM on every block; newest scores.json {int((time.time()-newest)/60)} min ago")
PY
)
  echo "[$(date '+%F %T')] HB unit=$(systemctl --user is-active $UNIT) | $n | unit mem $(systemctl --user show $UNIT -p MemoryCurrent --value 2>/dev/null | awk '{printf "%.1fG", $1/1e9}') | disk free $(df -h "$ROOT" | awk 'NR==2{print $4}')" >> "$LOGS/heartbeat.log"
}
t=0
while true; do
  st=$(systemctl --user is-active $UNIT)
  if grep -q "chain complete" "$LOGS/driver.log" 2>/dev/null; then echo "[$(date '+%F %T')] WATCH: chain complete" >> "$LOGS/heartbeat.log"; hb; exit 0; fi
  if grep -q "^FAILED" "$LOGS/driver.log" 2>/dev/null; then echo "[$(date '+%F %T')] WATCH: FAILED marker" >> "$LOGS/heartbeat.log"; exit 1; fi
  if [ "$st" != "active" ] && [ "$st" != "activating" ]; then
    echo "[$(date '+%F %T')] WATCH: unit $st WITHOUT completion marker — failure" >> "$LOGS/heartbeat.log"
    ping "PIM $UNIT: unit $st without completion marker" "$(tail -15 "$LOGS/master_eval.log" 2>/dev/null)" warning; exit 1
  fi
  if [ $((t % 1800)) -eq 0 ]; then hb; fi
  sleep 60; t=$((t+60))
done
