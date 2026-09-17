#!/usr/bin/env bash
# ── stage watcher + 30-min heartbeat for the oth_adjflip_seeds chain (its own unit) ──────────
#   Reads logs/oth_adjflip_seeds/driver.log; exits on `chain complete` / `FAILED`; pings if the
#   unit dies without either marker. Heartbeat (flushed files only): stage, the training
#   replicate's step / losses, probe-seed progress, GPU, unit memory, disk.
set -u; cd "$(dirname "$0")/../.." || exit 1; ROOT=$PWD; PY=$ROOT/.pim/bin/python
UNIT=oth_adjflip_seeds; LOGS=$ROOT/logs/$UNIT
NT=https://ntfy.sh/swirling-tornado-ai691k
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
hb() {
  st=$(grep "STAGE" "$LOGS/driver.log" 2>/dev/null | tail -1 | sed 's/.*STAGE //; s/ ===//')
  tr=$("$PY" - <<'PY'
import glob, json, os
rows = []
for p in glob.glob("runs/adjacent_flip_ablation/L-oth-adjacent-flip-20m__seed*/metrics.jsonl"):
    try:
        r = [json.loads(x) for x in open(p) if x.strip()]
    except Exception:
        continue
    if r:
        rows.append((os.path.getmtime(p), p.split("/")[2], r[-1], min(x["val_loss"] for x in r)))
if not rows:
    print("train [no replicate started]")
else:
    _, name, r, best = max(rows)
    f = lambda v: "[not set]" if v is None else f"{v:.4f}"
    print(f"{name} step {r.get('step','?')}/390000 train {f(r.get('train_loss'))} val {f(r.get('val_loss'))} "
          f"best {f(best)} elapsed {(r.get('elapsed_s') or 0)/3600:.1f}h")
PY
)
  ps=$(grep -h "seed " "$LOGS"/*probeseeds*.log 2>/dev/null | tail -1 | sed 's/^ *//' | cut -c1-70)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null | tr -d ' ' | head -1)
  echo "[$(date '+%F %T')] HB unit=$(systemctl --user is-active $UNIT) | stage: ${st:-[none]} | $tr | probes: ${ps:-none yet} | gpu $gpu | unit mem $(systemctl --user show $UNIT -p MemoryCurrent --value 2>/dev/null | awk '{printf "%.1fG", $1/1e9}') | disk free $(df -h "$ROOT" | awk 'NR==2{print $4}')" >> "$LOGS/heartbeat.log"
}
t=0
while true; do
  st=$(systemctl --user is-active $UNIT)
  if grep -q "chain complete" "$LOGS/driver.log" 2>/dev/null; then echo "[$(date '+%F %T')] WATCH: chain complete" >> "$LOGS/heartbeat.log"; hb; exit 0; fi
  if grep -q "^FAILED" "$LOGS/driver.log" 2>/dev/null; then echo "[$(date '+%F %T')] WATCH: FAILED marker" >> "$LOGS/heartbeat.log"; exit 1; fi
  if [ "$st" != "active" ] && [ "$st" != "activating" ]; then
    echo "[$(date '+%F %T')] WATCH: unit $st WITHOUT completion marker" >> "$LOGS/heartbeat.log"
    ping "PIM oth-adjflip-seeds: unit $st without completion marker" "$(tail -12 "$LOGS/unit.log" 2>/dev/null)
(if you stopped it deliberately, ignore — relaunch the same unit command to continue)" warning; exit 1
  fi
  if [ $((t % 1800)) -eq 0 ]; then hb; fi
  sleep 60; t=$((t+60))
done
