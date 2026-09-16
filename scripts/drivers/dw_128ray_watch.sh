#!/usr/bin/env bash
# ── dw_128ray_watch.sh — stage watcher + 30-min heartbeat for the dw_128ray chain (its own unit) ──
#   Watches logs/ray_ablation/dw_128ray/driver.log; exits on `chain complete` / `FAILED`; pings if
#   the unit dies without either marker. Heartbeat (flushed files only): stage, corpus shards done,
#   train step / losses / best val, GPU mem+util, unit memory, disk free.
set -u; cd "$(dirname "$0")/../.." || exit 1; ROOT=$PWD; PY=$ROOT/.pim/bin/python
UNIT=dw_128ray; LOGS=$ROOT/logs/ray_ablation/dw_128ray; RUN=$ROOT/runs/ray_ablation/L-dw-128ray-20m
NT=https://ntfy.sh/swirling-tornado-ai691k
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
hb() {
  st=$(grep "STAGE" "$LOGS/driver.log" 2>/dev/null | tail -1 | sed 's/.*STAGE //; s/ ===//')
  sh=$(ls "$ROOT/datasets/discworld/dw-128ray/train/" 2>/dev/null | grep -c '^_done_')
  tr=$("$PY" - "$RUN/metrics.jsonl" <<'PY'
import json, sys, os
p = sys.argv[1]
if not os.path.exists(p): print("train [not started]"); sys.exit()
rows = [json.loads(l) for l in open(p) if l.strip()]
if not rows: print("train [no rows yet]"); sys.exit()
r = rows[-1]; best = min((x.get("val_loss") for x in rows if x.get("val_loss") is not None), default=None)
f = lambda v, d=5: "[not set]" if v is None else f"{v:.{d}f}"
print(f"step {r.get('step','?')}/780000 train {f(r.get('train_loss'))} val {f(r.get('val_loss'))} best {f(best)} "
      f"elapsed {(r.get('elapsed_s') or 0)/3600:.1f}h")
PY
)
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1 || echo "[nvml n/a]")
  echo "[$(date '+%F %T')] HB unit=$(systemctl --user is-active $UNIT) | stage: ${st:-[none]} | shards $sh/40 | $tr | gpu $gpu | unit mem $(systemctl --user show $UNIT -p MemoryCurrent --value 2>/dev/null | awk '{printf "%.1fG", $1/1e9}') | disk free $(df -h "$ROOT" | awk 'NR==2{print $4}')" >> "$LOGS/heartbeat.log"
}
t=0
while true; do
  st=$(systemctl --user is-active $UNIT)
  if grep -q "chain complete" "$LOGS/driver.log" 2>/dev/null; then echo "[$(date '+%F %T')] WATCH: chain complete" >> "$LOGS/heartbeat.log"; hb; exit 0; fi
  if grep -q "^FAILED" "$LOGS/driver.log" 2>/dev/null; then echo "[$(date '+%F %T')] WATCH: FAILED marker" >> "$LOGS/heartbeat.log"; exit 1; fi
  if [ "$st" != "active" ] && [ "$st" != "activating" ]; then
    echo "[$(date '+%F %T')] WATCH: unit $st WITHOUT completion marker — failure" >> "$LOGS/heartbeat.log"
    ping "PIM dw-128ray: unit $st without completion marker" "$(tail -15 "$LOGS/unit.log" 2>/dev/null)" warning; exit 1
  fi
  if [ $((t % 1800)) -eq 0 ]; then hb; fi
  sleep 60; t=$((t+60))
done
