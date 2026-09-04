#!/usr/bin/env bash
# Stall check, self-contained so it survives context loss. Prints STATE + the evidence.
cd /home/sevan/research/PIM/physically-implicit-modeling || exit 1
L=runs/dw_noiseless/logs
PID=$(cat $L/driver.pid 2>/dev/null)
# ⛔ liveness by PID, never `pgrep -f <name>` (matches the checking shell itself)
if kill -0 "$PID" 2>/dev/null; then ALIVE=yes; else ALIVE=NO; fi
echo "== dw-noiseless @ $(date '+%F %T') =="
echo "driver PID $PID alive=$ALIVE   stage: $(grep '=== \[' $L/driver.log | tail -1 | sed 's/.*STAGE //')"
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)"
for f in a_rescore b1_eval b2_probe b3_corpus c_train d_score d_table; do
  [ -f "$L/$f.log" ] || continue
  AGE=$(( $(date +%s) - $(stat -c %Y "$L/$f.log") ))
  printf '  %-10s last-write %5ss ago | %s\n' "$f" "$AGE" "$(tail -1 "$L/$f.log" | cut -c1-100)"
done
# Stage C progress: metrics.jsonl is written+flushed per val pass, so it is the
# AUTHORITATIVE liveness signal for training (c_train.log is block-buffered and will
# look stale for hours even while training runs normally).
M=runs/noise_ablation/L-dw-noiseless-20m/metrics.jsonl
if [ -f "$M" ]; then
  AGE=$(( $(date +%s) - $(stat -c %Y "$M") ))
  echo "  TRAIN: $(tail -1 "$M" | python3 -c "import json,sys; r=json.load(sys.stdin); print(f\"step {r['step']:,}/780,000 ({100*r['step']/780000:.1f}%)  val {r['val_loss']:.6f}  {r['elapsed_s']/3600:.2f}h elapsed, ETA {(780000/r['step']-1)*r['elapsed_s']/3600:.2f}h\")" 2>/dev/null)"
  echo "         metrics.jsonl last-write ${AGE}s ago (val passes are every 5,000 steps ~ every 185s; >600s = STALLED)"
fi
DONE=$(ls datasets/discworld/dw-noiseless/train/_done_* 2>/dev/null | wc -l)
echo "  corpus shards done: $DONE/40   (~$(du -sh datasets/discworld/dw-noiseless 2>/dev/null | cut -f1) on disk)"
# ⚠ b3_corpus.log is BLOCK-BUFFERED (the driver calls python without -u), so its
# last-write age looks stale even while shards land normally. During stage B the
# SHARD COUNT is the authoritative progress signal, not the log age. Likewise
# a_rescore/d_score write only at nbconvert start and end — expect long quiet gaps.
[ -f runs/noise_ablation/L-dw-noiseless-20m/scores.json ] && echo "  *** SCORES WRITTEN — chain complete ***"
if [ "$ALIVE" = NO ]; then
  if [ -f runs/noise_ablation/L-dw-noiseless-20m/scores.json ]; then echo "VERDICT: COMPLETE"
  else echo "VERDICT: DRIVER DEAD BEFORE COMPLETION — read the newest log above, fix, and RESTART: bash scripts/drivers/dw_noiseless.sh (it is idempotent/resumable)"; fi
else
  echo "VERDICT: running. STALL RULE — if the newest active log's last-write age keeps growing across two consecutive checks AND the GPU/shard count has not advanced, the chain is STUCK: diagnose and restart rather than wait."
fi
