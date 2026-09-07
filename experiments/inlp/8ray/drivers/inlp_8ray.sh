#!/usr/bin/env bash
# ── INLP on L-dw-8ray-20m: the nullspace cascade at every residual point, both bases ──
# Runs the canonical INLP experiment script (experiments/inlp/scripts/inlp_dw.py) on the
# 8-ray transformer: per-probe aggregate AND per-component held-out R² of the cascade,
# and the multi-probe edit sweep (K x alpha, uniform / R²-shrunk targets) scored with the
# canonical Edit Index + fidelity. Every fitted cascade is persisted into the run's
# probes/ through ProbeCache (a re-run is a cache hit). One capped unit:
#   systemd-run --user --unit=inlp_8ray -p MemoryMax=30G --collect \
#       --working-directory=$PWD bash experiments/inlp/8ray/drivers/inlp_8ray.sh
set -u
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
export PYTHONPATH=$ROOT
NT=https://ntfy.sh/swirling-tornado-ai691k
RUN=runs/ray_ablation/L-dw-8ray-20m
N_SEQ=${N_SEQ:-20000}
OUTD=$ROOT/experiments/inlp/8ray/scores
LOGS=$ROOT/logs/inlp/8ray
mkdir -p "$OUTD" "$LOGS"
echo $$ > "$LOGS/driver.pid"
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" \
         -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM inlp_8ray FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM inlp_8ray: started" "INLP cascades on L-dw-8ray-20m, frustum then cartesian, n_seq $N_SEQ (~1-1.5 h each with the GPU shared)."
for basis in frustum cartesian; do
  stage "INLP $basis"
  "$PY" experiments/inlp/scripts/inlp_dw.py "$RUN" "$basis" "$N_SEQ" \
      "$OUTD/inlp_L-dw-8ray-20m_$basis.json" > "$LOGS/inlp_$basis.log" 2>&1 \
      || fail "INLP $basis" "$(tail -15 "$LOGS/inlp_$basis.log")"
  ping "PIM inlp_8ray: $basis DONE" "$(grep -E 'wiring check|^point [0-9]+:' "$LOGS/inlp_$basis.log" | tail -4 | cut -c1-200)"
done
stage "chain complete"
