#!/usr/bin/env bash
# ── fixed-resolution grid targets on the 5-ray and 16-ray models (2026-09-15, Sevan) ─────────────
#
# The ray axis (5 / 8 / 16 / 128) mixes observation resolution with the GRANULARITY of the
# observation-exact target (13 / 20 / 34 / 500+ factor classes). Under a fixed grid the 8-ray and
# 128-ray models are nearly indistinguishable (grid-8x4: PI +0.31 vs +0.26, ND +0.28 vs +0.35, GS +0.31
# vs +0.28), so: fit grid-8x4 and grid-16x8 on L-dw-5ray-20m and L-dw-16ray-20m — model probes, the
# random-init and observation floors, then master_eval + tables — and compare the four models under
# the SAME targets. Four sequential probe_target_fit.sh calls (each rescoring); one GPU job at a time.
#   systemd-run --user --unit=grid_targets_5_16 -p MemoryMax=45G --collect --working-directory=$PWD \
#       /usr/bin/bash -c 'bash scripts/drivers/grid_targets_5_16.sh > logs/grid_targets_5_16/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
NT=https://ntfy.sh/swirling-tornado-ai691k
LOGS=$ROOT/logs/grid_targets_5_16
mkdir -p "$LOGS"
ping()  { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail()  { ping "PIM grid_targets_5_16 FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM grid targets: started" "grid-8x4 + grid-16x8 on L-dw-5ray-20m and L-dw-16ray-20m (probes, floors, rescore each); ~4 h on the 5090."
for RUN in ray_ablation/L-dw-5ray-20m ray_ablation/L-dw-16ray-20m; do
  for TARGET in grid-8x4 grid-16x8; do
    NAME=grid_$(basename "$RUN" | sed 's/L-dw-//; s/-20m//')_${TARGET//-/_}
    stage "$RUN · $TARGET  ($NAME)"
    bash scripts/drivers/probe_target_fit.sh "$RUN" "$TARGET" "$NAME" > "$LOGS/$NAME.log" 2>&1 \
        || fail "$RUN $TARGET" "$(tail -20 "$LOGS/$NAME.log")"
  done
done
ping "PIM grid targets: ALL DONE" "$("$ROOT/.pim/bin/python" - <<'PYEOF'
import json
for r in ("ray_ablation/L-dw-5ray-20m", "ray_ablation/L-dw-8ray-20m", "ray_ablation/L-dw-16ray-20m", "noise_ablation/L-dw-noiseless-20m"):
    s = json.load(open(f"runs/{r}/scores.json"))
    for blk in ("grid-8x4", "grid-16x8"):
        T = s["bases"].get(blk)
        if T: print(r.split("/")[1], blk, " ".join(f"{k} {v['edit_index']:+.2f}/{v['fidelity_ratio']:.2f}" for k, v in T["best"].items() if v))
PYEOF
)" white_check_mark
stage "chain complete"
