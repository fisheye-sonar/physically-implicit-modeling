#!/usr/bin/env bash
# ── grid-target control: discretised discworld state on L-dw-noiseless-20m (2026-09-08) ──
#   A  fit the grid-cell probes (LIN + MLP-128, 9 points, 200k seq, 50 epochs)   (~2 h)
#   B  PI / ND / GS through them on the canonical bench                           (~20 min)
#   C  floors: random-init probes (all points) + observation floor                (~2 h)
# Under a unit:  systemd-run --user --unit=grid_ctl -p MemoryMax=45G --collect \
#   --working-directory=$PWD /usr/bin/bash -c 'bash experiments/grid_target_control/drivers/grid_target_control.sh > logs/grid_target_control/unit.log 2>&1'
set -u
export GRID=${GRID:-16x8}          # grid resolution; a non-default value suffixes every output
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
S=$ROOT/experiments/grid_target_control/scripts
LOGS=$ROOT/logs/grid_target_control
NT=https://ntfy.sh/swirling-tornado-ai691k
mkdir -p "$LOGS"
ping() { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail() { ping "PIM grid-control FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

ping "PIM grid-target control ($GRID): started" "A probes (200k seq, 9 pts, ~45 min) -> B edits (~20 min) -> D regression-to-cells -> C floors (~50 min)"
stage "A grid probes"
"$PY" -u "$S/fit_grid_probes.py" > "$LOGS/a_probes.log" 2>&1 || fail "A probes" "$(tail -15 "$LOGS/a_probes.log")"
ping "PIM grid-control: probes DONE" "$(grep -E 'best points|linear: err|mlp: err' "$LOGS/a_probes.log" | tail -3)"
stage "B edits"
"$PY" -u "$S/edit_grid.py" > "$LOGS/b_edit.log" 2>&1 || fail "B edits" "$(tail -15 "$LOGS/b_edit.log")"
ping "PIM grid-control: EDIT RESULT" "$(sed -n '/^| target/,$p' "$LOGS/b_edit.log" | head -6)" white_check_mark
stage "D regression probes on the grid axis"
"$PY" -u "$S/regression_to_cells.py" > "$LOGS/d_regression_to_cells.log" 2>&1 || fail "D regression-to-cells" "$(tail -15 "$LOGS/d_regression_to_cells.log")"
stage "C floors"
"$PY" -u "$S/fit_grid_probes.py" --random-init > "$LOGS/c_random_init.log" 2>&1 || fail "C random-init" "$(tail -15 "$LOGS/c_random_init.log")"
"$PY" -u "$S/fit_grid_obs_floor.py" > "$LOGS/c_obs_floor.log" 2>&1 || fail "C obs floor" "$(tail -15 "$LOGS/c_obs_floor.log")"
ping "PIM grid-control: ALL DONE" "$(grep -E 'best points' "$LOGS/c_random_init.log" | tail -1; grep -E 'skill' "$LOGS/c_obs_floor.log" | tail -2)" white_check_mark
stage "chain complete"
