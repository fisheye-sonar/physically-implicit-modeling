#!/usr/bin/env bash
# The unified-protocol RESCORE (2026-09-12, Sevan): every listed run in both environments under
# the new edit protocol — 1000 cases, fixed edit position, per-instance benches, full-state
# writes, shared alpha grids and GS layers, Othello headline = symmetric difference — plus the
# 5-ray floors, the Table 3 alignment refresh + Haufe editability, and both table notebooks.
#
# Launch (GPU must be free; nothing else may score or fit while it runs):
#   systemd-run --user --unit=rescore_protocol -p MemoryMax=45G --collect --working-directory=$PWD \
#       bash scripts/drivers/rescore_2026-09-12.sh
# Then arm a watcher (harness/OVERNIGHT.md): poll `systemctl --user is-active rescore_protocol`
# and count `grep -l '"eval_version": "2026-09-12' runs/*/*/scores.json`.
#
# Stages (each logged to logs/rescore_2026-09-12/, each pinged):
#   1  fit the dw-5ray appearance-fac floors (random-init + observation) — the one thing the
#      scorer will not fit itself (require_cached)
#   2  master_eval — rescores every discworld run (EVAL_VERSION_BY_ENV discworld 2026-09-12.2)
#      and every Othello run (othello 2026-09-12.1) on the new benches; computes the dw-5ray
#      canonical floors (the baselines cell fits floors for every scored instance)
#   3  table3_alignment.py — refresh Table 3's alignment rows at the NEW best PI points
#   4  table3_haufe_edit.py — Table 3's PI-after-Haufe columns
#   5  test_loss.py — Table 4 refresh (cheap)
#   6  build_paper_tables + build_full_tables
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=$ROOT/.pim/bin/python
LOGS=$ROOT/logs/rescore_2026-09-12
mkdir -p "$LOGS"
NT=https://ntfy.sh/swirling-tornado-ai691k
cd "$ROOT"

ping()  { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail()  { ping "PIM rescore FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }
nb()    { ( cd "$ROOT/notebooks" && "$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook --execute --inplace "$1.ipynb" \
            --ExecutePreprocessor.timeout=57600 ) > "$LOGS/$2.log" 2>&1 || fail "$2" "$(tail -25 "$LOGS/$2.log")"; }

ping "PIM rescore: chain started" "1 dw-5ray fac floors → 2 master_eval (both envs, 1000-case benches; ~9-11 h) → 3 alignment refresh → 4 Haufe editability → 5 test loss → 6 tables." rocket

stage "1 dw-5ray appearance-fac floors"
"$PY" -W ignore scripts/fit_probes.py --run ray_ablation/L-dw-5ray-20m --target appearance-fac --random-init \
    > "$LOGS/fit_5ray_fac_random.log" 2>&1 || fail "5ray random-init floor" "$(tail -15 "$LOGS/fit_5ray_fac_random.log")"
"$PY" -W ignore scripts/fit_probes.py --run ray_ablation/L-dw-5ray-20m --target appearance-fac --observation \
    > "$LOGS/fit_5ray_fac_obs.log" 2>&1 || fail "5ray observation floor" "$(tail -15 "$LOGS/fit_5ray_fac_obs.log")"
ping "PIM rescore: 5-ray floors DONE" "$(grep -h 'skill' "$LOGS"/fit_5ray_fac_*.log | tail -4)"

stage "2 master_eval (both environments)"
nb master_eval master_eval
N_NEW=$(grep -l '"eval_version": "2026-09-12' runs/*/*/scores.json 2>/dev/null | wc -l)
ping "PIM rescore: master_eval DONE" "$N_NEW scores.json at the new versions." white_check_mark

stage "3 Table 3 alignment refresh"
"$PY" -W ignore experiments/edit_direction_alignment/scripts/table3_alignment.py > "$LOGS/table3_alignment.log" 2>&1 \
    || fail "table3_alignment" "$(tail -15 "$LOGS/table3_alignment.log")"

stage "4 Table 3 Haufe editability"
"$PY" -W ignore experiments/edit_direction_alignment/scripts/table3_haufe_edit.py > "$LOGS/table3_haufe.log" 2>&1 \
    || fail "table3_haufe_edit" "$(tail -15 "$LOGS/table3_haufe.log")"
ping "PIM rescore: Haufe editability DONE" "$(grep -h 'PI-haufe' "$LOGS/table3_haufe.log" | head -16)"

stage "5 test loss"
"$PY" -W ignore experiments/bayes_floor/scripts/test_loss.py > "$LOGS/test_loss.log" 2>&1 || true

stage "6 tables"
nb build_paper_tables build_paper_tables
nb build_full_tables build_full_tables
"$PY" - > "$LOGS/headline.txt" 2>&1 <<'EOF'
import json, glob
for p in sorted(glob.glob("runs/*/*/scores.json")):
    if p.startswith("runs/_") or "/archive/" in p or "training_curve" in p: continue
    s = json.load(open(p))
    if not str(s.get("eval_version", "")).startswith("2026-09-12"): continue
    if s["env"] == "othello":
        u = s["unedited"]["edit_index_symdiff"]; b = {e: s["best"][e] for e in ("PI", "ND", "GS") if s["best"].get(e)}
        print(f"{s['run']:<46} uned {u:+.3f}  " + "  ".join(f"{e} {v['edit_index_symdiff']:+.3f}/{v['fidelity_ratio']:.2f}" for e, v in b.items()))
    else:
        T = s["bases"].get("frustum") or next(iter(s["bases"].values()))
        u = T["unedited"]["edit_index"]; b = {e: T["best"][e] for e in ("PI", "GS") if T["best"].get(e)}
        print(f"{s['run']:<46} uned {u:+.3f}  " + "  ".join(f"{e} {v['edit_index']:+.3f}/{v['fidelity_ratio']:.2f}" for e, v in b.items()))
EOF
ping "PIM rescore: ALL DONE" "$(head -30 "$LOGS/headline.txt")" white_check_mark
stage "chain complete"
