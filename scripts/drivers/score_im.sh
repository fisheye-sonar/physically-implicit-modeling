#!/usr/bin/env bash
# ── score_im.sh — add the IM / IM-NN editors to every scored run, both discworld bases (2026-09-15, Sevan) ──
#   master_eval with PIM_DW_BASES=frustum,cartesian: for every run at the current eval version the
#   scorer fits (or loads) the inverse map per point per basis, writes IM (overwrite) and IM-NN on
#   every block's bench, APPENDS the arms + best['IM'] / best['IM-NN'] to scores.json (dated backup
#   under runs/<run>/scores_backup/, atomic replace) and skips runs that already carry IM.
#   systemd-run --user --unit=score_im -p MemoryMax=45G --collect --working-directory=$PWD \
#     /usr/bin/bash -c 'bash scripts/drivers/score_im.sh > logs/score_im/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD; export PYTHONPATH=$ROOT; export PIM_DW_BASES=frustum,cartesian
PY=$ROOT/.pim/bin/python; LOGS=$ROOT/logs/score_im; NT=https://ntfy.sh/swirling-tornado-ai691k
mkdir -p "$LOGS"
ping()  { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail()  { ping "PIM score_im FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }
count() { "$PY" - <<'PY'
import json, glob
n=t=0
for p in glob.glob("runs/*/*/scores.json"):
    top=p.split("/")[1]
    if top.startswith("_") or top=="archive": continue
    try: d=json.load(open(p))
    except Exception: continue
    blocks=list(d.get("bases",{}).values()) + ([d] if "arms" in d else [])
    if not blocks: continue
    t+=1; n+= all(any(a.get("editor")=="IM" for a in b.get("arms",[])) for b in blocks)
print(f"{n}/{t}")
PY
}
ping "PIM score_im: started" "IM / IM-NN on every run, both discworld bases ($(count) runs already complete)"
stage "master_eval (IM fold-in, PIM_DW_BASES=$PIM_DW_BASES)"
"$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
    notebooks/master_eval.ipynb > "$LOGS/master_eval.log" 2>&1 || fail "master_eval" "$(tail -25 "$LOGS/master_eval.log")"
stage "verify"
"$PY" scripts/drivers/verify_im.py > "$LOGS/verify.txt" 2>&1
cat "$LOGS/verify.txt" >> "$LOGS/driver.log"
ping "PIM score_im: DONE" "$(count) runs carry IM on every block. $(tail -3 "$LOGS/verify.txt")" white_check_mark
stage "chain complete"
