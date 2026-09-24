#!/usr/bin/env bash
# ── score_cartesian.sh — add a CARTESIAN regression block to every discworld run (2026-09-14, Sevan) ──
#   master_eval with PIM_DW_BASES=frustum,cartesian: the scorer ADDS the cartesian block (probes, PI/ND/GS,
#   floors) to every discworld run that lacks one and skips everything current; no table rebuild — the
#   tables hide cartesian beside frustum, the block is a record in scores.json for the paper's back pocket.
#   systemd-run --user --unit=score_cartesian -p MemoryMax=45G --collect --working-directory=$PWD \
#     /usr/bin/bash -c 'bash scripts/drivers/score_cartesian.sh > logs/score_cartesian/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD; export PYTHONPATH=$ROOT; export PIM_DW_BASES=frustum,cartesian
PY=$ROOT/.pim/bin/python; LOGS=$ROOT/logs/score_cartesian; NT=https://ntfy.sh/swirling-tornado-ai691k
mkdir -p "$LOGS"
ping()  { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail()  { ping "PIM score_cartesian FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }
count() { "$PY" - <<'PY'
import json, glob
n=t=0
for p in glob.glob("runs/*/*/scores.json"):
    if p.split("/")[1].startswith("_") or p.split("/")[1]=="archive": continue
    try: d=json.load(open(p))
    except Exception: continue
    if not isinstance(d.get("bases"), dict) or "frustum" not in d["bases"] and "cartesian" not in d["bases"]: continue
    t+=1; n+= "cartesian" in d["bases"]
print(f"{n}/{t}")
PY
}
ping "PIM score_cartesian: started" "cartesian block for every discworld run ($(count) already have one); ETA ~10 h"
stage "master_eval (PIM_DW_BASES=$PIM_DW_BASES)"
"$PY" "$ROOT/.pim/bin/jupyter-nbconvert" --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
    notebooks/master_eval.ipynb > "$LOGS/master_eval.log" 2>&1 || fail "master_eval" "$(tail -25 "$LOGS/master_eval.log")"
stage "verify"
"$PY" - > "$LOGS/verify.txt" 2>&1 <<'PY'
import json, glob
rows=[]
for p in sorted(glob.glob("runs/*/*/scores.json")):
    t=p.split("/")[1]
    if t.startswith("_") or t=="archive": continue
    d=json.load(open(p)); b=d.get("bases")
    if not isinstance(b, dict) or not ({"frustum","cartesian"} & set(b)): continue
    c=b.get("cartesian"); f=b.get("frustum")
    def s(x): return "—" if x is None else f"PI {x['best']['PI']['edit_index']:+.2f}/{x['best']['PI']['fidelity_ratio']:.2f} GS {x['best']['GS']['edit_index']:+.2f}/{x['best']['GS']['fidelity_ratio']:.2f}"
    rows.append(f"{p.split('/')[1]}/{p.split('/')[2]:36s} frustum: {s(f):34s} cartesian: {s(c)}")
print("\n".join(rows)); print(f"\ncartesian blocks: {sum('cartesian' in r.split('cartesian: ')[1] and '—' not in r.split('cartesian: ')[1] for r in rows)}/{len(rows)}")
PY
cat "$LOGS/verify.txt" >> "$LOGS/driver.log"
ping "PIM score_cartesian: DONE" "$(count) discworld runs now carry a cartesian block. $(tail -1 "$LOGS/verify.txt")" white_check_mark
stage "chain complete"
