#!/usr/bin/env python3
"""ledger.py — where the numbers are landing, in three human-readable forms.

Reads the shortlist runs' scores.json and their seed replicates THROUGH ``pim.figures.tables``
(``collect`` → ``pool_replicates``), so every number here is the table's number: the canonical
value, and per (run, block, metric) the replicate set's n, mean, SD (the readout), t-based 95%
half-width (secondary), the members' values in seed order, the pooled budget and the budgets
left out. Writes ``dashboard/ledger.json`` (the dashboard reads it), ``dashboard/ledger.csv``
(long format, one row per run × block × metric) and ``dashboard/ledger.md`` (one table per
run). Run by the dispatcher whenever a job finishes; safe to run by hand any time.

    .pim/bin/python experiments/paper_ci/scripts/ledger.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
Q = ROOT / "experiments" / "paper_ci"
DASH = Q / "dashboard"
CFG = json.loads((Q / "config.json").read_text())["ledger"]

from pim.figures import tables as T  # noqa: E402

METRICS = ["skill_LIN", "skill_MLP", "unedited"] + [f"{e} EI" for e in T.EDITORS_ALL] + [f"{e} fid" for e in T.EDITORS_ALL]
GROUP_OF = {"L-oth-20m": "oth-standard", "L-oth-adjacent-flip-20m": "oth-adjflip", "L-oth-adjacent-20m": "oth-adjacent",
            "L-oth-noflip-20m": "oth-noflip", "L-dw-noiseless-20m": "dw-noiseless", "L-dw-blink-20m": "dw-blink",
            "L-dw-128ray-20m": "dw-128ray", "L-dw-16ray-20m": "dw-16ray", "L-dw-8ray-20m": "dw-8ray", "L-dw-5ray-20m": "dw-5ray"}


def f(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)


def main():
    T.set_basis(CFG["basis"])
    F = T.collect(CFG["runs_oth"], CFG["runs_dw"], label="paper_ci")
    groups, rows = {}, []
    for run in CFG["runs_oth"] + CFG["runs_dw"]:
        env = "othello" if run.startswith("L-oth") else "discworld"
        want = CFG["oth_blocks"] if env == "othello" else CFG["dw_blocks"]
        sub = F.df[F.df["run"] == run]
        g = groups.setdefault(GROUP_OF.get(run, run), {"run": run, "env": env, "blocks": {}})
        for blk in want:
            r = sub[sub["basis"] == blk]
            if r.empty:
                continue
            r = r.iloc[0].to_dict()
            v = F.rep_sd.get((run, blk), {})
            out = {}
            for m in METRICS:
                if m not in r:
                    continue
                d = {"canonical": f(r.get(m)), "n": int(v.get("n", 1)) if m in v else (1 if v else 1),
                     "mean": f(v.get(f"{m}_mean")), "sd": f(v.get(m)), "ci95": f(v.get(f"{m}_ci95")),
                     "values": [f(x) for x in v.get(f"{m}_values", [])], "steps": v.get("steps", []),
                     "seeds": v.get("seeds", []), "dropped_steps": v.get("dropped_steps", [])}
                if m.endswith(" EI") or m.endswith(" fid"):
                    d["arm"] = r.get(m.replace(" EI", " arm").replace(" fid", " arm"))
                out[m] = d
                rows.append({"run": run, "group": GROUP_OF.get(run, run), "block": blk, "metric": m, **{k: d[k] for k in
                             ("canonical", "n", "mean", "sd", "ci95")}, "values": " ".join(f"{x:+.4f}" for x in d["values"] if x is not None),
                             "steps": "/".join(str(s) for s in d["steps"]), "seeds": ",".join(str(s) for s in d["seeds"])})
            g["blocks"][blk] = out
    ledger = {"generated": time.time(), "generated_iso": time.strftime("%FT%T"), "basis": CFG["basis"],
              "groups": groups, "missing": F.missing}
    DASH.mkdir(parents=True, exist_ok=True)
    (DASH / "ledger.json").write_text(json.dumps(ledger, indent=1))
    with open(DASH / "ledger.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "group", "block", "metric", "canonical", "n", "mean", "sd", "ci95", "values", "steps", "seeds"])
        w.writeheader()
        w.writerows(rows)
    md = [f"# Ledger — {ledger['generated_iso']} · basis {CFG['basis']} · ± = SD over training seeds (n), [CI] = t-based 95% half-width\n",
          "Editor cells are the tables' REPORTED arm (pim.metrics.selection, 2026-09-19): the best Edit Index among arms inside the fidelity guard; the unguarded best only when no arm passes. `fid` = FIDELITY = 1 − the stored RMSE ratio (2026-09-22): 1 perfect, 0 = the guard, < 0 degraded.\n"]
    for gname, g in groups.items():
        md.append(f"\n## {gname} — `{g['run']}`\n")
        for blk, metrics in g["blocks"].items():
            md.append(f"\n**{blk}**\n\n| metric | canonical | n | mean ± SD | 95% ±h | members | budget |\n|---|---|---|---|---|---|---|")
            for m, d in metrics.items():
                if d["canonical"] is None:
                    continue
                ms = f"{d['mean']:+.3f} ± {d['sd']:.3f}" if d["n"] >= 2 and d["sd"] is not None else "—"
                ci = f"{d['ci95']:.3f}" if d.get("ci95") is not None else "—"
                vals = " ".join(f"{x:+.3f}" for x in d["values"] if x is not None) or "—"
                md.append(f"| {m} | {d['canonical']:+.3f} | {d['n']} | {ms} | {ci} | {vals} | {'/'.join(str(s // 1000) + 'k' for s in d['steps']) or '—'} |")
    (DASH / "ledger.md").write_text("\n".join(md) + "\n")
    n_rep = sum(1 for g in groups.values() for b in g["blocks"].values() for d in b.values() if d["n"] >= 2)
    print(f"ledger: {len(groups)} runs, {len(rows)} rows, {n_rep} with a replicate spread")


if __name__ == "__main__":
    main()
