"""Probe-corpus-size control, Othello (2026-09-19, Sevan): do decodability and EDITABILITY move when
the mine/theirs LINEAR probes and the inverse map are fitted on more games than the canonical 20k?
The canonical pipeline with the number of games as the only knob.

    python -u experiments/probe_corpus_size/scripts/corpus_size_oth.py \
        --run initial_othello_comparison/L-oth-20m --sizes 20000 60000 100000

For each size n (the first n games of the instance's ``probe_large`` split, 170k games — one
corpus for every size; the run's canonical numbers, fitted on the 20k-game ``probe`` split, are
the reference): fit the LINEAR probe grid (``arms.fit_probe_grid``, 80/20 by game, seed 0; cached
in this experiment's ``probes/``), sweep PI and ND at EVERY residual point over the run's own alpha
grids (the symmetric-difference Edit Index, the guard attached), and — for n ≤ ``--im-max-n`` — fit
the inverse map and write IM / IM-NN (``arms.inverse_arms``). Written after every size to
``scores/oth_<run>.json``.

Not here, on purpose: GS. Its MLP-128 grid costs ~90 min at 20k games, so ~7 h at 100k; the probe-
capacity sweep (``findings/probe-capacity.md``) already fitted MLP-128 on all 170k games (skill 0.976
vs 0.977 canonical). Sizes stop at 100k: harvesting one residual point for 170k games holds ~41 GB.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments.othello import arms as oa, corpus as oc  # noqa: E402
from pim.environments.othello import case_targets, load_benchmark  # noqa: E402
from pim.metrics.decodability import probe_skill_from_stats  # noqa: E402
from pim.metrics.set_editability import move_fidelity_ci95, move_fidelity_ratio  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402

EI = "edit_index_symdiff"
EXP = REPO / "experiments" / "probe_corpus_size"
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--sizes", nargs="+", type=int, default=(20_000, 60_000, 100_000))
ap.add_argument("--im-max-n", type=int, default=60_000, help="largest n the inverse map is fitted at (GPU: bank + fit tensors)")
ap.add_argument("--smoke", action="store_true", help="the canonical 20k probe split (labels cached), 2 alphas, 2 points, scratch output")
a = ap.parse_args()

run = REPO / "runs" / a.run
scores = json.loads((run / "scores.json").read_text())
S = scores["settings"]
inst = json.loads((run / "config.json").read_text())["data"]["instance"]
rules = oc.rules_of(inst)
model, _ = load_checkpoint(run / "best_model.pt", device=oa.DEV)
model.eval()
NP = n_points(model)
split = "probe" if a.smoke else "probe_large"
path = oc.build(oc.LADDER["D"], log=lambda s: None, only=(split,), instance=inst)[split]
cache_dir = EXP / "probes" if not a.smoke else REPO / ".scratch" / "probe_corpus_size_smoke"
out_path = (EXP / "scores" / f"oth_{run.name}.json") if not a.smoke else REPO / ".scratch" / "probe_corpus_size_smoke_oth.json"
sizes = [20_000] if a.smoke else list(a.sizes)
a_pi, a_nd = S["oth_alpha_pi"], S["oth_alpha_nd"]
points = range(NP)
if a.smoke:
    a_pi, a_nd, points = a_pi[4:6], a_nd[4:6], range(3, 5)

bench = load_benchmark(inst)
cur, tgt = case_targets(bench)
uns = oa.unsteered_probs(model, bench)
u = oa.unsteered(model, bench)


def canon_best(ed):
    sub = [r for r in scores["arms"] if r["editor"] == ed]
    if not sub:
        return None
    b = max(sub, key=lambda r: r[EI])
    return {"edit_index": b[EI], "fidelity_ratio": b.get("fidelity_ratio"), "point": b["point"], "alpha": b["alpha"]}


res = json.loads(out_path.read_text()) if out_path.exists() else {}
res.update({"run": a.run, "instance": inst, "corpus": split, "edit_index": EI, "unedited": u[EI],
            "canonical": {"n_games": S["oth_probe_games"], "corpus": "probe",
                          "skill_LIN": max(scores["probe_skill"]["mine|linear|sequence"]),
                          "best": {e: canon_best(e) for e in ("PI", "ND", "IM", "IM-NN")},
                          "g_r2_max": max(scores["inverse_map"]["g_r2"]) if scores.get("inverse_map") else None}})
res.setdefault("sizes", {})
t0 = time.time()
for n in sizes:
    data = oc.probe_data(path, n=int(n), **rules)
    rec = res["sizes"].get(str(n), {})
    rec["rows"] = int(data.mask.sum())
    grid = oa.fit_probe_grid(model, data, families=("linear",), seed=0, cache_dir=cache_dir, log=None)
    lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(NP)}
    rec["skill_LIN_by_point"] = [probe_skill_from_stats(next(st for st in grid.stats if st["point"] == p and st["family"] == "linear"))
                                 for p in range(NP)]
    rec["skill_LIN"] = max(rec["skill_LIN_by_point"])
    print(f"n={n:,} games ({rec['rows']:,} rows): skill LIN {rec['skill_LIN']:.4f}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    rec["best"] = {}
    for mode, name, alphas in (("pinv", "PI", a_pi), ("add_sub", "ND", a_nd)):
        arms = []
        for pt in points:
            for al in alphas:
                pr, card = oa.linear_arm(model, bench, lin, tgt, cur, mode=mode, alpha=al, points={pt})
                arms.append({"point": int(pt), "alpha": al, "edit_index": card[EI],
                             "edit_index_case_se": card.get(f"{EI}_case_se"),
                             "fidelity_ratio": move_fidelity_ratio(pr, uns, bench.legal_post),
                             "_probs": pr})
        bb = max(arms, key=lambda r: r["edit_index"])
        bb.update(move_fidelity_ci95(bb.pop("_probs"), uns, bench.legal_post))
        rec["best"][name] = {k: v for k, v in bb.items() if not k.startswith("_")}
    print("   " + " · ".join(f"{k} {v['edit_index']:+.3f}/{v['fidelity_ratio']:.2f} (pt {v['point']}, α {v['alpha']:g})"
                             for k, v in rec["best"].items()) + f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    res["sizes"][str(n)] = rec
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, default=float))
    del grid, lin
    if n <= a.im_max_n:
        recs, st = oa.inverse_arms(model, bench, data, rules=rules, cache_dir=cache_dir, n_games=int(n), seed=0,
                                   points=(list(points) if a.smoke else None), uns_probs=uns, log=None)
        for name in ("IM", "IM-NN"):
            sub = [r for r in recs if r["editor"] == name]
            bb = max(sub, key=lambda r: r[EI])
            rec["best"][name] = {"edit_index": bb[EI], "fidelity_ratio": bb.get("fidelity_ratio"), "point": bb["point"],
                                 "g_r2": bb["g_r2"], "fidelity_ci95_lo": bb.get("fidelity_ci95_lo"), "fidelity_ci95_hi": bb.get("fidelity_ci95_hi")}
        rec["g_r2_by_point"], rec["nn_r2_by_point"] = st["g_r2"], st.get("nn_r2")
        print("   " + " · ".join(f"{k} {rec['best'][k]['edit_index']:+.3f}/{rec['best'][k]['fidelity_ratio']:.2f} (pt {rec['best'][k]['point']})"
                                 for k in ("IM", "IM-NN")) + f" · g R² max {max(st['g_r2']):+.3f}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    else:
        rec["im_skipped"] = f"n > --im-max-n {a.im_max_n}"
    res["sizes"][str(n)] = rec
    res["written"] = time.strftime("%Y-%m-%d %H:%M")
    out_path.write_text(json.dumps(res, indent=1, default=float))
    del data
print("wrote", out_path.relative_to(REPO), f"[{(time.time() - t0) / 60:.1f} min]")
