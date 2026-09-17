"""Probe-seed replicates on an OTHELLO run (2026-09-14): the LINEAR mine/theirs probe grid
refitted with N seeds (init + held-out split), PI and ND swept at each seed's own best point
and at the seed-0 best point, written to ``runs/<run>/variance.json`` under
``probe_seeds["mine"]`` — the Othello sibling of ``probe_seeds.py``.

    python experiments/seed_variance/scripts/probe_seeds_othello.py --run initial_othello_comparison/L-oth-20m --seeds 10

Fits are linear only (``families=("linear",)``), so seed 0 is a fresh linear-only grid, not
the canonical two-family cache entry; its numbers reproduce the canonical linear probes.

**IM (2026-09-16, Sevan).** The inverse map is reseeded the same way and lands in
``probe_seeds["inverse_map"]``: g refitted under each seed (the seed drives g's init AND the
80/20 split by game, so the retrieval bank is reseeded with it), then IM and IM-NN written at
the run's canonical IM point (``--im-points all`` fits every point instead, ~9x the cost).
Seed 0 is a cache HIT on the canonical g, so its row reproduces the scored number exactly.
``--im-seeds 0`` skips the IM section.
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
from pim.environments.othello.data import canonical_vocab, tokens_and_labels  # noqa: E402
from pim.metrics.decodability import probe_skill_from_stats  # noqa: E402
from pim.metrics.set_editability import move_fidelity_ratio  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402

EI = "edit_index_symdiff"                # the Othello headline construction (tables, 2026-09-12)
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--im-seeds", type=int, default=None, help="seeds for the inverse map (default: --seeds; 0 skips)")
ap.add_argument("--im-points", choices=("canonical", "all"), default="canonical",
                help="'canonical' = the run's scored IM point only; 'all' = every residual point")
ap.add_argument("--out", default="variance.json", help="file under the run dir (smokes write elsewhere)")
a = ap.parse_args()
a.im_seeds = a.seeds if a.im_seeds is None else a.im_seeds
run = REPO / "runs" / a.run
S = json.loads((run / "scores.json").read_text())["settings"]
inst = json.loads((run / "config.json").read_text())["data"]["instance"]
rules = oc.rules_of(inst)
model, _ = load_checkpoint(run / "best_model.pt", device=oa.DEV); model.eval(); NP = n_points(model)
tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",), instance=inst)["probe"])
itos = {v: k for k, v in canonical_vocab().items()}
n_games = S["oth_probe_games"]
data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(tok[:n_games], ln[:n_games])], **rules)
bench = load_benchmark(inst)


def _canon_point(editor: str) -> int:
    """The run's best-EDIT point for one editor under EI — the construction this script reports.
    ``scores.json["best"]`` is keyed on the UNION Edit Index (the scorer's key), so on a run whose
    two constructions disagree it names a different point than the tables quote (2026-09-16)."""
    arms = json.loads((run / "scores.json").read_text())["arms"]
    sub = [r for r in arms if r["editor"] == editor or r["editor"].startswith(editor + "[")
           or r["editor"].startswith(editor + "@")]
    return int(max(sub, key=lambda r: r[EI])["point"])


canon_point = {name: {"point": _canon_point(name)} for name in ("PI", "ND")}
cur, tgt = case_targets(bench)
uns = oa.unsteered_probs(model, bench)
u = oa.unsteered(model, bench)
out_path = run / a.out
var = json.loads(out_path.read_text()) if out_path.exists() else {}
var.setdefault("probe_seeds", {})
seeds, t0 = {}, time.time()
for seed in range(a.seeds):   # --seeds 0 runs the IM section alone and leaves probe_seeds["mine"] as it is
    grid = oa.fit_probe_grid(model, data, families=("linear",), seed=seed, cache_dir=run / "probes", log=None)
    lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(NP)}
    skill = [probe_skill_from_stats(next(st for st in grid.stats if st["point"] == p and st["family"] == "linear"))
             for p in range(NP)]
    best = int(np.argmax(skill))
    if seed == 0:
        best0 = best
    rec = {"skill_by_point": skill, "best_point": best, "editors": {}}
    # three evaluation points: the seed's own best-DECODABILITY point, the seed-0 one, and the
    # run's canonical best-EDIT point per editor (scores.json) — on Othello these differ (the
    # deepest points read best and edit worst), so the spread is reported where editing happens too
    canon = canon_point  # by EI (symdiff), not scores.json["best"] (union) — see _canon_point
    for pt_label, pt in (("own_best", best), ("seed0_best", best0), ("canonical_edit_best", None)):
        ed = {}
        for mode, name, alphas in (("pinv", "PI", S["oth_alpha_pi"]), ("add_sub", "ND", S["oth_alpha_nd"])):
            if pt is None:
                pt = int(canon[name]["point"])
            arms = []
            for al in alphas:
                pr, card = oa.linear_arm(model, bench, lin, tgt, cur, mode=mode, alpha=al, points={pt})
                arms.append({"alpha": al, "edit_index": card[EI], "fidelity_ratio": move_fidelity_ratio(pr, uns, bench.legal_post)})
            bb = max(arms, key=lambda r: r["edit_index"])
            ed[name] = {**bb, "point": pt}
        rec["editors"][pt_label] = ed
    seeds[str(seed)] = rec
    e = rec["editors"]["own_best"]
    print(f"  mine seed {seed}: best pt {best} skill {skill[best]:.4f} | "
          + " ".join(f"{k} {v['edit_index']:+.3f}/{v['fidelity_ratio']:.2f}" for k, v in e.items())
          + f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
# ── IM / IM-NN under the same seeds (2026-09-16) ─────────────────────────────────────────
if a.im_seeds:
    pts = None if a.im_points == "all" else [_canon_point("IM")]
    im_seeds, t1 = {}, time.time()
    for seed in range(a.im_seeds):
        recs, st = oa.inverse_arms(model, bench, data, rules=rules, cache_dir=run / "probes",
                                   n_games=n_games, seed=seed, points=pts, uns_probs=uns, log=None)
        rec = {"g_r2": st["g_r2"], "nn_r2": st.get("nn_r2"), "editors": {}}
        for name in ("IM", "IM-NN"):
            sub = [r for r in recs if r["editor"] == name]
            b = max(sub, key=lambda r: r[EI])
            rec["editors"][name] = {"edit_index": b[EI], "fidelity_ratio": b.get("fidelity_ratio"),
                                    "point": b["point"], "g_r2": b["g_r2"]}
        im_seeds[str(seed)] = rec
        print(f"  inverse map seed {seed}: g R² {max(st['g_r2']):+.3f} | "
              + " ".join(f"{k} {v['edit_index']:+.3f}/{(v['fidelity_ratio'] or float('nan')):.2f} (pt {v['point']})"
                         for k, v in rec["editors"].items())
              + f"  [{(time.time() - t1) / 60:.1f} min]", flush=True)
    im_summ = {"n_seeds": a.im_seeds, "points": a.im_points, "n_games": n_games, "edit_index": EI,
               "canonical_point": _canon_point("IM"), "canonical_point_by": EI, "editors": {}}
    for name in ("IM", "IM-NN"):
        vals = [x["editors"][name] for x in im_seeds.values()]
        ei = np.array([v["edit_index"] for v in vals], float)
        fd = np.array([np.nan if v["fidelity_ratio"] is None else v["fidelity_ratio"] for v in vals], float)
        gr = np.array([v["g_r2"] for v in vals], float)
        sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else None   # noqa: E731
        im_summ["editors"][name] = {"edit_index": {"mean": float(ei.mean()), "sd": sd(ei)},
                                    "fidelity_ratio": {"mean": float(np.nanmean(fd)), "sd": sd(fd)},
                                    "g_r2_at_arm": {"mean": float(gr.mean()), "sd": sd(gr)},
                                    "points_used": sorted({v["point"] for v in vals})}
    nn = [x["nn_r2"] for x in im_seeds.values() if x["nn_r2"]]
    if nn:
        im_summ["nn_r2_max"] = {"mean": float(np.mean([max(x) for x in nn])),
                                "sd": float(np.std([max(x) for x in nn], ddof=1)) if len(nn) > 1 else None}
    var["probe_seeds"]["inverse_map"] = {"summary": im_summ, "seeds": im_seeds}
    print("inverse map: " + " ".join(
        f"{k}: EI {v['edit_index']['mean']:+.3f}±{v['edit_index']['sd'] or 0:.3f}"
        for k, v in im_summ["editors"].items()) + f"  [{(time.time() - t1) / 60:.1f} min]")

if not a.seeds:
    var["written"] = time.strftime("%Y-%m-%d %H:%M")
    out_path.write_text(json.dumps(var, indent=1, default=float))
    print("wrote", out_path.relative_to(REPO), "(inverse map only; linear record untouched)")
    raise SystemExit

sk_best = np.array([s["skill_by_point"][s["best_point"]] for s in seeds.values()])
sk_pts = np.array([s["skill_by_point"] for s in seeds.values()])
summ = {"n_seeds": a.seeds, "families": ["linear"], "n_games": n_games, "canonical_point_by": EI,
        "best_skill": {"mean": float(sk_best.mean()), "sd": float(sk_best.std(ddof=1)) if a.seeds > 1 else None},
        "best_point": {"values": [s["best_point"] for s in seeds.values()], "n_distinct": len({s["best_point"] for s in seeds.values()})},
        "skill_sd_across_points_per_seed": {"mean": float(sk_pts.std(axis=1, ddof=1).mean()), "min": float(sk_pts.std(axis=1, ddof=1).min()), "max": float(sk_pts.std(axis=1, ddof=1).max())},
        "skill_sd_across_seeds_per_point": [float(x) for x in sk_pts.std(axis=0, ddof=1)] if a.seeds > 1 else None,
        "unedited": {k: v for k, v in u.items() if isinstance(v, (int, float))}, "edit_index": EI, "editors": {}}
for name in ("PI", "ND"):
    for pt_label in ("own_best", "seed0_best", "canonical_edit_best"):
        vals = [s["editors"][pt_label][name] for s in seeds.values()]
        ei = np.array([v["edit_index"] for v in vals]); fd = np.array([v["fidelity_ratio"] for v in vals])
        summ["editors"][f"{name}@{pt_label}"] = {"edit_index": {"mean": float(ei.mean()), "sd": float(ei.std(ddof=1)) if len(ei) > 1 else None},
                                                 "fidelity_ratio": {"mean": float(fd.mean()), "sd": float(fd.std(ddof=1)) if len(fd) > 1 else None},
                                                 "points_used": sorted({v["point"] for v in vals})}
var["probe_seeds"]["mine"] = {"summary": summ, "seeds": seeds}
var["written"] = time.strftime("%Y-%m-%d %H:%M")
out_path.write_text(json.dumps(var, indent=1, default=float))
print(f"mine: best skill {summ['best_skill']['mean']:.4f} ± {summ['best_skill']['sd'] or 0:.4f} over {a.seeds} seeds; "
      f"best point {summ['best_point']['n_distinct']} distinct; "
      + " ".join(f"{k}: EI {v['edit_index']['mean']:+.3f}±{v['edit_index']['sd'] or 0:.3f}" for k, v in summ["editors"].items()))
print("wrote", out_path.relative_to(REPO), f"[{(time.time() - t0) / 60:.1f} min]")
