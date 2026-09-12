"""Probe-seed replicates: how much do decodability and the linear editors move when only the
PROBE's seed (init + held-out split) changes? (2026-09-11, Sevan's variance pilot.)

    python experiments/seed_variance/scripts/probe_seeds.py --run noise_ablation/L-dw-noiseless-20m \
        --targets full appearance-fac --seeds 20 6

For each target and each seed 0..N−1 (seed 0 = the canonical cached probes): fit the LINEAR
probe at EVERY residual point with the target's canonical recipe (cached in the run's probes/,
keyed by seed), then run the linear editors' full α sweep — PI and, on a categorical target,
ND — at the seed's OWN best point and at the seed-0 best point. Written to
``runs/<run>/variance.json`` under ``probe_seeds[<target>]``: per seed the skill by point, the
best point, each editor's best arm at both points; and summaries — SD across seeds of the best
skill, of the editors' best Edit Index / guard, and of the spread across residual points (SD
of skill over points, per seed). Nothing canonical changes; the seed-0 numbers reproduce the
run's scores.json.
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
from pim.environments.discworld import arms as dwa, bench as dwb  # noqa: E402
from pim.environments.discworld.grid_target import categorical_target  # noqa: E402
from pim.metrics.decodability import probe_skill_from_stats  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--targets", nargs="+", default=("full", "appearance-fac"))
ap.add_argument("--seeds", nargs="+", type=int, default=(20, 6), help="seeds per target (same order)")
ap.add_argument("--smoke", action="store_true", help="tiny n_seq into a scratch cache (never the run's)")
a = ap.parse_args()
run = REPO / "runs" / a.run
inst = json.loads((run / "config.json").read_text())["data"]["instance"]
S = json.loads((run / "scores.json").read_text())["settings"]
model, _ = load_checkpoint(run / "best_model.pt", device=dwa.DEV); model.eval(); NP = n_points(model)
cache_dir = run / "probes" if not a.smoke else REPO / ".scratch" / "seed_variance_smoke"
out_path = run / "variance.json" if not a.smoke else REPO / ".scratch" / "variance_smoke.json"
var = json.loads(out_path.read_text()) if out_path.exists() else {}
var.setdefault("probe_seeds", {})
t0 = time.time()
for target, n_seeds in zip(a.targets, a.seeds):
    cat = categorical_target(target)
    recipe = dwa.probe_recipe(target, inst, n_seq=S["dw_probe_seqs"])
    if a.smoke:
        recipe["n_seq"] = 2000; recipe["epochs"] = 2 if cat else None
    a_nd, a_pi = (S["dw_grid_alpha_nd"], S["dw_grid_alpha_pi"]) if cat else (S["dw_alpha_nd"], S["dw_alpha_pi"])
    dimsets = ("all",) if cat else tuple(S["dw_edit_dims"])
    b = dwb.load_bench(model, n=S["dw_bench_n"], target=target, basis_name="frustum", instance=inst)
    u = dwa.unsteered(model, b)
    seeds = {}
    for seed in range(n_seeds):
        fits = dwa.fit_probes(model, target=target, family="linear", basis_name="frustum",
                              cache_dir=cache_dir, log=None, seed=seed, **recipe)
        skill = [probe_skill_from_stats(fits[e][1]) for e in sorted(fits)]
        best = int(np.argmax(skill))
        if seed == 0:
            best0 = best
        rec = {"skill_by_point": skill, "best_point": best, "editors": {}}
        for pt_label, pt in (("own_best", best), ("seed0_best", best0)):
            arms = []
            for dims in dimsets:
                arms += dwa.pinv_arm(model, b, {pt: fits[pt]}, a_pi, space="zspace", dims=dims)
                if cat:
                    arms += dwa.nanda_arm(model, b, fits[pt][0], pt, a_nd, dims=dims)
            for r in arms:
                r["fidelity_ratio"] = dwa.fidelity_ratio(r, u)
            ed = {}
            for name in ("PI", "ND"):
                sub = [r for r in arms if r["editor"].startswith(name)]
                if sub:
                    bb = max(sub, key=lambda r: r["edit_index"])
                    ed[name] = {"edit_index": bb["edit_index"], "fidelity_ratio": bb["fidelity_ratio"],
                                "alpha": bb["alpha"], "dims": bb.get("dims"), "point": pt}
            rec["editors"][pt_label] = ed
        seeds[str(seed)] = rec
        e = rec["editors"]["own_best"]
        print(f"  {target} seed {seed}: best pt {best} skill {skill[best]:.4f} | "
              + " ".join(f"{k} {v['edit_index']:+.3f}/{v['fidelity_ratio']:.2f}" for k, v in e.items())
              + f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    sk_best = np.array([s["skill_by_point"][s["best_point"]] for s in seeds.values()])
    sk_pts = np.array([s["skill_by_point"] for s in seeds.values()])           # (seeds, points)
    summ = {"n_seeds": n_seeds, "recipe": {k: (v if not isinstance(v, Path) else str(v)) for k, v in recipe.items()},
            "best_skill": {"mean": float(sk_best.mean()), "sd": float(sk_best.std(ddof=1)) if n_seeds > 1 else None},
            "best_point": {"values": [s["best_point"] for s in seeds.values()],
                           "n_distinct": len({s["best_point"] for s in seeds.values()})},
            "skill_sd_across_points_per_seed": {"mean": float(sk_pts.std(axis=1, ddof=1).mean()),
                                                "min": float(sk_pts.std(axis=1, ddof=1).min()),
                                                "max": float(sk_pts.std(axis=1, ddof=1).max())},
            "skill_sd_across_seeds_per_point": [float(x) for x in sk_pts.std(axis=0, ddof=1)] if n_seeds > 1 else None,
            "editors": {}}
    for name in ("PI", "ND"):
        for pt_label in ("own_best", "seed0_best"):
            vals = [s["editors"][pt_label].get(name) for s in seeds.values()]
            vals = [v for v in vals if v]
            if not vals:
                continue
            ei = np.array([v["edit_index"] for v in vals]); fd = np.array([v["fidelity_ratio"] for v in vals])
            summ["editors"][f"{name}@{pt_label}"] = {
                "edit_index": {"mean": float(ei.mean()), "sd": float(ei.std(ddof=1)) if len(ei) > 1 else None},
                "fidelity_ratio": {"mean": float(fd.mean()), "sd": float(fd.std(ddof=1)) if len(fd) > 1 else None},
                "points_used": sorted({v["point"] for v in vals})}
    var["probe_seeds"][target] = {"summary": summ, "seeds": seeds}
    print(f"{target}: best skill {summ['best_skill']['mean']:.4f} ± {summ['best_skill']['sd'] or 0:.4f} over {n_seeds} seeds; "
          f"best point {summ['best_point']['n_distinct']} distinct; "
          + " ".join(f"{k}: EI {v['edit_index']['mean']:+.3f}±{v['edit_index']['sd'] or 0:.3f}" for k, v in summ["editors"].items()), flush=True)
var["written"] = time.strftime("%Y-%m-%d %H:%M")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(var, indent=1, default=float))
print("wrote", out_path.relative_to(REPO), f"[{(time.time() - t0) / 60:.1f} min]")
