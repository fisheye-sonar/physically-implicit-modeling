"""Categorical inverse map — PREVIEW on a parent run, into this experiment's own folders (2026-09-20).

    PYTHONPATH=$PWD .pim/bin/python -u experiments/categorical_inverse/scripts/preview.py \
        --run ray_ablation/L-dw-8ray-20m [--target appearance-fac]

Fits the categorical inverse map at the PRODUCTION recipe (the target's forward-probe recipe: the large
probe corpus, its n_seq and epochs, streamed) at every residual point and writes IM on the run's own
categorical bench — through exactly the functions the scorer calls (``arms.inverse_arms`` with
``target=<categorical target>``). NOTHING under runs/ is written: the fitted maps are cached in
``experiments/categorical_inverse/probes/``, the result goes to ``…/scores/preview_<run>_<target>.json``.
Read-only use of the run: its checkpoint, its cached categorical probes (for the landing check) and
its scores.json (for the number being replaced).

Beside the new arm the record keeps: the OLD arm (the continuous full-state map of the block's basis
scored on this same bench — what every categorical block's "IM" was until 2026-09-20), the unedited
floor, PI / ND / GS from scores.json, the map's held-out AND in-sample R² per point (the overfit check
Sevan asked about), and LANDING — the fraction of cases whose cached categorical probe (LIN and MLP)
reads the target labels off the written residual.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.editors.inverse import inverse_overwrite  # noqa: E402
from pim.environments.discworld import arms as dwa, bench as dwb  # noqa: E402
from pim.environments.discworld.bench import full_state_pair  # noqa: E402
from pim.metrics.selection import best_arm  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.inverse import CATEGORICAL_STATE, encode_categorical_state  # noqa: E402

EXP = REPO / "experiments" / "categorical_inverse"
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--target", default="appearance-fac")
ap.add_argument("--n-seq", type=int, default=None, help="override the recipe's n_seq (smoke)")
ap.add_argument("--epochs", type=int, default=None, help="override the recipe's epochs (smoke)")
ap.add_argument("--points", nargs="*", type=int, default=None)
ap.add_argument("--tag", default="", help="suffix for the output file (smoke)")
a = ap.parse_args()

run = REPO / "runs" / a.run
scores = json.loads((run / "scores.json").read_text())
S, inst = scores["settings"], json.loads((run / "config.json").read_text())["data"]["instance"]
blk = scores["bases"][a.target]
basis = blk["basis"]
model, _ = load_checkpoint(run / "best_model.pt", device=dwa.DEV)
model.eval()
recipe = dwa.probe_recipe(a.target, inst, n_seq=S["dw_probe_seqs"])
if a.n_seq:
    recipe["n_seq"] = a.n_seq
if a.epochs:
    recipe["epochs"] = a.epochs
b = dwb.load_bench(model, n=S["dw_bench_n"], target=a.target, basis_name=basis, instance=inst)
u = dwa.unsteered(model, b)
t0 = time.time()
print(f"{a.run} · {a.target} (basis {basis}) · recipe {recipe} · bench {b.n} cases · state {CATEGORICAL_STATE}", flush=True)
arms, st = dwa.inverse_arms(model, {a.target: b}, basis_name=basis, target=a.target, unsteered_cards={a.target: u},
                            cache_dir=EXP / "probes", points=a.points, log=print, **recipe)
recs = [{k: v for k, v in r.items() if np.isscalar(v)} for r in arms[a.target]]
new_guarded = best_arm(recs, "IM", "edit_index")
new_raw = best_arm(recs, "IM", "edit_index", guard=None)

# LANDING: does the run's own cached categorical probe read the target labels off the written residual?
landing = {}
try:
    fp = dwa.probe_recipe(a.target, inst, n_seq=S["dw_probe_seqs"])
    probes = {fam: dwa.fit_probes(model, target=a.target, family=fam, basis_name=basis, cache_dir=run / "probes",
                                  log=None, require_cached=True, **fp) for fam in ("linear", "mlp")}
    s_post = torch.from_numpy(full_state_pair(b.pos, b.vel, b.edit_object, b.sim, "cartesian")[1]).to(dwa.DEV)
    for ell, g, _, gst in dwa.iter_inverse_maps(model, basis_name=basis, target=a.target, cache_dir=EXP / "probes",
                                                points=[new_guarded["point"]], log=None, **recipe):
        h_new = inverse_overwrite(g, encode_categorical_state(b.tgt, s_post[:, -4:], gst["n_classes"]))
        dwa.as_activations(model, ell)
        h0 = model.flat_state(b.state)
        for fam, fits in probes.items():
            landing[fam] = {"written": dwa.readout_landed(h_new, fits[ell][0], b),
                            "unedited": dwa.readout_landed(h0, fits[ell][0], b), "point": int(ell)}
except RuntimeError as e:
    landing = {"error": str(e).splitlines()[0][:160]}

old_arms = [r for r in blk.get("arms", []) if r.get("editor") == "IM"]
out = {"run": a.run, "instance": inst, "target": a.target, "basis": basis, "state": CATEGORICAL_STATE,
       "recipe": {k: (v if not isinstance(v, Path) else str(v)) for k, v in recipe.items()}, "bench_n": int(b.n),
       "unedited": u["edit_index"],
       "new": {"reported (best inside the guard)": new_guarded, "unguarded best": new_raw,
               "g_r2_by_point": st["g_r2"], "g_r2_insample_by_point": st["g_r2_insample"], "by_point": recs},
       "old_continuous_map_on_this_bench": {"reported (best inside the guard)": best_arm(old_arms, "IM", "edit_index"),
                                            "unguarded best": best_arm(old_arms, "IM", "edit_index", guard=None),
                                            "g_r2_by_point": (blk.get("inverse_map") or {}).get("g_r2")},
       "other_editors_from_scores_json": {e: best_arm(blk.get("arms", []), e, "edit_index") for e in ("PI", "ND", "GS")},
       "landing": landing, "minutes": round((time.time() - t0) / 60, 1), "written": time.strftime("%Y-%m-%d %H:%M")}
p = EXP / "scores" / f"preview_{run.name}_{a.target}{a.tag}.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=1, default=float))
f = lambda r: f"{r['edit_index']:+.3f} / {r['fidelity_ratio']:.2f} (pt {r['point']})" if r else "—"   # noqa: E731
print(f"\n{run.name} · {a.target}:  NEW categorical-state IM {f(new_guarded)}   [unguarded {f(new_raw)}]")
print(f"   OLD continuous-map-on-this-bench {f(out['old_continuous_map_on_this_bench']['reported (best inside the guard)'])}"
      f"   unedited {u['edit_index']:+.3f}   PI {f(out['other_editors_from_scores_json']['PI'])}"
      f"   GS {f(out['other_editors_from_scores_json']['GS'])}")
print(f"   g R² held-out max {max(st['g_r2']):+.3f} (in-sample at that point {st['g_r2_insample'][int(np.argmax(st['g_r2']))]:+.3f})"
      f"   landing {landing}   [{out['minutes']} min]  -> {p.relative_to(REPO)}")
