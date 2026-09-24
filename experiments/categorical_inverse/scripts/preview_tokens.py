"""Categorical inverse map — PREVIEW on the frames-as-tokens model (2026-09-20), the token twin of preview.py.

    PYTHONPATH=$PWD .pim/bin/python -u experiments/categorical_inverse/scripts/preview_tokens.py \
        --run interface_ablation/L-dw-8ray-tok-20m [--n-seq 3000 --epochs 3 --points 4 --tag _smoke]

Same contract as preview.py: the production recipe unless overridden, the scorer's own functions
(``token_bench.inverse_arms`` with ``target=<categorical target>``, set up exactly as
``pim.scoring.discworld.score_discworld_tokens`` sets it up), NOTHING under runs/ written (maps cached in
this experiment's probes/, result in scores/preview_<run>_<target>_tokens<tag>.json). The index here is the
token model's frame-set Edit Index — it shares an axis with the frame models' ray-zone index, not a formula.
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
from pim.environments.discworld import arms as dwa, bench as dwb, token_bench as tkb  # noqa: E402
from pim.environments.discworld.tokens import FrameVocab  # noqa: E402
from pim.metrics.selection import best_arm  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.inverse import CATEGORICAL_STATE  # noqa: E402

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
vocab = FrameVocab.load(run / "vocab.npz")                 # the run's OWN vocabulary
recipe = dwa.probe_recipe(a.target, inst, n_seq=S["dw_probe_seqs"])
if a.n_seq:
    recipe["n_seq"] = a.n_seq
if a.epochs:
    recipe["epochs"] = a.epochs
tb = tkb.load_token_bench(vocab, n=S["dw_bench_n"], target=a.target, basis_name=basis, instance=inst)
arrays = dwb.bench_arrays(S["dw_bench_n"], a.target, basis, instance=inst)      # the same cases' world state
uns, u = tkb.unsteered(model, tb)
t0 = time.time()
print(f"{a.run} · {a.target} (basis {basis}, tokens) · recipe {recipe} · bench {tb.n} cases "
      f"({int(tb.keep.sum())} kept) · state {CATEGORICAL_STATE}", flush=True)
arms, st = tkb.inverse_arms(model, {a.target: tb}, {a.target: arrays}, vocab, basis_name=basis, target=a.target,
                            uns={a.target: uns}, cache_dir=EXP / "probes", points=a.points, log=print, **recipe)
recs = [{k: v for k, v in r.items() if np.isscalar(v)} for r in arms[a.target]]
new_guarded = best_arm(recs, "IM", "edit_index")
new_raw = best_arm(recs, "IM", "edit_index", guard=None)
old_arms = [r for r in blk.get("arms", []) if r.get("editor") == "IM"]
out = {"run": a.run, "instance": inst, "target": a.target, "basis": basis, "repr": "tokens", "state": CATEGORICAL_STATE,
       "recipe": {k: (v if not isinstance(v, Path) else str(v)) for k, v in recipe.items()}, "bench_n": int(tb.n),
       "n_cases_kept": int(tb.keep.sum()), "unedited": u["edit_index"],
       "new": {"reported (best inside the guard)": new_guarded, "unguarded best": new_raw,
               "g_r2_by_point": st["g_r2"], "g_r2_insample_by_point": st["g_r2_insample"], "by_point": recs},
       "old_continuous_map_on_this_bench": {"reported (best inside the guard)": best_arm(old_arms, "IM", "edit_index"),
                                            "unguarded best": best_arm(old_arms, "IM", "edit_index", guard=None),
                                            "g_r2_by_point": (blk.get("inverse_map") or {}).get("g_r2")},
       "other_editors_from_scores_json": {e: best_arm(blk.get("arms", []), e, "edit_index") for e in ("PI", "ND", "GS")},
       "minutes": round((time.time() - t0) / 60, 1), "written": time.strftime("%Y-%m-%d %H:%M")}
p = EXP / "scores" / f"preview_{run.name}_{a.target}_tokens{a.tag}.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=1, default=float))
f = lambda r: f"{r['edit_index']:+.3f} / {r['fidelity_ratio']:.2f} (pt {r['point']})" if r else "—"   # noqa: E731
print(f"\n{run.name} · {a.target} (tokens):  NEW categorical-state IM {f(new_guarded)}   [unguarded {f(new_raw)}]")
print(f"   OLD continuous-map-on-this-bench {f(out['old_continuous_map_on_this_bench']['reported (best inside the guard)'])}"
      f"   unedited {u['edit_index']:+.3f}   PI {f(out['other_editors_from_scores_json']['PI'])}"
      f"   GS {f(out['other_editors_from_scores_json']['GS'])}")
print(f"   g R² held-out max {max(st['g_r2']):+.3f} (in-sample at that point {st['g_r2_insample'][int(np.argmax(st['g_r2']))]:+.3f})"
      f"   [{out['minutes']} min]  -> {p.relative_to(REPO)}")
