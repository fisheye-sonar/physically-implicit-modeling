"""Predictive quality, for the appendix: per Othello variant one bench board with its true legal set (uniform
over the legal moves, the Bayes-optimal target) beside the same board tinted by the trained model's unedited
next-move distribution. Both come from the qualitative Othello figure's cache
(``.scratch/othello_edits_guarded_cache.pkl``: ``board_pre``, ``legal_pre``, ``probs["Unedited"]`` per variant, the
runs named in ``predictive.json``); no model is loaded. Squares are tinted exactly as that figure tints them
(``draw_board``: full at 0.02 probability mass and above). One random case per variant (``--seed``).

    .pim/bin/python paper/figs/environments_overview/othello/make_predictive.py
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import make_figure as mf                       # this folder's board drawing and page layouts
from pim.metrics.set_editability import uniform_over_legal

HERE = Path(__file__).resolve().parent
CACHE = mf.REPO / ".scratch" / "othello_edits_guarded_cache.pkl"
COLUMNS = [("standard", "Standard"), ("adjacent_flip", "Adjacent Flip"),
           ("adjacent_noflip", "Adjacent NoFlip"), ("standard_noflip", "Standard NoFlip")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cols = pickle.load(open(CACHE, "rb"))
    rng = np.random.default_rng(a.seed)
    all_views, record = {}, {}
    for name, key in COLUMNS:
        c = cols[key]
        i = int(rng.integers(len(c["board_pre"])))
        X = {"board": c["board_pre"][i], "legal": [int(s) for s in c["legal_pre"][i]]}
        all_views[name] = V = {"legal": dict(X=X, probs=uniform_over_legal(X["legal"])),
                               "model": dict(X=X, probs=c["probs"]["Unedited"][i])}
        record[name] = {"cache_column": key, "run": c["run"], "instance": c["instance"], "bench_case_id": i,
                        "board_after_move": int(c["lengths"][i]), "n_legal": len(X["legal"]),
                        "pieces": {k: f"pieces/predictive_{name}_{k}.pdf" for k in list(V) + ["pair"]}}
        print(f"{name:<16} {c['run']:<48} case {i:4d}  {len(X['legal'])} legal moves")
        for k, spec in V.items():
            mf.row([spec], HERE / "pieces" / f"predictive_{name}_{k}", arrows=False)
        mf.row(list(V.values()), HERE / "pieces" / f"predictive_{name}_pair", arrows=False, gap=0.25)
    mf.composite(all_views, ["legal", "model"], HERE / "predictive_composite", arrows=False,
                 labels=["legal moves", "model"], key=False)
    json.dump({"seed": a.seed, "cache": str(CACHE.relative_to(mf.REPO)),
               "tint": {"gamma": 0.6, "full_at_probability": mf._qual.TINT_SCALE},
               "selection_rule": "one case per variant, rng.integers over the 1000 cached bench cases",
               "variants": record}, open(HERE / "predictive.json", "w"), indent=1)
    print("->", HERE / "predictive.json")
