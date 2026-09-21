"""Predictive quality, for the appendix: per Othello variant one bench board with its true legal set (uniform
over the legal moves, the Bayes-optimal target) beside the same board tinted by the trained model's unedited
next-move distribution. Both come from the qualitative Othello figure's cache
(``.scratch/othello_edits_guarded_cache.pkl``: ``board_pre``, ``legal_pre``, ``probs["Unedited"]`` per variant, the
runs named in ``predictive.json``); no model is loaded. Squares are tinted exactly as that figure tints them
(``draw_board``: full at 0.02 probability mass and above). One random case per variant (``--seed``).
Moved here 2026-09-21 from paper/figs/environments_overview/othello/make_predictive.py; the board drawing and
page layouts are loaded by path from that folder's make_figure.py (which itself loads ``draw_board`` from
paper/figs/qualitative_edits_othello/make_figure.py). Outputs beside this script: ``othello.{pdf,png}``,
``othello.json``, ``pieces/predictive_<variant>_{legal,model,pair}.pdf``.

    .pim/bin/python paper/figs/predictive_quality/othello.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))


def _import(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem + "_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# the Othello overview's board drawing and page layouts (it applies paper_style and loads draw_board from
# paper/figs/qualitative_edits_othello/make_figure.py)
sys.dont_write_bytecode = True                # no __pycache__ in the figure folders
mf = _import(REPO / "paper" / "figs" / "environments_overview" / "othello" / "make_figure.py")
from pim.metrics.set_editability import uniform_over_legal  # noqa: E402

CACHE = REPO / ".scratch" / "othello_edits_guarded_cache.pkl"
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
    mf.composite(all_views, ["legal", "model"], HERE / "othello", arrows=False,
                 labels=["legal moves", "model"], key=False)
    json.dump({"seed": a.seed, "cache": str(CACHE.relative_to(REPO)),
               "tint": {"gamma": 0.6, "full_at_probability": mf._qual.TINT_SCALE},
               "selection_rule": "one case per variant, rng.integers over the 1000 cached bench cases",
               "variants": record}, open(HERE / "othello.json", "w"), indent=1)
    print("->", HERE / "othello.json")
