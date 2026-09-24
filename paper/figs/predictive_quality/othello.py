"""Predictive quality on Othello, for the appendix (2026-09-21, round 4): per rule variant three bench boards,
each with its true legal set (uniform over the legal moves, the Bayes-optimal target) above the same board
tinted by the trained model's unedited next-move distribution. Everything comes from the qualitative Othello
figure's cache (``.scratch/othello_edits_guarded_cache.pkl``: ``board_pre``, ``legal_pre``, ``probs["Unedited"]``
per variant, the runs named in ``othello.json``); no model is loaded. Squares are tinted exactly as that figure
tints them (``draw_board``: full at 0.02 probability mass and above).

Layout: four variant columns (a) standard, (b) adjacent-flip, (c) adjacent-noflip, (d) standard-noflip; three
example blocks stacked vertically, each a (legal moves / model) row pair, a thin gap between the blocks. Cases:
one ``numpy.random.default_rng(--seed)`` shared across the variants in column order, three distinct cases per
variant (``rng.choice(n_cached, 3, replace=False)``, sorted); ids in ``othello.json``.

Moved here 2026-09-21 from paper/figs/environments_overview/othello/make_predictive.py; the board drawing is
loaded by path from that folder's make_figure.py (which itself loads ``draw_board`` from
paper/figs/qualitative_edits_othello/make_figure.py). Outputs beside this script: ``othello.{pdf,png}``,
``othello.json``, ``pieces/othello_<variant>_<case>_{legal,model}.pdf``.

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


# the Othello overview's board drawing (it applies paper_style and loads draw_board from
# paper/figs/qualitative_edits_othello/make_figure.py)
sys.dont_write_bytecode = True                # no __pycache__ in the figure folders
mf = _import(REPO / "paper" / "figs" / "environments_overview" / "othello" / "make_figure.py")
import matplotlib.pyplot as plt  # noqa: E402
import paper_style as ps  # noqa: E402  (paper/figs is on sys.path once mf is loaded)

from pim.metrics.set_editability import uniform_over_legal  # noqa: E402

CACHE = REPO / ".scratch" / "othello_edits_guarded_cache.pkl"
COLUMNS = [("standard", "Standard"), ("adjacent_flip", "Adjacent Flip"),
           ("adjacent_noflip", "Adjacent NoFlip"), ("standard_noflip", "Standard NoFlip")]
N_EX = 3
LABELS = ("legal moves", "model")


def stacked(all_views: dict, stem: Path) -> None:
    """The four rule sets across (bold panel letters once, at the top); the examples stacked, each a
    (legal moves / model) row pair with its two labels, a thin gap between the example blocks."""
    W, cg, rg, bg, top, left = ps.TEXT_WIDTH_IN, 0.12, 0.12, 0.30, 0.2, 0.62
    s = (W - left - 3 * cg) / 4
    n_ex = len(all_views[COLUMNS[0][0]])
    block = 2 * s + rg
    H = top + n_ex * block + (n_ex - 1) * bg
    fig = plt.figure(figsize=(W, H))
    for c, (name, _) in enumerate(COLUMNS):
        x0 = left + c * (s + cg)
        fig.text(x0 / W, 1 - 0.5 * top / H, f"({'abcd'[c]})", ha="left", va="center", fontsize=9, fontweight="bold")
        for e, V in enumerate(all_views[name]):
            for r, k in enumerate(("legal", "model")):
                y0 = H - top - e * (block + bg) - (r + 1) * s - r * rg
                mf.board(fig.add_axes([x0 / W, y0 / H, s / W, s / H]), **V[k], lw=1.1, dot_r=0.1)
                if c == 0:
                    fig.text((left - 0.08) / W, (y0 + s / 2) / H, LABELS[r], ha="right", va="center", fontsize=8)
    ps.save(fig, stem)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cols = pickle.load(open(CACHE, "rb"))
    rng = np.random.default_rng(a.seed)
    (HERE / "pieces").mkdir(exist_ok=True)
    all_views, record, piece_files = {}, {}, []
    for name, key in COLUMNS:
        c = cols[key]
        ids = sorted(int(i) for i in rng.choice(len(c["board_pre"]), N_EX, replace=False))
        all_views[name], record[name] = [], {"cache_column": key, "run": c["run"], "instance": c["instance"],
                                             "n_cached": int(len(c["board_pre"])), "cases": []}
        for i in ids:
            X = {"board": c["board_pre"][i], "legal": [int(s) for s in c["legal_pre"][i]]}
            V = {"legal": dict(X=X, probs=uniform_over_legal(X["legal"])), "model": dict(X=X, probs=c["probs"]["Unedited"][i])}
            all_views[name].append(V)
            record[name]["cases"].append({"bench_case_id": i, "board_after_move": int(c["lengths"][i]),
                                          "n_legal": len(X["legal"]),
                                          "pieces": {k: f"pieces/othello_{name}_{i}_{k}.pdf" for k in V}})
            for k, spec in V.items():
                mf.row([spec], HERE / "pieces" / f"othello_{name}_{i}_{k}", arrows=False)
                piece_files.append(f"othello_{name}_{i}_{k}.pdf")
        print(f"{name:<16} {c['run']:<48} cases {ids}  legal {[len(v['legal']['X']['legal']) for v in all_views[name]]}")
    stacked(all_views, HERE / "othello")
    json.dump({"seed": a.seed, "cache": str(CACHE.relative_to(REPO)),
               "tint": {"gamma": 0.6, "full_at_probability": mf._qual.TINT_SCALE},
               "selection_rule": f"one numpy.random.default_rng(seed) shared across the variants in column order; per variant "
                                 f"rng.choice(n_cached, {N_EX}, replace=False), sorted",
               "rows_per_example": {"legal moves": "uniform_over_legal(legal_pre)", "model": 'cached probs["Unedited"]'},
               "variants": record, "files": {"figure": "othello", "pieces": piece_files}},
              open(HERE / "othello.json", "w"), indent=1)
    print("->", HERE / "othello.json")
