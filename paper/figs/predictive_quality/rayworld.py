"""Predictive quality on Rayworld, for the appendix (2026-09-21): the model's free-run against the clean
render of the UNEDITED world on three held-out sequences, no edit anywhere.

Reads the SAME array file as ``paper/figs/history_rewrite/draw_paper.py``,
``.scratch/history_rewrite_arrays.npz`` (written on the GPU by ``paper/figs/history_rewrite/make_figure.py``
for the 32 bench cases of ``runs/noise_ablation/L-dw-noiseless-20m``): ``obs_hist`` the observed frames 0..19,
``gt_unedited_roll`` the simulator's clean unedited continuation over frames 20..34 (the scorer's ghost
trajectory), ``roll_unsteered`` the model's free-run from frame 20 with no edit. It imports that script's panel
code, so the strips are drawn exactly as the history-rewrite figure draws them (gray on the dark panel, fixed
0-1 range, nearest, 600 ppi). No model, no metric.

Layout, transposed to be wider than tall: two rows (Ground truth, Prediction) by three example columns; time
runs downward within each panel, the last ``N_CTX`` observed frames above the dashed line and the K = 15
free-run steps below it; the "free-run start" key once. Cases: ``numpy.random.default_rng(0).choice(32, 3,
replace=False)``, sorted (no edit is involved, so no displacement filter). Outputs beside this script:
``rayworld.{pdf,png}``, ``rayworld.json``, and under ``pieces/`` one PDF per panel (``rayworld_case<idx>_gt``,
``rayworld_case<idx>_pred``) plus ``rayworld_key.pdf``.

    .pim/bin/python paper/figs/predictive_quality/rayworld.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))          # paper/figs
import paper_style as ps  # noqa: E402

ps.apply()
import matplotlib.pyplot as plt  # noqa: E402


def _import(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem + "_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.dont_write_bytecode = True                # no __pycache__ in the figure folders
hr = _import(REPO / "paper" / "figs" / "history_rewrite" / "draw_paper.py")   # the panel code, the npz path, the geometry
ROWS = [("Ground truth", "gt_unedited_roll"), ("Prediction", "roll_unsteered")]
LEFT = 0.52                                   # room for the row label, the time arrow and its label
SEED = 0


def figure(d, cases, *, width: float):
    EF, K = d["obs_hist"].shape[1], d["roll_unsteered"].shape[1]
    panel_w = (width - LEFT - hr.RIGHT - hr.COL_GAP * (len(cases) - 1)) / len(cases)
    panel_h = hr.FRAME_H * (hr.N_CTX + K)
    top = 0.02
    height = top + len(ROWS) * panel_h + (len(ROWS) - 1) * hr.ROW_GAP + hr.BOTTOM
    fig = plt.figure(figsize=(width, height), dpi=hr.PDF_PPI)
    for r, (label, body) in enumerate(ROWS):
        y = top + r * (panel_h + hr.ROW_GAP)
        for c, case in enumerate(cases):
            ax = hr._box(fig, LEFT + c * (panel_w + hr.COL_GAP), y, panel_w, panel_h)
            hr.waterfall(ax, d["obs_hist"][case, EF - hr.N_CTX:EF], d[body][case, :K])
        hr._arrow(fig, LEFT - 0.12, y, LEFT - 0.12, y + panel_h)                    # time runs downward
        yc = 1 - (y + panel_h / 2) / height
        fig.text((LEFT - 0.19) / width, yc, "time", rotation=90, ha="center", va="center", fontsize=8, color=ps.TEXT)
        fig.text((LEFT - 0.38) / width, yc, label, rotation=90, ha="center", va="center", fontsize=9, color=ps.TEXT)
    y_bot = top + len(ROWS) * panel_h + (len(ROWS) - 1) * hr.ROW_GAP
    hr._arrow(fig, LEFT, y_bot + 0.10, LEFT + panel_w, y_bot + 0.10)                 # rays run left to right
    fig.text((LEFT + panel_w / 2) / width, 1 - (y_bot + 0.20) / height, "ray", ha="center", va="center",
             fontsize=8, color=ps.TEXT)
    hr.key(fig, LEFT + panel_w + hr.COL_GAP + 0.08, y_bot + 0.10, hr.KEY_FREE)
    return fig


def pieces(d, cases) -> list[str]:
    EF, K = d["obs_hist"].shape[1], d["roll_unsteered"].shape[1]
    panel_w = (ps.TEXT_WIDTH_IN - LEFT - hr.RIGHT - hr.COL_GAP * (len(cases) - 1)) / len(cases)
    (HERE / "pieces").mkdir(exist_ok=True)
    out = []
    for case in cases:
        for tag, body in (("gt", "gt_unedited_roll"), ("pred", "roll_unsteered")):
            fig = plt.figure(figsize=(panel_w, hr.FRAME_H * (hr.N_CTX + K)), dpi=hr.PDF_PPI)
            hr.waterfall(fig.add_axes([0, 0, 1, 1]), d["obs_hist"][case, EF - hr.N_CTX:EF], d[body][case, :K])
            p = HERE / "pieces" / f"rayworld_case{case:02d}_{tag}.pdf"
            fig.savefig(p, bbox_inches="tight", pad_inches=0); plt.close(fig); out.append(p.name)
    fig = plt.figure(figsize=(2.0, 0.2), dpi=hr.PDF_PPI)
    hr.key(fig, 0.0, 0.05, hr.KEY_FREE)
    fig.savefig(HERE / "pieces" / "rayworld_key.pdf", bbox_inches="tight", pad_inches=0); plt.close(fig)
    return out + ["rayworld_key.pdf"]


if __name__ == "__main__":
    d = dict(np.load(hr.ARRAYS))
    scores = json.loads(str(d["scores_json"]))
    cases = sorted(int(i) for i in np.random.default_rng(SEED).choice(d["obs_hist"].shape[0], 3, replace=False))
    fig = figure(d, cases, width=ps.TEXT_WIDTH_IN)
    ps.save(fig, HERE / "rayworld"); plt.close(fig)
    piece_files = pieces(d, cases)
    json.dump({"run": scores["run"], "instance": "dw-noiseless", "arrays": str(hr.ARRAYS.relative_to(REPO)),
               "rows": {label: body for label, body in ROWS}, "context": f"obs_hist frames {d['obs_hist'].shape[1] - hr.N_CTX}.."
               f"{d['obs_hist'].shape[1] - 1} above the line", "k_roll": int(d["roll_unsteered"].shape[1]),
               "selection": {"rule": f"numpy.random.default_rng({SEED}).choice(32, 3, replace=False), sorted; no edit, "
                                     "so no displacement filter", "seed": SEED, "cases": cases},
               "files": {"figure": "rayworld", "pieces": piece_files}},
              open(HERE / "rayworld.json", "w"), indent=1)
    print("cases", cases, "-> rayworld.pdf/.png, rayworld.json +", len(piece_files), "pieces")
