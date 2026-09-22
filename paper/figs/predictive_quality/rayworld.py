"""Predictive quality on Rayworld, for the appendix (2026-09-21, round 4): the model's free-run against the
clean render of the UNEDITED world on six held-out sequences, no edit anywhere, and their difference.

Reads the SAME array file as ``paper/figs/history_rewrite/draw_paper.py``,
``.scratch/history_rewrite_arrays.npz`` (written on the GPU by ``paper/figs/history_rewrite/make_figure.py``
for the 32 bench cases of ``runs/noise_ablation/L-dw-noiseless-20m``): ``obs_hist`` the observed frames 0..19,
``gt_unedited_roll`` the simulator's clean unedited continuation over frames 20..34 (the scorer's ghost
trajectory), ``roll_unsteered`` the model's free-run from frame 20 with no edit. No model, no metric.

Rows x columns: three rows (Ground truth, Prediction, Difference) by six sequences
(``numpy.random.default_rng(0).choice(32, 6, replace=False)``, sorted). Time runs downward within each panel:
the last ``N_CTX`` observed frames above the dashed free-run-start line, the K = 15 free-run steps below it.
Ground truth and Prediction are drawn by ``draw_paper.py``'s panel code (imported by path: gray on the dark
panel, fixed 0-1 range, nearest, 600 ppi). Difference = prediction - truth per ray on the canonical signed-error
map (``pim.figures.waterfall.DIFF_CMAP``: under-prediction red, over-prediction green, zero = the dark
background, fixed +-1), drawn with ``error`` and ``_panel`` imported from
``paper/figs/qualitative_edits/make_figure.py`` (the prediction clipped to [0, 1] first, as that figure's
``_diff`` variant does); above the line the difference row is left empty (dark): those frames are observations,
not predictions. One "free-run start" key, one colour bar ("prediction - truth") beside the difference row.

Outputs beside this script: ``rayworld.{pdf,png}``, ``rayworld.json``, and under ``pieces/`` one PDF per panel
(``rayworld_case<idx>_{gt,pred,diff}``) plus ``rayworld_key.pdf`` and ``rayworld_colorbar.pdf``.

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
sys.path[:0] = [str(HERE.parent), str(REPO)]          # paper/figs, then the repo
import paper_style as ps  # noqa: E402

ps.apply()
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from pim.figures.waterfall import DIFF_CMAP  # noqa: E402


def _import(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem + "_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.dont_write_bytecode = True                # no __pycache__ in the figure folders
hr = _import(REPO / "paper" / "figs" / "history_rewrite" / "draw_paper.py")      # waterfall panels, key, npz path
qe = _import(REPO / "paper" / "figs" / "qualitative_edits" / "make_figure.py")   # error() and the signed-error _panel
ps.apply()                                    # the reference modules set rcParams at import; one look wins

ROWS = [("Ground truth", "gt_unedited_roll"), ("Prediction", "roll_unsteered"), ("Difference", None)]
N_EX, SEED = 6, 0
DIFF_SCALE = 1.0                              # fixed +-1, the qualitative figures' scale
LEFT, RIGHT = 0.46, 0.44                      # row label + time arrow at the left; the colour bar at the right
COL_GAP, ROW_GAP = 0.06, 0.10
FRAME_H = 0.034                               # one frame, inches (six narrow columns want shorter panels than 0.042)


def diff_panel(ax, n_ctx: int, pred: np.ndarray, gt: np.ndarray) -> None:
    """The difference row: empty above the free-run start (observations, not predictions), the signed error
    prediction - truth below it, on the fixed +-DIFF_SCALE map."""
    err = np.concatenate([np.zeros((n_ctx, pred.shape[1]), np.float32), qe.error(pred, gt)])
    qe._panel(ax, err, diff=True, diff_scale=DIFF_SCALE)
    ax.axhline(n_ctx - 0.5, color=hr.EDIT_LINE, lw=hr.EDIT_LW, ls=(0, (3, 2)))


def colorbar(cax) -> None:
    cb = mpl.colorbar.ColorbarBase(cax, cmap=DIFF_CMAP, norm=mpl.colors.Normalize(-DIFF_SCALE, DIFF_SCALE),
                                   orientation="vertical")
    cb.set_ticks([-DIFF_SCALE, 0, DIFF_SCALE])
    cb.ax.tick_params(labelsize=7, length=2, width=0.5, colors=ps.TEXT)
    cb.outline.set_edgecolor(ps.FRAME); cb.outline.set_linewidth(0.5)
    cb.set_label("prediction − truth", fontsize=8, color=ps.TEXT, labelpad=3)


def panel(ax, d, case: int, body: str | None, EF: int, K: int) -> None:
    if body is None:
        diff_panel(ax, hr.N_CTX, d["roll_unsteered"][case, :K], d["gt_unedited_roll"][case, :K])
    else:
        hr.waterfall(ax, d["obs_hist"][case, EF - hr.N_CTX:EF], d[body][case, :K])


def figure(d, cases, *, width: float):
    EF, K = d["obs_hist"].shape[1], d["roll_unsteered"].shape[1]
    panel_w = (width - LEFT - RIGHT - COL_GAP * (len(cases) - 1)) / len(cases)
    panel_h = FRAME_H * (hr.N_CTX + K)
    top = 0.02
    height = top + len(ROWS) * panel_h + (len(ROWS) - 1) * ROW_GAP + hr.BOTTOM
    fig = plt.figure(figsize=(width, height), dpi=hr.PDF_PPI)
    for r, (label, body) in enumerate(ROWS):
        y = top + r * (panel_h + ROW_GAP)
        for c, case in enumerate(cases):
            panel(hr._box(fig, LEFT + c * (panel_w + COL_GAP), y, panel_w, panel_h), d, case, body, EF, K)
        hr._arrow(fig, LEFT - 0.12, y, LEFT - 0.12, y + panel_h)                    # time runs downward
        yc = 1 - (y + panel_h / 2) / height
        fig.text((LEFT - 0.19) / width, yc, "time", rotation=90, ha="center", va="center", fontsize=8, color=ps.TEXT)
        fig.text((LEFT - 0.36) / width, yc, label, rotation=90, ha="center", va="center", fontsize=9, color=ps.TEXT)
        if body is None:
            colorbar(hr._box(fig, width - RIGHT + 0.07, y, 0.07, panel_h))
    y_bot = top + len(ROWS) * panel_h + (len(ROWS) - 1) * ROW_GAP
    hr._arrow(fig, LEFT, y_bot + 0.10, LEFT + panel_w, y_bot + 0.10)                 # rays run left to right
    fig.text((LEFT + panel_w / 2) / width, 1 - (y_bot + 0.20) / height, "ray", ha="center", va="center",
             fontsize=8, color=ps.TEXT)
    hr.key(fig, LEFT + panel_w + COL_GAP + 0.08, y_bot + 0.10, hr.KEY_FREE)
    return fig


def pieces(d, cases) -> list[str]:
    EF, K = d["obs_hist"].shape[1], d["roll_unsteered"].shape[1]
    panel_w = (ps.TEXT_WIDTH_IN - LEFT - RIGHT - COL_GAP * (len(cases) - 1)) / len(cases)
    panel_h = FRAME_H * (hr.N_CTX + K)
    (HERE / "pieces").mkdir(exist_ok=True)
    out = []
    for case in cases:
        for tag, body in (("gt", "gt_unedited_roll"), ("pred", "roll_unsteered"), ("diff", None)):
            fig = plt.figure(figsize=(panel_w, panel_h), dpi=hr.PDF_PPI)
            panel(fig.add_axes([0, 0, 1, 1]), d, case, body, EF, K)
            p = HERE / "pieces" / f"rayworld_case{case:02d}_{tag}.pdf"
            fig.savefig(p, bbox_inches="tight", pad_inches=0); plt.close(fig); out.append(p.name)
    fig = plt.figure(figsize=(2.0, 0.2), dpi=hr.PDF_PPI)
    hr.key(fig, 0.0, 0.05, hr.KEY_FREE)
    fig.savefig(HERE / "pieces" / "rayworld_key.pdf", bbox_inches="tight", pad_inches=0); plt.close(fig)
    fig = plt.figure(figsize=(0.6, panel_h + 0.3), dpi=hr.PDF_PPI)          # room for the end tick labels
    colorbar(fig.add_axes([0.1, 0.15 / (panel_h + 0.3), 0.07 / 0.6, panel_h / (panel_h + 0.3)]))
    fig.savefig(HERE / "pieces" / "rayworld_colorbar.pdf", bbox_inches="tight", pad_inches=0); plt.close(fig)
    return out + ["rayworld_key.pdf", "rayworld_colorbar.pdf"]


if __name__ == "__main__":
    d = dict(np.load(hr.ARRAYS))
    scores = json.loads(str(d["scores_json"]))
    n = d["obs_hist"].shape[0]
    cases = sorted(int(i) for i in np.random.default_rng(SEED).choice(n, N_EX, replace=False))
    fig = figure(d, cases, width=ps.TEXT_WIDTH_IN)
    ps.save(fig, HERE / "rayworld"); plt.close(fig)
    piece_files = pieces(d, cases)
    EF, K = d["obs_hist"].shape[1], d["roll_unsteered"].shape[1]
    json.dump({"run": scores["run"], "instance": "dw-noiseless", "arrays": str(hr.ARRAYS.relative_to(REPO)),
               "rows": {"Ground truth": "gt_unedited_roll (the simulator's clean unedited continuation)",
                        "Prediction": "roll_unsteered (the model's free-run, no edit)",
                        "Difference": "prediction - truth per ray, the prediction clipped to [0, 1] first "
                                      "(paper/figs/qualitative_edits/make_figure.error, raw=False), on "
                                      f"pim.figures.waterfall.DIFF_CMAP with a fixed +-{DIFF_SCALE:g} scale; empty above the line"},
               "context": f"obs_hist frames {EF - hr.N_CTX}..{EF - 1} above the line", "k_roll": int(K),
               "selection": {"rule": f"numpy.random.default_rng({SEED}).choice({n}, {N_EX}, replace=False), sorted; no edit, "
                                     "so no displacement filter", "seed": SEED, "cases": cases},
               "files": {"figure": "rayworld", "pieces": piece_files}},
              open(HERE / "rayworld.json", "w"), indent=1)
    print("cases", cases, "-> rayworld.pdf/.png, rayworld.json +", len(piece_files), "pieces")
