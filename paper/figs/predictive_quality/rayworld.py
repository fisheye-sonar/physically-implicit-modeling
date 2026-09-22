"""Predictive quality on Rayworld, for the appendix (2026-09-21, round 5): the model's free-run against the
clean render of the UNEDITED world, no edit anywhere, and their difference, on three Rayworld variants.

Three panels of two held-out sequences each: (a) Standard, ``runs/noise_ablation/L-dw-noiseless-20m`` on
dw-noiseless; (b) Blink, ``runs/blink_ablation/L-dw-blink-20m`` on dw-blink; (c) 5-ray,
``runs/ray_ablation/L-dw-5ray-20m`` on dw-5ray. Rows: Ground truth (the simulator's clean render of the unedited
world continuing, the scorer's ghost trajectory), Prediction (the model's free-run from the observed frames),
Difference (prediction - truth). Time runs downward within each panel: the last ``N_CTX`` observed frames above
the dashed free-run-start line, the K = 15 free-run steps below it. Panel (a) reads the SAME array file as
``paper/figs/history_rewrite/draw_paper.py``, ``.scratch/history_rewrite_arrays.npz`` (written on the GPU by
``paper/figs/history_rewrite/make_figure.py``); (b) and (c) read ``.scratch/predictive_quality_<instance>.npz``,
written by ``compute_rayworld.py`` beside this script through the same canonical calls. Cases:
``numpy.random.default_rng(0).choice(n_bench, 2, replace=False)`` per instance, sorted.

Ground truth and Prediction are drawn by ``draw_paper.py``'s panel code (imported by path: gray on the dark
panel, fixed 0-1 range, nearest, 600 ppi). On dw-blink the observed context is drawn as the model saw it,
blackouts and the 0.5 edge-ray markers included, and the ground truth is rendered under the same blackout
schedule; on dw-5ray every strip is 5 rays wide, stretched to the panel. Difference = prediction - truth per
ray on the canonical signed-error map (``pim.figures.waterfall.DIFF_CMAP``: under-prediction red,
over-prediction green, zero = the dark background, fixed +-1), drawn with ``error`` and ``_panel`` imported
from ``paper/figs/qualitative_edits/make_figure.py`` (the prediction clipped to [0, 1] first, as that figure's
``_diff`` variant does); above the line the difference row is left empty (dark): observations, not predictions.
One "free-run start" key, one colour bar ("prediction - truth") beside the difference row. No model, no metric.

Outputs beside this script: ``rayworld.{pdf,png}``, ``rayworld.json``, and under ``pieces/`` one PDF per panel
(``rayworld_<instance>_case<idx>_{gt,pred,diff}``) plus ``rayworld_key.pdf`` and ``rayworld_colorbar.pdf``.

    .pim/bin/python paper/figs/predictive_quality/compute_rayworld.py   # GPU, once, for (b) and (c)
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

PANELS = [("a", "Standard", "dw-noiseless"), ("b", "Blink", "dw-blink"), ("c", "5-ray", "dw-5ray")]
ROWS = [("Ground truth", "gt_unedited_roll"), ("Prediction", "roll_unsteered"), ("Difference", None)]
N_PER, SEED = 2, 0                            # sequences per panel; the draw's seed (one default_rng(SEED) per instance)
DIFF_SCALE = 1.0                              # fixed +-1, the qualitative figures' scale
LEFT, RIGHT = 0.46, 0.44                      # row label + time arrow at the left; the colour bar at the right
COL_GAP, PAIR_GAP, ROW_GAP, TOP = 0.06, 0.18, 0.10, 0.20
FRAME_H = 0.034                               # one frame, inches (six narrow columns want shorter panels than 0.042)


def arrays(inst: str):
    """The three arrays and their provenance: the history-rewrite npz for dw-noiseless, compute_rayworld.py's
    cache otherwise."""
    path = hr.ARRAYS if inst == "dw-noiseless" else REPO / ".scratch" / f"predictive_quality_{inst}.npz"
    d = dict(np.load(path))
    meta = (json.loads(str(d["meta_json"])) if "meta_json" in d
            else {"run": json.loads(str(d["scores_json"]))["run"], "instance": inst, "n": int(d["obs_hist"].shape[0])})
    return d, meta, path


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


def panel(ax, d, case: int, body: str | None) -> None:
    EF, K = d["obs_hist"].shape[1], d["roll_unsteered"].shape[1]
    if body is None:
        diff_panel(ax, hr.N_CTX, d["roll_unsteered"][case, :K], d["gt_unedited_roll"][case, :K])
    else:
        hr.waterfall(ax, d["obs_hist"][case, EF - hr.N_CTX:EF], d[body][case, :K])


def geometry(width: float):
    ncol, npan = N_PER * len(PANELS), len(PANELS)
    panel_w = (width - LEFT - RIGHT - COL_GAP * (ncol - npan) - PAIR_GAP * (npan - 1)) / ncol
    return panel_w, FRAME_H * (hr.N_CTX + 15)


def figure(data: list, *, width: float):
    """``data``: (letter, name, inst, d, cases) per panel; the panels are pairs of columns under a bold letter."""
    panel_w, panel_h = geometry(width)
    height = TOP + len(ROWS) * panel_h + (len(ROWS) - 1) * ROW_GAP + hr.BOTTOM
    fig = plt.figure(figsize=(width, height), dpi=hr.PDF_PPI)
    x = LEFT
    for letter, _, _, d, cases in data:
        fig.text(x / width, 1 - 0.5 * TOP / height, f"({letter})", ha="left", va="center", fontsize=9,
                 fontweight="bold", color=ps.TEXT)
        for case in cases:
            for r, (_, body) in enumerate(ROWS):
                panel(hr._box(fig, x, TOP + r * (panel_h + ROW_GAP), panel_w, panel_h), d, case, body)
            x += panel_w + COL_GAP
        x += PAIR_GAP - COL_GAP
    for r, (label, body) in enumerate(ROWS):
        y = TOP + r * (panel_h + ROW_GAP)
        hr._arrow(fig, LEFT - 0.12, y, LEFT - 0.12, y + panel_h)                    # time runs downward
        yc = 1 - (y + panel_h / 2) / height
        fig.text((LEFT - 0.19) / width, yc, "time", rotation=90, ha="center", va="center", fontsize=8, color=ps.TEXT)
        fig.text((LEFT - 0.36) / width, yc, label, rotation=90, ha="center", va="center", fontsize=9, color=ps.TEXT)
        if body is None:
            colorbar(hr._box(fig, width - RIGHT + 0.07, y, 0.07, panel_h))
    y_bot = TOP + len(ROWS) * panel_h + (len(ROWS) - 1) * ROW_GAP
    hr._arrow(fig, LEFT, y_bot + 0.10, LEFT + panel_w, y_bot + 0.10)                 # rays run left to right
    fig.text((LEFT + panel_w / 2) / width, 1 - (y_bot + 0.20) / height, "ray", ha="center", va="center",
             fontsize=8, color=ps.TEXT)
    hr.key(fig, LEFT + panel_w + COL_GAP + 0.08, y_bot + 0.10, hr.KEY_FREE)
    return fig


def pieces(data: list) -> list[str]:
    panel_w, panel_h = geometry(ps.TEXT_WIDTH_IN)
    (HERE / "pieces").mkdir(exist_ok=True)
    out = []
    for _, _, inst, d, cases in data:
        for case in cases:
            for tag, body in (("gt", "gt_unedited_roll"), ("pred", "roll_unsteered"), ("diff", None)):
                fig = plt.figure(figsize=(panel_w, panel_h), dpi=hr.PDF_PPI)
                panel(fig.add_axes([0, 0, 1, 1]), d, case, body)
                p = HERE / "pieces" / f"rayworld_{inst}_case{case:02d}_{tag}.pdf"
                fig.savefig(p, bbox_inches="tight", pad_inches=0); plt.close(fig); out.append(p.name)
    fig = plt.figure(figsize=(2.0, 0.2), dpi=hr.PDF_PPI)
    hr.key(fig, 0.0, 0.05, hr.KEY_FREE)
    fig.savefig(HERE / "pieces" / "rayworld_key.pdf", bbox_inches="tight", pad_inches=0); plt.close(fig)
    fig = plt.figure(figsize=(0.6, panel_h + 0.3), dpi=hr.PDF_PPI)          # room for the end tick labels
    colorbar(fig.add_axes([0.1, 0.15 / (panel_h + 0.3), 0.07 / 0.6, panel_h / (panel_h + 0.3)]))
    fig.savefig(HERE / "pieces" / "rayworld_colorbar.pdf", bbox_inches="tight", pad_inches=0); plt.close(fig)
    return out + ["rayworld_key.pdf", "rayworld_colorbar.pdf"]


if __name__ == "__main__":
    data, side = [], []
    for letter, name, inst in PANELS:
        d, meta, path = arrays(inst)
        n = int(d["obs_hist"].shape[0])
        cases = sorted(int(i) for i in np.random.default_rng(SEED).choice(n, N_PER, replace=False))
        data.append((letter, name, inst, d, cases))
        side.append({"panel": letter, "name": name, "run": meta["run"], "instance": inst,
                     "arrays": str(path.relative_to(REPO)), "n_bench": n, "rays": int(d["obs_hist"].shape[2]),
                     "cases": cases})
        print(f"({letter}) {name:<9} {meta['run']:<36} {d['obs_hist'].shape[2]:>3} rays  cases {cases}")
    fig = figure(data, width=ps.TEXT_WIDTH_IN)
    ps.save(fig, HERE / "rayworld"); plt.close(fig)
    piece_files = pieces(data)
    d0 = data[0][3]
    EF, K = d0["obs_hist"].shape[1], d0["roll_unsteered"].shape[1]
    json.dump({"panels": side,
               "rows": {"Ground truth": "gt_unedited_roll (the simulator's clean render of the unedited world continuing, "
                                        "b.zones.gt_unedited_traj; on dw-blink under the same blackout schedule as the observations)",
                        "Prediction": "roll_unsteered (dwa.unsteered_rollout: the model's free-run from the observed frames, no edit)",
                        "Difference": "prediction - truth per ray, the prediction clipped to [0, 1] first "
                                      "(paper/figs/qualitative_edits/make_figure.error, raw=False), on "
                                      f"pim.figures.waterfall.DIFF_CMAP with a fixed +-{DIFF_SCALE:g} scale; empty above the line"},
               "context": f"b.obs frames {EF - hr.N_CTX}..{EF - 1} above the line, as the model saw them (blackouts and "
                          "markers included on dw-blink)", "k_roll": int(K),
               "selection": {"rule": f"numpy.random.default_rng({SEED}).choice(n_bench, {N_PER}, replace=False) per instance, "
                                     "sorted; no edit, so no displacement filter", "seed": SEED},
               "files": {"figure": "rayworld", "pieces": piece_files}},
              open(HERE / "rayworld.json", "w"), indent=1)
    print("-> rayworld.pdf/.png, rayworld.json +", len(piece_files), "pieces")
