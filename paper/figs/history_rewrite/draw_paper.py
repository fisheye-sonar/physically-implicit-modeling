"""The paper figure for history rewriting (appendix): three edits, four columns (2026-09-21, round 2).

Reads ``.scratch/history_rewrite_arrays.npz``, written by ``make_figure.py`` beside this script, and
touches no model and computes no metric. Each panel is one 128-ray waterfall, time downward. In the main
figure EVERY column shows the same ground-truth pre-edit history above the edit-frame line (the last
``N_CTX`` original observed frames, ``obs_hist``; asserted identical across the columns of a row); below the
line each column shows its own free-run:

    Ground truth       the clean edited-world rollout (gt_roll)
    Unedited           the model's free-run, no edit
    Single-point edit  the IM write at the edit frame (one point, one step), then free-run
    History rewrite    free-run from the REWRITTEN history plus the same write at the edit frame (hist+IM);
                       ``_histonly``: from the rewritten history with NO write at the edit frame (hist)

The rewritten frames themselves are the appendix piece ``history_frames_<rows>``: the original observed
frames | the rewritten frames | the simulator's clean render of the counterfactual history, all 20 history
frames, no edit-frame line (the edit frame is the frame after the last row).

Rows: ``top3`` = the three cases with the largest teleport displacement |target_x - ghost_x| in rays (the
origin / destination ray centres the scorer's zones give); ``random3`` = three cases drawn with
``numpy.random.default_rng(0)``. Case indices, arms and the rule land in ``history_rewrite_paper.json``.

Outputs beside this script: ``history_rewrite_<rows>[_k10][_histonly].{pdf,png}``,
``history_frames_<rows>.{pdf,png}``, ``prediction_quality.{pdf,png}`` (stretch: the clean UNEDITED world
against the model's free-run, no edit), and every panel plus both legend keys as their own PDF under ``pieces/``.

    .pim/bin/python paper/figs/history_rewrite/draw_paper.py
"""
from __future__ import annotations

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
from matplotlib.patches import FancyArrowPatch  # noqa: E402

sys.path.insert(0, str(REPO))
from pim.figures.waterfall import DARK_BG, EDIT_LINE  # noqa: E402

ARRAYS = REPO / ".scratch" / "history_rewrite_arrays.npz"
PIECES = HERE / "pieces"
N_CTX = 8                                     # history frames above the edit-frame line
# (column label, drawn above the line, drawn below it; None = history alone, no line)
COLUMNS = [("Ground truth", "obs_hist", "gt_roll"),
           ("Unedited", "obs_hist", "roll_unsteered"),
           ("Single-point edit", "obs_hist", "roll_IM"),
           ("History rewrite", "obs_hist", "roll_hist+IM")]
HIST_ONLY = COLUMNS[:3] + [("History rewrite", "obs_hist", "roll_hist")]
FRAMES = [("Original frames", "obs_hist", None), ("Rewritten frames", "obs_cf", None),
          ("Counterfactual render", "cf_clean", None)]
QUALITY = [("Ground truth", "obs_hist", "gt_unedited_roll"), ("Prediction", "obs_hist", "roll_unsteered")]
PIECE_KEY = {"gt_roll": "gt", "roll_unsteered": "unedited", "roll_IM": "im", "roll_hist+IM": "hist_im",
             "roll_hist": "hist", "obs_hist": "frames_original", "obs_cf": "frames_rewritten",
             "cf_clean": "frames_cfrender"}

# geometry, in inches, at the printed size
FRAME_H = 0.042                               # one observation frame
COL_GAP, ROW_GAP = 0.07, 0.11
LEFT, RIGHT, TOP, BOTTOM = 0.30, 0.02, 0.20, 0.30
EDIT_LW, LOC_LW = 0.8, 0.7
PDF_PPI = 600        # the PDF backend resamples rasters at the figure's dpi: at the default 100 a 1.24 in panel
                     # would hold 125 pixels for 128 rays and drop three of them; 600 keeps every ray (5-6 px each)


def _panel(ax, img: np.ndarray) -> None:
    """A raw-observation panel exactly as paper/figs/qualitative_edits/make_figure.py draws one:
    gray on the dark panel, fixed 0-1 range, nearest, a thin frame."""
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    ax.set_facecolor(DARK_BG)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_edgecolor(ps.FRAME)


def waterfall(ax, ctx: np.ndarray, body: np.ndarray | None = None,
              origin_x: float = np.nan, dest_x: float = np.nan) -> None:
    """History above the line, the free-run below it, one fixed intensity scale; ``body`` None: the
    history alone and no line."""
    _panel(ax, ctx if body is None else np.concatenate([ctx, body], axis=0))
    if body is not None:
        ax.axhline(ctx.shape[0] - 0.5, color=EDIT_LINE, lw=EDIT_LW, ls=(0, (3, 2)))
    for x, colr in ((origin_x, ps.ORIGIN_C), (dest_x, ps.DEST_C)):
        if np.isfinite(x):
            ax.axvline(x, color=colr, lw=LOC_LW)


def _box(fig, x: float, y: float, w: float, h: float):
    """Axes at (x, y from the top) in inches."""
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, 1.0 - (y + h) / H, w / W, h / H])


def _arrow(fig, x0, y0, x1, y1) -> None:
    W, H = fig.get_size_inches()
    fig.add_artist(FancyArrowPatch((x0 / W, 1 - y0 / H), (x1 / W, 1 - y1 / H), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=5, lw=0.6, color=ps.TEXT, shrinkA=0, shrinkB=0))


def key(fig, x: float, y: float, entries, gap: float = 0.14) -> float:
    """Legend: a dark swatch carrying each line as it is drawn on the panels, its label beside it.
    Returns the x (inches) where the key ends."""
    W, H = fig.get_size_inches()
    for kind, colr, label in entries:
        ax = _box(fig, x, y, 0.22, 0.11)
        ax.set_facecolor(DARK_BG); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_edgecolor(ps.FRAME)
        if kind == "h":
            ax.axhline(0.5, color=colr, lw=EDIT_LW, ls=(0, (3, 2)))
        else:
            ax.axvline(0.5, color=colr, lw=LOC_LW)
        t = fig.text((x + 0.27) / W, 1 - (y + 0.055) / H, label, ha="left", va="center", fontsize=8, color=ps.TEXT)
        x += 0.27 + t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi + gap   # measured label width
    return x - gap


KEY_EDIT = [("h", EDIT_LINE, "edit frame"), ("v", ps.ORIGIN_C, "origin position"), ("v", ps.DEST_C, "destination position")]
KEY_POS = KEY_EDIT[1:]
KEY_FREE = [("h", EDIT_LINE, "free-run start")]


def figure(d, cases, columns, *, n_ctx: int, K: int, width: float, locators: bool = True, legend=KEY_EDIT,
           same_context: bool = True):
    """Rows = cases, columns as given. With ``same_context`` every column of a row must show the identical
    frames above the line (asserted on the arrays)."""
    EF = d["obs_hist"].shape[1]
    panel_w = (width - LEFT - RIGHT - COL_GAP * (len(columns) - 1)) / len(columns)
    panel_h = FRAME_H * (n_ctx + K)
    height = TOP + len(cases) * panel_h + (len(cases) - 1) * ROW_GAP + BOTTOM
    fig = plt.figure(figsize=(width, height), dpi=PDF_PPI)
    for r, case in enumerate(cases):
        y = TOP + r * (panel_h + ROW_GAP)
        ctxs = [d[top][case, EF - n_ctx:EF] for _, top, _ in columns]
        if same_context:
            assert all(np.array_equal(ctxs[0], c) for c in ctxs[1:]), f"case {case}: the context differs across columns"
        for c, (name, _, bottom) in enumerate(columns):
            ax = _box(fig, LEFT + c * (panel_w + COL_GAP), y, panel_w, panel_h)
            waterfall(ax, ctxs[c], None if bottom is None else d[bottom][case, :K],
                      *((d["ghost_x"][case], d["target_x"][case]) if locators else ()))
            if r == 0:
                ax.set_title(name, pad=3)
        _arrow(fig, LEFT - 0.12, y, LEFT - 0.12, y + panel_h)                       # time runs downward
        fig.text((LEFT - 0.19) / width, 1 - (y + panel_h / 2) / height, "time", rotation=90,
                 ha="center", va="center", fontsize=8, color=ps.TEXT)
    y_bot = TOP + len(cases) * panel_h + (len(cases) - 1) * ROW_GAP
    _arrow(fig, LEFT, y_bot + 0.10, LEFT + panel_w, y_bot + 0.10)                    # rays run left to right
    fig.text((LEFT + panel_w / 2) / width, 1 - (y_bot + 0.20) / height, "ray", ha="center", va="center",
             fontsize=8, color=ps.TEXT)
    if legend:
        end = key(fig, LEFT + panel_w + COL_GAP + 0.08, y_bot + 0.10, legend)
        if end > width:
            print(f"  note: the key runs {end - width:.2f} in past the panels' right edge")
    return fig


def pieces(d, cases, K: int) -> list[str]:
    """Every panel as its own PDF: the four columns and the hist-alone fourth column (history + K steps), the
    three history_frames panels (20 frames), and both legend keys."""
    EF = d["obs_hist"].shape[1]
    PIECES.mkdir(exist_ok=True)
    specs = [(N_CTX, K, len(COLUMNS), col) for col in COLUMNS + [HIST_ONLY[-1]]] + [(EF, 0, len(FRAMES), col) for col in FRAMES]
    out = []
    for case in cases:
        for n_ctx, k, ncol, (_, top, bottom) in specs:
            panel_w = (ps.TEXT_WIDTH_IN - LEFT - RIGHT - COL_GAP * (ncol - 1)) / ncol
            fig = plt.figure(figsize=(panel_w, FRAME_H * (n_ctx + k)), dpi=PDF_PPI)
            waterfall(fig.add_axes([0, 0, 1, 1]), d[top][case, EF - n_ctx:EF],
                      None if bottom is None else d[bottom][case, :k], d["ghost_x"][case], d["target_x"][case])
            p = PIECES / f"case{case:02d}_{PIECE_KEY[bottom or top]}.pdf"
            fig.savefig(p, bbox_inches="tight", pad_inches=0); plt.close(fig); out.append(p.name)
    for name, entries in (("legend_key", KEY_EDIT), ("legend_key_positions", KEY_POS)):
        fig = plt.figure(figsize=(4.5, 0.2), dpi=PDF_PPI)
        key(fig, 0.0, 0.05, entries)
        fig.savefig(PIECES / f"{name}.pdf", bbox_inches="tight", pad_inches=0); plt.close(fig); out.append(f"{name}.pdf")
    return out


if __name__ == "__main__":
    d = dict(np.load(ARRAYS))
    scores = json.loads(str(d["scores_json"]))
    EF, K_FULL = d["obs_hist"].shape[1], d["gt_roll"].shape[1]
    disp = np.abs(d["target_x"] - d["ghost_x"])                    # teleport displacement in rays; NaN = origin not visible
    ranked = [int(i) for i in np.argsort(-np.nan_to_num(disp, nan=-1.0)) if np.isfinite(disp[i])]
    rows = {"top3": ranked[:3],
            "random3": sorted(int(i) for i in np.random.default_rng(0).choice(len(disp), 3, replace=False))}
    written = []
    for sel, cases in rows.items():
        for K, ktag in ((K_FULL, ""), (10, "_k10")):
            for cols, htag in ((COLUMNS, ""), (HIST_ONLY, "_histonly")):
                stem = HERE / f"history_rewrite_{sel}{ktag}{htag}"
                fig = figure(d, cases, cols, n_ctx=N_CTX, K=K, width=ps.TEXT_WIDTH_IN)
                ps.save(fig, stem); plt.close(fig); written.append(stem.name)
        # the rewritten frames themselves, beside the original and the counterfactual render (appendix piece)
        fig = figure(d, cases, FRAMES, n_ctx=EF, K=0, width=ps.TEXT_WIDTH_IN, legend=KEY_POS, same_context=False)
        ps.save(fig, HERE / f"history_frames_{sel}"); plt.close(fig); written.append(f"history_frames_{sel}")
    # stretch: predictive quality with no edit, the clean unedited world against the model's free-run
    fig = figure(d, rows["random3"], QUALITY, n_ctx=N_CTX, K=K_FULL, width=ps.HALF_WIDTH_IN, locators=False, legend=KEY_FREE)
    ps.save(fig, HERE / "prediction_quality"); plt.close(fig); written.append("prediction_quality")
    piece_files = pieces(d, sorted(set(rows["top3"]) | set(rows["random3"])), K_FULL)
    side = {
        "run": scores["run"], "instance": "dw-noiseless", "block": scores["block"], "im_point": scores["point"],
        "n_cases": int(scores["n"]), "n_ctx": N_CTX, "k_roll": K_FULL, "edit_frame": int(EF),
        "context": f"the original observed frames {EF - N_CTX}..{EF - 1} (obs_hist) above the line in every column "
                   "of history_rewrite_*; asserted identical across the columns of each row",
        "columns_below_the_line": {name: bottom for name, _, bottom in COLUMNS},
        "histonly_fourth_column": HIST_ONLY[-1][2],
        "selection": {
            "top3": {"rule": "the three of the 32 bench cases with the largest |target_x - ghost_x| (teleport "
                             "displacement in rays, from the scorer's target / ghost ray zones); a case whose origin "
                             "is not visible at the edit frame (ghost_x NaN) is excluded", "cases": rows["top3"],
                     "displacement_rays": [float(disp[i]) for i in rows["top3"]]},
            "random3": {"rule": "numpy.random.default_rng(0).choice(32, 3, replace=False), sorted", "seed": 0,
                        "cases": rows["random3"], "displacement_rays": [float(disp[i]) for i in rows["random3"]]},
            "prediction_quality": {"rule": "the random3 cases; no edit, unedited world vs free-run", "cases": rows["random3"]}},
        "per_case": {"displacement_rays": [None if not np.isfinite(x) else float(x) for x in disp],
                     "ghost_x": [None if not np.isfinite(x) else float(x) for x in d["ghost_x"]],
                     "target_x": [float(x) for x in d["target_x"]], "edit_object": [int(x) for x in d["edit_object"]]},
        "arms": {k: {"edit_index": c["edit_index"], "fidelity_ratio": c["fidelity_ratio"],
                     "edit_index_by_step": c["edit_index_by_step"]} for k, c in scores["cards"].items()},
        "history_rmse": scores["history_rmse"],
        "files": {"figures": written, "pieces": piece_files},
    }
    json.dump(side, open(HERE / "history_rewrite_paper.json", "w"), indent=1)
    print("rows:", rows, "\ncontext identical across the columns of every row: asserted for every history_rewrite_* composite",
          "\n->", ", ".join(written), f"+ {len(piece_files)} pieces, history_rewrite_paper.json")
