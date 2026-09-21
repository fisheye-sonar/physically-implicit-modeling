"""Main-text Othello panel (b): three options from the appendix cache, no model loaded.

    T1  full boards, the squares whose legality the flip changes outlined (the squares the Edit Index
        scores under the symmetric-difference construction), yellow tint by predicted mass as in the appendix
    T2  full boards, tint ONLY those changed squares by mass (everything else plain), edited tile pink
    T3  zoom: every board of a column cropped to the same S x S window around {edited tile} + changed squares
        (one-square margin); the same outlines and tint as T1

Each option for all four variants (columns) and for the two-column cut Standard vs Adjacent NoFlip; rows
Unedited (pre-edit board), Ground truth (post-edit board, uniform over its legal moves), PI, IM (+ GS with
``--gs``). Cases: one per variant by the rule in ``common.select`` (at least 3 changed squares, window
fits 5 x 5, then random with --seed; ``--rule typical`` picks the eligible case whose per-case indices are
closest to the population means). Output beside this script: ``othello_<opt>_<cut>[_gs].{pdf,png}``,
``othello_cases_<rule>.json``, and one PDF per board under ``pieces/othello_<opt>_<cut>/``.

    .pim/bin/python paper/figs/qualitative_main/othello_panel.py [--rule random|typical] [--seed 0] [--gs]
"""
from __future__ import annotations

import argparse

import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle

import common as C
from common import oth, plt, ps

CONDS = ("Unedited", "Ground truth", "PI", "IM")
CONDS_GS = ("Unedited", "Ground truth", "PI", "GS", "IM")
TWO = ("Standard", "Adjacent NoFlip")
OUTLINE_C, OUTLINE_LW = ps.BOARD_LINE, 1.1        # the changed-legality squares
GAP_IN, LEFT_IN, TOP_IN, PAD_IN = 0.06, 0.78, 0.18, 0.03
OPTIONS = {"T1": dict(zoom=False, tint="all"), "T2": dict(zoom=False, tint="symdiff"), "T3": dict(zoom=True, tint="all")}


def variants(cut):
    return list(oth.VARIANTS) if cut == "4col" else [(n, C.OTH_RUNS[n]) for n in TWO]


def size_in(n_cols, n_rows, board_in):
    return (LEFT_IN + n_cols * board_in + (n_cols - 1) * GAP_IN + PAD_IN, TOP_IN + n_rows * board_in + (n_rows - 1) * GAP_IN + PAD_IN)


def board_for_width(width_in, n_cols):
    return (width_in - LEFT_IN - PAD_IN - (n_cols - 1) * GAP_IN) / n_cols


def windows(cols, picks, names):
    """The common S x S crop per column: S = the largest window the picked cases need."""
    S = max(C.side(C.bbox(cols[n], picks[n])) for n in names)
    return {n: C.square(C.bbox(cols[n], picks[n]), S) for n in names}


def board(ax, col, i, cond, *, tint="all", outline=True, window=None):
    """One board as ``draw_board`` draws it, plus the changed-square outlines and the crop."""
    b = col["board_pre"] if cond == "Unedited" else col["board_post"]
    p, sd = np.asarray(col["probs"][cond][i], float), C.symdiff(col, i)
    if tint == "symdiff":
        m = np.zeros(64)
        m[sd] = 1.0
        p = p * m
    oth.draw_board(ax, b[i], p, int(col["pos"][i]))
    if outline:
        for s in sd:
            r, c = divmod(s, 8)
            ax.add_patch(Rectangle((c, 7 - r), 1, 1, fill=False, edgecolor=OUTLINE_C, linewidth=OUTLINE_LW, zorder=4))
    if window is not None:
        r0, r1, c0, c1 = window
        ax.set_xlim(c0, c1 + 1)
        ax.set_ylim(7 - r1, 8 - r0)


def panel(F, cols, picks, names, conds, *, zoom=False, tint="all", outline=True, titles=True, letter=None, transpose=False):
    """Draw into Figure / SubFigure ``F`` (already sized by ``size_in``). Default: variants across, conditions
    down; ``transpose``: variants down, conditions across (the compact two-variant cut)."""
    W, H = F.bbox.width / F.dpi, F.bbox.height / F.dpi
    rows, columns = (names, conds) if transpose else (conds, names)
    n, m = len(columns), len(rows)
    aw = (W - LEFT_IN - PAD_IN - (n - 1) * GAP_IN) / n
    ah = (H - TOP_IN - PAD_IN - (m - 1) * GAP_IN) / m
    gs = GridSpec(m, n, figure=F, left=LEFT_IN / W, right=1 - PAD_IN / W, top=1 - TOP_IN / H, bottom=PAD_IN / H,
                  wspace=GAP_IN / aw, hspace=GAP_IN / ah)
    win = windows(cols, picks, names) if zoom else {}
    wrap = lambda t, narrow: t.replace(" ", "\n") if narrow and t in names and " " in t else t   # noqa: E731
    for c, cname in enumerate(columns):
        for r, rname in enumerate(rows):
            name, cond = (rname, cname) if transpose else (cname, rname)
            ax = F.add_subplot(gs[r, c])
            board(ax, cols[name], picks[name], cond, tint=tint, outline=outline, window=win.get(name))
            if r == 0 and titles:
                ax.set_title(wrap(cname, aw < 0.85), pad=3, fontsize=9, color=ps.TEXT, fontweight="bold" if cname == "Ground truth" else "normal")
            if c == 0:
                ax.annotate(wrap(rname, True), xy=(0, 0.5), xycoords="axes fraction", xytext=(-4, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=8, color=ps.TEXT, fontweight="bold" if rname == "Ground truth" else "normal")
    if letter:
        F.text(0.06 / W, 1 - 0.04 / H, letter, ha="left", va="top", fontsize=10, fontweight="bold", color=ps.TEXT)
    return win


def thumbnail(ax, col, i, window):
    """A faded full board with the crop rectangle."""
    board(ax, col, i, "Ground truth", outline=False)
    ax.add_patch(Rectangle((0, 0), 8, 8, facecolor="white", alpha=0.55, zorder=6))
    r0, r1, c0, c1 = window
    ax.add_patch(Rectangle((c0, 7 - r1), c1 - c0 + 1, r1 - r0 + 1, fill=False, edgecolor=ps.TEXT, linewidth=1.0, zorder=7))


def pieces(cols, picks, names, conds, out_dir, *, zoom, tint, board_in=1.4):
    out_dir.mkdir(parents=True, exist_ok=True)
    win = windows(cols, picks, names) if zoom else {}
    for name in names:
        for cond in conds:
            fig = plt.figure(figsize=(board_in, board_in))
            board(fig.add_axes([0, 0, 1, 1]), cols[name], picks[name], cond, tint=tint, window=win.get(name))
            ps.save(fig, out_dir / f"{name.replace(' ', '')}_case{picks[name]}_{cond.replace(' ', '_').lower()}")
            plt.close(fig)
        if zoom:
            fig = plt.figure(figsize=(0.7, 0.7))
            thumbnail(fig.add_axes([0, 0, 1, 1]), cols[name], picks[name], win[name])
            ps.save(fig, out_dir / f"{name.replace(' ', '')}_case{picks[name]}_thumbnail")
            plt.close(fig)
    fig = plt.figure(figsize=(1.9, 0.6))
    fig.legend(handles=[Patch(facecolor=ps.BOARD_TINT, edgecolor=ps.BOARD_LINE, lw=0.4),
                        Patch(facecolor="none", edgecolor=OUTLINE_C, lw=OUTLINE_LW),
                        Patch(facecolor="none", edgecolor=ps.BOARD_EDIT, lw=2.0)],
               labels=["predicted mass", "legality changed", "edited tile"], loc="center", fontsize=8, handlelength=1.2)
    ps.save(fig, out_dir / "key_boards")
    plt.close(fig)


def sidecar(cols, picks, rule, seed):
    out = {"rule": rule, "seed": seed, "eligibility": "at least 3 squares change legality; window (edited tile + changed squares + 1 margin) fits 5 x 5",
           "tint": "yellow = predicted next-move probability, fully tinted at 0.02 and above, (p / 0.02)^0.6 below (draw_board defaults)",
           "outline": "thin dark outline = squares whose legality the flip changes (legal_pre XOR legal_post), the symdiff support of the Edit Index",
           "cases": {}}
    for name, run in oth.VARIANTS:
        col, i = cols[name], picks[name]
        box = C.bbox(col, i)
        out["cases"][name] = {"run": run, "instance": col["instance"], "case": i, "moves_played": int(col["lengths"][i]),
                              "edited_tile": int(col["pos"][i]), "edited_tile_rc": list(divmod(int(col["pos"][i]), 8)),
                              "changed_squares": C.symdiff(col, i), "legal_pre": sorted(col["legal_pre"][i]), "legal_post": sorted(col["legal_post"][i]),
                              "window_rc": list(box), "window_side": C.side(box),
                              "per_case_edit_index_symdiff": C.case_index(col, i),
                              "arms_drawn": {e: {"point": p, "alpha": a} for e, (p, a) in col["arms"].items()},
                              "table2": C.table2_othello(run)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=("random", "typical"), default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gs", action="store_true", help="add the GS row")
    ap.add_argument("--options", nargs="+", default=list(OPTIONS))
    ap.add_argument("--no-pieces", action="store_true")
    a = ap.parse_args()
    cols = C.othello()
    picks = C.select(cols, oth.VARIANTS, rule=a.rule, seed=a.seed)
    conds = CONDS_GS if a.gs else CONDS
    tag = ("" if a.rule == "random" else f"_{a.rule}") + ("_gs" if a.gs else "")
    for name, i in picks.items():
        print(f"  {name:<16} case {i}  changed squares {C.symdiff(cols[name], i)}  window {C.bbox(cols[name], i)}  "
              f"per-case EI {{{', '.join(f'{k} {v:+.2f}' for k, v in C.case_index(cols[name], i).items())}}}", flush=True)
    for opt in a.options:
        # 4col: all variants across (full width); 2col: Standard vs Adjacent NoFlip across (half width);
        # 2row: the same two variants down, conditions across (full width; its boards are 2col's pieces)
        for cut, width, transpose in (("4col", ps.TEXT_WIDTH_IN, False), ("2col", ps.HALF_WIDTH_IN, False), ("2row", ps.TEXT_WIDTH_IN, True)):
            names = [n for n, _ in variants(cut)]
            n_cols, n_rows = (len(conds), len(names)) if transpose else (len(names), len(conds))
            size = size_in(n_cols, n_rows, board_for_width(width, n_cols))
            fig = plt.figure(figsize=size)
            panel(fig, cols, picks, names, conds, transpose=transpose, **OPTIONS[opt])
            ps.save(fig, C.HERE / f"othello_{opt}_{cut}{tag}")
            plt.close(fig)
            if not a.no_pieces and not transpose:
                pieces(cols, picks, names, conds, C.HERE / "pieces" / f"othello_{opt}_{cut}{tag}", **OPTIONS[opt])
            print(f"→ othello_{opt}_{cut}{tag}.pdf  {size[0]:.2f} x {size[1]:.2f} in", flush=True)
    C.dump(sidecar(cols, picks, a.rule, a.seed), C.HERE / f"othello_cases_{a.rule}.json")
