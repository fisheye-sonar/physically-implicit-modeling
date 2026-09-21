"""Main-text Othello panel (b): drawn from the appendix cache, no model loaded.

    T1  full boards, yellow tint by predicted mass as in the appendix
    T2  full boards, tint ONLY the squares whose legality the flip changes (everything else plain)
    T3  zoom: every board of a variant cropped to the same S x S window around {flipped tile} + changed squares
        (one-square margin)

Marking (Sevan, 2026-09-21, the appendix's ``mark``): on the Unedited board the flipped tile and every square
whose legality the flip switches are outlined in cyan; on every other board the same squares in pink. Nothing
else is outlined. Rows / columns: variants and conditions (Unedited, Ground truth, PI, [GS,] IM); the gap
before the PI condition is a touch wider than the others. Cases: one per variant by ``common.select`` (at
least 3 changed squares, window fits 5 x 5, then random with --seed; ``--rule typical --rank k`` takes the
k-th eligible case nearest the population means). Output beside this script: ``othello_<opt>_<cut><tag>.{pdf,png}``,
``othello_cases_<rule>[_r<rank>].json``, one PDF per board under ``pieces/othello_<opt>_<cut><tag>/``.

    .pim/bin/python paper/figs/qualitative_main/othello_panel.py [--rule random|typical] [--rank 1] [--seed 0] [--gs]
"""
from __future__ import annotations

import argparse

import numpy as np
from matplotlib.patches import Patch, Rectangle

import common as C
from common import oth, plt, ps

CONDS = ("Unedited", "Ground truth", "PI", "IM")
CONDS_GS = ("Unedited", "Ground truth", "PI", "GS", "IM")
TWO = ("Standard", "Adjacent NoFlip")
THREE = ("Standard", "Adjacent Flip", "Adjacent NoFlip")
GAP_IN, PI_GAP_IN, TOP_IN, PAD_IN, KEY_IN = 0.05, 0.10, 0.18, 0.03, 0.17      # inches
S_MIN = 5                                    # the zoom is 5 x 5 (T3; the eligibility cap) even when a case needs less
LEFT_IN = C.GUTTER_IN
OPTIONS = {"T1": dict(zoom=False, tint="all"), "T2": dict(zoom=False, tint="symdiff"), "T3": dict(zoom=True, tint="all")}
DISPLAY = {"Unedited": "Unedited Pred"}                                        # condition key -> label
WRAPPED = {"Unedited": "Unedited\nPred", "Ground truth": "Ground\ntruth"}      # row labels in the narrow gutter


def variants(cut):
    return list(oth.VARIANTS) if cut == "4col" else [(n, C.OTH_RUNS[n]) for n in TWO]


def gaps(items, along_conds):
    """The gap after each item but the last; the Ground truth -> PI step is a touch wider than the rest."""
    return [PI_GAP_IN if along_conds and items[k + 1] == "PI" else GAP_IN for k in range(len(items) - 1)]


def axes_of(names, conds, transpose):
    """(columns, rows, column gaps, row gaps) of a layout: variants across (default) or down (``transpose``)."""
    columns, rows = (conds, names) if transpose else (names, conds)
    return columns, rows, gaps(columns, transpose), gaps(rows, not transpose)


def size_in(names, conds, board_in, *, transpose=False, key=False, top_in=TOP_IN):
    columns, rows, gx, gy = axes_of(names, conds, transpose)
    return (LEFT_IN + len(columns) * board_in + sum(gx) + PAD_IN,
            top_in + len(rows) * board_in + sum(gy) + PAD_IN + (KEY_IN if key else 0.0))


def board_for_width(width_in, names, conds, *, transpose=False):
    columns, _, gx, _ = axes_of(names, conds, transpose)
    return (width_in - LEFT_IN - PAD_IN - sum(gx)) / len(columns)


def windows(cols, picks, names):
    """The common S x S crop per variant: S = the largest window the picked cases need, at least ``S_MIN``."""
    S = max([S_MIN] + [C.side(C.bbox(cols[n], picks[n])) for n in names])
    return {n: C.square(C.bbox(cols[n], picks[n]), S) for n in names}


def board(ax, col, i, cond, *, tint="all", marks=True, lw=oth.MARK_LW, window=None):
    """One board as ``draw_board`` draws it, the cyan / pink marks, and the crop."""
    b = col["board_pre"] if cond == "Unedited" else col["board_post"]
    p = np.asarray(col["probs"][cond][i], float)
    if tint == "symdiff":
        m = np.zeros(64)
        m[C.symdiff(col, i)] = 1.0
        p = p * m
    oth.draw_board(ax, b[i], p, int(col["pos"][i]), locator=False)
    if marks:
        oth.mark(ax, C.marked(col, i), oth.mark_color(cond), lw=lw)
    if window is not None:
        r0, r1, c0, c1 = window
        ax.set_xlim(c0, c1 + 1)
        ax.set_ylim(7 - r1, 8 - r0)


def panel(F, cols, picks, names, conds, *, zoom=False, tint="all", titles=True, letter=None, transpose=False,
          lw=oth.MARK_LW, key=False, letter_size=10, top_in=TOP_IN):
    """Draw into Figure / SubFigure ``F`` (sized by ``size_in`` with the same arguments). Default: variants
    across, conditions down; ``transpose``: variants down, conditions across. ``key``: the marks key in the
    bottom-right corner (the composites draw it at the top right of the figure instead)."""
    W, H = F.bbox.width / F.dpi, F.bbox.height / F.dpi
    columns, rows, gx, gy = axes_of(names, conds, transpose)
    b = (W - LEFT_IN - PAD_IN - sum(gx)) / len(columns)                      # board side, inches
    x0 = [LEFT_IN + k * b + sum(gx[:k]) for k in range(len(columns))]
    y0 = [H - top_in - (k + 1) * b - sum(gy[:k]) for k in range(len(rows))]
    win = windows(cols, picks, names) if zoom else {}
    wrap = lambda t: t.replace(" ", "\n") if t in names and " " in t else t   # noqa: E731  (two-line variant names)
    for c, cname in enumerate(columns):
        for r, rname in enumerate(rows):
            name, cond = (rname, cname) if transpose else (cname, rname)
            ax = F.add_axes([x0[c] / W, y0[r] / H, b / W, b / H])
            board(ax, cols[name], picks[name], cond, tint=tint, lw=lw, window=win.get(name))
            if r == 0 and titles:
                ax.set_title(wrap(cname) if b < 0.9 else DISPLAY.get(cname, cname), pad=3, fontsize=9, color=ps.TEXT,
                             linespacing=0.95, fontweight="bold" if cname == "Ground truth" else "normal")
            if c == 0:
                ax.annotate(WRAPPED.get(rname, wrap(rname)), xy=(0, 0.5), xycoords="axes fraction",
                            xytext=(-4, 0), textcoords="offset points", ha="right", va="center", fontsize=8, color=ps.TEXT,
                            linespacing=0.95, fontweight="bold" if rname == "Ground truth" else "normal")
    if key:
        oth.mark_key(F, fontsize=7.5, markersize=4.5, loc="lower right", bbox_to_anchor=(1 - PAD_IN / W, 0.0), ncol=2)
    if letter:
        F.text(0.02 / W, 1 - 0.02 / H, letter, ha="left", va="top", fontsize=letter_size, fontweight="bold", color=ps.TEXT)
    return win


def thumbnail(ax, col, i, window):
    """A faded full board with the crop rectangle."""
    board(ax, col, i, "Ground truth", marks=False)
    ax.add_patch(Rectangle((0, 0), 8, 8, facecolor="white", alpha=0.55, zorder=6))
    r0, r1, c0, c1 = window
    ax.add_patch(Rectangle((c0, 7 - r1), c1 - c0 + 1, r1 - r0 + 1, fill=False, edgecolor=ps.TEXT, linewidth=1.0, zorder=7))


def pieces(cols, picks, names, conds, out_dir, *, zoom, tint, lw=oth.MARK_LW, board_in=1.4):
    out_dir.mkdir(parents=True, exist_ok=True)
    win = windows(cols, picks, names) if zoom else {}
    for name in names:
        for cond in conds:
            fig = plt.figure(figsize=(board_in, board_in))
            board(fig.add_axes([0, 0, 1, 1]), cols[name], picks[name], cond, tint=tint, lw=lw * board_in / 0.95,
                  window=win.get(name))
            ps.save(fig, out_dir / f"{name.replace(' ', '')}_case{picks[name]}_{cond.replace(' ', '_').lower()}")
            plt.close(fig)
        if zoom:
            fig = plt.figure(figsize=(0.7, 0.7))
            thumbnail(fig.add_axes([0, 0, 1, 1]), cols[name], picks[name], win[name])
            ps.save(fig, out_dir / f"{name.replace(' ', '')}_case{picks[name]}_thumbnail")
            plt.close(fig)
    fig = plt.figure(figsize=(1.2, 0.5))
    oth.mark_key(fig, fontsize=8, markersize=5, loc="center", ncol=1)
    ps.save(fig, out_dir / "key_marks")
    plt.close(fig)
    fig = plt.figure(figsize=(1.4, 0.3))
    fig.legend(handles=[Patch(facecolor=ps.BOARD_TINT, edgecolor=ps.BOARD_LINE, lw=0.4)], labels=["predicted mass"],
               loc="center", fontsize=8, handlelength=1.0, handleheight=1.0)
    ps.save(fig, out_dir / "key_tint")
    plt.close(fig)


def sidecar(cols, picks, rule, seed, rank=1, names=None):
    out = {"rule": rule, "seed": seed, "rank": rank,
           "eligibility": "at least 3 squares change legality; window (flipped tile + changed squares + 1 margin) fits 5 x 5",
           "tint": "yellow = predicted next-move probability, fully tinted at 0.02 and above, (p / 0.02)^0.6 below (draw_board defaults)",
           "marks": "outlined squares = the flipped tile + every square whose legality the flip switches (legal_pre XOR legal_post, "
                    "the squares the symmetric-difference Edit Index scores): cyan on the pre-edit (Unedited) board, pink on every "
                    "post-edit board; nothing else is outlined",
           "cases": {}}
    for name, run in oth.VARIANTS:
        if names is not None and name not in names:
            continue
        col, i = cols[name], picks[name]
        box = C.bbox(col, i)
        out["cases"][name] = {"run": run, "instance": col["instance"], "case": i, "moves_played": int(col["lengths"][i]),
                              "edited_tile": int(col["pos"][i]), "edited_tile_rc": list(divmod(int(col["pos"][i]), 8)),
                              "changed_squares": C.symdiff(col, i), "marked_squares": C.marked(col, i),
                              "legal_pre": sorted(col["legal_pre"][i]), "legal_post": sorted(col["legal_post"][i]),
                              "window_rc": list(box), "window_side": C.side(box), "n_eligible": len(C.eligible(col)),
                              "per_case_edit_index_symdiff": C.case_index(col, i),
                              "population_mean_edit_index_symdiff": {e: round(float(x), 3) for e, x in col["ei"].items()},
                              "arms_drawn": {e: {"point": p, "alpha": a} for e, (p, a) in col["arms"].items()},
                              "table2": C.table2_othello(run)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=("random", "typical"), default="random")
    ap.add_argument("--rank", type=int, default=1, help="typical rule: the k-th eligible case nearest the population means")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gs", action="store_true", help="add the GS condition")
    ap.add_argument("--options", nargs="+", default=list(OPTIONS))
    ap.add_argument("--no-pieces", action="store_true")
    a = ap.parse_args()
    cols = C.othello()
    picks = C.select(cols, oth.VARIANTS, rule=a.rule, seed=a.seed, rank=a.rank)
    conds = CONDS_GS if a.gs else CONDS
    rtag = ("" if a.rule == "random" else f"_{a.rule}") + (f"_r{a.rank}" if a.rank != 1 else "")
    tag = rtag + ("_gs" if a.gs else "")
    for name, i in picks.items():
        print(f"  {name:<16} case {i}  marked squares {C.marked(cols[name], i)}  window {C.bbox(cols[name], i)}  "
              f"per-case EI {{{', '.join(f'{k} {v:+.2f}' for k, v in C.case_index(cols[name], i).items())}}}  "
              f"population {{{', '.join(f'{k} {v:+.2f}' for k, v in cols[name]['ei'].items())}}}", flush=True)
    for opt in a.options:
        # 4col: all variants across (full width); 2col: Standard vs Adjacent NoFlip across (half width);
        # 2row: the same two variants down, conditions across (full width; its boards are 2col's pieces)
        for cut, width, transpose in (("4col", ps.TEXT_WIDTH_IN, False), ("2col", ps.HALF_WIDTH_IN, False), ("2row", ps.TEXT_WIDTH_IN, True)):
            names = [n for n, _ in variants(cut)]
            size = size_in(names, conds, board_for_width(width, names, conds, transpose=transpose), transpose=transpose, key=True)
            fig = plt.figure(figsize=size)
            panel(fig, cols, picks, names, conds, transpose=transpose, key=True, **OPTIONS[opt])
            ps.save(fig, C.HERE / f"othello_{opt}_{cut}{tag}")
            plt.close(fig)
            if not a.no_pieces and not transpose:
                pieces(cols, picks, names, conds, C.HERE / "pieces" / f"othello_{opt}_{cut}{tag}", **OPTIONS[opt])
            print(f"→ othello_{opt}_{cut}{tag}.pdf  {size[0]:.2f} x {size[1]:.2f} in", flush=True)
    C.dump(sidecar(cols, picks, a.rule, a.seed, a.rank), C.HERE / f"othello_cases{rtag or '_random'}.json")
