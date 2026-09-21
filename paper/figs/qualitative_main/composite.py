"""The composed main-text figure: (a) a Rayworld option above (b) an Othello option, panel letters only.
Also a side-by-side arrangement. Both panels are drawn by the panel scripts into SubFigures.

    .pim/bin/python paper/figs/qualitative_main/composite.py [--rayworld R3] [--othello T3] [--cut 4col] [--rule random]
"""
from __future__ import annotations

import argparse

import common as C
import othello_panel as T
import rayworld_panel as R
from common import oth, plt, ps


def stacked(cols, picks, r_opt, t_opt, names, conds, out, *, transpose=False, board=None):
    """(a) above (b) at full width. ``transpose``: (b) with the variants down and the conditions across."""
    n_cols, n_rows = (len(conds), len(names)) if transpose else (len(names), len(conds))
    ha = R.height_in(r_opt, R.UNIT[r_opt])
    wb, hb = T.size_in(n_cols, n_rows, board or T.board_for_width(ps.TEXT_WIDTH_IN, n_cols))
    gap = 0.12
    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, ha + hb + gap))
    sa, sb = fig.subfigures(2, 1, height_ratios=[ha, hb], hspace=gap / (ha + hb))
    R.panel(sa, r_opt, unit=R.UNIT[r_opt], letter="(a)")
    T.panel(sb, cols, picks, names, conds, letter="(b)", transpose=transpose, **T.OPTIONS[t_opt])
    ps.save(fig, out)
    plt.close(fig)
    return ha + hb + gap


def side_by_side(cols, picks, r_opt, t_opt, conds, out, *, unit=0.2):
    """(a) left, (b) the two-column cut right, both the same height."""
    names = list(T.TWO)
    h = R.height_in(r_opt, unit)
    b = (h - T.TOP_IN - T.PAD_IN - (len(conds) - 1) * T.GAP_IN) / len(conds)
    wb = T.size_in(len(names), len(conds), b)[0]
    wa = ps.TEXT_WIDTH_IN - wb
    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, h))
    sa, sb = fig.subfigures(1, 2, width_ratios=[wa, wb], wspace=0.0)
    R.panel(sa, r_opt, unit=unit, letter="(a)")
    T.panel(sb, cols, picks, names, conds, letter="(b)", **T.OPTIONS[t_opt])
    ps.save(fig, out)
    plt.close(fig)
    return wa, wb, b


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rayworld", default="R3")
    ap.add_argument("--othello", default="T3")
    ap.add_argument("--rule", choices=("random", "typical"), default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gs", action="store_true")
    a = ap.parse_args()
    cols = C.othello()
    picks = C.select(cols, oth.VARIANTS, rule=a.rule, seed=a.seed)
    conds = T.CONDS_GS if a.gs else T.CONDS
    tag = f"{a.rayworld}_{a.othello}" + ("" if a.rule == "random" else f"_{a.rule}") + ("_gs" if a.gs else "")
    all4, two = [n for n, _ in oth.VARIANTS], list(T.TWO)
    h = stacked(cols, picks, a.rayworld, a.othello, all4, conds, C.HERE / f"composite_{tag}_4col")
    print(f"→ composite_{tag}_4col.pdf  5.50 x {h:.2f} in   (b) all four variants across, conditions down")
    h = stacked(cols, picks, a.rayworld, a.othello, two, conds, C.HERE / f"composite_{tag}_2row", transpose=True)
    print(f"→ composite_{tag}_2row.pdf  5.50 x {h:.2f} in   (b) Standard / Adjacent NoFlip down, conditions across")
    wa, wb, b = side_by_side(cols, picks, a.rayworld, a.othello, conds, C.HERE / f"composite_{tag}_sidebyside")
    print(f"→ composite_{tag}_sidebyside.pdf  (a) {wa:.2f} in wide, (b) {wb:.2f} in wide, boards {b:.2f} in")
