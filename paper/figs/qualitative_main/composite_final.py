"""The final main-text figure (round 2, Sevan's spec), two arrangements, two data versions of the Rayworld panel:

    composite_final_<A1|A2>             (a) Standard (continuous) x 2 scenarios | Standard (categorical) x the same 2,
                                        rows Context, Unedited, Ground truth, PI, GS, IM with signed-error strips;
                                        above (b) Othello Standard / Adjacent NoFlip (rows) x Unedited, Ground truth,
                                        PI, GS, IM (columns), 5 x 5 zoom, typical cases of rank RANK, cyan / pink marks
    composite_final_sidebyside_<A1|A2>  (a) left; (b) right with the variants Standard / Adjacent Flip / Adjacent NoFlip
                                        as columns and Unedited, Ground truth, PI, IM as rows (no GS), thinner rims
    A1: Standard = dw-noiseless (the categorical IM cells are blank, no arm); A2: the 128-ray model of the ray family.

Pieces (every strip, board, key) under pieces/final_*; sidecars composite_final[_sidebyside]_<version>.json.

    .pim/bin/python paper/figs/qualitative_main/composite_final.py [--versions A1 A2] [--rank 2] [--no-pieces]
"""
from __future__ import annotations

import argparse

import common as C
import othello_panel as T
import rayworld_panel as R
from common import oth, plt, ps

LETTER = 11.5            # panel letters, pt (bold)
RANK = 2                 # the typical rule's rank (1 = the case nearest the population means; Sevan asked for the next)
SIDE_BOARD_IN = 0.64     # side-by-side boards: round 1's 0.58 in plus 10 %
SIDE_LW = 1.0            # thinner rims on the smaller boards
SIDE_TOP_IN = 0.30       # room for two-line variant titles over the narrow boards
GAP_IN = 0.16            # between (a) and (b) when stacked


def stacked(version, cols, picks, out):
    names, conds = list(T.TWO), list(T.CONDS_GS)
    b = T.board_for_width(ps.TEXT_WIDTH_IN, names, conds, transpose=True)
    wb, hb = T.size_in(names, conds, b, transpose=True)
    unit = R.unit_for_height(version, hb)                      # (a) as tall as (b): vertically balanced
    ha = R.height_in(version, unit)
    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, ha + hb + GAP_IN))
    sa, sb = fig.subfigures(2, 1, height_ratios=[ha, hb], hspace=GAP_IN / (ha + hb))
    R.panel(sa, version, unit=unit, letter="(a)", letter_size=LETTER)
    T.panel(sb, cols, picks, names, conds, zoom=True, transpose=True, letter="(b)", letter_size=LETTER)
    ps.save(fig, out)
    plt.close(fig)
    return {"width_in": ps.TEXT_WIDTH_IN, "height_in": round(ha + hb + GAP_IN, 3), "rayworld_height_in": round(ha, 3),
            "othello_height_in": round(hb, 3), "board_in": round(b, 3), "strip_unit_in": round(unit, 4)}


def side_by_side(version, cols, picks, out):
    names, conds = list(T.THREE), list(T.CONDS)
    wb, hb = T.size_in(names, conds, SIDE_BOARD_IN, top_in=SIDE_TOP_IN)
    unit = R.unit_for_height(version, hb, narrow=True)
    wa = ps.TEXT_WIDTH_IN - wb
    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, hb))
    sa, sb = fig.subfigures(1, 2, width_ratios=[wa, wb], wspace=0.0)
    R.panel(sa, version, unit=unit, letter="(a)", letter_size=LETTER, narrow=True)
    T.panel(sb, cols, picks, names, conds, zoom=True, letter="(b)", letter_size=LETTER, lw=SIDE_LW, top_in=SIDE_TOP_IN)
    ps.save(fig, out)
    plt.close(fig)
    return {"width_in": ps.TEXT_WIDTH_IN, "height_in": round(hb, 3), "rayworld_width_in": round(wa, 3),
            "othello_width_in": round(wb, 3), "board_in": SIDE_BOARD_IN, "strip_unit_in": round(unit, 4), "mark_lw": SIDE_LW}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=list(R.FINAL))
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--no-pieces", action="store_true")
    a = ap.parse_args()
    cols = C.othello()
    picks = C.select(cols, oth.VARIANTS, rule="typical", rank=a.rank)
    for name in T.THREE:
        i = picks[name]
        print(f"  {name:<16} case {i} (rank {a.rank})  marked {C.marked(cols[name], i)}  per-case EI "
              f"{{{', '.join(f'{k} {v:+.2f}' for k, v in C.case_index(cols[name], i).items())}}}  population "
              f"{{{', '.join(f'{k} {v:+.2f}' for k, v in cols[name]['ei'].items())}}}", flush=True)
    for v in a.versions:
        geo = stacked(v, cols, picks, C.HERE / f"composite_final_{v}")
        C.dump({"version": v, "geometry": geo, "rayworld": R.sidecar(v),
                "othello": T.sidecar(cols, picks, "typical", 0, a.rank, names=T.TWO)}, C.HERE / f"composite_final_{v}.json")
        print(f"→ composite_final_{v}.pdf  {geo}", flush=True)
        geo = side_by_side(v, cols, picks, C.HERE / f"composite_final_sidebyside_{v}")
        C.dump({"version": v, "geometry": geo, "rayworld": R.sidecar(v),
                "othello": T.sidecar(cols, picks, "typical", 0, a.rank, names=T.THREE)}, C.HERE / f"composite_final_sidebyside_{v}.json")
        print(f"→ composite_final_sidebyside_{v}.pdf  {geo}", flush=True)
        if not a.no_pieces:
            R.pieces(v, C.HERE / "pieces" / f"final_rayworld_{v}")
    if not a.no_pieces:
        T.pieces(cols, picks, list(T.TWO), list(T.CONDS_GS), C.HERE / "pieces" / f"final_othello_2row_rank{a.rank}", zoom=True, tint="all")
        T.pieces(cols, picks, list(T.THREE), list(T.CONDS), C.HERE / "pieces" / f"final_othello_sidebyside_rank{a.rank}", zoom=True, tint="all", lw=SIDE_LW)
