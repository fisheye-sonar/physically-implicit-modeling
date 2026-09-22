"""The main-text qualitative figure (Sevan's spec, rounds 2-5), two files:

    composite_final              (a) four columns: Standard (continuous) on Examples 1 and 2 (dw-noiseless, Cartesian block),
                                 128-ray (categorical) on Example 3 (dw-128ray, appearance-fac block) and 5-ray (categorical)
                                 on the SAME Example 3 (dw-5ray); rows Context, Unedited GT, Edited GT, PI, GS, IM with
                                 signed-error strips; above (b) Othello Standard / Adjacent NoFlip (rows) x Unedited GT,
                                 Edited GT, PI, GS, IM (columns), 5 x 5 zoom, typical cases of rank RANK, cyan / pink marks.
    composite_final_sidebyside   (a) left; (b) right with Standard / Adjacent Flip / Adjacent NoFlip as columns and Unedited
                                 GT, Edited GT, PI, IM as rows (no GS), thinner rims and headers that clear each other.
    composite_final_sidebyside_single
                                 the same side-by-side with a SINGLE Standard (continuous) example in (a) (Rayworld option A4),
                                 which fits the 5.5 in text width with nothing scaled at include time.

Rows two and three are both ground truth since round 6: Unedited GT is the clean observation of the world in which the
teleport never happened (Rayworld: the scorer's own ``zones.gt_unedited``; Othello: uniform over ``legal_pre``), Edited GT
the world in which it did. Neither is a model prediction; the three rows below them are.

Every drawn scenario passes the 5-ray visibility filter (``rayworld_panel`` option A3, ``common.passing_seeds``); the
two categorical columns share the second passing seed (``rayworld_panel.CAT_SLOT``), the continuous columns the other two. The
pre-edit / post-edit key (dots) sits at the top right of each figure, above the "prediction - truth" colour bar; the figure
is 0.22 in wider than the text width to make room for it (LaTeX scales it to the column). Pieces under
pieces/composite_final{,_sidebyside}; sidecars composite_final{,_sidebyside}.json.

    .pim/bin/python paper/figs/qualitative_main/composite_final.py [--rank 2] [--no-pieces]
"""
from __future__ import annotations

import argparse

import common as C
import othello_panel as T
import rayworld_panel as R
from common import oth, plt, ps

VERSION = "A3"           # the Rayworld option (rayworld_panel.ROWS): the four-column cut
SINGLE = "A4"            # the side-by-side variant with one Standard (continuous) example
WIDTH_IN = ps.TEXT_WIDTH_IN + 0.22   # the marks key at the top right needs the room (Sevan: wider is fine, taller is not)
LETTER = 11.5            # panel letters, pt (bold)
RANK = 2                 # the typical rule's rank (1 = the case nearest the population means; Sevan asked for the next)
SIDE_BOARD_IN = 0.60     # side-by-side boards (height-bound: 4 rows + gaps + titles = the figure's 2.93 in)
SIDE_LW = 1.0            # thinner rims on the smaller boards
SIDE_TOP_IN = 0.30       # room for two-line variant titles over the narrow boards
SIDE_COL_GAP_IN = 0.16   # round 6: the gap between (b)'s variant columns, so "Adjacent Flip" / "Adjacent NoFlip" clear
SIDE_TITLE = 8           # and their headers set one point smaller (was 9 over a 0.65 in pitch: they collided)
SIDE_WIDTH = {"A3": 6.10, "A4": ps.TEXT_WIDTH_IN}    # the wider cut buys (a) four legible columns; the single fits 5.5 in
SIDE_SPACER = {"A3": 0.0, "A4": 0.18}                # group spacers only where the columns are wide enough
SIDE_EXAMPLE = {"A3": 7, "A4": 8}                    # (a)'s example titles, pt
GAP_IN = 0.16            # between (a) and (b) when stacked


def key(sf):
    """The marks key at the top right of the Rayworld subfigure, above its colour bar."""
    W, H = sf.bbox.width / sf.dpi, sf.bbox.height / sf.dpi
    oth.mark_key(sf, fontsize=7.5, markersize=4.5, loc="upper right", bbox_to_anchor=(1 - 0.02 / W, 1 - 0.02 / H), ncol=1)


def stacked(version, cols, picks, out):
    names, conds = list(T.TWO), list(T.CONDS_GS)
    b = T.board_for_width(WIDTH_IN, names, conds, transpose=True)
    wb, hb = T.size_in(names, conds, b, transpose=True)
    unit = R.unit_for_height(version, hb)                      # (a) as tall as (b): vertically balanced
    ha = R.height_in(version, unit)
    fig = plt.figure(figsize=(WIDTH_IN, ha + hb + GAP_IN))
    sa, sb = fig.subfigures(2, 1, height_ratios=[ha, hb], hspace=GAP_IN / (ha + hb))
    R.panel(sa, version, unit=unit, letter="(a)", letter_size=LETTER)
    key(sa)
    T.panel(sb, cols, picks, names, conds, zoom=True, transpose=True, letter="(b)", letter_size=LETTER)
    ps.save(fig, out)
    plt.close(fig)
    return {"width_in": WIDTH_IN, "height_in": round(ha + hb + GAP_IN, 3), "rayworld_height_in": round(ha, 3),
            "othello_height_in": round(hb, 3), "board_in": round(b, 3), "strip_unit_in": round(unit, 4)}


def side_by_side(version, cols, picks, out):
    """(a) left, (b) right. ``version`` A3 = two Standard examples (6.10 in wide), A4 = one (5.50 in)."""
    names, conds = list(T.THREE), list(T.CONDS)
    width = SIDE_WIDTH[version]
    wb, hb = T.size_in(names, conds, SIDE_BOARD_IN, top_in=SIDE_TOP_IN, col_gap_in=SIDE_COL_GAP_IN)
    unit = R.unit_for_height(version, hb, narrow=True)
    wa = width - wb
    fig = plt.figure(figsize=(width, hb))
    sa, sb = fig.subfigures(1, 2, width_ratios=[wa, wb], wspace=0.0)
    R.panel(sa, version, unit=unit, letter="(a)", letter_size=LETTER, narrow=True,
            spacer=SIDE_SPACER[version], title_size=SIDE_EXAMPLE[version])
    key(sa)
    T.panel(sb, cols, picks, names, conds, zoom=True, letter="(b)", letter_size=LETTER, lw=SIDE_LW,
            top_in=SIDE_TOP_IN, col_gap_in=SIDE_COL_GAP_IN, title_size=SIDE_TITLE)
    ps.save(fig, out)
    plt.close(fig)
    return {"width_in": width, "height_in": round(hb, 3), "rayworld_width_in": round(wa, 3),
            "othello_width_in": round(wb, 3), "board_in": SIDE_BOARD_IN, "column_gap_in": SIDE_COL_GAP_IN,
            "othello_title_pt": SIDE_TITLE, "example_title_pt": SIDE_EXAMPLE[version],
            "strip_unit_in": round(unit, 4), "mark_lw": SIDE_LW, "rayworld_option": version,
            "rayworld_columns": len(R.columns(version))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
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
    print(f"  Rayworld columns: {[(t, v, s) for t, v, s, *_ in R.columns(VERSION)]}", flush=True)
    stem = "composite_final"
    geo = stacked(VERSION, cols, picks, C.HERE / stem)
    C.dump({"version": VERSION, "geometry": geo, "rayworld": R.sidecar(VERSION),
            "othello": T.sidecar(cols, picks, "typical", 0, a.rank, names=T.TWO)}, C.HERE / f"{stem}.json")
    print(f"→ {stem}.pdf  {geo}", flush=True)
    if not a.no_pieces:
        R.pieces(VERSION, C.HERE / "pieces" / stem)                                # the strips of (a)
        T.pieces(cols, picks, list(T.TWO), list(T.CONDS_GS), C.HERE / "pieces" / stem, zoom=True, tint="all")   # the boards of (b)
    for v, stem_v in ((VERSION, "composite_final_sidebyside"), (SINGLE, "composite_final_sidebyside_single")):
        geo = side_by_side(v, cols, picks, C.HERE / stem_v)
        C.dump({"version": v, "geometry": geo, "rayworld": R.sidecar(v),
                "othello": T.sidecar(cols, picks, "typical", 0, a.rank, names=T.THREE)}, C.HERE / f"{stem_v}.json")
        print(f"→ {stem_v}.pdf  {geo}", flush=True)
    if not a.no_pieces:      # the twelve boards of the side-by-side (both variants share them); strips are composite_final's
        T.pieces(cols, picks, list(T.THREE), list(T.CONDS), C.HERE / "pieces" / "composite_final_sidebyside", zoom=True, tint="all", lw=SIDE_LW)
