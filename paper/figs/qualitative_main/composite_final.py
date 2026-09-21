"""The main-text qualitative figure (Sevan's spec, rounds 2-3), two files:

    composite_final              (a) Standard (continuous) x examples 1, 2 | Standard (categorical) x the same two; rows
                                 Context, Unedited Pred, Ground truth, PI, GS, IM with signed-error strips; above (b) Othello
                                 Standard / Adjacent NoFlip (rows) x Unedited Pred, Ground truth, PI, GS, IM (columns), 5 x 5
                                 zoom, typical cases of rank RANK, cyan / pink marks. Standard = the 128-ray model (A2).
    composite_final_sidebyside   (a) left; (b) right with Standard / Adjacent Flip / Adjacent NoFlip as columns and Unedited
                                 Pred, Ground truth, PI, IM as rows (no GS), thinner rims. Same model.
    (--version A1 renders composite_final_A1, the dw-noiseless alternative whose categorical IM cells are blank: no arm.
    Dropped from the deliverables in round 3; the option stays so it can be regenerated.)

The pre-edit / post-edit key (dots) sits at the top right of each figure, above the "prediction - truth" colour bar; the
figure is 0.22 in wider than the text width to make room for it (LaTeX scales it to the column). Pieces under
pieces/composite_final{,_sidebyside}; sidecars composite_final{,_sidebyside}.json.

    .pim/bin/python paper/figs/qualitative_main/composite_final.py [--rank 2] [--no-pieces] [--version A2]
"""
from __future__ import annotations

import argparse

import common as C
import othello_panel as T
import rayworld_panel as R
from common import oth, plt, ps

WIDTH_IN = ps.TEXT_WIDTH_IN + 0.22   # the marks key at the top right needs the room (Sevan: wider is fine, taller is not)
LETTER = 11.5            # panel letters, pt (bold)
RANK = 2                 # the typical rule's rank (1 = the case nearest the population means; Sevan asked for the next)
SIDE_BOARD_IN = 0.64     # side-by-side boards: round 1's 0.58 in plus 10 %
SIDE_LW = 1.0            # thinner rims on the smaller boards
SIDE_TOP_IN = 0.30       # room for two-line variant titles over the narrow boards
GAP_IN = 0.16            # between (a) and (b) when stacked
FILES = {"A2": "composite_final", "A1": "composite_final_A1"}      # A1 only on request (--version A1)


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
    names, conds = list(T.THREE), list(T.CONDS)
    wb, hb = T.size_in(names, conds, SIDE_BOARD_IN, top_in=SIDE_TOP_IN)
    unit = R.unit_for_height(version, hb, narrow=True)
    wa = WIDTH_IN - wb
    fig = plt.figure(figsize=(WIDTH_IN, hb))
    sa, sb = fig.subfigures(1, 2, width_ratios=[wa, wb], wspace=0.0)
    R.panel(sa, version, unit=unit, letter="(a)", letter_size=LETTER, narrow=True)
    key(sa)
    T.panel(sb, cols, picks, names, conds, zoom=True, letter="(b)", letter_size=LETTER, lw=SIDE_LW, top_in=SIDE_TOP_IN)
    ps.save(fig, out)
    plt.close(fig)
    return {"width_in": WIDTH_IN, "height_in": round(hb, 3), "rayworld_width_in": round(wa, 3),
            "othello_width_in": round(wb, 3), "board_in": SIDE_BOARD_IN, "strip_unit_in": round(unit, 4), "mark_lw": SIDE_LW}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--no-pieces", action="store_true")
    ap.add_argument("--version", choices=list(FILES), default="A2", help="the Rayworld model: A2 = 128-ray (the deliverable)")
    a = ap.parse_args()
    cols = C.othello()
    picks = C.select(cols, oth.VARIANTS, rule="typical", rank=a.rank)
    for name in T.THREE:
        i = picks[name]
        print(f"  {name:<16} case {i} (rank {a.rank})  marked {C.marked(cols[name], i)}  per-case EI "
              f"{{{', '.join(f'{k} {v:+.2f}' for k, v in C.case_index(cols[name], i).items())}}}  population "
              f"{{{', '.join(f'{k} {v:+.2f}' for k, v in cols[name]['ei'].items())}}}", flush=True)
    v, stem = a.version, FILES[a.version]
    geo = stacked(v, cols, picks, C.HERE / stem)
    C.dump({"version": v, "geometry": geo, "rayworld": R.sidecar(v),
            "othello": T.sidecar(cols, picks, "typical", 0, a.rank, names=T.TWO)}, C.HERE / f"{stem}.json")
    print(f"→ {stem}.pdf  {geo}", flush=True)
    if not a.no_pieces:
        R.pieces(v, C.HERE / "pieces" / stem)                                # the strips of (a)
        T.pieces(cols, picks, list(T.TWO), list(T.CONDS_GS), C.HERE / "pieces" / stem, zoom=True, tint="all")   # the boards of (b)
    if v == "A2":
        geo = side_by_side(v, cols, picks, C.HERE / "composite_final_sidebyside")
        C.dump({"version": v, "geometry": geo, "rayworld": R.sidecar(v),
                "othello": T.sidecar(cols, picks, "typical", 0, a.rank, names=T.THREE)}, C.HERE / "composite_final_sidebyside.json")
        print(f"→ composite_final_sidebyside.pdf  {geo}", flush=True)
        if not a.no_pieces:      # its twelve boards; its strips are composite_final's
            T.pieces(cols, picks, list(T.THREE), list(T.CONDS), C.HERE / "pieces" / "composite_final_sidebyside", zoom=True, tint="all", lw=SIDE_LW)
