"""Main-text Rayworld panel (a), drawn from the appendix caches (``.scratch/*_catim_*``); no model is loaded here.

    R1  five variants (Standard, Blink, 16-ray, 8-ray, 5-ray) x PI, GS (categorical), IM      seed 0
    R2  Standard, 8-ray, 5-ray x PI, GS, GS (categorical), IM                                   seed 0
    R3  Standard only, three scenarios (seeds 0, 1, 2) x PI, GS, IM                            continuous
    R4  R3 plus a 5-ray column on scenario 1, with the GS (categorical) row (the coarse foil)
    A3  the final cut (rounds 4-6, Sevan's spec), four columns: Standard (continuous) on Examples 1 and 2 (dw-noiseless,
        Cartesian block), 128-ray (categorical) on Example 3 (dw-128ray, appearance-fac block, IM = the categorical
        inverse map) and 5-ray (categorical) on the SAME Example 3 (dw-5ray); rows Context, Unedited Pred, Ground
        truth, PI, GS, IM (rows 2-3 are both ground truth since round 6). The three scenarios are the first three
        seeds passing the 5-ray visibility filter
        (``common.passing_seeds``); the categorical pair takes the second of them (``CAT_SLOT``), the continuous
        columns the other two. Rounds 2-3's A1 / A2 cuts (two scenarios x continuous | categorical, one model) were
        retired in round 4.
    A4  A3 with a SINGLE Standard (continuous) example, for composite_final_sidebyside_single.

Every column: the last 8 observed frames (time downward), then the two clean reference frames at the edit step
(Unedited GT: the world in which the teleport never happened, the scorer's own ``zones.gt_unedited``; Edited GT:
the world in which it did) and each editor's next-step write at the run's guarded best arm, with the signed error
(prediction minus Edited GT, clipped prediction) directly beneath. Neither GT row is a model output. Cyan / pink lines mark the edited disc's rays before / after
the edit. Output beside this script: ``rayworld_<opt>.{pdf,png,json}`` and one PDF per strip under
``pieces/rayworld_<opt>/``.

    .pim/bin/python paper/figs/qualitative_main/rayworld_panel.py [--options R3 A3] [--no-pieces]
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import common as C
from common import plt, ps, rw

CTX, STRIP, DIFF, GAP, BIG = 0.42, 1.0, 0.7, 0.3, 0.75        # row heights, in strip units
GAP_U, BIG_F = 0.65, 0.9        # the final cut: room for the two-line "Unedited / Pred" and "Ground / truth" labels
RIGHT_IN, RIGHT_F, BOT_IN = 0.5, 0.72, 0.03                   # gutters, inches; RIGHT_F leaves room for the marks key
SPACER = {False: 0.18, True: 0.0}    # the final cut: an empty column between model groups, in column widths (none in the
                                     # narrow side-by-side, whose 7 pt "Example k" titles need every bit of column width)
UNIT = {"R1": 0.12, "R2": 0.15, "R3": 0.15, "R4": 0.13, "A3": 0.155, "A4": 0.155}   # inches per strip unit
FINAL = ("A3", "A4")
N_CONT = {"A3": 2, "A4": 1}      # continuous Standard examples the cut draws (A4: the single-example side-by-side)
CAT_SLOT = 1     # which of the three passing seeds the two categorical columns share (Sevan, round 5: the second,
                 # whose teleport crosses the frame; the other two are the continuous examples, in seed order)
LABELS = {"context": "History", "unedited": "Unedited GT", "gt": "Edited GT"}
WRAPPED = {"unedited": "Unedited\nGT", "gt": "Edited\nGT"}          # the final cut's narrow gutter


def edit(blk, ed, label=None):
    return ("edit", blk, ed, label or ed)          # blk None: the column's own block


ROWS = {
    "R1": [("context",), ("unedited",), ("gt",), edit("cont", "PI"), edit("cat", "GS", "GS (categorical)"), edit("cont", "IM")],
    "R2": [("context",), ("unedited",), ("gt",), edit("cont", "PI"), edit("cont", "GS"), edit("cat", "GS", "GS (categorical)"), edit("cont", "IM")],
    "R3": [("context",), ("unedited",), ("gt",), edit("cont", "PI"), edit("cont", "GS"), edit("cont", "IM")],
    "R4": [("context",), ("unedited",), ("gt",), edit("cont", "PI"), edit("cont", "GS"), edit("cat", "GS", "GS (categorical)"), edit("cont", "IM")],
    "A3": [("context",), ("unedited",), ("gt",), edit(None, "PI"), edit(None, "GS"), edit(None, "IM")],
}
ROWS["A4"] = ROWS["A3"]


def left_in(option):
    return C.GUTTER_IN if option in FINAL else 0.8


def right_in(option, key=True):
    """``key`` False: the marks key is drawn outside the panel (the side-by-side puts it in the gap between
    (a) and (b)), so the wider gutter it needs is not reserved."""
    return RIGHT_F if option in FINAL and key else RIGHT_IN


def top_in(option, narrow=False):
    """The final cut carries a group title over each model's columns; ``narrow`` (the side-by-side, strips under
    0.6 in) sets it in two lines and the example titles at 7 pt."""
    if option in FINAL:
        return 0.46 if narrow else 0.36
    return 0.2


def columns(option):
    """(title, variant, seed, source, block, group) per column. block None = the row decides (R1..R4); group = the
    title over a run of consecutive columns of one model (the final cut only, else None)."""
    if option == "R1":
        return [(n, n, 0, "appendix", None, None) for n in ("Standard", "Blink", "16-ray", "8-ray", "5-ray")]
    if option == "R2":
        return [(n, n, 0, "appendix", None, None) for n in ("Standard", "8-ray", "5-ray")]
    if option == "R3":
        return [(f"Scenario {k + 1}", "Standard", s, "appendix", None, None) for k, s in enumerate(C.HERO_SEEDS)]
    if option == "R4":
        return ([(f"Standard\nscenario {k + 1}", "Standard", s, "appendix", None, None) for k, s in enumerate(C.HERO_SEEDS)]
                + [("5-ray\nscenario 1", "5-ray", C.HERO_SEEDS[0], "appendix", None, None)])
    if option in FINAL:
        seeds = C.passing_seeds(C.rw.N_MAIN)         # the first three seeds that pass the 5-ray visibility filter
        cat = seeds[CAT_SLOT]                        # the scenario the two categorical columns share
        cont = [s for k, s in enumerate(seeds) if k != CAT_SLOT][:N_CONT[option]]
        last = f"Example {len(cont) + 1}"
        return ([(f"Example {k + 1}", "Standard", s, "appendix", "cont", "Standard (continuous)")
                 for k, s in enumerate(cont)]
                + [(last, "128-ray", cat, "128ray", "cat", "128-ray (categorical)"),
                   (last, "5-ray", cat, "appendix", "cat", "5-ray (categorical)")])
    raise ValueError(option)


def runs(cols, f=lambda col: col[5]):
    """(value, first column, last column) of every run of consecutive columns on which ``f`` agrees; falsy values
    (no group) are skipped. Default ``f``: the group title."""
    out = []
    for c, col in enumerate(cols):
        v = f(col)
        if out and out[-1][0] == v:
            out[-1][2] = c
        else:
            out.append([v, c, c])
    return [tuple(r) for r in out if r[0]]


def model_of(col):
    """The model half of a group title ("Standard (continuous)" -> "Standard")."""
    return col[5] and col[5].split(" (")[0]


def block_of(col):
    """The block half of a group title ("Standard (continuous)" -> "(continuous)")."""
    return col[5] and "(" + col[5].split(" (")[1]


def grid_columns(cols, spacer):
    """The GridSpec column of every figure column and the width ratios: an empty column ``spacer`` column-widths
    wide wherever the group changes (``spacer`` 0: none)."""
    idx, widths = [], []
    for c, col in enumerate(cols):
        if c and spacer and col[5] != cols[c - 1][5]:
            widths.append(spacer)
        idx.append(len(widths))
        widths.append(1.0)
    return idx, widths


def run_of(variant, source):
    """(instance, run) behind a column."""
    return tuple(C.RAY128[1:]) if source == "128ray" else C.RW_RUNS[variant]


def geometry(option):
    cols = columns(option)
    data = {(src, v, s): C.rayworld(s, source=src)["cols"][v] for _, v, s, src, *_ in cols}
    n_ctx = next(iter(data.values()))["context"].shape[0]
    lay = []                                            # (kind, row spec, height)
    for r in ROWS[option]:
        kind = r[0]
        if kind == "context":
            lay.append(("context", r, n_ctx * CTX))
        elif kind == "edit":
            lay += [("pred", r, STRIP), ("diff", r, DIFF)]
        else:
            lay.append((kind, r, STRIP))
        final = option in FINAL
        lay.append(("gap", None, (BIG_F if final else BIG) if kind == "gt" else (GAP_U if final and kind == "unedited" else GAP)))
    lay.pop()
    return cols, data, lay


def units(option):
    return sum(h for *_, h in geometry(option)[2])


def height_in(option, unit, narrow=False):
    return unit * units(option) + top_in(option, narrow) + BOT_IN


def unit_for_height(option, h_in, narrow=False):
    return (h_in - top_in(option, narrow) - BOT_IN) / units(option)


def frame_of(col, spec, blk_col):
    """(prediction, block) for an edit row in a column; the prediction is None where the block has no arm."""
    blk = spec[1] or blk_col
    return col[blk][spec[2]], blk


def cell(ax, kind, spec, col, blk_col=None, diff_scale=1.0):
    """One strip, drawn as the appendix draws it; a blank framed cell where the editor has no arm."""
    if kind == "context":
        rw._panel(ax, col["context"])
    elif kind == "unedited":
        rw._panel(ax, col["cont"]["unedited_gt"])      # the clean UNEDITED world at the edit frame (zones.gt_unedited)
    elif kind == "gt":
        rw._panel(ax, col["gt"])
    else:
        pred, _ = frame_of(col, spec, blk_col)
        if pred is None:
            rw._blank(ax)
            return
        if kind == "pred":
            rw._panel(ax, pred)
        else:
            rw._panel(ax, rw.error(pred, col["gt"]), diff=True, diff_scale=diff_scale)
    if kind != "context":
        for key, colr in (("ghost_x", ps.ORIGIN_C), ("target_x", ps.DEST_C)):
            x = col["cont"].get(key, float("nan"))
            if np.isfinite(x):
                ax.axvline(x, color=colr, lw=0.8, alpha=0.95)


def time_label(ax, kind, n_ctx):
    """Time down the RIGHT of the three reference rows (Sevan, round 7): the history strip runs 0 to t-1, the two
    ground-truth strips are the single frame t. ⚠ The strip draws the LAST ``n_ctx`` observed frames of a longer
    warm-up, so "0" names the first row DRAWN, not sequence frame 0."""
    put = lambda y, s: ax.annotate(s, xy=(1, y), xycoords="axes fraction", xytext=(3, 0),  # noqa: E731
                                   textcoords="offset points", ha="left", va="center", fontsize=7, color=ps.TEXT)
    if kind == "context":
        put(1 - 0.5 / n_ctx, "0")
        put(0.5 / n_ctx, "t-1")
    else:
        put(0.5, "t")


def row_label(ax, kind, spec, wrap=False):
    text = spec[3] if kind == "pred" else (WRAPPED.get(kind, LABELS[kind]) if wrap else LABELS[kind])
    y = 0.5 if kind != "pred" else (1 - DIFF / STRIP) / 2            # centred on prediction + error pair
    ax.annotate(text, xy=(0, y), xycoords="axes fraction", xytext=(-4, 0), textcoords="offset points",
                ha="right", va="center", fontsize=8, color=ps.TEXT, linespacing=0.95,
                fontweight="bold" if kind == "gt" else "normal")


def panel(F, option, *, unit, titles=True, letter=None, colorbar=True, diff_scale=1.0, letter_size=10, narrow=False,
          spacer=None, title_size=None, key_gutter=True):
    """Draw option ``option`` into Figure / SubFigure ``F`` (already sized: width x height_in(option, unit, narrow)).
    ``spacer``: empty column between model groups, in column widths (default ``SPACER[narrow]``)."""
    cols, data, lay = geometry(option)
    W, H = F.bbox.width / F.dpi, F.bbox.height / F.dpi
    final = option in FINAL
    gcol, widths = grid_columns(cols, (SPACER[narrow] if spacer is None else spacer) if final else 0)
    gs = GridSpec(len(lay), len(widths), figure=F, height_ratios=[h for *_, h in lay], width_ratios=widths,
                  left=left_in(option) / W, right=1 - right_in(option, key_gutter) / W, top=1 - top_in(option, narrow) / H,
                  bottom=BOT_IN / H, wspace=0.13 if narrow else 0.06, hspace=0.0)
    for c, (title, v, s, src, blk, _) in enumerate(cols):
        col = data[(src, v, s)]
        for r, (kind, spec, _) in enumerate(lay):
            if kind == "gap":
                continue
            ax = F.add_subplot(gs[r, gcol[c]])
            cell(ax, kind, spec, col, blk, diff_scale)
            if c == len(cols) - 1 and kind in ("context", "unedited", "gt"):
                time_label(ax, kind, next(h for k, _, h in lay if k == "context") / CTX)
            if r == 0 and titles:
                # the narrow cut's column pitch is about 0.45 in: "Example 1" at 7 pt is 0.44 in and the titles touch
                shown = title.replace("Example ", "Ex. ") if narrow else title
                ax.set_title(shown, pad=3, fontsize=title_size or ((7 if narrow else 8) if final else 9), color=ps.TEXT)
            if c == 0 and kind != "diff":
                row_label(ax, kind, spec, wrap=final)
    if final and titles:                                # one group title over each model's run of columns
        y = gs[0, 0].get_position(F).y1 + (0.18 if narrow else 0.21) / H
        xc = lambda c0, c1: (gs[0, gcol[c0]].get_position(F).x0 + gs[0, gcol[c1]].get_position(F).x1) / 2  # noqa: E731
        text = lambda x, y_, s, size: F.text(x, y_, s, ha="center", va="bottom", fontsize=size, color=ps.TEXT)  # noqa: E731
        if narrow:
            # Two lines: the model over its own columns; the block qualifier once over the neighbouring groups that share
            # it ("(categorical)" at 7 pt is 0.54 in, wider than a narrow column, so it cannot sit over each single).
            for blk, c0, c1 in runs(cols, block_of):
                text(xc(c0, c1), y + 0.135 / H, blk.strip("()"), 8)   # the qualifier band, over the groups sharing it
            for name, c0, c1 in runs(cols, model_of):
                text(xc(c0, c1), y, name, 8)
        else:
            for label, c0, c1 in runs(cols):
                text(xc(c0, c1), y, label, 9)
    if colorbar:
        r0 = next(r for r, (k, *_) in enumerate(lay) if k == "pred")
        r1 = max(r for r, (k, *_) in enumerate(lay) if k == "diff")
        b0, b1 = gs[r0, -1].get_position(F), gs[r1, -1].get_position(F)
        cax = F.add_axes([b0.x1 + 0.07 / W, b1.y0, 0.05 / W, b0.y1 - b1.y0])
        error_key(cax, diff_scale)
    if letter:
        F.text(0.02 / W, 1 - 0.02 / H, letter, ha="left", va="top", fontsize=letter_size, fontweight="bold", color=ps.TEXT)


def error_key(cax, diff_scale=1.0):
    cb = mpl.colorbar.ColorbarBase(cax, cmap=rw.DIFF_CMAP, norm=mpl.colors.Normalize(-diff_scale, diff_scale), orientation="vertical")
    cb.set_ticks([-diff_scale, 0, diff_scale])
    cb.ax.tick_params(labelsize=7, length=1.5, pad=1.5, width=0.5, colors=ps.TEXT)
    cb.outline.set_edgecolor(ps.FRAME)
    cb.outline.set_linewidth(0.5)
    cb.set_label(r"prediction$_t$ − truth$_t$", fontsize=8, labelpad=2, color=ps.TEXT)


def pieces(option, out_dir, *, w_in=1.6, unit=0.16):
    """Every strip of the option as its own PDF (+ PNG), plus the two keys. A blank cell (no arm) exports nothing.
    Names: ``col<k>_<instance>_seed<s>_<row>`` (rows ``context``, ``unedited_gt``, ``edited_gt``,
    ``<block>_<editor>_{prediction,error}``)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols, data, lay = geometry(option)
    for c, (title, v, s, src, blk, _) in enumerate(cols):
        col = data[(src, v, s)]
        tag = f"col{c + 1}_{run_of(v, src)[0]}_seed{s}"
        for kind, spec, h in lay:
            if kind == "gap" or (kind in ("pred", "diff") and frame_of(col, spec, blk)[0] is None):
                continue
            fig = plt.figure(figsize=(w_in, unit * h))
            cell(fig.add_axes([0, 0, 1, 1]), kind, spec, col, blk)
            name = {"context": "context", "unedited": "unedited_gt", "gt": "edited_gt"}.get(kind) \
                or f"{spec[1] or blk}_{spec[2]}_{'error' if kind == 'diff' else 'prediction'}"
            ps.save(fig, out_dir / f"{tag}_{name}")
            plt.close(fig)
    fig = plt.figure(figsize=(0.5, 1.2))
    error_key(fig.add_axes([0.05, 0.05, 0.16, 0.9]))
    ps.save(fig, out_dir / "key_error_scale")
    plt.close(fig)
    fig = plt.figure(figsize=(1.6, 0.4))
    fig.legend(handles=[Line2D([], [], color=ps.ORIGIN_C, lw=1.2), Line2D([], [], color=ps.DEST_C, lw=1.2)],
               labels=["pre-edit", "post-edit"], loc="center", ncol=1, fontsize=8, handlelength=1.5)
    ps.save(fig, out_dir / "key_locators")
    plt.close(fig)


def sidecar(option):
    cols, data, _ = geometry(option)
    rows = [LABELS.get(r[0], f"{r[2]} [{r[1] or 'column block'}]" if r[0] == "edit" else r[0]) for r in ROWS[option]]
    out = {"option": option, "rows": rows, "columns": [], "blank_cells": []}
    for c, (title, v, s, src, blk, grp) in enumerate(cols):
        col, full = data[(src, v, s)], C.rayworld(s, source=src)
        inst, run = run_of(v, src)
        arms = {b: {ed: None if a is None else {"point": int(a[0]), "alpha": float(a[1])} for ed, a in col[b]["arms"].items()}
                for b in ("cont", "cat")}
        rays = {i: C.changed_rays(s, i) for i in dict.fromkeys((rw.FILTER_INST, inst, "dw-noiseless", "dw-128ray"))}
        deltas = list(rw.change_at(s)[1])
        out["columns"].append({
            "title": title.replace("\n", ", "), "group": grp, "variant": v, "instance": inst, "run": run, "seed": s,
            "source_cache": C.SOURCES[src].format(seed=s, context=8), "block": None if blk is None else C.DW_BLOCK[blk],
            "edit_object": full["edit_object"], "origin_x": col["cont"]["ghost_x"], "destination_x": col["cont"]["target_x"],
            "changes_categorical_tile": bool(col["cat"]["changes_tile"]),
            "n_changed_rays_5ray": len(rays[rw.FILTER_INST]), "changed_rays": rays,
            "delta_intensity_5ray": [round(x, 3) for x in deltas],
            "arms_drawn": arms, "table2": C.table2_rayworld(run)})
        if blk is not None:
            out["blank_cells"] += [f"col{c + 1} {ed}" for ed in C.EDITORS if col[blk][ed] is None]
    out["selection"] = (
        f"Scenario filter (2026-09-21, tightened in round 5): a seed is drawn only if its teleport changes at least "
        f"{rw.MIN_RAYS} rays of the clean {rw.FILTER_INST} frame at the edit frame, at least {rw.MIN_STRONG} of them by "
        f"at least {rw.MIN_DELTA} in intensity (the scorer's differing-ray zone and the gap between its two clean "
        f"reference renders under the 5-ray renderer). A3's three scenarios are the first three passing seeds; the two "
        f"categorical columns share the SECOND of them (Sevan, round 5) and are titled Example 3, the other two are the "
        f"continuous Examples 1 and 2 in seed order. R1 / R2 use seed 0, R3 / R4 seeds 0, 1, 2 (round 1, unfiltered). "
        f"Arms are the tables' guarded best arms (pim.metrics.selection.best_arm); a block with no arm for an editor is "
        f"a blank cell.")
    out["matching"] = (
        "One scenario = one world (positions, velocities, teleport) generated under the tightest geometry (radius 1.0) "
        "and rendered under each instance's own renderer (radius 0.5 / 128 rays for dw-noiseless, radius 1.0 / N rays "
        "for the ray family); positions are identical across columns, only the rendering differs. changed_rays = the "
        "rays on which the clean edited and unedited frames at the edit frame differ, per renderer.")
    out["rows_drawn"] = (
        "Context = the last 8 observed frames. Unedited GT = the clean observation of the world in which the teleport "
        "never happened, at the edit frame: the scorer's own counterfactual reference (zones.gt_unedited, step 0 of "
        "zones.gt_unedited_traj), the unedited pole arms.score compares a prediction against. Edited GT = the clean "
        "observation of the edited world at the same frame, the other pole. NEITHER is a model prediction (round 6: the "
        "second row used to be the model's unedited next frame); PI / GS / IM are. The two GT rows differ exactly on the "
        "scorer's differing-ray zone, the support of the Edit Index.")
    out["error_strip"] = "prediction (clipped to [0, 1]) minus Edited GT at the edit frame, drawn on the canonical signed-error map, fixed scale ±1"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--options", nargs="+", default=list(ROWS))
    ap.add_argument("--no-pieces", action="store_true")
    a = ap.parse_args()
    for opt in a.options:
        fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, height_in(opt, UNIT[opt])))
        panel(fig, opt, unit=UNIT[opt])
        ps.save(fig, C.HERE / f"rayworld_{opt}")
        plt.close(fig)
        C.dump(sidecar(opt), C.HERE / f"rayworld_{opt}.json")
        if not a.no_pieces:
            pieces(opt, C.HERE / "pieces" / f"rayworld_{opt}")
        print(f"→ rayworld_{opt}.pdf ({height_in(opt, UNIT[opt]):.2f} in tall)  columns {[(t, s, src) for t, _, s, src, *_ in columns(opt)]}", flush=True)
