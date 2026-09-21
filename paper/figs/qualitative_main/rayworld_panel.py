"""Main-text Rayworld panel (a), drawn from the appendix caches (``.scratch/*_catim_*``); no model is loaded here.

    R1  five variants (Standard, Blink, 16-ray, 8-ray, 5-ray) x PI, GS (categorical), IM      seed 0
    R2  Standard, 8-ray, 5-ray x PI, GS, GS (categorical), IM                                   seed 0
    R3  Standard only, three scenarios (seeds 0, 1, 2) x PI, GS, IM                            continuous
    R4  R3 plus a 5-ray column on scenario 1, with the GS (categorical) row (the coarse foil)
    A1  the final cut (round 2): Standard (continuous) x two scenarios | Standard (categorical) x the SAME two;
        rows Context, Unedited, Ground truth, PI, GS, IM. Standard = dw-noiseless, whose categorical block
        carries no IM arm: those two cells are blank (thin frame, nothing drawn), as in the table.
    A2  the same cut on the 128-ray model of the ray family (dw-128ray, radius 1.0): every cell a scored arm,
        the categorical IM through the categorical inverse map.

Every column: the last 8 observed frames (time downward), then single next-step frames: Unedited, Ground
truth, and each editor's write at the run's guarded best arm, with the signed error (prediction minus
truth, clipped prediction) directly beneath. Cyan / pink lines mark the edited disc's rays before / after
the edit. Output beside this script: ``rayworld_<opt>.{pdf,png,json}`` and one PDF per strip under
``pieces/rayworld_<opt>/``.

    .pim/bin/python paper/figs/qualitative_main/rayworld_panel.py [--options R3 A1 A2] [--no-pieces]
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
UNIT = {"R1": 0.12, "R2": 0.15, "R3": 0.15, "R4": 0.13, "A1": 0.155, "A2": 0.155}   # inches per strip unit
FINAL = ("A1", "A2")
SOURCE = {"A1": "appendix", "A2": "128ray"}
GROUP = {"cont": "Standard (continuous)", "cat": "Standard (categorical)"}
LABELS = {"context": "Context", "unedited": "Unedited Pred", "gt": "Ground truth"}
WRAPPED = {"unedited": "Unedited\nPred", "gt": "Ground\ntruth"}      # the final cut's narrow gutter


def edit(blk, ed, label=None):
    return ("edit", blk, ed, label or ed)          # blk None: the column's own block


ROWS = {
    "R1": [("context",), ("unedited",), ("gt",), edit("cont", "PI"), edit("cat", "GS", "GS (categorical)"), edit("cont", "IM")],
    "R2": [("context",), ("unedited",), ("gt",), edit("cont", "PI"), edit("cont", "GS"), edit("cat", "GS", "GS (categorical)"), edit("cont", "IM")],
    "R3": [("context",), ("unedited",), ("gt",), edit("cont", "PI"), edit("cont", "GS"), edit("cont", "IM")],
    "R4": [("context",), ("unedited",), ("gt",), edit("cont", "PI"), edit("cont", "GS"), edit("cat", "GS", "GS (categorical)"), edit("cont", "IM")],
    "A1": [("context",), ("unedited",), ("gt",), edit(None, "PI"), edit(None, "GS"), edit(None, "IM")],
}
ROWS["A2"] = ROWS["A1"]


def left_in(option):
    return C.GUTTER_IN if option in FINAL else 0.8


def right_in(option):
    return RIGHT_F if option in FINAL else RIGHT_IN


def top_in(option, narrow=False):
    """The final cut carries a group title over each pair of columns; ``narrow`` (the side-by-side, strips under
    0.6 in) sets it in two lines and the scenario titles at 7 pt."""
    if option in FINAL:
        return 0.46 if narrow else 0.36
    return 0.2


def final_seeds(source, n=2, seeds=range(6)):
    """The first ``n`` cached seeds whose edited disc is visible before the edit (finite origin locator) AND whose
    teleport changes a categorical tile, both judged on the drawn model."""
    out = []
    for s in seeds:
        try:
            col = C.rayworld(s, source=source)["cols"]["Standard"]
        except FileNotFoundError:
            continue
        if np.isfinite(col["cont"]["ghost_x"]) and col["cat"]["changes_tile"]:
            out.append(s)
    return tuple(out[:n])


def columns(option):
    """(title, variant, seed, source, block) per column; block None = the row decides (R1..R4)."""
    if option == "R1":
        return [(n, n, 0, "appendix", None) for n in ("Standard", "Blink", "16-ray", "8-ray", "5-ray")]
    if option == "R2":
        return [(n, n, 0, "appendix", None) for n in ("Standard", "8-ray", "5-ray")]
    if option == "R3":
        return [(f"Scenario {k + 1}", "Standard", s, "appendix", None) for k, s in enumerate(C.HERO_SEEDS)]
    if option == "R4":
        return ([(f"Standard\nscenario {k + 1}", "Standard", s, "appendix", None) for k, s in enumerate(C.HERO_SEEDS)]
                + [("5-ray\nscenario 1", "5-ray", C.HERO_SEEDS[0], "appendix", None)])
    if option in FINAL:
        src = SOURCE[option]
        return [(f"Example {k + 1}", "Standard", s, src, blk) for blk in ("cont", "cat") for k, s in enumerate(final_seeds(src))]
    raise ValueError(option)


def run_of(variant, source):
    """(instance, run) behind a column."""
    return tuple(C.A2_VARIANT[1:]) if source == "128ray" else C.RW_RUNS[variant]


def geometry(option):
    cols = columns(option)
    data = {(src, v, s): C.rayworld(s, source=src)["cols"][v] for _, v, s, src, _ in cols}
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
        rw._panel(ax, col["cont"]["unedited"])
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


def row_label(ax, kind, spec, wrap=False):
    text = spec[3] if kind == "pred" else (WRAPPED.get(kind, LABELS[kind]) if wrap else LABELS[kind])
    y = 0.5 if kind != "pred" else (1 - DIFF / STRIP) / 2            # centred on prediction + error pair
    ax.annotate(text, xy=(0, y), xycoords="axes fraction", xytext=(-4, 0), textcoords="offset points",
                ha="right", va="center", fontsize=8, color=ps.TEXT, linespacing=0.95,
                fontweight="bold" if kind == "gt" else "normal")


def panel(F, option, *, unit, titles=True, letter=None, colorbar=True, diff_scale=1.0, letter_size=10, narrow=False):
    """Draw option ``option`` into Figure / SubFigure ``F`` (already sized: width x height_in(option, unit, narrow))."""
    cols, data, lay = geometry(option)
    W, H = F.bbox.width / F.dpi, F.bbox.height / F.dpi
    final = option in FINAL
    gs = GridSpec(len(lay), len(cols), figure=F, height_ratios=[h for *_, h in lay], left=left_in(option) / W,
                  right=1 - right_in(option) / W, top=1 - top_in(option, narrow) / H, bottom=BOT_IN / H,
                  wspace=0.13 if narrow else 0.06, hspace=0.0)
    for c, (title, v, s, src, blk) in enumerate(cols):
        col = data[(src, v, s)]
        for r, (kind, spec, _) in enumerate(lay):
            if kind == "gap":
                continue
            ax = F.add_subplot(gs[r, c])
            cell(ax, kind, spec, col, blk, diff_scale)
            if r == 0 and titles:
                ax.set_title(title, pad=3, fontsize=(7 if narrow else 8) if final else 9, color=ps.TEXT)
            if c == 0 and kind != "diff":
                row_label(ax, kind, spec, wrap=final)
    if final and titles:                                # one group title over each pair of scenario columns
        y = gs[0, 0].get_position(F).y1 + (0.18 if narrow else 0.21) / H
        for k, blk in enumerate(("cont", "cat")):
            x = (gs[0, 2 * k].get_position(F).x0 + gs[0, 2 * k + 1].get_position(F).x1) / 2
            F.text(x, y, GROUP[blk].replace(" (", "\n(") if narrow else GROUP[blk], ha="center", va="bottom",
                   fontsize=8 if narrow else 9, linespacing=0.95, color=ps.TEXT)
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
    cb.set_label("prediction − truth", fontsize=8, labelpad=2, color=ps.TEXT)


def pieces(option, out_dir, *, w_in=1.6, unit=0.16):
    """Every strip of the option as its own PDF (+ PNG), plus the two keys. A blank cell (no arm) exports nothing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols, data, lay = geometry(option)
    for c, (title, v, s, src, blk) in enumerate(cols):
        col = data[(src, v, s)]
        for kind, spec, h in lay:
            if kind == "gap" or (kind in ("pred", "diff") and frame_of(col, spec, blk)[0] is None):
                continue
            fig = plt.figure(figsize=(w_in, unit * h))
            cell(fig.add_axes([0, 0, 1, 1]), kind, spec, col, blk)
            name = {"context": "context", "unedited": "unedited", "gt": "ground_truth"}.get(kind) \
                or f"{spec[1] or blk}_{spec[2]}_{'error' if kind == 'diff' else 'prediction'}"
            tag = f"col{c + 1}_{v.replace(' ', '')}{'-128ray' if src == '128ray' else ''}_seed{s}"
            ps.save(fig, out_dir / f"{tag}_{name}")
            plt.close(fig)
    fig = plt.figure(figsize=(0.5, 1.2))
    error_key(fig.add_axes([0.05, 0.05, 0.16, 0.9]))
    ps.save(fig, out_dir / "key_error_scale")
    plt.close(fig)
    fig = plt.figure(figsize=(1.6, 0.4))
    fig.legend(handles=[Line2D([], [], color=ps.ORIGIN_C, lw=1.2), Line2D([], [], color=ps.DEST_C, lw=1.2)],
               labels=["disc before edit", "disc after edit"], loc="center", ncol=1, fontsize=8, handlelength=1.5)
    ps.save(fig, out_dir / "key_locators")
    plt.close(fig)


def sidecar(option):
    cols, data, _ = geometry(option)
    rows = [LABELS.get(r[0], f"{r[2]} [{r[1] or 'column block'}]" if r[0] == "edit" else r[0]) for r in ROWS[option]]
    out = {"option": option, "rows": rows, "columns": [], "blank_cells": []}
    for c, (title, v, s, src, blk) in enumerate(cols):
        col, full = data[(src, v, s)], C.rayworld(s, source=src)
        inst, run = run_of(v, src)
        arms = {b: {ed: None if a is None else {"point": int(a[0]), "alpha": float(a[1])} for ed, a in col[b]["arms"].items()}
                for b in ("cont", "cat")}
        out["columns"].append({
            "title": title.replace("\n", ", "), "group": GROUP.get(blk), "variant": v, "instance": inst, "run": run, "seed": s,
            "source_cache": C.SOURCES[src].format(seed=s, context=8), "block": None if blk is None else C.DW_BLOCK[blk],
            "edit_object": full["edit_object"], "origin_x": col["cont"]["ghost_x"], "destination_x": col["cont"]["target_x"],
            "changes_categorical_tile": bool(col["cat"]["changes_tile"]), "arms_drawn": arms, "table2": C.table2_rayworld(run)})
        if blk is not None:
            out["blank_cells"] += [f"col{c + 1} {ed}" for ed in C.EDITORS if col[blk][ed] is None]
    out["selection"] = (
        "Scenarios are the appendix caches (seed = generator seed of the teleport case, radius-1.0 geometry, rendered "
        "under each variant); R1 / R2 use seed 0 like the appendix figure; R3 / R4 the first three cached seeds whose "
        "edited disc is visible before the edit on Standard (0, 1, 2); A1 / A2 the first two cached seeds whose edited "
        "disc is visible before the edit AND whose teleport changes a categorical tile, on the drawn model. Arms are "
        "the tables' guarded best arms (pim.metrics.selection.best_arm); a block with no arm for an editor is a blank cell.")
    out["error_strip"] = "prediction (clipped to [0, 1]) minus clean ground truth at the edit frame, drawn on the canonical signed-error map, fixed scale ±1"
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
        print(f"→ rayworld_{opt}.pdf ({height_in(opt, UNIT[opt]):.2f} in tall)  columns {[(t, s, src) for t, _, s, src, _ in columns(opt)]}", flush=True)
