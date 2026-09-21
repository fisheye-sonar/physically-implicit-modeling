"""Main-text Rayworld panel (a): four options drawn from the appendix caches, no model loaded.

    R1  five variants (Standard, Blink, 16-ray, 8-ray, 5-ray) x PI, GS (categorical), IM      seed 0
    R2  Standard, 8-ray, 5-ray x PI, GS, GS (categorical), IM                                   seed 0
    R3  the hero: Standard only, three scenarios (seeds 0, 1, 2) x PI, GS, IM                  continuous
    R4  R3 plus a 5-ray column on scenario 1, with the GS (categorical) row (the coarse foil)

Every column: the last 8 observed frames (time downward), then single next-step frames: Unedited,
Ground truth, and each editor's write at the run's guarded best arm, with the signed error
(prediction minus truth, clipped prediction) directly beneath. Cyan / pink lines mark the edited
disc's rays before / after the edit. Output beside this script: ``rayworld_<opt>.{pdf,png,json}``
and one PDF per strip under ``pieces/rayworld_<opt>/``.

    .pim/bin/python paper/figs/qualitative_main/rayworld_panel.py [--options R1 R3] [--no-pieces]
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
LEFT_IN, RIGHT_IN, TOP_IN, BOT_IN = 0.8, 0.5, 0.2, 0.03        # gutters, inches
UNIT = {"R1": 0.12, "R2": 0.15, "R3": 0.15, "R4": 0.13}         # inches per strip unit


def edit(blk, ed, label=None):
    return ("edit", blk, ed, label or ed)


ROWS = {
    "R1": [("context",), ("unedited",), ("gt",), edit("cont", "PI"), edit("cat", "GS", "GS (categorical)"), edit("cont", "IM")],
    "R2": [("context",), ("unedited",), ("gt",), edit("cont", "PI"), edit("cont", "GS"), edit("cat", "GS", "GS (categorical)"), edit("cont", "IM")],
    "R3": [("context",), ("unedited",), ("gt",), edit("cont", "PI"), edit("cont", "GS"), edit("cont", "IM")],
    "R4": [("context",), ("unedited",), ("gt",), edit("cont", "PI"), edit("cont", "GS"), edit("cat", "GS", "GS (categorical)"), edit("cont", "IM")],
}


def columns(option):
    """(title, variant, seed) per column."""
    if option == "R1":
        return [(n, n, 0) for n in ("Standard", "Blink", "16-ray", "8-ray", "5-ray")]
    if option == "R2":
        return [(n, n, 0) for n in ("Standard", "8-ray", "5-ray")]
    if option == "R3":
        return [(f"Scenario {k + 1}", "Standard", s) for k, s in enumerate(C.HERO_SEEDS)]
    if option == "R4":
        return [(f"Standard\nscenario {k + 1}", "Standard", s) for k, s in enumerate(C.HERO_SEEDS)] + [("5-ray\nscenario 1", "5-ray", C.HERO_SEEDS[0])]
    raise ValueError(option)


def geometry(option):
    cols = columns(option)
    data = {(v, s): C.rayworld(s)["cols"][v] for _, v, s in cols}
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
        lay.append(("gap", None, BIG if kind == "gt" else GAP))
    lay.pop()
    return cols, data, lay


def height_in(option, unit):
    return unit * sum(h for *_, h in geometry(option)[2]) + TOP_IN + BOT_IN


def cell(ax, kind, spec, col, diff_scale=1.0):
    """One strip, drawn as the appendix draws it."""
    if kind == "context":
        rw._panel(ax, col["context"])
    elif kind == "unedited":
        rw._panel(ax, col["cont"]["unedited"])
    elif kind == "gt":
        rw._panel(ax, col["gt"])
    elif kind == "pred":
        rw._panel(ax, col[spec[1]][spec[2]])
    else:
        rw._panel(ax, rw.error(col[spec[1]][spec[2]], col["gt"]), diff=True, diff_scale=diff_scale)
    if kind != "context":
        for key, colr in (("ghost_x", ps.ORIGIN_C), ("target_x", ps.DEST_C)):
            x = col["cont"].get(key, float("nan"))
            if np.isfinite(x):
                ax.axvline(x, color=colr, lw=0.8, alpha=0.95)


def row_label(ax, kind, spec):
    text = spec[3] if kind == "pred" else {"context": "Context", "unedited": "Unedited", "gt": "Ground truth"}[kind]
    y = 0.5 if kind != "pred" else (1 - DIFF / STRIP) / 2            # centred on prediction + error pair
    ax.annotate(text, xy=(0, y), xycoords="axes fraction", xytext=(-4, 0), textcoords="offset points",
                ha="right", va="center", fontsize=8, color=ps.TEXT, fontweight="bold" if kind == "gt" else "normal")


def panel(F, option, *, unit, titles=True, letter=None, colorbar=True, diff_scale=1.0):
    """Draw option ``option`` into Figure / SubFigure ``F`` (already sized: width x height_in(option, unit))."""
    cols, data, lay = geometry(option)
    W, H = F.bbox.width / F.dpi, F.bbox.height / F.dpi
    gs = GridSpec(len(lay), len(cols), figure=F, height_ratios=[h for *_, h in lay], left=LEFT_IN / W,
                  right=1 - RIGHT_IN / W, top=1 - TOP_IN / H, bottom=BOT_IN / H, wspace=0.06, hspace=0.0)
    for c, (title, v, s) in enumerate(cols):
        col = data[(v, s)]
        for r, (kind, spec, _) in enumerate(lay):
            if kind == "gap":
                continue
            ax = F.add_subplot(gs[r, c])
            cell(ax, kind, spec, col, diff_scale)
            if r == 0 and titles:
                ax.set_title(title, pad=3, fontsize=9, color=ps.TEXT)
            if c == 0 and kind != "diff":
                row_label(ax, kind, spec)
    if colorbar:
        r0 = next(r for r, (k, *_) in enumerate(lay) if k == "pred")
        r1 = max(r for r, (k, *_) in enumerate(lay) if k == "diff")
        b0, b1 = gs[r0, -1].get_position(F), gs[r1, -1].get_position(F)
        cax = F.add_axes([b0.x1 + 0.07 / W, b1.y0, 0.05 / W, b0.y1 - b1.y0])
        error_key(cax, diff_scale)
    if letter:
        F.text(0.06 / W, 1 - 0.04 / H, letter, ha="left", va="top", fontsize=10, fontweight="bold", color=ps.TEXT)


def error_key(cax, diff_scale=1.0):
    cb = mpl.colorbar.ColorbarBase(cax, cmap=rw.DIFF_CMAP, norm=mpl.colors.Normalize(-diff_scale, diff_scale), orientation="vertical")
    cb.set_ticks([-diff_scale, 0, diff_scale])
    cb.ax.tick_params(labelsize=7, length=1.5, pad=1.5, width=0.5, colors=ps.TEXT)
    cb.outline.set_edgecolor(ps.FRAME)
    cb.outline.set_linewidth(0.5)
    cb.set_label("prediction − truth", fontsize=8, labelpad=2, color=ps.TEXT)


def pieces(option, out_dir, *, w_in=1.6, unit=0.16):
    """Every strip of the option as its own PDF (+ PNG), plus the two keys."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols, data, lay = geometry(option)
    for c, (title, v, s) in enumerate(cols):
        col = data[(v, s)]
        for kind, spec, h in lay:
            if kind == "gap":
                continue
            fig = plt.figure(figsize=(w_in, unit * h))
            cell(fig.add_axes([0, 0, 1, 1]), kind, spec, col)
            name = {"context": "context", "unedited": "unedited", "gt": "ground_truth"}.get(kind) \
                or f"{spec[1]}_{spec[2]}_{'error' if kind == 'diff' else 'prediction'}"
            ps.save(fig, out_dir / f"col{c + 1}_{v.replace(' ', '')}_seed{s}_{name}")
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
    rows = [{"context": "Context", "unedited": "Unedited", "gt": "Ground truth"}.get(r[0], f"{r[2]} [{r[1]}]" if r[0] == "edit" else r[0])
            for r in ROWS[option]]
    out = {"option": option, "rows": rows, "columns": []}
    for title, v, s in cols:
        col, full = data[(v, s)], C.rayworld(s)
        inst, run = C.RW_RUNS[v]
        out["columns"].append({
            "title": title.replace("\n", ", "), "variant": v, "instance": inst, "run": run, "seed": s,
            "edit_object": full["edit_object"], "origin_x": col["cont"]["ghost_x"], "destination_x": col["cont"]["target_x"],
            "changes_categorical_tile": bool(col["cat"]["changes_tile"]),
            "arms_drawn": {blk: {ed: {"point": int(p), "alpha": float(a)} for ed, (p, a) in col[blk]["arms"].items()} for blk in ("cont", "cat")},
            "table2": C.table2_rayworld(run)})
    out["selection"] = ("Scenarios are the appendix caches (seed = generator seed of the teleport case, radius-1.0 geometry, "
                        "rendered under each variant); R1 / R2 use seed 0 like the appendix figure; R3 / R4 use the first three "
                        "cached seeds whose edited disc is visible before the edit on the Standard variant (finite origin "
                        "locator): 0, 1, 2. Arms are the tables' guarded best arms (pim.metrics.selection.best_arm).")
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
        print(f"→ rayworld_{opt}.pdf ({height_in(opt, UNIT[opt]):.2f} in tall)", flush=True)
