"""Edit Index against ray count with the seed spread (Results, beside the editability table, or appendix).

Numbers come from the tables' own reader, ``pim.figures.tables.collect``, so they are the paper's. Per (run,
block, editor): the parent run's reported arm (``pim.metrics.selection.best_arm`` — the best Edit Index inside
the fidelity guard; ``guarded`` False when no arm passes) and, through ``F.rep_sd``
(``pim.metrics.replicates.pool_replicates``), the seed replicates' pooled mean and SD (n − 1) at a matched
training budget. Plotted: the pooled mean with ± SD bars where n > 1, else the parent's value; the parent's
reported arm outside the guard is drawn hollow. "continuous" = the ``cartesian`` block, "categorical" = the
``appearance-fac`` block (its IM is the categorical inverse map). Nothing is computed here.

    .pim/bin/python paper/figs/editability_trends/by_rays.py         # by_rays (two panels), by_rays_half, pieces/, values
    .pim/bin/python paper/figs/editability_trends/by_rays.py --all   # also, under extra/: the one-panel full-width form
                                                                     # and the two-panel form without ND
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))            # paper/figs
import paper_style as ps                                                # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.figures import tables as T                                     # noqa: E402
from pim.figures.theme import style_ax                                  # noqa: E402

matplotlib.rcdefaults()                     # importing pim.figures.tables applies seaborn's theme; undo it
ps.apply()
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.lines import Line2D                                     # noqa: E402
from matplotlib.ticker import NullLocator                               # noqa: E402

OUT = Path(__file__).resolve().parent
PIECES, EXTRA_DIR = OUT / "pieces", OUT / "extra"
RAYS = (5, 8, 16, 128)
RUN = {r: f"L-dw-{r}ray-20m" for r in RAYS}
BLOCK = {"continuous": "cartesian", "categorical": "appearance-fac"}
LS = {"continuous": "-", "categorical": "--"}
EDITORS = ("PI", "GS", "IM")
MARK = {"PI": "o", "GS": "s", "IM": "^", "ND": "D"}
GREY, ZERO = "#7f7f7f", "#c8c8c8"
LEGEND_GAP = 0.18           # inches of clear space between the x-axis label and the legend row beneath it

T.set_basis("cartesian")
F = T.collect([], list(RUN.values()))
assert not F.missing, F.missing


def cell(r: int, block: str, ed: str) -> dict:
    """What the figure shows for one (rays, block, editor): parent's reported arm + the replicates' spread."""
    row = F.df[(F.df["run"] == RUN[r]) & (F.df["basis"] == BLOCK[block])].iloc[0]
    rep = F.rep_sd.get((RUN[r], BLOCK[block]), {})
    n = int(rep.get("n", 1))
    mean, sd = float(rep.get(f"{ed} EI_mean", np.nan)), float(rep.get(f"{ed} EI", np.nan))
    return dict(rays=r, basis=block, editor=ed, n=n, steps=rep.get("steps"),
                canonical=float(row[f"{ed} EI"]), fid=float(row[f"{ed} fid"]), arm=str(row[f"{ed} arm"]),
                guarded=bool(row[f"{ed} guarded"]), mean=mean, sd=sd,
                plotted=mean if n > 1 and np.isfinite(mean) else float(row[f"{ed} EI"]),
                values=rep.get(f"{ed} EI_values"), fid_values=rep.get(f"{ed} fid_values"))


CELLS = [cell(r, block, ed) for block in BLOCK for ed in EDITORS + ("ND",) for r in RAYS]


def series(block: str, ed: str) -> list[dict]:
    return [c for c in CELLS if c["basis"] == block and c["editor"] == ed]


# ── drawing ──────────────────────────────────────────────────────────────────


def _ax(ax):
    style_ax(ax)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, labelcolor=ps.TEXT, length=2.5)
    ax.set_xscale("log")
    ax.set_xticks(RAYS)
    ax.set_xticklabels([str(r) for r in RAYS])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(RAYS[0] / 1.3, RAYS[-1] * 1.3)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])


def draw(ax, blocks, editors) -> bool:
    """Mean ± SD per ray count; hollow = the parent's reported arm is outside the guard. Returns whether any is."""
    ax.axhline(0, color=ZERO, lw=0.6, zorder=0)
    any_hollow = False
    for block in blocks:
        for ed in editors:
            S = series(block, ed)
            if all(np.isnan(s["plotted"]) for s in S):
                continue
            x, y, c = [s["rays"] for s in S], [s["plotted"] for s in S], ps.EDITOR_COLORS[ed]
            e = [s["sd"] if s["n"] > 1 else np.nan for s in S]
            ax.errorbar(x, y, yerr=e if np.isfinite(e).any() else None, color=c, ls=LS[block], marker=MARK[ed],
                        ms=3.2, lw=1.0, elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=3)
            hollow = [(s["rays"], s["plotted"]) for s in S if not s["guarded"]]
            if hollow:
                any_hollow = True
                ax.plot(*zip(*hollow), ls="none", marker=MARK[ed], ms=3.2, mfc="white", mec=c, zorder=4)
    _ax(ax)
    return any_hollow


def key_editor(ed):
    return Line2D([], [], color=ps.EDITOR_COLORS[ed], marker=MARK[ed], ms=3.2, lw=1.0, label=ed)


def key_style(block):
    return Line2D([], [], color=GREY, ls=LS[block], lw=1.0, label=block)


def key_hollow():
    return Line2D([], [], ls="none", marker="o", ms=3.2, mfc="white", mec=ps.TEXT, label="outside fidelity guard")


LEGEND_KW = dict(handlelength=2.4, columnspacing=1.0, handletextpad=0.5)


def legend_below(fig, handles, ncol: int) -> None:
    """A figure legend on the page's bottom edge with LEGEND_GAP inches of clear space above it: constrained
    layout lays the axes out in the region above the legend strip (``rect``), so the gap is exact."""
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), borderaxespad=0.0,
                     ncol=ncol, **LEGEND_KW)
    strip = leg.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi + LEGEND_GAP
    frac = strip / fig.get_figheight()
    fig.get_layout_engine().set(rect=(0, frac, 1, 1 - frac))


def save_pdf(fig, stem: Path) -> None:
    """ps.save, PDF only (the pieces ship without previews)."""
    ps.save(fig, stem)
    Path(stem).with_suffix(".png").unlink()


def fig_one_panel(stem: Path, width: float, height: float, legend_below_axes: bool) -> None:
    """Both bases in one panel: continuous solid, categorical dashed."""
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
    hollow = draw(ax, BLOCK, EDITORS)
    ax.set_xlabel("Rays")
    ax.set_ylabel("Edit Index")
    eds, sty, hol = [key_editor(e) for e in EDITORS], [key_style(b) for b in BLOCK], [key_hollow()] if hollow else []
    if legend_below_axes:
        # a legend fills column-wise: interleaved, the two rows read "PI GS IM" / "continuous categorical"
        legend_below(fig, [eds[0], sty[0], eds[1], sty[1], eds[2]] + hol, ncol=3)
    else:
        fig.legend(handles=eds + sty + hol, loc="outside right center", ncol=1, **LEGEND_KW)
    ps.save(fig, stem)
    plt.close(fig)


def fig_two_panels(stem: Path, with_nd: bool = True) -> None:
    """Continuous state | categorical state, sharing y."""
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(ps.TEXT_WIDTH_IN, 2.3), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04)
    cat = EDITORS + (("ND",) if with_nd else ())
    hollow = draw(axes[0], ["continuous"], EDITORS) | draw(axes[1], ["categorical"], cat)
    axes[0].set_title("Continuous state", pad=3)
    axes[1].set_title("Categorical state", pad=3)
    axes[0].set_ylabel("Edit Index")
    for ax in axes:
        ax.set_xlabel("Rays")
    h = [key_editor(e) for e in cat] + ([key_hollow()] if hollow else [])
    legend_below(fig, h, ncol=len(h))
    ps.save(fig, stem)
    plt.close(fig)


def piece(block: str, editors, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(ps.HALF_WIDTH_IN, 2.2), layout="constrained")
    draw(ax, [block], editors)
    ax.set_xlabel("Rays")
    ax.set_ylabel("Edit Index")
    save_pdf(fig, stem)
    plt.close(fig)


def legend_piece(handles, stem: Path) -> None:
    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, 0.3))
    fig.legend(handles=handles, loc="center", ncol=len(handles), **LEGEND_KW)
    save_pdf(fig, stem)
    plt.close(fig)


# ── the printed table (checked against experiments/paper_ci/dashboard/ledger.md) ─


def table() -> str:
    L = ["| rays | basis | editor | plotted (mean) | SD | n | budget | members | parent | parent fid | parent arm | inside guard | member fids |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in CELLS:
        if np.isnan(c["plotted"]):
            continue
        vals = " ".join(f"{v:+.3f}" for v in c["values"]) if c["values"] else ""
        fids = " ".join(f"{v:.2f}" for v in c["fid_values"]) if c["fid_values"] else ""
        steps = "/".join(f"{s // 1000}k" for s in c["steps"]) if c["steps"] else ""
        L.append(f"| {c['rays']} | {c['basis']} | {c['editor']} | {c['plotted']:+.3f} | {c['sd']:.3f} | {c['n']} | {steps} | {vals} | "
                 f"{c['canonical']:+.3f} | {c['fid']:.2f} | {c['arm']} | {'yes' if c['guarded'] else 'no'} | {fids} |")
    L.append("")
    L.append("plotted = pooled mean over the seed replicates (n > 1) else the parent's value; SD with n − 1; parent = the "
             "parent run's reported arm (best Edit Index inside the fidelity guard) with its fidelity ratio and arm "
             "(point · step size); inside guard = that arm's fidelity ratio ≤ 1; member fids = the replicates' "
             "fidelity ratios at their own reported arms.")
    return "\n".join(L)


if __name__ == "__main__":
    PIECES.mkdir(exist_ok=True)
    fig_two_panels(OUT / "by_rays", with_nd=True)                                       # the paper's figure
    fig_one_panel(OUT / "by_rays_half", ps.HALF_WIDTH_IN, 2.9, legend_below_axes=True)  # for a wrap beside the table
    if "--all" in sys.argv[1:]:                                                         # variants, kept out of the top level
        EXTRA_DIR.mkdir(exist_ok=True)
        fig_one_panel(EXTRA_DIR / "by_rays_one_panel", ps.TEXT_WIDTH_IN, 2.3, legend_below_axes=False)
        fig_two_panels(EXTRA_DIR / "by_rays_noND", with_nd=False)
    piece("continuous", EDITORS, PIECES / "B_continuous")
    piece("categorical", EDITORS + ("ND",), PIECES / "B_categorical")
    piece("categorical", EDITORS, PIECES / "B_categorical_noND")
    legend_piece([key_editor(e) for e in EDITORS + ("ND",)] + [key_hollow()], PIECES / "B_legend")          # by_rays
    legend_piece([key_editor(e) for e in EDITORS] + [key_style(b) for b in BLOCK] + [key_hollow()],
                 PIECES / "B_legend_half")                                                                  # by_rays_half
    md = table()
    print(md)
    (OUT / "by_rays_values.md").write_text(md + "\n")
    (OUT / "by_rays_values.json").write_text(json.dumps(
        [{k: v for k, v in c.items()} for c in CELLS if not np.isnan(c["plotted"])], indent=1, default=float))
