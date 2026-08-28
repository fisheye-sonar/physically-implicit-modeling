"""The canonical editability-scaling figure: Edit Index vs training steps.

One figure per **(architecture, editor)**. Within a panel:

* **x** — optimiser steps, log scale. *Not epochs.* An epoch is a different amount of compute at
  every data volume (250 epochs is 88k steps at 100k games and 17.6M at 20M — a 200x difference
  behind one label), so an epoch axis hides the data/steps trade-off that the whole grid exists to
  expose.
* **y** — Edit Index, the **absolute post-edit value**, never a gain over the model's own null. The
  null falls as a model becomes a better predictor of the *unedited* world, so a gain axis rises
  with model quality even when editability is flat (measured 2026-08-22: gain rose 3.8x across a
  ladder on which absolute editability moved +0.059 -> +0.098).
* **colour** — the setting (Othello vs discworld). **shade** — the data volume, light to dark.
* **marker fill** — hollow means the point is mocked, not measured.

Purity: this module takes assembled records and returns a `Figure`. It never loads a checkpoint,
runs a model, or computes a metric (`CLAUDE.md` invariant 1).

Record schema
-------------
A record is a plain dict::

    {"setting": "othello"|"discworld", "arch": str, "editor": str,
     "games": int, "steps": int, "edit_index": float,
     "measured": bool, "label": str|None}

`collect.py` in `notebooks/experiments/editability/scaling/` builds these from the result JSONs.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from pim.figures.theme import style_ax

# Setting -> base colour. Two clearly distinct hues; shade carries data volume within each.
# Two hues chosen to stay separable in greyscale and for the common colour-vision deficiencies:
# a cool teal-blue against a warm magenta-red, rather than the blue/orange pair which read as the
# same value when printed.
SETTING_CMAP = {"othello": "GnBu", "discworld": "RdPu"}
SETTING_LABEL = {"othello": "Othello", "discworld": "Discworld"}


def _shades(cmap: str, n: int) -> list:
    """`n` shades light->dark, kept clear of the near-white and near-black ends."""
    m = plt.get_cmap(cmap)
    return [m(x) for x in np.linspace(0.38, 0.92, max(n, 1))]


def edit_index_vs_steps(
    records: list[dict],
    arch: str,
    editor: str,
    *,
    ax=None,
    reference: dict | None = None,
    ylim: tuple[float, float] = (-1.0, 1.0),
    xlim: tuple[float, float] | None = None,
    show_ylabel: bool = True,
    title_extra: str = "",
):
    """One panel: Edit Index vs optimiser steps, coloured by setting, shaded by data volume."""
    made = ax is None
    if made:
        _, ax = plt.subplots(figsize=(7.2, 5.0), facecolor="white")
    sel = [r for r in records if r["arch"] == arch and r["editor"] == editor]
    volumes = sorted({r["games"] for r in sel})
    handles: list[Line2D] = []

    for setting in ("othello", "discworld"):
        shades = _shades(SETTING_CMAP[setting], len(volumes))
        for vi, games in enumerate(volumes):
            pts = sorted([r for r in sel if r["setting"] == setting and r["games"] == games],
                         key=lambda r: r["steps"])
            if not pts:
                continue
            x = [p["steps"] for p in pts]
            y = [p["edit_index"] for p in pts]
            mock = [not p["measured"] for p in pts]
            c = shades[vi]
            ax.plot(x, y, color=c, lw=2.0, ls="--" if all(mock) else "-", zorder=3)
            # measured points filled, mocked points hollow — visible per-point, not per-line
            for xi, yi, mk in zip(x, y, mock):
                ax.plot([xi], [yi], marker="o" if setting == "othello" else "s", ms=7,
                        mfc="none" if mk else c, mec=c, mew=1.8, zorder=4)
            handles.append(Line2D([], [], color=c, lw=2.0,
                                  marker="o" if setting == "othello" else "s",
                                  label=f"{SETTING_LABEL[setting]} · {_fmt_games(games)}"))

    # Zero is drawn as a reference but deliberately NOT labelled "no edit": an Edit Index of 0
    # means the output moved as far from the unedited world as toward the edited one, which is a
    # real change, often a destructive one. Labelling it "no edit" invited exactly that misreading.
    ax.axhline(0.0, color="0.25", lw=1.4, zorder=1)
    if reference:
        for lbl, val in reference.items():
            ax.axhline(val, color="0.55", ls=":", lw=1.5, zorder=1)
            ax.text(0.995, val, f" {lbl} ", transform=ax.get_yaxis_transform(), ha="right",
                    va="bottom", fontsize=7.5, color="0.45")
    ax.set_xscale("log")
    ax.set_xlabel("Optimizer steps (log)")
    if show_ylabel:
        ax.set_ylabel("Edit Index ↑")
    ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_title(f"{arch} · {editor}{title_extra}", fontsize=11)
    if handles:
        ax.legend(handles=handles, fontsize=7.5, loc="lower right", ncol=1)
    style_ax(ax)
    return ax.figure if made else ax


def _fmt_games(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    return f"{n / 1_000:g}k"


def grid(records: list[dict], archs: list[str], editors: list[str], *,
         reference: dict | None = None, suptitle: str = "", mock_note: bool = False):
    """The full set of panels: one row per architecture, one column per editor."""
    fig, axes = plt.subplots(len(archs), len(editors),
                             figsize=(6.2 * len(editors), 4.6 * len(archs)),
                             squeeze=False, facecolor="white", sharex=True, sharey=True)
    # One x-range for every panel, so a point at 10^5 sits at the same place in all of them.
    xs = [r["steps"] for r in records] or [1e4, 1e6]
    xlim = (min(xs) * 0.55, max(xs) * 1.8)
    for i, arch in enumerate(archs):
        for j, ed in enumerate(editors):
            edit_index_vs_steps(records, arch, ed, ax=axes[i][j], reference=reference,
                                xlim=xlim, show_ylabel=(j == 0))
    if mock_note:
        fig.text(0.5, 0.005,
                 "⚠ HOLLOW MARKERS / DASHED LINES ARE FABRICATED — illustration of layout only, "
                 "not results", ha="center", fontsize=10, color="#b00020", weight="bold")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0.03 if mock_note else 0, 1, 0.95), w_pad=2.2, h_pad=2.4)
    return fig


# ---------------------------------------------------------------------------------------------
# Loss curves: the data-volume regime figure.
#
# Added 2026-08-24 for `research/scratch/2026-08-24-saturation-is-overfitting.md`. Separate from
# the Edit Index panels above because it answers a prior question: *is this cell learning a world
# model at all, or memorising its pool?* An Edit Index measured inside the memorising regime is
# measuring something different from one measured outside it.
#
# Purity as above: these take arrays and a pre-computed fit, and never open a checkpoint.
# ---------------------------------------------------------------------------------------------

def loss_curves(curves: list[dict], bayes: float, *, fit: tuple[float, float] | None = None,
                fit_label: str = "", suptitle: str = ""):
    """Two panels: val loss vs steps, and excess-over-Bayes on log-log.

    `curves` — list of ``{"label": str, "steps": array, "val": array, "train": array,
    "color": str}``. `bayes` — the irreducible floor for the generator. `fit` — ``(A, b)`` for
    ``excess = A * step**b``, drawn on the right panel only; computed by the caller.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), facecolor="white")
    ax0, ax1 = axes

    for c in curves:
        s, v, t = np.asarray(c["steps"]), np.asarray(c["val"]), np.asarray(c.get("train", []))
        ax0.plot(s, v, color=c["color"], lw=2.2, label=f"{c['label']} — val", zorder=4)
        if t.size:
            ax0.plot(s, t, color=c["color"], lw=1.3, ls="--", alpha=0.75,
                     label=f"{c['label']} — train", zorder=3)
        i = int(np.argmin(v))
        if i < len(s) - 1:  # a genuine interior minimum => it turned around
            ax0.plot([s[i]], [v[i]], marker="v", ms=10, color=c["color"], mec="white",
                     mew=1.2, zorder=6)
            # placed ABOVE the marker: below it collides with the Bayes-floor label, which is
            # the one annotation on this panel that must stay legible.
            ax0.annotate(f"turns around\nstep {int(s[i]):,}", xy=(s[i], v[i]),
                         xytext=(-4, 16), textcoords="offset points", fontsize=8,
                         color=c["color"], ha="right", weight="bold")

    # The Bayes floor is the whole point of the left panel: a TRAIN curve below it is proof of
    # memorisation, since no model can beat the generator's own conditional entropy honestly.
    ax0.axhline(bayes, color="0.2", lw=1.6, ls="-", zorder=2)
    ax0.text(0.995, bayes, f" Bayes floor {bayes:.4f} ", transform=ax0.get_yaxis_transform(),
             ha="right", va="bottom", fontsize=8, color="0.2")
    ax0.axhspan(ax0.get_ylim()[0], bayes, color="#b00020", alpha=0.055, zorder=0)
    ax0.text(0.02, 0.03, "below the floor = memorising the pool", transform=ax0.transAxes,
             fontsize=8, color="#b00020", style="italic")
    ax0.set_xscale("log")
    ax0.set_xlabel("Optimizer steps (log)")
    ax0.set_ylabel("Cross-entropy ↓")
    ax0.set_title("Val and train loss against the irreducible floor", fontsize=11)
    ax0.legend(fontsize=7.5, loc="upper right")
    style_ax(ax0)

    for c in curves:
        s, v = np.asarray(c["steps"]), np.asarray(c["val"])
        e = v - bayes
        m = e > 0
        ax1.plot(s[m], e[m], color=c["color"], lw=2.2, label=c["label"], zorder=4)
    if fit is not None:
        A, b = fit
        xs = np.asarray([c["steps"] for c in curves][0], float)
        grid_x = np.logspace(np.log10(max(xs.min(), 1e3)), np.log10(xs.max() * 4.0), 80)
        ax1.plot(grid_x, A * grid_x**b, color="0.35", lw=1.4, ls=":", zorder=5,
                 label=fit_label or f"fit: {A:.1f}·step^{b:.3f}")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Optimizer steps (log)")
    ax1.set_ylabel("Excess over Bayes floor (log) ↓")
    ax1.set_title("Excess over the floor — a straight line here is a power law", fontsize=11)
    ax1.legend(fontsize=7.5, loc="lower left")
    style_ax(ax1)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if suptitle else 1.0), w_pad=2.4)
    return fig
