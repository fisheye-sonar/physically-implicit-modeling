"""Portable figure theme: colorblind-safe palette + Axes styling, light and dark.

Copy this into a project's figure module and import from there. Two themes, chosen
by what is being shown (see harness/STYLE.md §6):

  * light  — results, metrics, analysis
  * dark   — raw data artifacts (imagery, sensor output, model generations)

Intensity data on the dark theme uses a gray colormap. Do not invent a decorative
colormap for data whose absolute values carry meaning.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# ── Light / academic theme ───────────────────────────────────────────────────

LIGHT_BG = "#ffffff"
LIGHT_TEXT = "#172239"
LIGHT_TICK = "#555555"
LIGHT_SPINE = "#555555"

# ── Dark / raw-data theme ────────────────────────────────────────────────────

DARK_BG = "#0a0a14"
DARK_TICK = "#8892a6"

# Okabe-Ito — colorblind safe, in the conventional order.
PALETTE: list[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#E69F00",  # orange
    "#009E73",  # teal
    "#CC79A7",  # purple
    "#56B4E9",  # sky
    "#F0E442",  # yellow
]


def style_ax(ax: plt.Axes) -> None:
    """Light/academic theme for one Axes."""
    ax.set_facecolor(LIGHT_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(LIGHT_SPINE)
    ax.tick_params(colors=LIGHT_TICK, labelsize=9)
    ax.xaxis.label.set_color(LIGHT_TEXT)
    ax.yaxis.label.set_color(LIGHT_TEXT)
    ax.title.set_color(LIGHT_TEXT)


def style_ax_dark(ax: plt.Axes) -> None:
    """Dark theme for one Axes (raw-data panels)."""
    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(DARK_TICK)
    ax.tick_params(colors=DARK_TICK, labelsize=9)
    ax.xaxis.label.set_color(DARK_TICK)
    ax.yaxis.label.set_color(DARK_TICK)
    ax.title.set_color(DARK_TICK)


def legend_top(fig, handles, labels, ncol: int = 4, **kw) -> None:
    """Figure-top legend — the default for multi-column comparison panels.

    Legends belong above the figure, not inside a panel, so no panel is occluded
    and every column stays full size. `handlelength` is set so dashed and solid
    entries are actually distinguishable (STYLE.md §3).
    """
    kw.setdefault("handlelength", 2.6)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=ncol,
        frameon=False,
        **kw,
    )
