"""Visual theme: the Okabe-Ito palette + Axes styling for the academic light mode.

All result figures use this. Figures that show the simulator as an artifact (the
waterfall panels) use the dark palette defined once in ``pim.environments.discworld.viz``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# ── Light / academic theme ────────────────────────────────────────────────────

_BG_HEX = "#ffffff"
_TEXT_COLOR = "#172239"
_TICK_COLOR = "#555555"
_SPINE_COLOR = "#555555"

# Okabe-Ito palette — colorblind safe
PALETTE: list[tuple[float, float, float]] = [
    (0.00, 0.45, 0.70),  # blue      #0072B2
    (0.84, 0.37, 0.00),  # vermilion #D55E00
    (0.90, 0.62, 0.00),  # orange    #E69F00
    (0.00, 0.62, 0.45),  # teal      #009E73
    (0.80, 0.47, 0.65),  # purple    #CC79A7
    (0.34, 0.71, 0.91),  # sky       #56B4E9
    (0.94, 0.89, 0.26),  # yellow    #F0E442
]

def style_ax(ax: plt.Axes) -> None:
    """Light/academic theme for one Axes."""
    ax.set_facecolor(_BG_HEX)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE_COLOR)
    ax.tick_params(colors=_TICK_COLOR, labelsize=9)
