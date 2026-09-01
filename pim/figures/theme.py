"""Visual theme: palette + Axes styling for academic light + simulator dark modes.

All result figures (metrics, recovery, drift) use the light/academic theme.
Figures that show the simulator as an artifact (waterfall panels) use the
dark theme. The two are imported from pim.figures.theme and from
pim.environments.discworld.viz respectively.
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

# Mapping from simulator OBJECT_COLORS to Okabe-Ito.
_SIM_TO_OKABE: dict[tuple, tuple] = {
    (0.00, 0.83, 1.00): PALETTE[0],  # cyan   → blue
    (1.00, 0.42, 0.42): PALETTE[1],  # coral  → vermilion
    (1.00, 0.85, 0.24): PALETTE[2],  # amber  → orange
    (0.42, 0.80, 0.47): PALETTE[3],  # green  → teal
    (0.78, 0.48, 1.00): PALETTE[4],  # violet → purple
}


def plot_color(sim_color) -> tuple:
    """Map a simulator object color to its Okabe-Ito equivalent."""
    key = tuple(round(float(v), 2) for v in sim_color)
    return _SIM_TO_OKABE.get(key, tuple(float(v) for v in sim_color))


def style_ax(ax: plt.Axes) -> None:
    """Light/academic theme for one Axes."""
    ax.set_facecolor(_BG_HEX)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE_COLOR)
    ax.tick_params(colors=_TICK_COLOR, labelsize=9)


def style_ax_dark(ax: plt.Axes) -> None:
    """Dark/simulator theme for one Axes (waterfall panels)."""
    from pim.environments.discworld.viz import _BG_HEX as _DARK_BG_HEX, _TICK_COLOR as _DARK_TICK_COLOR
    ax.set_facecolor(_DARK_BG_HEX)
    for spine in ax.spines.values():
        spine.set_edgecolor(_DARK_TICK_COLOR)
    ax.tick_params(colors=_DARK_TICK_COLOR, labelsize=9)
