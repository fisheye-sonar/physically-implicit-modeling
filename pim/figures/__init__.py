"""pim.figures — figure builders. Pure: take pre-computed arrays/metrics, return Figure.

No model calls, no metric computation — notebooks/scripts compute numbers with
``pim.metrics`` and pass arrays here.

    theme.py      palette + style_ax / style_ax_dark + plot_color
    waterfall.py  waterfall_grid — THE canonical editor-comparison panel
                  (spec: research/specs/WATERFALL_SPEC.md)
    scaling.py    loss_curves (val/train vs steps with the Bayes floor) + the
                  Edit-Index-vs-steps grid
"""

from pim.figures.theme import PALETTE, plot_color, style_ax, style_ax_dark
from pim.figures.waterfall import waterfall_grid

__all__ = [
    "PALETTE",
    "plot_color",
    "style_ax",
    "style_ax_dark",
    "waterfall_grid",
]
