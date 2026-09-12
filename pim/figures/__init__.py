"""pim.figures — figure builders. Pure: take pre-computed arrays/metrics, return Figure.

No model calls, no metric computation — notebooks/scripts compute numbers with
``pim.metrics`` and pass arrays here.

    theme.py      palette + style_ax (light); the dark palette lives in discworld.viz
    waterfall.py  waterfall_grid — THE canonical editor-comparison panel
                  (spec: research/specs/WATERFALL_SPEC.md)
    probe_capacity.py  the probe-width sweep figure (experiments/probe_capacity)
    probe_targets.py   the probe-target resolution sweep (experiments/probe_targets)
"""

from pim.figures.theme import PALETTE, style_ax
from pim.figures.probe_targets import sweep_figure
from pim.figures.waterfall import waterfall_grid

__all__ = [
    "PALETTE",
    "style_ax",
    "sweep_figure",
    "waterfall_grid",
]
