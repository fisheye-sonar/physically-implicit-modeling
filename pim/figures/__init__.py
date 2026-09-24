"""pim.figures — figure builders. No model calls and no metric DEFINITIONS: numbers come from
``pim.metrics`` (or are passed in as arrays), and so do the rules that pick which number a table
cell reports (``pim.metrics.selection``, ``pim.metrics.replicates``).

    tables.py     the master tables — reads scores.json + runs/_baselines/, assembles rows, draws

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
