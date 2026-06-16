"""ProbeSpec — minimal wrapper bundling a probe with display metadata.

Used to pass a flexible list of probes through fit/eval/plot pipelines.
The probe itself owns its training logic via its fit() method; ProbeSpec
adds only what's needed for naming and visual differentiation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class ProbeSpec:
    """One probe + its display style.

    Attributes
    ----------
    name       : human-readable label for results dicts and plot legends
    probe      : the extractor module (LinearExtractor, MLPExtractor, ...)
    marker     : matplotlib marker for scatter plots
    color_idx  : index into pim.figures.theme.PALETTE for line/bar color
    linestyle  : line style for time-series plots
    """

    name: str
    probe: nn.Module
    marker: str = "x"
    color_idx: int = 0
    linestyle: str = "-"
