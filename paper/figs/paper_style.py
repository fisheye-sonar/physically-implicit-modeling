"""One look for every paper figure (2026-09-21). Import this FIRST, before pyplot.

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # paper/figs
    import paper_style as ps
    ps.apply()

Arial (Liberation Sans / Nimbus Sans fall back; Sevan 2026-09-21 — was Times New Roman until 11:50 PT), text
embedded as TrueType so the PDF stays editable and vector, white page, black text, one colour per editor in
every figure. ``save`` crops to the content with NO outer padding, so the spacing around a figure is set in LaTeX. Raw
observations are drawn as the canonical waterfall draws them: ``gray`` on the dark panel
(``pim.figures.waterfall.DARK_BG``), fixed 0–1 range, nearest interpolation.

Sizes: design at the FINAL printed width — ICLR's text width is 5.5 in (a half-width figure is
2.65 in) — with 8–9 pt text, so nothing is scaled at include time. If a script draws larger
and lets LaTeX scale it down, its text must still be at least 7 pt on the page.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans"],
    "mathtext.fontset": "custom",          # maths in Arial too (R², t*)
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "legend.frameon": False,
}

TEXT_WIDTH_IN = 5.5           # ICLR text width; a full-width figure
HALF_WIDTH_IN = 2.65

# Okabe-Ito, one editor one colour, everywhere.
EDITOR_COLORS = {"PI": "#0072B2", "GS": "#D55E00", "IM": "#009E73", "ND": "#E69F00", "IM-NN": "#CC79A7"}
TEXT = "black"
FRAME = "#6f6f6f"             # thin border around a raw-observation panel
# Locators shared with paper/figs/qualitative_edits: cyan = where the edited disc came from, pink = where it went.
ORIGIN_C, DEST_C = "#00bcd4", "#ff4fa3"
# Othello boards (paper/figs/qualitative_edits_othello): board green, grid line, probability tint, edited tile.
BOARD_GREEN, BOARD_LINE, BOARD_TINT, BOARD_EDIT = "#33a852", "#1e1e1e", "#ffe600", "#ff4fa3"


def apply() -> None:
    matplotlib.rcParams.update(RC)


def save(fig, stem: Path, dpi: int = 300, pad: float = 0.0) -> None:
    """Vector PDF for the paper and a PNG preview beside it (same stem), cropped to the content with no
    outer padding (``pad`` inches, default 0): the figure touches its own bounding box, LaTeX sets the rest."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=pad)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=pad)
