"""Fig 2 — Probe Skill against probe capacity (one-hidden-layer width), per source.

Reads the JSON files ``experiments/probe_capacity/scripts/probe_capacity.py`` writes after
EVERY fit, so the figure can be re-rendered while the sweep is still running and fills in
as cells land (missing cells break the line rather than being interpolated). One row per
environment; (a) held-out Probe Skill, (b) the in-sample gap. Canonical-corpus values
(30k / 20k sequences, the tables' numbers at the same residual point) are drawn as hollow
markers at LIN and 128 so the data effect is visible at those two widths.

Colours follow the ENTITY (Okabe-Ito, validated 2026-09-01): trained = blue, random-init =
vermilion, observation = teal; distinct markers; a legend always; text in ink tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pim.figures.theme import PALETTE

SOURCES = ("trained", "random_init", "observation")
LABEL = {"trained": "trained", "random_init": "random-init", "observation": "observation"}
STYLE = {"trained": (PALETTE[0], "o"), "random_init": (PALETTE[1], "s"),
         "observation": (PALETTE[3], "^")}
INK2, REF_GRAY, GRID = "#52514e", "#898781", "#e1e0d9"


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in rgb)


def capacity_figure(score_files: list[Path]) -> plt.Figure:
    """Render the sweep from whatever cells exist in ``score_files``."""
    envs = [json.loads(Path(p).read_text()) for p in score_files if Path(p).exists()]
    fig, axes = plt.subplots(max(1, len(envs)), 2, figsize=(12.5, 4.2 * max(1, len(envs))),
                             squeeze=False, gridspec_kw=dict(wspace=0.3, hspace=0.55))
    if not envs:
        axes[0, 0].text(0.5, 0.5, "no probe-capacity scores yet", ha="center", va="center",
                        transform=axes[0, 0].transAxes, color=INK2)
        return fig
    for row, S in enumerate(envs):
        widths = [str(w) for w in S["widths"]]                 # "LIN", "16", ...
        x = np.arange(len(widths))
        a, b = axes[row]
        for src in SOURCES:
            cells = S["cells"].get(src, {})
            skill = np.array([cells.get(w, {}).get("skill", np.nan) for w in widths], float)
            gap = np.array([cells.get(w, {}).get("insample_gap", np.nan) for w in widths], float)
            c, m = _hex(STYLE[src][0]), STYLE[src][1]
            a.plot(x, skill, color=c, lw=2, marker=m, ms=6, mec="white", mew=1.2,
                   solid_capstyle="round", label=LABEL[src])
            b.plot(x, gap, color=c, lw=2, marker=m, ms=6, mec="white", mew=1.2,
                   solid_capstyle="round", label=LABEL[src])
            ref = S.get("refs", {}).get(src, {})              # canonical corpus, same point
            for w, v in ref.items():
                if w in widths and v is not None:
                    a.plot([widths.index(w)], [v], marker=m, ms=9, mfc="none", mec=c, mew=1.6,
                           ls="none")
        a.plot([], [], marker="o", ms=9, mfc="none", mec=REF_GRAY, mew=1.6, ls="none",
               label=f"canonical corpus ({S['refs_label']})")
        for ax, title, ylab in ((a, "(a) held-out Probe Skill vs probe width", "Probe Skill"),
                                (b, "(b) overfit check: in-sample − held-out", "gap")):
            ax.set_xticks(x)
            ax.set_xticklabels(widths, fontsize=9)
            ax.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for sp in ax.spines.values():
                sp.set_edgecolor("#c3c2b7")
            ax.set_title(f"{title} — {S['env']} · {S['run']} · point {S['point']}",
                         fontsize=10, loc="left", pad=6)
            ax.set_xlabel("one-hidden-layer width (LIN = no hidden layer)", fontsize=9, color=INK2)
            ax.set_ylabel(ylab, fontsize=9, color=INK2)
            ax.tick_params(labelsize=8, colors=INK2)
            ax.legend(fontsize=8, frameon=False, handlelength=2.4, loc="best")
        a.set_ylim(0, 1.02)
        b.set_ylim(bottom=0)
        b.axhline(0, color="#c3c2b7", lw=0.8)
    fig.suptitle("Probe capacity sweep",
                 fontsize=11.5, y=1.0)
    return fig
