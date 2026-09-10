"""The probe-target resolution sweep: decodability and editability against cell count.

One row per run (the frame model's Edit Index is the ray-zone construction, the token
model's the frame-set one — different constructions, so they never share an axis: STYLE
§3). Columns: Probe Skill (LIN and MLP-128, best residual point), then the Edit Index at
each editor's best arm. Targets are grouped into three families, each its own colour and
marker: the product grids (uniform in the frustum basis), the appearance family (the
observation-exact partition and its coarser / finer relatives), and the misaligned
controls (the appearance partition's cell count, a product-grid structure). Arms whose
fidelity guard exceeds 1 (the edit left the output further from the truth than doing
nothing) are drawn hollow — a mark, not a number (STYLE §3).

Pure: takes score blocks and cell counts, returns a Figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from pim.figures.theme import PALETTE, style_ax

EDITORS = ("PI", "ND", "GS")
FAMILIES = ("grid", "appearance", "misaligned")
LABEL = {"grid": "product grid (frustum basis)", "appearance": "appearance family",
         "misaligned": "misaligned 30-cell grid"}
STYLE = {"grid": (PALETTE[0], "o"), "appearance": (PALETTE[1], "s"),
         "misaligned": (PALETTE[2], "D")}
MISALIGNED = {"grid-6x5", "grid-10x3"}
INK2, GRID_C = "#52514e", "#e1e0d9"


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in rgb)


def family_of(target: str) -> str:
    if target in MISALIGNED:
        return "misaligned"
    return "appearance" if target.startswith("appearance") else "grid"


def sweep_figure(rows: list[dict]) -> plt.Figure:
    """``rows``: one dict per run — ``{"label", "ei_name", "targets": {name: (cells, block)}}``
    where ``block`` is the run's ``scores.json["bases"][name]`` and ``ei_name`` names the
    Edit Index construction for that run's axis label."""
    n = len(rows)
    fig, axes = plt.subplots(n, 1 + len(EDITORS), figsize=(4.0 * (1 + len(EDITORS)), 3.6 * n),
                             squeeze=False, gridspec_kw=dict(wspace=0.3, hspace=0.62))
    for r, row in enumerate(rows):
        items = sorted(row["targets"].items(), key=lambda kv: kv[1][0])
        all_cells = sorted({c for _, (c, _) in items})
        for ax in axes[r]:
            style_ax(ax)
            ax.set_xscale("log", base=2)
            ticks = [c for i, c in enumerate(all_cells)
                     if i == 0 or c / all_cells[i - 1] > 1.15]      # 30 and 32 share one tick
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(c) for c in ticks], fontsize=8)
            ax.minorticks_off()
            ax.grid(True, color=GRID_C, lw=0.8)
            ax.set_xlabel("cells in the target partition")
        a = axes[r][0]
        for fam in FAMILIES:
            pts = [(c, b) for _, (c, b) in items if family_of(b["target"]) == fam]
            if not pts:
                continue
            col, m = _hex(STYLE[fam][0]), STYLE[fam][1]
            x = [c for c, _ in pts]
            a.plot(x, [max(b["probe_skill_mlp"]) for _, b in pts], color=col, lw=2, marker=m,
                   ms=6, mec="white", mew=1.0, label=LABEL[fam])
            a.plot(x, [max(b["probe_skill_linear"]) for _, b in pts], color=col, lw=1.4,
                   ls="--", marker=m, ms=5, mfc="white", mec=col)
            for k, ed in enumerate(EDITORS, start=1):
                ax = axes[r][k]
                ei = np.array([b["best"][ed]["edit_index"] for _, b in pts])
                fid = np.array([b["best"][ed]["fidelity_ratio"] for _, b in pts])
                ax.plot(x, ei, color=col, lw=2, label=LABEL[fam])
                ok = fid <= 1.0
                ax.plot(np.array(x)[ok], ei[ok], ls="none", marker=m, ms=7, color=col,
                        mec="white", mew=1.0)
                ax.plot(np.array(x)[~ok], ei[~ok], ls="none", marker=m, ms=7, mfc="white",
                        mec=col, mew=1.6)
                if fam == "grid":
                    ax.plot([], [], ls="none", marker="o", ms=7, mfc="white", mec=INK2,
                            mew=1.6, label="hollow: fidelity guard > 1 (degraded)")
        a.plot([], [], color=INK2, lw=2, label="solid: MLP-128")
        a.plot([], [], color=INK2, lw=1.4, ls="--", label="dashed: LIN")
        a.set_ylabel("Probe Skill, best residual point")
        a.set_ylim(-0.02, 1.02)
        a.set_title("decodability", loc="right", fontsize=9, color=INK2)
        a.legend(fontsize=7, loc="lower left", handlelength=2.6, frameon=False)
        for k, ed in enumerate(EDITORS, start=1):
            ax = axes[r][k]
            ax.axhline(0, color=INK2, lw=0.8, ls=":")
            ax.set_ylabel(f"{row['ei_name']}, edit step")
            ax.set_title(f"{ed}, each target's best arm", loc="right", fontsize=9, color=INK2)
            ax.set_ylim(-0.1, 0.75)
            if k == 1:
                ax.legend(fontsize=7, loc="upper left", handlelength=2.6, frameon=False)
        axes[r][0].text(0.0, 1.13, row["label"], transform=axes[r][0].transAxes, fontsize=10.5,
                        color=INK2, fontweight="bold", va="bottom")
    fig.suptitle("Probe-target resolution sweep — one recipe, one bench, one seed; "
                 "x = cells of the categorical target", fontsize=11, color=INK2, y=1.0)
    return fig
