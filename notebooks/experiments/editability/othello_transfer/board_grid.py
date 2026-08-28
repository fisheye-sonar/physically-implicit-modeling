"""The qualitative panel for this thread — boards, not observation waterfalls.

`pim.figures.waterfall_grid` is the canonical panel for a 1D intensity scan and has no
meaning here: Othello's "observation" is a distribution over 64 board squares. This is
the same idea in that space, and it follows the same universal rules from
`harness/STYLE.md` §2 — every arm gets a column, a ground-truth reference column is
always present, at least three rows, samples drawn at RANDOM with the rule stated in the
title, fixed colour scale across every cell, and the intervened square marked.

Kept deliberately ad hoc (Sevan, 2026-08-20): it is not promoted to a spec until we know
which qualitative view is actually informative for this setting.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D

BOARD, EDGE, LEGAL, MARK, TOPN = "#14342b", "#0b1f19", "#ffd166", "#ef476f", "#4cc9f0"
FG, BG, MUTE = "#e8e8e8", "#0d0d0f", "#9aa0a6"
ROWS = list("ABCDEFGH")
PROB_VMAX = 0.20  # fixed across every cell; the model's per-case max is ~0.10 median


def _frame(ax, title):
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(8) + 0.5)
    ax.set_xticklabels(range(1, 9), fontsize=6)
    ax.set_yticks(np.arange(8) + 0.5)
    ax.set_yticklabels(ROWS[::-1], fontsize=6)
    ax.tick_params(colors=MUTE, length=0)
    for s in ax.spines.values():
        s.set_color("#3a3a3a")
    ax.set_title(title, fontsize=8, color=FG, pad=4)


def draw_board(ax, state, legal, intervened=None, title=""):
    """`state` is an 8x8 array in the simulator's own encoding: +1 black, -1 white, 0 blank."""
    ax.set_facecolor(BOARD)
    for k in range(9):
        ax.plot([0, 8], [k, k], color=EDGE, lw=1.0, zorder=1)
        ax.plot([k, k], [0, 8], color=EDGE, lw=1.0, zorder=1)
    for sq in legal:
        r, c = sq // 8, sq % 8
        ax.add_patch(Circle((c + .5, 7.5 - r), .17, facecolor="none", edgecolor=LEGAL, lw=1.8, zorder=3))
    for r in range(8):
        for c in range(8):
            if state[r, c] == 0:
                continue
            ax.add_patch(Circle((c + .5, 7.5 - r), .36, zorder=2, lw=.7, edgecolor="#8a8a8a",
                                facecolor="#111111" if state[r, c] > 0 else "#f2f2f2"))
    if intervened is not None:
        r, c = intervened // 8, intervened % 8
        ax.add_patch(Rectangle((c + .04, 7.5 - r - .46), .92, .92, fill=False,
                               edgecolor=MARK, lw=2.6, zorder=4))
    _frame(ax, title)


def draw_probs(ax, probs, legal_post, intervened=None, title=""):
    """Predicted next-move distribution, with the post-flip legal set outlined."""
    im = ax.imshow(probs.reshape(8, 8), cmap="magma", vmin=0.0, vmax=PROB_VMAX,
                   extent=(0, 8, 0, 8), origin="upper")
    n = len(legal_post)
    top = set(np.argsort(-probs)[:n].tolist()) if n else set()
    for sq in legal_post:
        r, c = sq // 8, sq % 8
        ax.add_patch(Rectangle((c + .06, 7.5 - r - .44), .88, .88, fill=False,
                               edgecolor=LEGAL, lw=1.6, zorder=3))
    for sq in top:
        r, c = sq // 8, sq % 8
        ax.plot(c + .5, 7.5 - r, marker="x", ms=5, mew=1.6, color=TOPN, zorder=4)
    if intervened is not None:
        r, c = intervened // 8, intervened % 8
        ax.add_patch(Rectangle((c + .02, 7.5 - r - .48), .96, .96, fill=False,
                               edgecolor=MARK, lw=2.2, zorder=5))
    _frame(ax, title)
    return im


def board_panel(bench, boards_pre, boards_post, arms, case_ids, fig_no, seed,
                subtitle="", figsize_per=(2.55, 2.95)):
    """`arms` is an ordered dict ``{column label: (probs (n_cases,64), headline number)}``.

    Column 1 is always the post-flip ground-truth board — the reference `STYLE.md` §2
    requires. Every subsequent column is one arm's own output; nothing is ever painted
    across columns from a privileged arm.
    """
    n_rows, n_cols = len(case_ids), 1 + len(arms)
    # Reserve the header in INCHES, not as a fraction: a fraction that clears the titles at
    # 3 rows collides with them at 6, and the collision is only visible in the rendered file.
    header_in = 1.45
    fig_h = figsize_per[1] * n_rows + header_in
    plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG, "text.color": FG})
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(figsize_per[0] * n_cols, fig_h))
    axs = np.atleast_2d(axs)
    im = None
    for r, ci in enumerate(case_ids):
        sq = bench.pos_int[ci]
        draw_board(axs[r, 0], boards_post[ci], bench.legal_post[ci], intervened=sq,
                   title=(f"case {ci} — ground truth after the flip\n"
                          f"{len(bench.legal_post[ci])} legal moves"))
        for k, (label, (probs, head)) in enumerate(arms.items(), start=1):
            im = draw_probs(axs[r, k], probs[ci], bench.legal_post[ci], intervened=sq,
                            title=f"{label}\n{head}")
    handles = [
        Line2D([], [], marker="o", ls="none", mfc="none", mec=LEGAL, ms=9, mew=1.8,
               label="legal after the flip (ground truth)"),
        Line2D([], [], marker="s", ls="none", mfc="none", mec=MARK, ms=9, mew=2.2,
               label="the intervened square"),
        Line2D([], [], marker="x", ls="none", color=TOPN, ms=7, mew=1.8,
               label="model's top-N predicted moves"),
        Line2D([], [], marker="o", ls="none", mfc="#111111", mec="#8a8a8a", ms=9, label="black disc"),
        Line2D([], [], marker="o", ls="none", mfc="#f2f2f2", mec="#8a8a8a", ms=9, label="white disc"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8, frameon=False,
               labelcolor=FG, bbox_to_anchor=(0.5, 1 - 0.10 / fig_h), handlelength=1.6)
    fig.suptitle(f"Fig {fig_no} — the intervention on the board, "
                 f"{n_rows} cases drawn at random (seed {seed}) from all {bench.n_cases}\n" + subtitle,
                 fontsize=10, color=FG, y=1 - 0.45 / fig_h, va="top")
    cax = fig.add_axes([0.945, 0.28, 0.008, 0.32])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("predicted probability", color=FG, fontsize=7, labelpad=2)
    cb.ax.tick_params(colors=MUTE, labelsize=6)
    fig.subplots_adjust(top=1 - header_in / fig_h, bottom=0.02, left=0.035, right=0.93,
                        hspace=0.34, wspace=0.06)
    return fig
