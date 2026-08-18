"""The one waterfall implementation.

`notebooks/experiments/editability/WATERFALL_SPEC.md` is the specification; this is its
single implementation. Route **every** 1D-observation waterfall through `waterfall_grid`.
Eighteen separate copies of this panel existed as of 2026-08-17, which is exactly the drift
the spec was written to prevent — each copy re-decided the colormap, the context frames, and
the alignment.

Obeys the `pim/figures` contract: takes pre-computed arrays, returns a `Figure`, never calls
a model and never computes a metric. Pass metrics in via `metrics=`.

Two spec rules are enforced structurally rather than by documentation:

* **Each column shows its own free-run.** `columns` maps a name to that arm's own rollout, so
  a shared teacher-forced row painted across every column cannot be expressed at all.
* **Fixed intensity scaling.** `vmin`/`vmax` default to 0/1 and are applied to every cell, so
  per-cell autoscaling — which makes a collapsed arm look normal — cannot happen by accident.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

DARK_BG = "#0a0a14"
DARK_TEXT = "#a3adc2"
# The boundary marker is deliberately NEUTRAL, not warm: an earlier orange (#fa8850) was
# visually indistinguishable from the vermilion ghost locator (#d55e00) at figure scale, so
# two different meanings read as one colour. Locators keep the spec's green/red.
EDIT_LINE = "#e8ecf5"
TARGET_C = "#00b050"
GHOST_C = "#ff3b30"

MIN_SAMPLE_ROWS = 3


def waterfall_grid(
    columns: dict[str, np.ndarray],
    context: np.ndarray,
    gt: np.ndarray,
    *,
    title: str,
    sample_idx: np.ndarray | list[int] | None = None,
    target_x: np.ndarray | None = None,
    ghost_x: np.ndarray | None = None,
    metrics: dict[str, float] | None = None,
    metric_label: str = "Edit Index",
    leads_by_one: tuple[str, ...] = (),
    gt_label: str = "GT (sim clean obs)",
    vmin: float = 0.0,
    vmax: float = 1.0,
    col_width: float = 2.5,
    row_height: float = 3.2,
) -> Figure:
    """Build the canonical comparison waterfall.

    Parameters
    ----------
    columns
        ``{column name: rollout}``, each ``(n_samples, K, W)`` — **that arm's own free-run**,
        step 0 first. Column order is preserved; the GT column is prepended automatically.
    context
        ``(n_samples, N_CTX, W)`` — the **actual (noisy) observations** the model was
        teacher-forced on before the edit frame. Not the clean render: only the GT column is
        clean. Drawn above the edit-frame line in every column.
    gt
        ``(n_samples, K, W)`` — the clean reference, ``clean_obs[ef:ef+K]``. Becomes the first
        column. Every observation-space reference is scored and drawn against the clean render.
    title
        A single what-is-shown title. Results belong in a dated results block, not here.
    sample_idx
        Which rows of the arrays to draw. Defaults to the first ``MIN_SAMPLE_ROWS``. Fewer
        than three rows warns: a waterfall is judged qualitatively and two rows cannot
        distinguish a real effect from a lucky sample.
    target_x, ghost_x
        ``(n_samples,)`` pixel coordinates for the green target and red-dashed ghost locators.
    metrics
        ``{column name: value}``, appended to that column's title so the picture and the
        number are read together.
    leads_by_one
        Names of columns that were fed the post-edit frame and therefore lead the others by one
        step. Labelled as such — never re-align the other columns to them.
    """
    if not columns:
        raise ValueError("waterfall_grid needs at least one method column")

    n_samples = context.shape[0]
    if sample_idx is None:
        sample_idx = list(range(min(MIN_SAMPLE_ROWS, n_samples)))
    sample_idx = list(sample_idx)

    if len(sample_idx) < MIN_SAMPLE_ROWS:
        warnings.warn(
            f"waterfall_grid: {len(sample_idx)} sample row(s); the spec requires "
            f"at least {MIN_SAMPLE_ROWS}. Two rows is not a comparison.",
            stacklevel=2,
        )

    n_ctx = context.shape[1]
    names = [gt_label, *columns.keys()]
    bodies: dict[str, np.ndarray] = {gt_label: gt, **columns}

    fig, axes = plt.subplots(
        len(sample_idx),
        len(names),
        figsize=(col_width * len(names), row_height * len(sample_idx)),
        squeeze=False,
        facecolor=DARK_BG,
    )

    for r, smp in enumerate(sample_idx):
        for c, name in enumerate(names):
            ax = axes[r][c]
            ax.set_facecolor(DARK_BG)

            body = np.asarray(bodies[name])[smp]
            panel = np.concatenate([np.asarray(context)[smp], body], axis=0)

            # Fixed scaling on every cell — never per-cell autoscale.
            ax.imshow(
                panel,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                interpolation="nearest",
            )

            # The edit-frame boundary: context above, each column's own free-run below.
            ax.axhline(n_ctx - 0.5, color=EDIT_LINE, lw=1.4, ls="--")

            if target_x is not None and np.isfinite(target_x[smp]):
                ax.axvline(target_x[smp], color=TARGET_C, lw=1.2)
            if ghost_x is not None and np.isfinite(ghost_x[smp]):
                ax.axvline(ghost_x[smp], color=GHOST_C, lw=1.2, ls="--")

            ax.tick_params(colors=DARK_TEXT, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(DARK_TEXT)

            if r == 0:
                label = name
                if metrics is not None and name in metrics:
                    label = f"{label}\n{metric_label} {metrics[name]:+.2f}"
                if name in leads_by_one:
                    label = f"{label}\n(leads by one frame)"
                ax.set_title(label, color=DARK_TEXT, fontsize=8.5)
            if c == 0:
                ax.set_ylabel(f"sample {smp}\ntime ↓", color=DARK_TEXT, fontsize=8)

    handles = [
        Line2D([], [], color=EDIT_LINE, ls="--", lw=1.4, label="edit frame"),
        Line2D([], [], color=TARGET_C, lw=1.2, label="target"),
        Line2D([], [], color=GHOST_C, ls="--", lw=1.2, label="ghost"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        handlelength=2.6,
        labelcolor=DARK_TEXT,
        fontsize=8.5,
    )
    fig.suptitle(title, color=DARK_TEXT, fontsize=11, y=1.045)
    fig.text(
        0.5,
        0.0,
        f"context: {n_ctx} noisy observed frames above the line · "
        "below: each column's own free-run from step 0",
        ha="center",
        color=DARK_TEXT,
        fontsize=8,
    )
    fig.tight_layout()
    return fig
