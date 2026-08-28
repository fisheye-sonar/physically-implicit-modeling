"""Figure builders for the `latent_linearity` thread — pure: arrays in, `Figure` out.

Every builder takes **`models`** and **`arms`** as explicit ordered lists and draws one series
per arm, so adding an architecture or a mechanism is a data change rather than a re-layout
(`harness/STYLE.md` §7). Category order is shared across panels, bars are horizontal wherever
the labels are long, and each arm keeps one colour across every figure in the notebook.

No model calls and no metric computation happen here; the notebook wires
`edit_directions.py` into these.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pim.figures.theme import PALETTE, style_ax

#: One colour per mechanism, held fixed across every figure in the thread.
ARM_COLOR = {
    "Unsteered (no edit)": "0.55",
    "Counterfactual Overwriting": PALETTE[3],  # teal
    "Freeze-time Interp. TF @8": PALETTE[0],  # blue
    "Action Interface": PALETTE[2],  # orange
    "First Obs. TF": PALETTE[4],  # purple
}

#: Parenthetical appended to an arm's LEGEND entry only, where the arm needs a caveat carried
#: with it into every figure. `First Obs. TF` consumes the edit frame, so it is scored one frame
#: later than the others and must never be read as if it sat on the same step.
ARM_NOTE = {"First Obs. TF": "leads by one frame"}

_GRID = dict(color="0.85", lw=0.7, zorder=0)


def _lab(arm: str) -> str:
    return f"{arm} ({ARM_NOTE[arm]})" if arm in ARM_NOTE else arm


def _top_matter(fig, title, handles, labels, *, ncol=None, gap=None):
    """Grow the figure by exactly the header it needs, then place the title and legend in it.

    Callers size the figure for its **content**; the header is added here in inches from the
    number of title lines and legend rows, so a longer label that wraps the legend to another row
    moves the plot down instead of colliding with it. Fractional offsets alone cannot do this —
    they collide or leave a band of dead space the moment the content height changes, and that
    only shows up in the rendered image.
    """
    ncol = ncol or min(len(labels), 5)
    rows = int(np.ceil(len(labels) / ncol))
    n_title = title.count("\n") + 1
    header = 0.30 * n_title + 0.24 * rows + 0.12
    h = fig.get_figheight() + header
    fig.set_figheight(h)
    fig.suptitle(title, fontsize=11, y=1.0 - 0.10 / h, va="top")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0 - (0.30 * n_title) / h),
        ncol=ncol,
        frameon=False,
        handlelength=2.2,
        fontsize=9,
    )
    return 1.0 - header / h


def _fit(fig, top: float) -> None:
    """Lay the axes out, then hand the reserved header back — `tight_layout` re-reserves space
    for the suptitle and legend it can see, which double-counts what `_top_matter` already
    allowed for and leaves a band of dead space under the legend."""
    fig.tight_layout()
    fig.subplots_adjust(top=top)


def _group_positions(n_models: int, n_arms: int, span: float = 0.74):
    """y positions for grouped horizontal bars.

    Model groups run top-to-bottom and **arms run top-to-bottom within a group in the order
    given**, so the bars are read in the same order as the legend rather than mirrored.
    """
    h = span / n_arms
    base = np.arange(n_models)[::-1]
    return base, h, [((n_arms - 1) / 2 - a) * h for a in range(n_arms)]


def grouped_barh(
    ax,
    values: dict[str, list[float]],
    models: list[str],
    arms: list[str],
    *,
    errors: dict[str, list[float]] | None = None,
    flag: dict[str, list[bool]] | None = None,
) -> None:
    """Grouped horizontal bars: one group per model, one bar per arm. Shared by several figures.

    `flag` marks individual bars with a hatch — used for a guardrail failure, which stays as a
    *mark* rather than a printed second number on the bar (`harness/STYLE.md` §3).
    """
    base, h, offs = _group_positions(len(models), len(arms))
    for a, arm in enumerate(arms):
        v = np.asarray(values[arm], float)
        e = None if errors is None else np.asarray(errors[arm], float)
        hatch = None if flag is None else flag.get(arm)
        ax.barh(
            base + offs[a],
            v,
            height=h * 0.9,
            color=ARM_COLOR.get(arm, PALETTE[a % len(PALETTE)]),
            edgecolor="white",
            linewidth=0.6,
            xerr=e,
            error_kw=dict(ecolor="0.35", elinewidth=0.9, capsize=2),
            zorder=3,
        )
        for i, val in enumerate(v):
            if np.isnan(val):
                ax.text(
                    0.0,
                    base[i] + offs[a],
                    "  n/a",
                    ha="left",
                    va="center",
                    fontsize=7.5,
                    color="0.45",
                    zorder=5,
                )
        if hatch is not None:
            for i, bad in enumerate(hatch):
                if bad:
                    ax.barh(
                        base[i] + offs[a],
                        v[i],
                        height=h * 0.9,
                        facecolor="none",
                        edgecolor="0.15",
                        hatch="////",
                        linewidth=0.0,
                        zorder=4,
                    )
    for y in base[:-1]:
        ax.axhline(y - 0.5, color="0.88", lw=0.8, zorder=1)
    ax.set_yticks(base)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_ylim(base[-1] - 0.55, base[0] + 0.55)
    ax.xaxis.grid(True, **_GRID)
    ax.set_axisbelow(True)
    style_ax(ax)


# ── 1. Does each mechanism actually edit the generation? ──────────────────────


def plot_edit_gate(
    index: dict[str, list[float]],
    fidelity: dict[str, list[float]],
    models: list[str],
    arms: list[str],
    *,
    title: str,
    step_label: str = "at the edit frame (rollout step 0)",
) -> Figure:
    """Edit Index per model per mechanism, with a mark on any arm whose fidelity ratio > 1."""
    fig, ax = plt.subplots(figsize=(10.5, 0.30 * len(models) * len(arms) + 1.7))
    flag = {a: [bool(f > 1.0) for f in fidelity[a]] for a in arms}
    grouped_barh(ax, index, models, arms, flag=flag)
    ax.axvline(0.0, color="0.4", lw=0.9, ls=":", zorder=2)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel(
        f"Edit Index {step_label}   (+1 = edited world · 0 = equidistant · −1 = no edit)",
        fontsize=9,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ARM_COLOR.get(a, PALETTE[i % len(PALETTE)]))
        for i, a in enumerate(arms)
    ]
    labels = [_lab(a) for a in arms]
    handles.append(
        plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="0.15", hatch="////")
    )
    labels.append("fidelity ratio > 1 (edit degraded the rollout)")
    handles.append(Line2D([0], [0], color="0.4", lw=0.9, ls=":"))
    labels.append("Edit Index = 0")
    top = _top_matter(fig, title, handles, labels, ncol=3)
    _fit(fig, top)
    return fig


def plot_index_by_step(
    curves: dict[str, dict[str, list[float]]],
    models: list[str],
    arms: list[str],
    *,
    title: str,
) -> Figure:
    """Edit Index at every rollout step — landing an edit and holding it are different results."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        for arm in arms:
            y = curves[model].get(arm)
            if y is None:
                continue
            ax.plot(
                np.arange(len(y)),
                y,
                color=ARM_COLOR.get(arm, PALETTE[0]),
                lw=1.8,
                marker="o",
                ms=2.6,
            )
        ax.axhline(0.0, color="0.4", lw=0.9, ls=":")
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel("rollout step (0 = the edit frame)", fontsize=9)
        ax.set_title(model, fontsize=9.5)
        ax.yaxis.grid(True, **_GRID)
        ax.set_axisbelow(True)
        style_ax(ax)
    axes[0].set_ylabel("Edit Index (−1 … +1)", fontsize=9)
    handles = [
        Line2D([0], [0], color=ARM_COLOR.get(a, PALETTE[0]), lw=1.8) for a in arms
    ] + [Line2D([0], [0], color="0.4", lw=0.9, ls=":")]
    top = _top_matter(
        fig, title, handles, [_lab(a) for a in arms] + ["Edit Index = 0"], ncol=5
    )
    _fit(fig, top)
    return fig


# ── 2. Do two mechanisms' latent displacements point the same way? ────────────


def plot_cosine_violins(
    reports: dict[str, dict],
    models: list[str],
    *,
    title: str,
    pair_label: str,
) -> Figure:
    """Per-sample cosine between two mechanisms' Δh, one row per model, against its own control.

    The whole distribution rather than a mean-and-error-bar: the mean is drawn on it, so this
    panel carries the summary as well and no second panel repeats it.
    """
    fig, ax = plt.subplots(figsize=(9.5, 0.72 * len(models) + 2.0))
    pos = np.arange(len(models))[::-1]
    for i, model in enumerate(models):
        r = reports[model]
        for data, color, off in (
            (r["cos_shuffled_per_sample"], "0.72", -0.18),
            (r["cos_per_sample"], PALETTE[3], 0.18),
        ):
            v = ax.violinplot(
                [data],
                positions=[pos[i] + off],
                vert=False,
                widths=0.33,
                showextrema=False,
                showmedians=False,
            )
            for body in v["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.85)
                body.set_edgecolor("white")
            ax.plot(
                [np.mean(data)],
                [pos[i] + off],
                marker="|",
                ms=13,
                mew=2.0,
                color="0.15",
                zorder=5,
            )
    ax.axvline(0.0, color="0.4", lw=0.9, ls=":")
    ax.set_yticks(pos)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlim(-1.0, 1.05)
    ax.set_xlabel("cosine between the two mechanisms' Δh (per episode)", fontsize=9)
    ax.xaxis.grid(True, **_GRID)
    ax.set_axisbelow(True)
    style_ax(ax)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE[3]),
        plt.Rectangle((0, 0), 1, 1, color="0.72"),
        Line2D([0], [0], color="0.15", lw=2.0),
        Line2D([0], [0], color="0.4", lw=0.9, ls=":"),
    ]
    top = _top_matter(
        fig,
        f"{title}\n{pair_label}",
        handles,
        [
            "matched pairs (same episode)",
            "shuffled pairs (chance)",
            "mean",
            "cosine = 0",
        ],
        ncol=4,
    )
    _fit(fig, top)
    return fig


def plot_cos_matrix(
    matrices: dict[str, np.ndarray],
    arms: list[str],
    models: list[str],
    *,
    title: str,
    chance: dict[str, float] | None = None,
) -> Figure:
    """Mechanism × mechanism cosine, one matrix per model. Same arm order in every panel."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.9 * n, 4.3))
    axes = np.atleast_1d(axes)
    short = [a.replace(" Interp. TF @8", "").replace(" Overwriting", "") for a in arms]
    for ax, model in zip(axes, models):
        M = matrices[model]
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        for i in range(len(arms)):
            for j in range(len(arms)):
                if np.isnan(M[i, j]):
                    ax.text(
                        j, i, "n/a", ha="center", va="center", fontsize=8, color="0.3"
                    )
                    continue
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(M[i, j]) > 0.55 else "0.15",
                )
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(short, rotation=30, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels(short if ax is axes[0] else [], fontsize=8.5)
        sub = (
            model
            if chance is None
            else f"{model}\nshuffled-pair chance {chance[model]:+.3f}"
        )
        ax.set_title(sub, fontsize=9.5)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cb.set_label("cosine (per episode, then averaged)", fontsize=9)
    fig.suptitle(title, fontsize=11, y=0.99)
    return fig


# ── 3. How big is the edit, in scales that mean something ─────────────────────


def plot_magnitudes(
    rel_state: dict[str, list[float]],
    rel_state_sd: dict[str, list[float]],
    rel_step: dict[str, list[float]],
    rel_step_sd: dict[str, list[float]],
    models: list[str],
    arms: list[str],
    *,
    title: str,
) -> Figure:
    """Δh magnitude against two reference scales. Separate panels: they are different quantities.

    Left is normalised by the state's own norm — meaningful *within* an architecture. Right is
    normalised by one ordinary dynamics step, which is the scale that transfers across them.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 0.30 * len(models) * len(arms) + 1.9), sharey=True
    )
    grouped_barh(axes[0], rel_state, models, arms, errors=rel_state_sd)
    axes[0].set_xlabel("‖Δh‖ ÷ ‖h‖ of the unedited state", fontsize=9)
    axes[0].set_title("(a) relative to the state's own norm", fontsize=10)
    grouped_barh(axes[1], rel_step, models, arms, errors=rel_step_sd)
    axes[1].axvline(1.0, color="0.4", lw=0.9, ls=":")
    axes[1].set_xlabel("‖Δh‖ ÷ ‖one ordinary dynamics step‖", fontsize=9)
    axes[1].set_title(
        "(b) relative to one step of the model's own dynamics", fontsize=10
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ARM_COLOR.get(a, PALETTE[i % len(PALETTE)]))
        for i, a in enumerate(arms)
    ] + [Line2D([0], [0], color="0.4", lw=0.9, ls=":")]
    top = _top_matter(
        fig, title, handles, [_lab(a) for a in arms] + ["one dynamics step"], ncol=4
    )
    _fit(fig, top)
    return fig


# ── 4. Structure: one shared direction, or one per edit? ──────────────────────


def plot_consistency(
    mean_cos: dict[str, list[float]],
    sd: dict[str, list[float]],
    models: list[str],
    arms: list[str],
    *,
    title: str,
    xlim: tuple[float, float] = (-0.05, 0.35),
) -> Figure:
    """Mean cosine between the Δh of DIFFERENT edits — is there a generic "an object moved" axis?

    For unrelated directions the expected mean is 0 (`1/√H` is the per-pair standard deviation,
    not a floor), so the reference drawn is 0 and the error bars are the per-pair spread.
    """
    fig, ax = plt.subplots(figsize=(9.5, 0.30 * len(models) * len(arms) + 1.7))
    grouped_barh(ax, mean_cos, models, arms, errors=sd)
    ax.axvline(0.0, color="0.4", lw=0.9, ls=":")
    ax.set_xlim(*xlim)
    ax.set_xlabel(
        "mean cosine between Δh of different edit episodes (chance = 0; bars are ± the "
        "per-pair spread)",
        fontsize=9,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ARM_COLOR.get(a, PALETTE[i % len(PALETTE)]))
        for i, a in enumerate(arms)
    ] + [Line2D([0], [0], color="0.4", lw=0.9, ls=":")]
    top = _top_matter(
        fig,
        title,
        handles,
        [_lab(a) for a in arms] + ["chance (unrelated directions)"],
        ncol=3,
    )
    _fit(fig, top)
    return fig


def plot_rowspace(
    enrichment: dict[str, list[float]],
    models: list[str],
    arms: list[str],
    *,
    title: str,
) -> Figure:
    """Share of Δh visible to a linear position probe, ÷ the chance level for a random vector.

    Enrichment rather than the raw fraction: chance is `√(d/H)` and `H` differs by architecture
    here (64 … 320), so a raw fraction would manufacture a trend that is entirely the moving
    chance level (`harness/ANALYSIS.md` §8.3).
    """
    fig, ax = plt.subplots(figsize=(9.5, 0.30 * len(models) * len(arms) + 1.7))
    grouped_barh(ax, enrichment, models, arms)
    ax.axvline(1.0, color="0.4", lw=1.1, ls="--")
    ax.set_xlabel(
        "‖P_row·Δh‖ / ‖Δh‖ ÷ chance √(d/H)   (1.0 = as visible as a random direction)",
        fontsize=9,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ARM_COLOR.get(a, PALETTE[i % len(PALETTE)]))
        for i, a in enumerate(arms)
    ] + [Line2D([0], [0], color="0.4", lw=1.1, ls="--")]
    top = _top_matter(
        fig,
        title,
        handles,
        [_lab(a) for a in arms] + ["chance for a random direction"],
        ncol=3,
    )
    _fit(fig, top)
    return fig
