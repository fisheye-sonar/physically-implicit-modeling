"""Academic-aesthetic plotting for evaluation results.

All figures use the light/academic theme: white background, Okabe-Ito
colorblind-safe palette, clean spines, minimal decoration.

This module consolidates style_ax and plot_color from nb_viz.py along with
standard evaluation plot types.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

# ── Theme ─────────────────────────────────────────────────────────────────────

_BG_HEX   = "#ffffff"
_TEXT_COLOR = "#172239"   # dark navy for titles and labels
_TICK_COLOR = "#555555"   # medium gray for spines and ticks
_SPINE_COLOR = "#555555"

# Okabe-Ito palette — colorblind safe
PALETTE: list[tuple[float, float, float]] = [
    (0.00, 0.45, 0.70),  # blue      #0072B2
    (0.84, 0.37, 0.00),  # vermilion #D55E00
    (0.90, 0.62, 0.00),  # orange    #E69F00
    (0.00, 0.62, 0.45),  # teal      #009E73
    (0.80, 0.47, 0.65),  # purple    #CC79A7
    (0.34, 0.71, 0.91),  # sky       #56B4E9
    (0.94, 0.89, 0.26),  # yellow    #F0E442
]

# Mapping from simulator OBJECT_COLORS (cyan/coral/amber/green/violet) to Okabe-Ito
# Maintained here rather than importing from simulator to keep eval independent.
_SIM_TO_OKABE: dict[tuple, tuple] = {
    (0.00, 0.83, 1.00): PALETTE[0],  # cyan   → blue
    (1.00, 0.42, 0.42): PALETTE[1],  # coral  → vermilion
    (1.00, 0.85, 0.24): PALETTE[2],  # amber  → orange
    (0.42, 0.80, 0.47): PALETTE[3],  # green  → teal
    (0.78, 0.48, 1.00): PALETTE[4],  # violet → purple
}


def plot_color(sim_color) -> tuple:
    """Map a simulator object color to its Okabe-Ito equivalent.

    Use in all result figures so colors are consistent and accessible on
    white backgrounds.
    """
    key = tuple(round(float(v), 2) for v in sim_color)
    return _SIM_TO_OKABE.get(key, tuple(float(v) for v in sim_color))


def style_ax(ax: plt.Axes) -> None:
    """Apply light academic theme to a matplotlib Axes."""
    ax.set_facecolor(_BG_HEX)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE_COLOR)
    ax.tick_params(colors=_TICK_COLOR, labelsize=9)


# ── Standard evaluation plots ─────────────────────────────────────────────────


def plot_training_curves(
    metrics: list[dict],
    *,
    title: str = "Training curves",
    log_y: bool = False,
) -> plt.Figure:
    """Plot train and val loss curves from a list of epoch metric dicts.

    Parameters
    ----------
    metrics  : list of dicts with keys "epoch", "train_loss", "val_loss"
    title    : figure title
    log_y    : if True, use log scale on y-axis

    Returns
    -------
    fig : matplotlib Figure
    """
    epochs     = [m["epoch"] for m in metrics]
    train_loss = [m["train_loss"] for m in metrics]
    val_loss   = [m["val_loss"] for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
    style_ax(ax)
    ax.plot(epochs, train_loss, color=PALETTE[0], linewidth=1.8, label="train")
    ax.plot(epochs, val_loss,   color=PALETTE[1], linewidth=1.8, label="val",
            linestyle="--")
    ax.set_xlabel("epoch", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("MSE loss", color=_TEXT_COLOR, fontsize=10)
    ax.set_title(title, color=_TEXT_COLOR, fontsize=11)
    ax.legend(frameon=False, labelcolor=_TEXT_COLOR)
    if log_y:
        ax.set_yscale("log")
    ax.tick_params(colors=_TICK_COLOR, labelsize=9)
    plt.tight_layout()
    return fig


def plot_horizon_sweep(
    horizons: np.ndarray,
    mse: np.ndarray,
    *,
    title: str = "Next-n MSE (autoregressive rollout)",
    color: tuple | None = None,
) -> plt.Figure:
    """MSE at each AR horizon step.

    Parameters
    ----------
    horizons : (T_rollout,) or (n_horizons,) step indices
    mse      : (T_rollout,) MSE values
    """
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
    style_ax(ax)
    ax.plot(horizons, mse, color=color or PALETTE[0], linewidth=1.8)
    ax.set_xlabel("steps ahead", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("observation MSE", color=_TEXT_COLOR, fontsize=10)
    ax.set_title(title, color=_TEXT_COLOR, fontsize=11)
    ax.tick_params(colors=_TICK_COLOR, labelsize=9)
    plt.tight_layout()
    return fig


def plot_mse_by_context(
    context_lengths: np.ndarray,
    mse: np.ndarray,
    *,
    title: str = "MSE vs context length",
    color: tuple | None = None,
) -> plt.Figure:
    """MSE as a function of context length."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
    style_ax(ax)
    ax.plot(context_lengths, mse, color=color or PALETTE[0], linewidth=1.8)
    ax.set_xlabel("context frames", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("observation MSE", color=_TEXT_COLOR, fontsize=10)
    ax.set_title(title, color=_TEXT_COLOR, fontsize=11)
    ax.tick_params(colors=_TICK_COLOR, labelsize=9)
    plt.tight_layout()
    return fig


def plot_per_component_bars(
    component_labels: Sequence[str],
    mse_arrays: dict[str, np.ndarray],
    *,
    title: str = "Per-component RMSE",
    ylabel: str = "RMSE",
) -> plt.Figure:
    """Bar chart of per-component RMSE for one or more methods.

    Parameters
    ----------
    component_labels : labels for each state component
    mse_arrays       : dict mapping method name → (n_components,) RMSE array
    ylabel           : y-axis label (default "RMSE")
    """
    n_comp = len(component_labels)
    n_methods = len(mse_arrays)
    x = np.arange(n_comp)
    width = 0.8 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(max(5, n_comp * 1.2), 4), facecolor=_BG_HEX)
    style_ax(ax)

    for k, (name, vals) in enumerate(mse_arrays.items()):
        offset = (k - (n_methods - 1) / 2) * width
        ax.bar(x + offset, vals, width * 0.9, label=name, color=PALETTE[k % len(PALETTE)])

    ax.set_xticks(x)
    ax.set_xticklabels(component_labels, rotation=30, ha="right",
                       color=_TEXT_COLOR, fontsize=9)
    ax.set_ylabel(ylabel, color=_TEXT_COLOR, fontsize=10)
    ax.set_title(title, color=_TEXT_COLOR, fontsize=11)
    ax.tick_params(colors=_TICK_COLOR, labelsize=9)
    if n_methods > 1:
        ax.legend(frameon=False, labelcolor=_TEXT_COLOR)
    plt.tight_layout()
    return fig


def plot_trajectory_comparison(
    gt: np.ndarray,                          # (T, n_obj, 2) ground-truth positions
    predictions: dict[str, np.ndarray],      # name → (T, n_obj, 2)
    scene_colors: np.ndarray | None = None,  # (n_obj, 3) simulator RGB colors
    *,
    title: str = "Trajectory comparison",
) -> plt.Figure:
    """Overlay GT and predicted trajectories for each object.

    Parameters
    ----------
    gt           : (T, n_obj, 2) ground-truth x-y positions
    predictions  : dict of method name → (T, n_obj, 2) predicted positions
    scene_colors : optional (n_obj, 3) RGB colors from scene; mapped to Okabe-Ito
    title        : figure title
    """
    T, n_obj, _ = gt.shape

    fig, axes = plt.subplots(1, n_obj, figsize=(4 * n_obj, 4), facecolor=_BG_HEX)
    if n_obj == 1:
        axes = [axes]

    for obj_idx, ax in enumerate(axes):
        style_ax(ax)
        if scene_colors is not None:
            obj_color = plot_color(scene_colors[obj_idx])
        else:
            obj_color = PALETTE[obj_idx % len(PALETTE)]

        # Ground truth
        ax.plot(gt[:, obj_idx, 0], gt[:, obj_idx, 1],
                color=obj_color, linewidth=2.0, label="GT", zorder=3)
        ax.scatter(gt[0, obj_idx, 0], gt[0, obj_idx, 1],
                   color=obj_color, s=40, zorder=4)

        # Predictions
        line_styles = ["--", ":", "-."]
        for k, (name, pred) in enumerate(predictions.items()):
            ax.plot(pred[:, obj_idx, 0], pred[:, obj_idx, 1],
                    color=PALETTE[(obj_idx + k + 1) % len(PALETTE)],
                    linewidth=1.5, linestyle=line_styles[k % len(line_styles)],
                    label=name, alpha=0.8)

        ax.set_title(f"object {obj_idx}", color=_TEXT_COLOR, fontsize=10)
        ax.set_xlabel("x", color=_TEXT_COLOR, fontsize=9)
        ax.set_ylabel("y", color=_TEXT_COLOR, fontsize=9)
        ax.tick_params(colors=_TICK_COLOR, labelsize=8)

    axes[0].legend(frameon=False, labelcolor=_TEXT_COLOR, fontsize=8)
    fig.suptitle(title, color=_TEXT_COLOR, fontsize=11)
    plt.tight_layout()
    return fig


def plot_coherence_distribution(
    scores: dict[str, np.ndarray],
    *,
    title: str = "Rollout coherence score distribution",
) -> plt.Figure:
    """Histogram of coherence scores for one or more methods.

    Parameters
    ----------
    scores : dict mapping method name → (N,) array of per-sample scores
    """
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
    style_ax(ax)

    for k, (name, vals) in enumerate(scores.items()):
        ax.hist(vals, bins=30, alpha=0.6, color=PALETTE[k % len(PALETTE)],
                label=f"{name}  (μ={vals.mean():.3f})", density=True)

    ax.set_xlabel("coherence score (lower = smoother)", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("density", color=_TEXT_COLOR, fontsize=10)
    ax.set_title(title, color=_TEXT_COLOR, fontsize=11)
    ax.tick_params(colors=_TICK_COLOR, labelsize=9)
    ax.legend(frameon=False, labelcolor=_TEXT_COLOR)
    plt.tight_layout()
    return fig
