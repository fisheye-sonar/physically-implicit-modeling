"""Recovery-stage figures: per-object / per-coord bars, RMSE vs context, traj viz."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pim.eval.baselines import PosBaselines
from pim.eval.recovery import RecoveryMetrics
from pim.extractors.spec import ProbeSpec
from pim.figures.theme import (
    PALETTE,
    _BG_HEX,
    _TEXT_COLOR,
    _TICK_COLOR,
    plot_color,
    style_ax,
)


def plot_recovery_bars(
    metrics: dict[str, RecoveryMetrics],
    probes: list[ProbeSpec],
    *,
    n_obj: int,
) -> Figure:
    """Two-panel bar chart: per-object RMSE (X+Y averaged) and X vs Y overall."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), facecolor=_BG_HEX)

    # Per-object
    style_ax(ax1)
    obj_labels = [f"obj {i}" for i in range(n_obj)] + ["overall"]
    n_p = len(probes)
    x = np.arange(len(obj_labels))
    width = 0.8 / max(n_p, 1)
    for k, p in enumerate(probes):
        per_obj = np.sqrt(metrics[p.name].per_component_mse.reshape(n_obj, 2).mean(1))
        overall = np.sqrt(metrics[p.name].overall_mse)
        vals = np.append(per_obj, overall)
        offset = (k - (n_p - 1) / 2) * width
        ax1.bar(x + offset, vals, width * 0.9, label=p.name, color=PALETTE[p.color_idx])
    ax1.set_xticks(x)
    ax1.set_xticklabels(obj_labels, rotation=30, ha="right", color=_TEXT_COLOR, fontsize=9)
    ax1.set_ylabel("RMSE", color=_TEXT_COLOR, fontsize=10)
    ax1.set_title("Per-object position recovery RMSE", color=_TEXT_COLOR, fontsize=11)
    if n_p > 1:
        ax1.legend(frameon=False, labelcolor=_TEXT_COLOR)

    # X vs Y overall
    style_ax(ax2)
    coord_labels = ["X (overall)", "Y (overall)"]
    xc = np.arange(2)
    for k, p in enumerate(probes):
        m = metrics[p.name].per_component_mse
        x_rmse = float(np.sqrt(m[0::2].mean()))
        y_rmse = float(np.sqrt(m[1::2].mean()))
        offset = (k - (n_p - 1) / 2) * width
        ax2.bar(xc + offset, [x_rmse, y_rmse], width * 0.9,
                label=p.name, color=PALETTE[p.color_idx])
    ax2.set_xticks(xc)
    ax2.set_xticklabels(coord_labels, color=_TEXT_COLOR, fontsize=9)
    ax2.set_ylabel("RMSE", color=_TEXT_COLOR, fontsize=10)
    ax2.set_title("Position recovery RMSE by coordinate", color=_TEXT_COLOR, fontsize=11)
    if n_p > 1:
        ax2.legend(frameon=False, labelcolor=_TEXT_COLOR)

    plt.tight_layout()
    return fig


def plot_recovery_by_context(
    metrics: dict[str, RecoveryMetrics],
    probes: list[ProbeSpec],
    *,
    baselines: PosBaselines | None = None,
    include: set[str] | None = None,
) -> Figure:
    """Position recovery RMSE vs context length, one line per probe."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
    style_ax(ax)
    for p in probes:
        m = metrics[p.name]
        ax.plot(
            np.arange(1, len(m.mse_by_context) + 1),
            np.sqrt(m.mse_by_context),
            color=PALETTE[p.color_idx], linewidth=1.8,
            linestyle=p.linestyle, label=p.name,
        )
    if baselines is not None:
        _pos_lines = [
            ("random_rmse",   baselines.random_rmse,   f"random baseline ({baselines.random_rmse:.3f})",   _TICK_COLOR, ":", 0.7),
            ("identity_rmse", baselines.identity_rmse, f"identity RMSE ({baselines.identity_rmse:.3f})",   PALETTE[5],  ":", 0.8),
            ("noise_std",     baselines.noise_std,     f"position noise σ ({baselines.noise_std:.3f})",    PALETTE[2],  ":", 0.8),
        ]
        for name, value, label, color, ls, alpha in _pos_lines:
            if include is None or name in include:
                ax.axhline(value, color=color, linewidth=1.2, linestyle=ls, alpha=alpha, label=label)
    ax.set_xlabel("context frames", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("position recovery RMSE", color=_TEXT_COLOR, fontsize=10)
    ax.set_title("Position recovery RMSE vs context length", color=_TEXT_COLOR, fontsize=11)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(colors=_TICK_COLOR)
    ax.legend(frameon=False, labelcolor=_TEXT_COLOR, fontsize=8)
    plt.tight_layout()
    return fig


def plot_recovery_trajectory(
    positions_gt: np.ndarray,           # (T_align, n_obj, 2)  GT aligned to states (no last frame)
    decoded_per_probe: dict[str, np.ndarray],   # name → (T_align, n_obj, 2)
    probes: list[ProbeSpec],
    scene_colors: np.ndarray,           # (n_obj, 3) simulator RGB
    *,
    vis_mask: np.ndarray | None = None, # (T_align,) bool
    sample_idx: int = 0,
    title_suffix: str = "teacher forcing",
) -> Figure:
    """Per-coordinate x/y trajectory plot: GT solid line + each probe's scatter."""
    T = positions_gt.shape[0]
    n_obj = positions_gt.shape[1]
    timesteps = np.arange(T)
    vis_t = timesteps[vis_mask] if vis_mask is not None else timesteps

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=_BG_HEX)
    fig.suptitle(
        f"Sample {sample_idx}  —  position recovery ({title_suffix})",
        color=_TEXT_COLOR, fontsize=11, fontweight="bold",
    )
    for ax, coord, coord_lbl in zip(axes, [0, 1], ["x", "y (depth)"]):
        style_ax(ax)
        for obj in range(n_obj):
            color = plot_color(scene_colors[obj])
            ax.plot(timesteps, positions_gt[:, obj, coord], color=color, linewidth=1.8)
            for p in probes:
                pos = decoded_per_probe[p.name]
                if vis_mask is not None:
                    ax.scatter(vis_t, pos[vis_mask, obj, coord],
                               color=color, s=18, marker=p.marker, alpha=0.75)
                else:
                    ax.scatter(timesteps, pos[:, obj, coord],
                               color=color, s=18, marker=p.marker, alpha=0.75)
        ax.set_xlabel("frame", color=_TEXT_COLOR, fontsize=9)
        ax.set_ylabel(coord_lbl, color=_TEXT_COLOR, fontsize=9)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.tick_params(colors=_TICK_COLOR)

    handles = [Line2D([0], [0], color="gray", linewidth=1.8, label="GT")] + [
        Line2D([0], [0], color="gray", marker=p.marker, linestyle="none",
               markersize=6, label=p.name)
        for p in probes
    ]
    axes[0].legend(handles=handles, frameon=False, labelcolor=_TEXT_COLOR, fontsize=8)
    plt.tight_layout()
    return fig
