"""Prediction-stage figures: 1-step MSE vs context length, horizon MSE."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from pim.eval.baselines import ObsBaselines
from pim.figures.theme import (
    PALETTE,
    _BG_HEX,
    _TEXT_COLOR,
    _TICK_COLOR,
    style_ax,
)


def _add_obs_baselines(ax, baselines: ObsBaselines, include: set[str] | None = None) -> None:
    lines = [
        ("random_rmse",    baselines.random_rmse,    f"random baseline ({baselines.random_rmse:.3f})",    _TICK_COLOR, ":", 0.7),
        ("identity_rmse",  baselines.identity_rmse,  f"identity RMSE ({baselines.identity_rmse:.3f})",    PALETTE[5],  ":", 0.8),
        ("noise_std",      baselines.noise_std,       f"applied noise σ ({baselines.noise_std:.3f})",      PALETTE[2],  ":", 0.8),
        ("noise_floor_rmse", baselines.noise_floor_rmse, f"noise floor RMSE ({baselines.noise_floor_rmse:.3f})", PALETTE[3], ":", 0.8),
    ]
    for name, value, label, color, ls, alpha in lines:
        if include is None or name in include:
            ax.axhline(value, color=color, linewidth=1.2, linestyle=ls, alpha=alpha, label=label)


def plot_mse_by_context(
    context_lengths: np.ndarray,
    mse_noisy: np.ndarray,
    mse_clean: np.ndarray,
    *,
    n_context_warmup: int,
    baselines: ObsBaselines | None = None,
    include: set[str] | None = None,
) -> Figure:
    """1-step prediction RMSE vs context length, noisy + clean targets."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
    style_ax(ax)
    ax.plot(context_lengths, np.sqrt(mse_noisy), color=PALETTE[0], linewidth=1.8,
            label="vs noisy obs (AR warm-up)")
    ax.plot(context_lengths, np.sqrt(mse_clean), color=PALETTE[0], linewidth=1.8,
            linestyle="--", label="vs clean obs (TF warm-up)")
    if baselines is not None:
        _add_obs_baselines(ax, baselines, include)
    ax.set_xlabel("context frames", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("observation RMSE", color=_TEXT_COLOR, fontsize=10)
    ax.set_title(
        f"1-step prediction RMSE vs context length  (warm-up={n_context_warmup})",
        color=_TEXT_COLOR, fontsize=11,
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(colors=_TICK_COLOR)
    ax.legend(frameon=False, labelcolor=_TEXT_COLOR, fontsize=8)
    plt.tight_layout()
    return fig


def plot_horizon_rmse(
    horizon_mse_noisy: np.ndarray,
    horizon_mse_clean: np.ndarray,
    *,
    n_context: int,
    title: str = "Horizon RMSE (AR rollout)",
    baselines: ObsBaselines | None = None,
    include: set[str] | None = None,
) -> Figure:
    """Per-step RMSE over an AR rollout horizon, noisy + clean targets."""
    _steps = np.arange(1, len(horizon_mse_noisy) + 1)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
    style_ax(ax)
    ax.plot(_steps, np.sqrt(horizon_mse_noisy), color=PALETTE[0], linewidth=1.8,
            label="vs noisy obs")
    ax.plot(_steps, np.sqrt(horizon_mse_clean), color=PALETTE[0], linewidth=1.8,
            linestyle="--", label="vs clean obs")
    if baselines is not None:
        _add_obs_baselines(ax, baselines, include)
    ax.set_xlabel("steps ahead", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("observation RMSE", color=_TEXT_COLOR, fontsize=10)
    ax.set_title(f"{title}  (warm-up={n_context})", color=_TEXT_COLOR, fontsize=11)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(colors=_TICK_COLOR)
    ax.legend(frameon=False, labelcolor=_TEXT_COLOR, fontsize=8)
    plt.tight_layout()
    return fig
