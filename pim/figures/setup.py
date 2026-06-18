"""Setup-stage figures: training curves and dataset waterfall overview."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from pim.figures.theme import (
    PALETTE,
    _BG_HEX,
    _TEXT_COLOR,
    _TICK_COLOR,
    style_ax,
    style_ax_dark,
)
from pim.simulator.viz import (
    _BG_HEX as _DARK_BG_HEX,
    _TEXT_COLOR as _DARK_TEXT_COLOR,
    make_waterfall,
)
from pim.simulator.dataset import load_sample


def plot_training_curves(
    metrics_history: list[dict],
    *,
    best_epoch: int | None = None,
    run_name: str | None = None,
    show_components: bool = False,
    log_x: bool = False,
) -> Figure:
    """Train + val loss curves; optional log-x; optional recon/KL subplot (RSSM).

    Parameters
    ----------
    metrics_history : list of per-epoch dicts with keys epoch, train_loss, val_loss
                      (and optionally recon_loss, val_recon_loss, kl_loss, val_kl_loss)
    best_epoch      : marks a vertical dashed line at this epoch (optional)
    run_name        : figure title (optional)
    show_components : add a second subplot with recon/KL components (RSSM only)
    log_x           : log scale on the x axis
    """
    epochs = [m["epoch"] for m in metrics_history]
    train_loss = [m["train_loss"] for m in metrics_history]
    val_loss = [m["val_loss"] for m in metrics_history]

    n_rows = 1 + int(show_components)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(7, 4 + 3 * int(show_components)), facecolor=_BG_HEX,
    )
    if n_rows == 1:
        axes = [axes]

    ax = axes[0]
    style_ax(ax)
    ax.plot(epochs, train_loss, color=PALETTE[0], linewidth=1.8, label="train")
    ax.plot(epochs, val_loss, color=PALETTE[1], linewidth=1.8, linestyle="--", label="val")
    if best_epoch is not None:
        ax.axvline(
            best_epoch, color=_TICK_COLOR, linewidth=1.0, linestyle=":", alpha=0.7,
            label=f"best epoch {best_epoch}",
        )
    if log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epoch", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("loss", color=_TEXT_COLOR, fontsize=10)
    ax.legend(frameon=False, labelcolor=_TEXT_COLOR)
    ax.set_title(
        "training curves" + (" (log-log)" if log_x else ""),
        color=_TEXT_COLOR, fontsize=10,
    )

    if show_components:
        ax2 = axes[1]
        style_ax(ax2)
        ax2.plot(epochs, [m["recon_loss"] for m in metrics_history],
                 color=PALETTE[0], linewidth=1.5, label="train recon")
        ax2.plot(epochs, [m["val_recon_loss"] for m in metrics_history],
                 color=PALETTE[0], linewidth=1.5, linestyle="--", label="val recon")
        ax2.plot(epochs, [m["kl_loss"] for m in metrics_history],
                 color=PALETTE[2], linewidth=1.5, label="train KL")
        ax2.plot(epochs, [m["val_kl_loss"] for m in metrics_history],
                 color=PALETTE[2], linewidth=1.5, linestyle="--", label="val KL")
        if log_x:
            ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("epoch", color=_TEXT_COLOR, fontsize=10)
        ax2.set_ylabel("loss components", color=_TEXT_COLOR, fontsize=10)
        ax2.legend(frameon=False, labelcolor=_TEXT_COLOR, fontsize=8)

    if run_name:
        fig.suptitle(run_name, color=_TEXT_COLOR, fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    return fig


def plot_dataset_overview(
    test_h5_path: str,
    *,
    n_samples: int = 8,
) -> Figure:
    """Grid of waterfall thumbnails for the first n_samples test samples."""
    n_cols = max(1, n_samples // 2)
    n_rows = 2 if n_samples > n_cols else 1
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 1.8, 6 if n_rows == 2 else 3),
        facecolor=_DARK_BG_HEX,
    )
    fig.suptitle(
        "dataset overview — stored observations",
        color=_DARK_TEXT_COLOR, fontsize=12,
    )
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, idx in zip(axes_flat, range(n_samples)):
        scene_i, obs_depth_i, obs_id_i, obs_intensity_i = load_sample(test_h5_path, idx)
        wf = make_waterfall(obs_depth_i, obs_id_i, obs_intensity_i, scene_i)
        style_ax_dark(ax)
        ax.imshow(wf, aspect="auto", origin="upper", interpolation="nearest")
        ax.set_title(f"#{idx}", color=_DARK_TEXT_COLOR, fontsize=10)
        ax.axis("off")
    # Hide any unused axes
    for ax in axes_flat[n_samples:]:
        ax.axis("off")
    plt.tight_layout()
    return fig
