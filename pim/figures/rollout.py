"""Rollout-stage figures: observation drift, position drift, coherence, 3-panel waterfall."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pim.eval.baselines import ObsBaselines, PosBaselines
from pim.extractors.spec import ProbeSpec
from pim.figures.theme import (
    PALETTE,
    _BG_HEX,
    _TEXT_COLOR,
    _TICK_COLOR,
    plot_color,
    style_ax,
    style_ax_dark,
)
from pim.simulator.viz import (
    _BG as _DARK_BG_ARRAY,
    _BG_HEX as _DARK_BG_HEX,
    _TEXT_COLOR as _DARK_TEXT_COLOR,
    make_waterfall,
)
from pim.simulator.dataset import load_sample


def plot_observation_drift(
    drift_mse_noisy: np.ndarray,
    drift_mse_clean: np.ndarray | None,
    *,
    n_context: int,
    n_rollout: int,
    baselines: ObsBaselines | None = None,
    include: set[str] | None = None,
) -> Figure:
    """Per-step observation RMSE during AR rollout."""
    steps = np.arange(1, n_rollout + 1)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
    style_ax(ax)
    ax.plot(steps, np.sqrt(drift_mse_noisy), color=PALETTE[0], linewidth=1.8,
            label="vs noisy obs")
    if drift_mse_clean is not None:
        ax.plot(steps, np.sqrt(drift_mse_clean), color=PALETTE[0], linewidth=1.8,
                linestyle="--", label="vs clean obs")
    if baselines is not None:
        _obs_lines = [
            ("random_rmse",      baselines.random_rmse,      f"random baseline ({baselines.random_rmse:.3f})",      _TICK_COLOR, ":", 0.7),
            ("identity_rmse",    baselines.identity_rmse,    f"identity RMSE ({baselines.identity_rmse:.3f})",      PALETTE[5],  ":", 0.8),
            ("noise_std",        baselines.noise_std,        f"applied noise σ ({baselines.noise_std:.3f})",        PALETTE[2],  ":", 0.8),
            ("noise_floor_rmse", baselines.noise_floor_rmse, f"noise floor RMSE ({baselines.noise_floor_rmse:.3f})", PALETTE[3], ":", 0.8),
        ]
        for name, value, label, color, ls, alpha in _obs_lines:
            if include is None or name in include:
                ax.axhline(value, color=color, linewidth=1.2, linestyle=ls, alpha=alpha, label=label)
    ax.set_xlabel("steps ahead", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("observation RMSE", color=_TEXT_COLOR, fontsize=10)
    ax.set_title(
        f"Observation drift  (warm-up={n_context}, rollout={n_rollout})",
        color=_TEXT_COLOR, fontsize=11,
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(colors=_TICK_COLOR)
    ax.legend(frameon=False, labelcolor=_TEXT_COLOR, fontsize=8)
    plt.tight_layout()
    return fig


def plot_position_drift(
    drift_per_probe: dict[str, np.ndarray],   # name → (n_rollout,) MSE per step
    probes: list[ProbeSpec],
    *,
    n_context: int,
    n_rollout: int,
    baselines: PosBaselines | None = None,
    include: set[str] | None = None,
) -> Figure:
    """Per-step decoded-position RMSE per probe."""
    steps = np.arange(1, n_rollout + 1)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
    style_ax(ax)
    for p in probes:
        ax.plot(steps, np.sqrt(drift_per_probe[p.name]),
                color=PALETTE[p.color_idx], linewidth=1.8,
                linestyle=p.linestyle, label=p.name)
    if baselines is not None:
        _pos_lines = [
            ("random_rmse",   baselines.random_rmse,   f"random baseline ({baselines.random_rmse:.3f})",   _TICK_COLOR, ":", 0.7),
            ("identity_rmse", baselines.identity_rmse, f"identity RMSE ({baselines.identity_rmse:.3f})",   PALETTE[5],  ":", 0.8),
            ("noise_std",     baselines.noise_std,     f"position noise σ ({baselines.noise_std:.3f})",    PALETTE[2],  ":", 0.8),
        ]
        for name, value, label, color, ls, alpha in _pos_lines:
            if include is None or name in include:
                ax.axhline(value, color=color, linewidth=1.2, linestyle=ls, alpha=alpha, label=label)
    ax.set_xlabel("steps ahead", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("position RMSE", color=_TEXT_COLOR, fontsize=10)
    ax.set_title(
        f"Decoded position drift  (warm-up={n_context}, rollout={n_rollout})",
        color=_TEXT_COLOR, fontsize=11,
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(colors=_TICK_COLOR)
    ax.legend(frameon=False, labelcolor=_TEXT_COLOR, fontsize=8)
    plt.tight_layout()
    return fig


def plot_coherence_bar(
    gt_scores: np.ndarray,
    probe_scores: dict[str, np.ndarray],
    probes: list[ProbeSpec],
    *,
    score_label: str = "smoothness",
) -> Figure:
    """Mean coherence/smoothness across GT and each probe."""
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=_BG_HEX)
    style_ax(ax)
    names = ["GT"] + [p.name for p in probes]
    means = [gt_scores.mean()] + [probe_scores[p.name].mean() for p in probes]
    colors = [PALETTE[3]] + [PALETTE[p.color_idx] for p in probes]
    ax.bar(names, means, color=colors, width=0.5)
    ax.set_ylabel(f"mean {score_label} score", color=_TEXT_COLOR, fontsize=10)
    ax.set_title(f"Trajectory {score_label} (lower = smoother)",
                 color=_TEXT_COLOR, fontsize=11)
    ax.tick_params(colors=_TICK_COLOR)
    plt.tight_layout()
    return fig


def plot_coherence_distribution(
    gt_scores: np.ndarray,
    probe_scores: dict[str, np.ndarray],
    probes: list[ProbeSpec],
    *,
    title: str = "Trajectory smoothness score distribution",
) -> Figure:
    """Histogram of per-sample smoothness scores, GT + each probe."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
    style_ax(ax)
    ax.hist(gt_scores, bins=30, alpha=0.6, color=PALETTE[3],
            label=f"GT  (μ={gt_scores.mean():.3f})", density=True)
    for p in probes:
        s = probe_scores[p.name]
        ax.hist(s, bins=30, alpha=0.6, color=PALETTE[p.color_idx],
                label=f"{p.name}  (μ={s.mean():.3f})", density=True)
    ax.set_xlabel("smoothness score (lower = smoother)", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("density", color=_TEXT_COLOR, fontsize=10)
    ax.set_title(title, color=_TEXT_COLOR, fontsize=11)
    ax.tick_params(colors=_TICK_COLOR)
    ax.legend(frameon=False, labelcolor=_TEXT_COLOR)
    plt.tight_layout()
    return fig


def plot_rollout_trajectory(
    positions_gt: np.ndarray,            # (T_full, n_obj, 2) full GT sequence
    decoded_per_probe: dict[str, np.ndarray],   # name → (n_rollout, n_obj, 2)
    probes: list[ProbeSpec],
    scene_colors: np.ndarray,            # (n_obj, 3)
    *,
    sample_idx: int,
    n_context: int,
    n_rollout: int,
    n_ctx_show: int = 8,
    score_label: str = "smoothness",
    sample_scores: dict[str, float] | None = None,
) -> Figure:
    """x/y trajectory plot for one sample over the rollout window (GT + decoded)."""
    n_obj = positions_gt.shape[1]
    n_ctx_show = min(n_ctx_show, n_context)
    ctx_frames = np.arange(n_context - n_ctx_show, n_context)
    roll_frames = np.arange(n_context, n_context + n_rollout)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=_BG_HEX)
    score_str = ""
    if sample_scores is not None:
        parts = [f"{k}={v:.3f}" for k, v in sample_scores.items()]
        score_str = f"  {score_label}: " + "  ".join(parts)
    fig.suptitle(
        f"Sample {sample_idx}  —  rollout positions{score_str}",
        color=_TEXT_COLOR, fontsize=11, fontweight="bold",
    )
    for ax, coord, coord_lbl in zip(axes, [0, 1], ["x", "y (depth)"]):
        style_ax(ax)
        for obj in range(n_obj):
            color = plot_color(scene_colors[obj])
            ax.plot(
                ctx_frames, positions_gt[n_context - n_ctx_show:n_context, obj, coord],
                color=color, linewidth=1.5, alpha=0.3,
            )
            ax.plot(
                roll_frames, positions_gt[n_context:n_context + n_rollout, obj, coord],
                color=color, linewidth=1.8, alpha=0.9,
            )
            for p in probes:
                pos = decoded_per_probe[p.name]
                ax.scatter(roll_frames, pos[:, obj, coord],
                           color=color, s=20, marker=p.marker, alpha=0.9)
        ax.axvline(n_context - 0.5, color=_TICK_COLOR, linewidth=1.0, linestyle="--", alpha=0.5)
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


def plot_rollout_3panel(
    test_h5_path: str,
    positions_gt: np.ndarray,            # (T_full, n_obj, 2)
    obs_rollout: np.ndarray,             # (n_rollout, R) predicted observations
    decoded_per_probe: dict[str, np.ndarray],
    probes: list[ProbeSpec],
    *,
    sample_idx: int,
    n_context: int,
    n_rollout: int,
    pred_panel_title: str = "predicted",
    suptitle: str = "",
) -> Figure:
    """3-panel dark waterfall: actual | predicted | decoded 2D positions."""
    scene_i, obs_depth_i, obs_id_i, obs_intensity_i = load_sample(test_h5_path, sample_idx)
    colors_i = scene_i.colors

    n_obj = positions_gt.shape[1]
    wf_actual = make_waterfall(obs_depth_i, obs_id_i, obs_intensity_i, scene_i, mode="model")

    wf_pred = np.zeros_like(wf_actual)
    wf_pred[:, :, :3] = _DARK_BG_ARRAY
    wf_pred[:, :, 3] = 1.0
    wf_pred[:n_context] = wf_actual[:n_context] * np.array([1.0, 1.0, 1.0, 0.35])
    wf_pred[:n_context, :, 3] = 1.0
    gray = np.clip(obs_rollout, 0.0, 1.0)
    wf_pred[n_context:n_context + n_rollout, :, 0] = gray
    wf_pred[n_context:n_context + n_rollout, :, 1] = gray
    wf_pred[n_context:n_context + n_rollout, :, 2] = gray

    fig = plt.figure(figsize=(18, 5.5), facecolor=_DARK_BG_HEX)
    if suptitle:
        fig.suptitle(suptitle, color=_DARK_TEXT_COLOR, fontsize=11, y=0.99)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.12, wspace=0.18)
    ax_fa = fig.add_subplot(1, 3, 1)
    ax_fp = fig.add_subplot(1, 3, 2)
    ax_2d = fig.add_subplot(1, 3, 3)

    for ax, img, ttl in zip(
        [ax_fa, ax_fp], [wf_actual, wf_pred],
        ["actual", f"{pred_panel_title}  (warm-up={n_context} frames)"],
    ):
        style_ax_dark(ax)
        ax.imshow(img, aspect="auto", origin="upper", interpolation="nearest")
        ax.axhline(n_context - 0.5, color="#fa8850", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.set_title(ttl, color=_DARK_TEXT_COLOR, fontsize=10)
        ax.set_xlabel("ray position", color=_DARK_TEXT_COLOR, fontsize=9)
        ax.set_ylabel("frame", color=_DARK_TEXT_COLOR, fontsize=9)

    style_ax_dark(ax_2d)
    ax_2d.set_title("decoded positions (2D)", color=_DARK_TEXT_COLOR, fontsize=10)
    ax_2d.set_xlabel("x", color=_DARK_TEXT_COLOR, fontsize=9)
    ax_2d.set_ylabel("y (depth)", color=_DARK_TEXT_COLOR, fontsize=9)
    for obj in range(n_obj):
        color = colors_i[obj]
        ax_2d.plot(positions_gt[:, obj, 0], positions_gt[:, obj, 1],
                   color=color, linewidth=1.0, alpha=0.25)
        ax_2d.plot(
            positions_gt[n_context:n_context + n_rollout, obj, 0],
            positions_gt[n_context:n_context + n_rollout, obj, 1],
            color=color, linewidth=2.0, alpha=0.9,
        )
        for p in probes:
            pos = decoded_per_probe[p.name]
            ax_2d.scatter(pos[:, obj, 0], pos[:, obj, 1],
                          color=color, s=18, marker=p.marker, alpha=0.9)

    return fig
