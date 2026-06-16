"""Controllability-stage figures: per-step RMSE curves, position trajectories, 3-panel waterfall."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pim.eval.baselines import ObsBaselines, PosBaselines
from pim.eval.controllability import ControllabilityMetrics
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
    _BG_HEX as _DARK_BG_HEX,
    _TEXT_COLOR as _DARK_TEXT_COLOR,
)


def plot_controllability_obs(
    metrics: ControllabilityMetrics,
    *,
    baselines: ObsBaselines | None = None,
    include: set[str] | None = None,
) -> Figure:
    """Per-step observation RMSE: steered vs unsteered, noisy + clean targets."""
    steps = np.arange(len(metrics.steered_obs_step))
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
    style_ax(ax)
    ax.plot(steps, np.sqrt(metrics.unsteered_obs_step), color=PALETTE[1], linewidth=1.8,
            label="unsteered (vs noisy)")
    ax.plot(steps, np.sqrt(metrics.steered_obs_step), color=PALETTE[0], linewidth=1.8,
            label="steered (vs noisy)")
    ax.plot(steps, np.sqrt(metrics.clean_unsteered_obs_step), color=PALETTE[1], linewidth=1.8,
            linestyle="--", label="unsteered (vs clean)")
    ax.plot(steps, np.sqrt(metrics.clean_steered_obs_step), color=PALETTE[0], linewidth=1.8,
            linestyle="--", label="steered (vs clean)")
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
    ax.set_xlabel("rollout step (0 = edit frame)", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("observation RMSE", color=_TEXT_COLOR, fontsize=10)
    ax.set_title("Per-step observation RMSE: steered vs unsteered",
                 color=_TEXT_COLOR, fontsize=11)
    ax.legend(frameon=False, labelcolor=_TEXT_COLOR, fontsize=8)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(colors=_TICK_COLOR)
    plt.tight_layout()
    return fig


def plot_controllability_positions(
    pos_rmse: dict[str, dict[str, np.ndarray]],   # {probe_name: {"steered": (n,), "unsteered": (n,)}}
    probes: list[ProbeSpec],
    *,
    baselines: PosBaselines | None = None,
    include: set[str] | None = None,
) -> Figure:
    """Per-step position RMSE: steered (solid) and unsteered (dashed) per probe."""
    n_rollout = len(next(iter(pos_rmse.values()))["steered"])
    steps = np.arange(n_rollout)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
    style_ax(ax)
    for p in probes:
        st = pos_rmse[p.name]["steered"]
        un = pos_rmse[p.name]["unsteered"]
        ax.plot(steps, np.sqrt(un), color=PALETTE[1], linewidth=1.8,
                linestyle=p.linestyle, label=f"unsteered ({p.name})")
        ax.plot(steps, np.sqrt(st), color=PALETTE[0], linewidth=1.8,
                linestyle=p.linestyle, label=f"steered ({p.name})")
    if baselines is not None:
        _pos_lines = [
            ("random_rmse",   baselines.random_rmse,   f"random baseline ({baselines.random_rmse:.3f})",   _TICK_COLOR, ":", 0.7),
            ("identity_rmse", baselines.identity_rmse, f"identity RMSE ({baselines.identity_rmse:.3f})",   PALETTE[5],  ":", 0.8),
            ("noise_std",     baselines.noise_std,     f"position noise σ ({baselines.noise_std:.3f})",    PALETTE[2],  ":", 0.8),
        ]
        for name, value, label, color, ls, alpha in _pos_lines:
            if include is None or name in include:
                ax.axhline(value, color=color, linewidth=1.2, linestyle=ls, alpha=alpha, label=label)
    ax.set_xlabel("rollout step (0 = edit frame)", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("position RMSE", color=_TEXT_COLOR, fontsize=10)
    ax.set_title("Per-step position RMSE: steered vs unsteered",
                 color=_TEXT_COLOR, fontsize=11)
    ax.legend(frameon=False, labelcolor=_TEXT_COLOR, fontsize=8)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(colors=_TICK_COLOR)
    plt.tight_layout()
    return fig


def plot_controllability_trajectory(
    pre_edit_gt: np.ndarray,             # (n_ctx_show, n_obj, 2)
    pre_edit_decoded: dict[str, np.ndarray],   # name → (n_ctx_show, n_obj, 2)
    post_edit_gt: np.ndarray,            # (n_rollout, n_obj, 2)
    steered_decoded: dict[str, np.ndarray],    # name → (n_rollout, n_obj, 2)
    probes: list[ProbeSpec],
    scene_colors: np.ndarray,            # (n_obj, 3)
    *,
    sample_idx: int,
    edit_frame: int,
    n_rollout: int,
    show_unsteered: bool = False,
    unsteered_decoded: dict[str, np.ndarray] | None = None,
) -> Figure:
    """x/y position trajectory: pre-edit context (faint) + post-edit GT + decoded."""
    n_ctx_show = pre_edit_gt.shape[0]
    n_obj = pre_edit_gt.shape[1]
    ctx_frames = np.arange(edit_frame - n_ctx_show, edit_frame)
    roll_frames = np.arange(edit_frame, edit_frame + n_rollout)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=_BG_HEX)
    fig.suptitle(
        f"Sample {sample_idx}  —  steered positions  (edit at frame {edit_frame}, step 0 = edit frame)",
        color=_TEXT_COLOR, fontsize=11, fontweight="bold",
    )
    for ax, coord, coord_lbl in zip(axes, [0, 1], ["x", "y (depth)"]):
        style_ax(ax)
        for obj in range(n_obj):
            color = plot_color(scene_colors[obj])
            # Pre-edit GT (faint solid)
            ax.plot(ctx_frames, pre_edit_gt[:, obj, coord],
                    color=color, linewidth=1.5, alpha=0.3)
            # Pre-edit decoded (faint scatter)
            for p in probes:
                ax.scatter(ctx_frames, pre_edit_decoded[p.name][:, obj, coord],
                           color=color, s=14, marker=p.marker, alpha=0.3)
            # Post-edit GT (solid)
            ax.plot(roll_frames, post_edit_gt[:, obj, coord],
                    color=color, linewidth=1.8, alpha=0.9)
            # Steered decoded (solid scatter)
            for p in probes:
                ax.scatter(roll_frames, steered_decoded[p.name][:, obj, coord],
                           color=color, s=20, marker=p.marker, alpha=0.9)
            # Unsteered decoded (open scatter, optional)
            if show_unsteered and unsteered_decoded is not None:
                for p in probes:
                    ax.scatter(roll_frames, unsteered_decoded[p.name][:, obj, coord],
                               color=color, s=20, marker=p.marker, alpha=0.6, facecolors="none")
        ax.axvline(edit_frame - 0.5, color=_TICK_COLOR, linewidth=1.0, linestyle="--", alpha=0.5)
        ax.set_xlabel("frame", color=_TEXT_COLOR, fontsize=9)
        ax.set_ylabel(coord_lbl, color=_TEXT_COLOR, fontsize=9)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.tick_params(colors=_TICK_COLOR)

    handles = [Line2D([0], [0], color="gray", linewidth=1.8, label="GT")]
    handles += [
        Line2D([0], [0], color="gray", marker=p.marker, linestyle="none",
               markersize=6, label=p.name)
        for p in probes
    ]
    if show_unsteered:
        handles += [
            Line2D([0], [0], color="gray", marker="s", linestyle="none",
                   markersize=6, alpha=0.9, label="steered (solid fill)"),
            Line2D([0], [0], color="gray", marker="s", linestyle="none",
                   markersize=6, alpha=0.6, markerfacecolor="none", label="unsteered (open)"),
        ]
    handles += [
        Line2D([0], [0], color="gray", marker="s", linestyle="none",
               markersize=6, alpha=0.3, label="pre-edit (faint)"),
    ]
    axes[0].legend(handles=handles, frameon=False, labelcolor=_TEXT_COLOR, fontsize=8)
    plt.tight_layout()
    return fig


def plot_controllability_waterfalls(
    pre_edit_obs: np.ndarray,            # (edit_frame, R) actual pre-edit obs
    gt_post_obs: np.ndarray,             # (n_rollout, R) actual post-edit obs
    steered_obs: np.ndarray,             # (n_rollout, R) predicted steered
    unsteered_obs: np.ndarray,           # (n_rollout, R) predicted unsteered
    *,
    sample_idx: int,
    edit_frame: int,
    n_rollout: int,
) -> Figure:
    """3-panel dark waterfall: GT | steered | unsteered (full sequence)."""
    total_frames = edit_frame + n_rollout
    R = pre_edit_obs.shape[1]

    gt_full = np.clip(np.concatenate([pre_edit_obs, gt_post_obs], axis=0), 0, 1)

    def _build(pred):
        panel = np.zeros((total_frames, R), dtype=np.float32)
        panel[:edit_frame] = np.clip(pre_edit_obs, 0, 1)
        panel[edit_frame:] = np.clip(pred, 0, 1)
        return panel

    steered_full = _build(steered_obs)
    unsteered_full = _build(unsteered_obs)

    fig = plt.figure(figsize=(18, 5.5), facecolor=_DARK_BG_HEX)
    fig.suptitle(
        f"Sample {sample_idx}  —  counterfactual controllability  (edit at frame {edit_frame})",
        color=_DARK_TEXT_COLOR, fontsize=11, y=0.99,
    )
    fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.12, wspace=0.18)

    for k, (img, ttl) in enumerate(
        [(gt_full, "GT (full sequence)"),
         (steered_full, "steered rollout"),
         (unsteered_full, "unsteered rollout")]
    ):
        ax = fig.add_subplot(1, 3, k + 1)
        style_ax_dark(ax)
        ax.imshow(img, aspect="auto", origin="upper", interpolation="nearest",
                  cmap="gray", vmin=0, vmax=1)
        ax.axhline(edit_frame - 0.5, color="#fa8850", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.set_title(ttl, color=_DARK_TEXT_COLOR, fontsize=10)
        ax.set_xlabel("ray position", color=_DARK_TEXT_COLOR, fontsize=9)
        ax.set_ylabel("frame", color=_DARK_TEXT_COLOR, fontsize=9)

    return fig
