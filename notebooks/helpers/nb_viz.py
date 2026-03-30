"""Shared visualization helpers for GRU evaluation / probe notebooks."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from pim.sim import Scene
from pim.viz import (
    make_waterfall,
    _BG,          # dark background array — kept for waterfall pixel fills
    _TRAIL_LEN,
    _trail_t0,
)
from pim.sim import OBJECT_COLORS as _SIM_COLORS

# Light-theme styling constants for notebook result figures.
# _BG (the numpy array used to fill waterfall image pixels) intentionally stays
# dark so waterfall image data renders correctly.
_BG_HEX       = "#ffffff"
_FRUSTUM_EDGE = "#888888"   # medium gray — clean boundary without competing with data
_TICK_COLOR   = "#555555"   # medium gray for spines and tick marks
_TEXT_COLOR   = "#172239"   # dark navy for titles and labels
_TRAIL_ALPHA  = 0.50        # more opaque than dark theme

# Okabe-Ito equivalents of OBJECT_COLORS, ordered to match the sim palette.
# Use plot_color() in notebook figures so colors read clearly on white backgrounds.
_PLOT_COLORS = [
    (0.00, 0.45, 0.70),  # blue      #0072B2  ← was cyan
    (0.84, 0.37, 0.00),  # vermilion #D55E00  ← was coral
    (0.90, 0.62, 0.00),  # orange    #E69F00  ← was amber
    (0.00, 0.62, 0.45),  # teal      #009E73  ← was green
    (0.80, 0.47, 0.65),  # purple    #CC79A7  ← was violet
]
_COLOR_MAP = {
    tuple(round(v, 3) for v in c): _PLOT_COLORS[i]
    for i, c in enumerate(_SIM_COLORS)
}


def plot_color(scene_color) -> tuple:
    """Map a sim object color to its Okabe-Ito equivalent for white-background figures."""
    key = tuple(round(float(v), 3) for v in scene_color)
    return _COLOR_MAP.get(key, tuple(float(v) for v in scene_color))


# Re-export styling constants so notebooks only need to import nb_viz
__all__ = [
    "style_ax", "plot_waterfall_pair", "animate_3panel",
    "_BG", "_BG_HEX", "_TEXT_COLOR", "_TICK_COLOR",
]


# ── Styling ───────────────────────────────────────────────────────────────────


def style_ax(ax) -> None:
    """Apply light-theme styling to an Axes."""
    ax.set_facecolor(_BG_HEX)
    for spine in ax.spines.values():
        spine.set_edgecolor(_TICK_COLOR)
    ax.tick_params(colors=_TICK_COLOR, labelsize=9)


# ── Waterfall comparison ──────────────────────────────────────────────────────


def plot_waterfall_pair(
    obs_depth: np.ndarray,
    obs_id: np.ndarray,
    obs_intensity: np.ndarray,
    scene: Scene,
    pred_rollout: np.ndarray,
    n_context: int,
    title: str = "",
) -> plt.Figure:
    """Side-by-side actual vs predicted waterfall.

    Parameters
    ----------
    pred_rollout : (T - n_context, R) — autoregressive predictions
    """
    T = obs_intensity.shape[0]
    wf_actual = make_waterfall(obs_depth, obs_id, obs_intensity, scene, mode="model")

    wf_pred = np.zeros_like(wf_actual)
    wf_pred[:, :, :3] = _BG
    wf_pred[:, :, 3]  = 1.0
    # warm-up rows: show dimmed actual
    wf_pred[:n_context] = wf_actual[:n_context] * np.array([1, 1, 1, 0.35])
    wf_pred[:n_context, :, 3] = 1.0
    # rollout rows: model predictions
    gray = np.clip(pred_rollout, 0.0, 1.0)
    wf_pred[n_context:, :, 0] = gray
    wf_pred[n_context:, :, 1] = gray
    wf_pred[n_context:, :, 2] = gray

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=_BG_HEX)
    if title:
        fig.suptitle(title, color=_TEXT_COLOR, fontsize=12)
    for ax, img, ttl in zip(
        axes,
        [wf_actual, wf_pred],
        ["actual", f"predicted  (warm-up={n_context} frames)"],
    ):
        style_ax(ax)
        ax.imshow(img, aspect="auto", origin="upper", interpolation="nearest")
        ax.axhline(n_context - 0.5, color="#fa8850", linewidth=1.2, linestyle="--")
        ax.set_title(ttl, color=_TEXT_COLOR, fontsize=11)
        ax.set_xlabel("scan position", color=_TEXT_COLOR, fontsize=10)
        ax.set_ylabel("frame", color=_TEXT_COLOR, fontsize=10)
    plt.tight_layout()
    return fig


# ── 3-panel animation ─────────────────────────────────────────────────────────


def animate_3panel(
    scene: Scene,
    obs_depth: np.ndarray,
    obs_id: np.ndarray,
    obs_intensity: np.ndarray,
    pred_rollout: np.ndarray,
    n_context: int,
    interval: int = 80,
    title: str = "",
) -> FuncAnimation:
    """3-panel animation: 2D scene | actual waterfall | predicted waterfall.

    Parameters
    ----------
    pred_rollout : (T - n_context, R) — autoregressive predictions
    """
    cfg   = scene.config
    T     = obs_intensity.shape[0]
    n_obj = scene.positions.shape[1]

    fig = plt.figure(figsize=(18, 5.5), facecolor=_BG_HEX)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.10, wspace=0.14)
    ax_w  = fig.add_subplot(1, 3, 1)
    ax_fa = fig.add_subplot(1, 3, 2)
    ax_fp = fig.add_subplot(1, 3, 3)

    if title:
        fig.suptitle(title, color=_TEXT_COLOR, fontsize=11, y=0.97)

    for ax in (ax_w, ax_fa, ax_fp):
        style_ax(ax)

    # ── 2D world panel ────────────────────────────────────────────────────
    mx, my = 0.7, 0.7
    ax_w.set_xlim(-cfg.x_far - mx, cfg.x_far + mx)
    ax_w.set_ylim(cfg.y_near - my, cfg.y_far + my)
    ax_w.set_aspect("equal")
    ax_w.set_xlabel("x", color=_TEXT_COLOR, fontsize=10)
    ax_w.set_ylabel("depth  y", color=_TEXT_COLOR, fontsize=10)
    ax_w.set_title("2D scene  (ground truth)", color=_TEXT_COLOR, fontsize=11, pad=6)

    corners = np.array([
        [-cfg.x_near, cfg.y_near], [cfg.x_near, cfg.y_near],
        [cfg.x_far,   cfg.y_far],  [-cfg.x_far, cfg.y_far],
    ])
    ax_w.add_patch(Polygon(corners, closed=True, fill=False,
                            edgecolor=_FRUSTUM_EDGE, linewidth=1.8, zorder=1))

    circles, trails, refl_labels = [], [], []
    for i in range(n_obj):
        x0, y0 = scene.positions[0, i]
        circ = plt.Circle((x0, y0), scene.radii[i], color=scene.colors[i],
                           zorder=3, linewidth=1.2, edgecolor="#cccccc", alpha=0.95)
        ax_w.add_patch(circ)
        circles.append(circ)
        (trail,) = ax_w.plot([x0], [y0], color=scene.colors[i], linewidth=1.5,
                              alpha=_TRAIL_ALPHA, zorder=2, solid_capstyle="round")
        trails.append(trail)
        lbl = ax_w.text(x0, y0, f"{scene.reflectivities[i]:.2f}",
                        ha="center", va="center", color="white",
                        fontsize=7, fontfamily="monospace", fontweight="bold", zorder=4)
        refl_labels.append(lbl)

    frame_text = ax_w.text(
        -cfg.x_far - mx + 0.2, cfg.y_far + my - 0.15, "frame   0",
        color=_TEXT_COLOR, fontsize=9, fontfamily="monospace", va="top",
    )

    # ── Waterfall panels ──────────────────────────────────────────────────
    disp_actual = np.zeros((T, cfg.obs_res, 4))
    disp_actual[:, :, :3] = _BG
    disp_actual[:, :, 3]  = 1.0

    disp_pred = np.zeros((T, cfg.obs_res, 4))
    disp_pred[:, :, :3] = _BG
    disp_pred[:, :, 3]  = 1.0

    extent = [0, cfg.obs_res, T, 0]
    img_actual = ax_fa.imshow(disp_actual, aspect="auto", origin="upper",
                               interpolation="nearest", extent=extent)
    img_pred   = ax_fp.imshow(disp_pred,   aspect="auto", origin="upper",
                               interpolation="nearest", extent=extent)

    for ax, ttl in [
        (ax_fa, "actual"),
        (ax_fp, f"predicted  (rollout from frame {n_context})"),
    ]:
        ax.set_xlabel("scan position", color=_TEXT_COLOR, fontsize=10)
        ax.set_ylabel("frame", color=_TEXT_COLOR, fontsize=10)
        ax.set_title(ttl, color=_TEXT_COLOR, fontsize=11, pad=6)
        ax.set_xlim(0, cfg.obs_res)
        ax.set_ylim(T, 0)
        ax.axhline(n_context - 0.5, color="#fa8850", linewidth=1.0,
                   linestyle="--", zorder=5, alpha=0.7)

    line_actual = ax_fa.plot([0, cfg.obs_res], [0.5, 0.5],
                              color="white", linewidth=1.4, alpha=0.45, zorder=5)[0]
    line_pred   = ax_fp.plot([0, cfg.obs_res], [0.5, 0.5],
                              color="white", linewidth=1.4, alpha=0.45, zorder=5)[0]

    wf_actual_full = make_waterfall(obs_depth, obs_id, obs_intensity, scene, mode="model")

    def update(f: int):
        for i, (circ, trail, lbl) in enumerate(zip(circles, trails, refl_labels)):
            pos = scene.positions[f, i]
            circ.center = pos
            lbl.set_position(pos)
            t0 = _trail_t0(scene.positions[:, i], max(0, f - _TRAIL_LEN), f, cfg)
            trail.set_data(scene.positions[t0:f+1, i, 0], scene.positions[t0:f+1, i, 1])
        frame_text.set_text(f"frame  {f:3d}")

        disp_actual[:f+1] = wf_actual_full[:f+1]
        img_actual.set_data(disp_actual)
        line_actual.set_ydata([f + 0.5, f + 0.5])

        if f < n_context:
            disp_pred[:f+1] = wf_actual_full[:f+1] * np.array([1, 1, 1, 0.35])
            disp_pred[:f+1, :, 3] = 1.0
        else:
            disp_pred[:n_context] = wf_actual_full[:n_context] * np.array([1, 1, 1, 0.35])
            disp_pred[:n_context, :, 3] = 1.0
            n_pred = f - n_context + 1
            gray = np.clip(pred_rollout[:n_pred], 0.0, 1.0)
            disp_pred[n_context:f+1, :, 0] = gray
            disp_pred[n_context:f+1, :, 1] = gray
            disp_pred[n_context:f+1, :, 2] = gray
        img_pred.set_data(disp_pred)
        line_pred.set_ydata([f + 0.5, f + 0.5])

        return (circles + trails + refl_labels
                + [frame_text, img_actual, line_actual, img_pred, line_pred])

    return FuncAnimation(fig, update, frames=T, interval=interval, blit=True)
