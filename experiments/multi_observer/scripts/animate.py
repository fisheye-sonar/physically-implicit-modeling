"""Render the multi-observer discworld to look at (2026-09-13) — pilot only, nothing canonical.

Top-down world (the circular arena, the pivot, N observers with their kept-ray fans, the discs
with trails) beside the N simultaneous 1D observations (one strip per observer, the current frame
as bars) and the observer-major flat vector the model would consume. Writes an MP4 + a GIF + a
contact-sheet PNG per scene into experiments/multi_observer/outputs/.

    .pim/bin/python experiments/multi_observer/scripts/animate.py --n-observers 5 --seeds 0 1 2

Geometry = dw-8ray's (2 discs, radius 1.0, 10 rays cast / 8 kept, no noise, open boundary,
stay-inside acceptance) with region="circle" and N observers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter  # noqa: E402

from pim.environments.discworld import observers as ob  # noqa: E402
from pim.environments.discworld.config import SimConfig, obs_dim  # noqa: E402
from pim.environments.discworld.renderer import _fov_scale, render_frame  # noqa: E402
from pim.environments.discworld.sim import simulate  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"


def dw8ray_cfg(n_observers: int, seed: int, rays: int = 10) -> SimConfig:
    return SimConfig(n_objects=2, n_frames=40, obs_res=rays, drop_edge_rays=True, radius=1.0,
                     boundary="open", position_noise_std=0.0, obs_noise_std=0.0,
                     fixed_reflectivities=True, always_in_frustum=True, max_gen_attempts=5000,
                     region="circle", n_observers=n_observers, seed=seed)


def kept_fan(cfg: SimConfig):
    """Unit directions of the KEPT rays in the canonical frame (edge rays dropped if asked)."""
    s = np.linspace(-1.0, 1.0, cfg.obs_res)
    if cfg.drop_edge_rays:
        s = s[1:-1]
    d = np.stack([s * _fov_scale(cfg), np.ones_like(s)], -1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def world_dirs(dirs, pose):
    a = pose[2] - np.pi / 2
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return dirs @ R.T


def render_scene(cfg: SimConfig):
    scene = simulate(cfg)
    frames = [render_frame(scene.positions[t], scene.radii, scene.reflectivities, cfg)
              for t in range(cfg.n_frames)]
    inten = np.stack([f[2] for f in frames])          # (T, N·R)
    ids = np.stack([f[1] for f in frames])
    return scene, inten, ids


def draw(cfg: SimConfig, scene, inten, ids, tag: str, fps: int = 8):
    N, R = cfg.n_observers, obs_dim(cfg) // cfg.n_observers
    poses = ob.observer_poses(cfg)
    c, Rc = ob.pivot(cfg), ob.region_radius(cfg)
    colours = plt.cm.tab10(np.arange(N) % 10)
    obj_col = ["#d62728", "#1f77b4"]

    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(N, 2, width_ratios=[1.15, 1.0], hspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    strips = [fig.add_subplot(gs[k, 1]) for k in range(N)]

    # static world: arena, pivot, observers + fans, near/far arcs
    ax.add_patch(plt.Circle(c, Rc, fill=False, lw=2, color="k"))
    ax.add_patch(plt.Circle(c, Rc - cfg.radius, fill=False, lw=0.8, ls="--", color="0.5"))
    ax.plot(*c, "k+", ms=10)
    fan = kept_fan(cfg)
    for k, pose in enumerate(poses):
        o = pose[:2]
        ax.plot(*o, "o", color=colours[k], ms=9, mec="k")
        ax.annotate(f"obs {k}", o, textcoords="offset points", xytext=(6, 6), color=colours[k], fontsize=9)
        d = world_dirs(fan, pose)
        for j, dj in enumerate(d):
            far = o + dj * (cfg.y_far / (dj @ world_dirs(np.array([[0, 1.0]]), pose)[0]))
            ax.plot([o[0], far[0]], [o[1], far[1]], color=colours[k], lw=0.6, alpha=0.35)
    ax.set_xlim(c[0] - 0.5 * (cfg.y_near + cfg.y_far) - 0.8, c[0] + 0.5 * (cfg.y_near + cfg.y_far) + 0.8)
    ax.set_ylim(c[1] - 0.5 * (cfg.y_near + cfg.y_far) - 0.8, c[1] + 0.5 * (cfg.y_near + cfg.y_far) + 0.8)
    ax.set_aspect("equal")
    ax.set_title(f"{tag}: {N} observers on the ring, circular arena (r = {Rc:g}), discs r = {cfg.radius:g}")
    discs = [ax.add_patch(plt.Circle(scene.positions[0, i], cfg.radius, color=obj_col[i], alpha=0.85)) for i in range(2)]
    trails = [ax.plot([], [], "-", color=obj_col[i], lw=1, alpha=0.6)[0] for i in range(2)]
    ttl = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)

    bars = []
    for k in range(N):
        s = strips[k]
        b = s.bar(np.arange(R), inten[0, k * R:(k + 1) * R], color=colours[k], width=0.9)
        s.set_ylim(0, 1.0)
        s.set_xlim(-0.6, R - 0.4)
        s.set_yticks([0, 0.5, 1.0])
        s.set_ylabel(f"obs {k}", color=colours[k], fontsize=9)
        s.tick_params(labelsize=8)
        if k < N - 1:
            s.set_xticklabels([])
        bars.append(b)
    strips[-1].set_xlabel(f"ray (each observer: {R} kept rays)  →  flat model input = observer-major concat, {N * R} values")

    def update(t):
        for i in range(2):
            discs[i].center = scene.positions[t, i]
            trails[i].set_data(scene.positions[:t + 1, i, 0], scene.positions[:t + 1, i, 1])
        for k in range(N):
            for j, rect in enumerate(bars[k]):
                v = inten[t, k * R + j]
                rect.set_height(v)
                o = ids[t, k * R + j]
                rect.set_color(obj_col[o] if o >= 0 else colours[k])
        ttl.set_text(f"frame {t:2d}/{cfg.n_frames - 1}")
        return discs + trails

    anim = FuncAnimation(fig, update, frames=cfg.n_frames, interval=1000 // fps, blit=False)
    OUT.mkdir(parents=True, exist_ok=True)
    anim.save(OUT / f"{tag}.mp4", writer=FFMpegWriter(fps=fps, bitrate=1800))
    anim.save(OUT / f"{tag}.gif", writer=PillowWriter(fps=fps))
    update(min(20, cfg.n_frames - 1))
    fig.savefig(OUT / f"{tag}_frame20.png", dpi=110)
    plt.close(fig)


def contact_sheet(cfg: SimConfig, scene, inten, tag: str):
    """All N observations as waterfalls (time down, ray across) — what the model sees."""
    N, R = cfg.n_observers, obs_dim(cfg) // cfg.n_observers
    fig, axes = plt.subplots(1, N, figsize=(2.2 * N, 5), sharey=True)
    for k in range(N):
        axes[k].imshow(inten[:, k * R:(k + 1) * R], aspect="auto", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[k].set_title(f"obs {k}", fontsize=10)
        axes[k].set_xlabel("ray")
    axes[0].set_ylabel("frame")
    fig.suptitle(f"{tag}: the {N} simultaneous observations over the 40 frames (grey = intensity)")
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}_waterfalls.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-observers", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fps", type=int, default=8)
    a = ap.parse_args()
    for seed in a.seeds:
        cfg = dw8ray_cfg(a.n_observers, seed)
        scene, inten, ids = render_scene(cfg)
        tag = f"dw8ray_obs{a.n_observers}_seed{seed}"
        draw(cfg, scene, inten, ids, tag, fps=a.fps)
        contact_sheet(cfg, scene, inten, tag)
        vis = (ids.reshape(cfg.n_frames, a.n_observers, -1) >= 0).any(-1)   # (T, N) observer sees anything
        print(f"{tag}: obs dim {obs_dim(cfg)}; P(observer sees >= 1 disc) {vis.mean():.2f}; "
              f"frames where some observer sees nothing: {(~vis).any(1).mean():.2f}  -> {OUT / tag}.{{mp4,gif}}")


if __name__ == "__main__":
    main()
