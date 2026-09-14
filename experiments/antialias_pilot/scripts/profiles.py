#!/usr/bin/env python
"""Candidate DISC PROFILES for a smooth-rendered discworld instance (2026-09-12) — pilot only.

Nothing here touches `pim/`: the ray geometry, edge coverage and depth ordering are copied
from `pim/environments/discworld/soft_render.py::render_frame_soft` line for line, and only
the SHADING term is swapped for a profile f(u) of u = (perpendicular distance to the disc
centre) / radius — baked into the disc, so identical at the frustum edge and centre:

    lambert   sqrt(1 − u²)                (the existing `soft_shading="lambert"`; ∞ slope at the rim)
    cosine    (1 + cos(π u)) / 2          (zero value AND zero slope at the rim — C¹ silhouette)
    power p   (1 − u²)^p                  (p = 0.5 lambert, 1 parabola, 2 quartic bell)
    gauss σ   exp(−u² / 2σ²), σ in radii  (smoothest; blurs the disc's extent → weaker depth cue)

    .pim/bin/python experiments/antialias_pilot/scripts/profiles.py
    -> outputs/profile_candidates.png + a printed sensitivity table

Once one is chosen it becomes a `SimConfig.soft_profile` knob in soft_render (both backends,
default = the current behaviour, pinned bit-identical by tests) and `sim_config_from` passes
it through so the bench's reference renders use the same profile.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
from pim.environments import layout  # noqa: E402
from pim.environments.discworld.soft_render import _EPS, _FAR, _ray_dirs, _sigmoid  # noqa: E402
from pim.metrics.zone_editability import object_constants, sim_config_from  # noqa: E402

PROFILES = {
    "lambert": lambda u: np.sqrt(np.maximum(1.0 - u**2, 0.0)),
    "cosine": lambda u: 0.5 * (1.0 + np.cos(np.pi * np.minimum(u, 1.0))),
    "power2": lambda u: np.maximum(1.0 - u**2, 0.0) ** 2,
    "gauss0.4": lambda u: np.exp(-(u**2) / (2 * 0.4**2)),
}


def render_profile(positions, radii, reflectivities, cfg, profile: str, edge: float) -> np.ndarray:
    """`render_frame_soft`'s intensity with the shading term replaced by PROFILES[profile](u)."""
    positions = np.asarray(positions, float)
    radii = np.asarray(radii, float)
    dx, dy = _ray_dirs(cfg)
    cx, cy = positions[:, 0], positions[:, 1]
    b_ = dx[:, None] * cx[None, :] + dy[:, None] * cy[None, :]
    perp2 = np.maximum(cx**2 + cy**2 - b_**2, 0.0)
    sq = np.sqrt(np.maximum(radii[None, :] ** 2 - perp2, 0.0))
    t_front, t_back = b_ - sq, b_ + sq
    hy_f, hy_b = dy[:, None] * t_front, dy[:, None] * t_back
    t_at_near = cfg.y_near / dy[:, None]
    visible = (t_front > _EPS) & (hy_f >= cfg.y_near) & (hy_f <= cfg.y_far)
    clamp_near = (hy_f < cfg.y_near) & (hy_b >= cfg.y_near)
    t_eff = np.where(visible, t_front, np.where(clamp_near, t_at_near, _FAR))
    gate = (t_eff < _FAR).astype(float)
    signed = radii[None, :] - np.sqrt(perp2)
    alpha = gate * (_sigmoid(signed / edge) if edge > 0 else (signed > 0))
    u = np.sqrt(perp2) / np.maximum(radii[None, :], _EPS)
    shade = reflectivities[None, :] * PROFILES[profile](u)
    dt = t_eff[..., None, :] - t_eff[..., :, None]
    front = (dt > 0).astype(float)
    eye = np.eye(alpha.shape[-1])
    keep = 1.0 - alpha[..., None, :] * front * (1.0 - eye)
    return np.clip((alpha * shade * keep.prod(-1)).sum(-1), 0.0, 1.0)


def main() -> None:
    sim = json.loads(layout.edits_manifest("discworld", "dw-noiseless").read_text())["sim"]
    cfg = sim_config_from(sim, 2)
    rad, refl = object_constants(sim, 2)
    CANDS = {"D  lambert + edge 0.10 (reference)": ("lambert", 0.10),
             "F  raised cosine": ("cosine", 0.0),
             "G  raised cosine + edge 0.10": ("cosine", 0.10),
             "H  power dome p=2": ("power2", 0.0),
             "I  gaussian σ=0.4 r + edge 0.10": ("gauss0.4", 0.10)}

    def frame(prof, edge, x, y, x2=-3.0, y2=10.0):
        pos = np.array([[x, y], [x2, y2]], np.float32)
        return render_profile(pos, rad, refl, dataclasses.replace(cfg), prof, edge)

    def ray_width(y):
        xw = sim["x_near"] + (sim["x_far"] - sim["x_near"]) * (y - sim["y_near"]) / (sim["y_far"] - sim["y_near"])
        return 2 * xw / sim["obs_res"]

    depths = [(3.6, "near, y=3.6"), (7.0, "mid, y=7.0"), (11.0, "far, y=11.0")]
    offs = np.linspace(0, 1, 6)[:-1]
    fig, axes = plt.subplots(len(CANDS), len(depths) + 1, figsize=(5.2 * (len(depths) + 1), 3.0 * len(CANDS)), squeeze=False)
    rows = []
    for i, (name, (prof, edge)) in enumerate(CANDS.items()):
        for j, (y, lab) in enumerate(depths):
            ax = axes[i, j]
            w = ray_width(y)
            P = np.stack([frame(prof, edge, 0.8 + o * w, y) for o in offs])
            for o, f in zip(offs, P):
                ax.plot(np.arange(128), f, lw=1.3, color=plt.cm.viridis(o), label=f"+{o:.1f} ray" if j == 0 else None)
            d = P[1:] - P[:-1]
            rmse = np.sqrt((d ** 2).mean(1)).mean()
            moving = int((np.abs(d) > 1e-4).any(0).sum())
            covered = int((P[0] > 1e-4).sum())
            rim = float(np.abs(np.diff(P[0])).max())
            rows.append((name, lab, rmse, moving, covered, rim))
            ax.set_title(f"{name} — {lab}\nΔframe/0.2-ray RMSE {rmse:.4f}; {moving}/{covered} rays move; max step {rim:.3f}", fontsize=8.5, loc="left")
            ax.set_xlim(20, 108)
            ax.set_ylim(-0.02, 0.9)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.legend(fontsize=6.5, frameon=False, title="sub-ray offset", title_fontsize=6.5)
            if i == len(CANDS) - 1:
                ax.set_xlabel("ray", fontsize=8)
        ax = axes[i, len(depths)]
        xs, ys = np.linspace(-1.0, 1.2, 40), np.linspace(4.0, 9.5, 40)
        W = np.stack([frame(prof, edge, x, y) for x, y in zip(xs, ys)])
        ax.imshow(W, aspect="auto", cmap="gray", vmin=0, vmax=0.9, interpolation="nearest")
        ax.set_title(f"{name} — 40-frame waterfall", fontsize=8.5, loc="left")
        ax.set_xlabel("ray", fontsize=8)
        ax.set_ylabel("frame", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Smooth disc PROFILES on the dw-noiseless geometry (128 rays, radius 0.5): one disc at five sub-ray offsets, three depths\n"
                 "profiles are functions of distance-to-centre / radius, baked into the disc; 'max step' = largest jump between neighbouring rays (silhouette sharpness)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    out = _REPO / "experiments" / "antialias_pilot" / "outputs" / "profile_candidates.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"{'candidate':<36}{'depth':<14}{'RMSE/0.2 ray':>13}{'moving/covered':>16}{'max step':>10}")
    for r in rows:
        print(f"{r[0]:<36}{r[1]:<14}{r[2]:>13.4f}{f'{r[3]}/{r[4]}':>16}{r[5]:>10.3f}")
    print(f"-> {out.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
