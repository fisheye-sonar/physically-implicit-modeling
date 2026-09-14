"""Multiple observers around a circular arena (2026-09-13, Sevan's "increasing observers" control).

The canonical discworld has ONE observer at the origin looking +y into a perspective frustum
``y ∈ [y_near, y_far]``. This module places ``N`` observers on a ring and renders the same
discs into ``N`` simultaneous 1D observations, concatenated along the ray axis.

The rule (Sevan, 2026-09-13):

* the **pivot** is the frustum's depth midpoint on the optical axis, ``(0, (y_near + y_far)/2)``;
* observer ``k`` is the canonical observer **rotated about the pivot** by ``2πk/N`` (observer 0 IS
  the canonical one, so ``N = 1`` reproduces every existing render bit for bit);
* the near and far planes stay hard constraints for EVERY observer, and the set of points
  satisfying both for all rotations is the **circle** of radius ``(y_far − y_near)/2`` about the
  pivot — tangent to each observer's near and far plane. ``region="circle"`` makes that circle the
  arena: discs are sampled in it and a trajectory is accepted only if every disc stays fully
  inside it (the analogue of ``always_in_frustum``).

What the rule does NOT guarantee: lateral visibility. The circle's radius (4.5 on the default
geometry) exceeds the frustum half-width at the pivot depth (3.75), so a disc near the rim is
outside some observers' field of view while inside others' — no single observer sees everything,
which is the point of the control. ``compute_visibility`` / ``is_visible`` remain the canonical
observer's (frame 0) and are not per-observer.

Everything here is a thin layer over the canonical pieces: ``render_frame`` renders each view
(so 8-ray / dropped edge rays / soft profiles / blink all apply per observer), and the two hooks
in ``sim.py`` (``sample_position``, ``fully_in_frustum``) route to ``sample_position_region`` /
``inside_region`` when ``cfg.region == "circle"``. Flat observation layout: observer-major,
``obs[k * R : (k + 1) * R]`` is observer ``k``'s ``R = obs_dim(single view)`` rays
(``observer_of_ray``).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .config import SimConfig


# ── geometry ──────────────────────────────────────────────────────────────────

def pivot(cfg: SimConfig) -> np.ndarray:
    """The point the observers rotate about: the optical axis at the frustum's depth midpoint."""
    return np.array([0.0, 0.5 * (cfg.y_near + cfg.y_far)])


def region_radius(cfg: SimConfig) -> float:
    """Radius of the circular arena — the intersection of every rotated observer's depth band."""
    return 0.5 * (cfg.y_far - cfg.y_near)


def observer_angles(cfg: SimConfig) -> np.ndarray:
    """(N,) rotation of each observer about the pivot; observer 0 is the canonical one."""
    n = int(getattr(cfg, "n_observers", 1))
    return 2.0 * np.pi * np.arange(n) / n


def observer_poses(cfg: SimConfig) -> np.ndarray:
    """(N, 3): observer position ``(x, y)`` and its view-direction angle ``phi`` (radians, from
    the +x axis; the canonical observer's is π/2 = looking +y). The observer sits at the pivot's
    distance ``d = (y_near + y_far)/2`` from the pivot, looking at it."""
    c = pivot(cfg)
    d = c[1]
    out = []
    for a in observer_angles(cfg):
        # canonical observer = pivot + d·(0, −1); rotate that offset by a
        off = d * np.array([np.sin(a), -np.cos(a)])
        out.append([c[0] + off[0], c[1] + off[1], np.pi / 2 + a])
    return np.array(out)


def to_observer_frame(positions: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """World coordinates ``(..., 2)`` → the observer's canonical frame (observer at the origin,
    looking +y). Inverse rotation by ``a = phi − π/2`` after translating the observer to the
    origin; for observer 0 this is the identity, exactly."""
    a = float(pose[2]) - np.pi / 2
    p = np.asarray(positions, float) - np.asarray(pose[:2], float)
    if a == 0.0:
        return p
    ca, sa = np.cos(-a), np.sin(-a)
    x, y = p[..., 0], p[..., 1]
    return np.stack([ca * x - sa * y, sa * x + ca * y], -1)


def observer_of_ray(cfg: SimConfig) -> np.ndarray:
    """(N·R,) which observer each flat observation entry belongs to."""
    from .config import obs_dim

    n = int(getattr(cfg, "n_observers", 1))
    return np.repeat(np.arange(n), obs_dim(cfg) // n)


# ── the circular arena ────────────────────────────────────────────────────────

def inside_region(positions: np.ndarray, radius: float, cfg: SimConfig) -> bool:
    """True when every disc is fully inside the circle at every frame — ``positions`` is
    ``(F, n, 2)`` (pass a single point as ``p[None, None, :]``), the same contract as
    ``sim.fully_in_frustum``."""
    d = np.linalg.norm(np.asarray(positions, float) - pivot(cfg), axis=-1)
    return bool((d + radius <= region_radius(cfg)).all())


def sample_position_region(rng: np.random.Generator, cfg: SimConfig, radius: float) -> tuple[float, float]:
    """One ``(x, y)`` uniform over the circle of radius ``region_radius − radius`` (disc fully
    inside). Two ``rng.uniform`` draws, like ``sim.sample_position``."""
    rmax = region_radius(cfg) - radius
    r = rmax * np.sqrt(rng.uniform(0.0, 1.0))
    t = rng.uniform(0.0, 2.0 * np.pi)
    c = pivot(cfg)
    return float(c[0] + r * np.cos(t)), float(c[1] + r * np.sin(t))


# ── rendering ─────────────────────────────────────────────────────────────────

def single_view_config(cfg: SimConfig) -> SimConfig:
    """The same config with ONE observer — what each view is rendered with."""
    return dataclasses.replace(cfg, n_observers=1)


def render_frame_multi(positions, radii, reflectivities, cfg: SimConfig, rng=None, visible=None):
    """``renderer.render_frame`` from every observer, concatenated observer-major along the ray
    axis: ``(hit_depth, hit_id, intensity)`` each of length ``N · R``. Depths are in each
    observer's own frame (distance along its rays), ids are the world object indices."""
    from .renderer import render_frame

    cfg1 = single_view_config(cfg)
    pos = np.asarray(positions, float)
    outs = [render_frame(to_observer_frame(pos, pose), radii, reflectivities, cfg1, rng=rng,
                         visible=visible) for pose in observer_poses(cfg)]
    return tuple(np.concatenate([o[j] for o in outs]) for j in range(3))
