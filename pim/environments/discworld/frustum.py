"""World coordinates → the frustum basis the renderer actually samples.

Why this basis and not (angle, range)
-------------------------------------
Read off `pim/simulator/renderer.py`, not assumed:

* The observer is at the origin and rays are indexed by ``s = linspace(-1, 1, obs_res)`` with
  direction proportional to ``(s * scale, 1)``, ``scale = x_far / y_far``. So the rays are uniform
  in **tan θ**, not in θ. An object at ``(x, y)`` falls on ray ``s = x / (scale * y)``. That makes

      u = x / (scale * y)          in [-1, 1] across the frustum

  the lateral coordinate that maps **linearly onto the observation cells**. Using the angle
  ``atan2(x, y)`` instead is a monotone but *nonlinear* reparameterisation which no longer lines up
  with the sensor, so a linear probe would have to undo an arctangent.
* ``render_frame`` returns ``hit_depth = dy * t``, i.e. the **y-coordinate** of the hit, and culls
  on ``y in [y_near, y_far]``. Depth is therefore ``y``, not Euclidean range ``sqrt(x^2 + y^2)``.

The reference direction is the frustum centreline (``u = 0`` at ``x = 0``). Choosing a wall instead
is an additive offset, which a linear probe absorbs into its bias — it cannot matter.

Velocity
--------
Exact, by differentiating the map:

    u = x / (scale * y)      ->   u̇ = (ẋ * y - x * ẏ) / (scale * y**2)
    d = y                    ->   ḋ = ẏ

The ``1 / y**2`` weight is the substantive part: the same world motion produces a large ``u̇`` near
the observer and a small one far away. If the model encodes lateral motion as *image-plane* motion —
which is what it can see — then ``ẋ`` is the wrong probe target and ``u̇`` is the right one. That is
the concrete hypothesis for why velocity decodability has been stuck near R² 0.28 in the Cartesian
basis while position reaches 0.80.
"""

from __future__ import annotations

import numpy as np


def fov_scale(sim: dict) -> float:
    """tan(half-FOV) — the same quantity `renderer._fov_scale` uses."""
    return float(sim["x_far"]) / float(sim["y_far"])


def world_to_frustum(pos: np.ndarray, vel: np.ndarray | None, sim: dict):
    """(..., 2) world (x, y) [and (ẋ, ẏ)] → frustum (u, d) [and (u̇, ḋ)].

    Parameters
    ----------
    pos : (..., 2) float — world positions, last axis (x, y).
    vel : (..., 2) float or None — world velocities, last axis (ẋ, ẏ).
    sim : the dataset's `sim` config dict (needs `x_far`, `y_far`).

    Returns
    -------
    fpos : (..., 2) — (u, d). `u` is dimensionless in [-1, 1] inside the frustum; `d` is `y`.
    fvel : (..., 2) or None — (u̇, ḋ).
    """
    scale = fov_scale(sim)
    x, y = pos[..., 0], pos[..., 1]
    # y is bounded below by y_near (3.0 by default) so no guard is needed in-frustum; the clip is
    # only for objects placed outside it by a synthetic edit.
    ys = np.where(np.abs(y) < 1e-6, 1e-6, y)
    u = x / (scale * ys)
    fpos = np.stack([u, y], axis=-1)
    if vel is None:
        return fpos, None
    vx, vy = vel[..., 0], vel[..., 1]
    du = (vx * ys - x * vy) / (scale * ys**2)
    fvel = np.stack([du, vy], axis=-1)
    return fpos, fvel


def frustum_to_world(fpos: np.ndarray, sim: dict) -> np.ndarray:
    """Inverse of the position map: (u, d) → (x, y). Needed to express an edit target."""
    scale = fov_scale(sim)
    u, d = fpos[..., 0], fpos[..., 1]
    return np.stack([u * scale * d, d], axis=-1)


def ray_index(u: np.ndarray, obs_res: int = 128) -> np.ndarray:
    """`u` → fractional observation-cell index, purely for interpretation.

    Affine in `u`, so a linear probe cannot tell the two apart; use `u`.
    """
    return (u + 1.0) * 0.5 * (obs_res - 1)


# ── depth parameterisations ───────────────────────────────────────────────────
#
# ⚠ `render_frame` returns a `hit_depth` field equal to the y-coordinate, but **the world model
# never sees it** — it is trained on `obs_intensity` alone. So `y` has no privileged status; it is
# a leftover of the renderer's internals. (Sevan, 2026-08-23.)
#
# What the model can actually observe about depth is the **apparent width** of an object, because
# radius is constant (0.5) and reflectivity encodes identity, not distance. Deriving that width:
#
#   ray s has direction proportional to (s*k, 1), k = x_far / y_far
#   the ray meets a disc of radius r centred at (x, y) iff  |x - y*k*s| <= r*sqrt(1 + s^2 k^2)
#   writing u = x / (k*y) and evaluating at s ~ u, the half-width in ray units is
#
#       w  =  r * sqrt(1 + u^2 k^2) / (k * y)   =   r * rho / (k * y^2),    rho = sqrt(x^2+y^2)
#
# So apparent width goes as **1/depth**, which is why an *inverse* depth coordinate (disparity) is
# the natural one: it is linear in the quantity the model can read off by counting lit rays. The
# candidates below are ranked by that argument but chosen **empirically** — see
# `research/scratch/2026-08-23-frustum-basis.md`.

# ``basis(..., depth="frustum")`` is the canonical alias: one name for "the frustum
# basis" in configs, tables and scores.json, so the depth coordinate can be re-settled
# without renaming every artifact that mentions it. Currently -> inv_y.
CANONICAL_DEPTH = "inv_y"

DEPTHS = {
    "y": lambda x, y, sim: y,                                   # axial depth (the sim's leftover)
    "rho": lambda x, y, sim: np.hypot(x, y),                    # Euclidean range
    "inv_y": lambda x, y, sim: 1.0 / y,                         # inverse axial depth
    "inv_rho": lambda x, y, sim: 1.0 / np.hypot(x, y),          # inverse range (disparity)
    "width": lambda x, y, sim: (                                # apparent half-width in ray units
        float(sim["radius"]) * np.hypot(x, y)
        / (fov_scale(sim) * np.maximum(y, 1e-6) ** 2)
    ),
}


def lateral(pos: np.ndarray, sim: dict) -> np.ndarray:
    """The ray coordinate `u = x / (scale * y)` — linear in observation-cell index."""
    scale = fov_scale(sim)
    x, y = pos[..., 0], pos[..., 1]
    return x / (scale * np.where(np.abs(y) < 1e-6, 1e-6, y))


def basis(pos: np.ndarray, vel: np.ndarray | None, sim: dict, depth: str = "frustum"):
    """(u, g(x,y)) and their time derivatives, for any named depth parameterisation.

    ``depth="frustum"`` (the default) resolves to ``CANONICAL_DEPTH``; the five concrete
    names stay available for pilots that re-settle which one that should be.

    Velocities are taken by central differences on the analytic map rather than by hand-derived
    Jacobians — one code path for five candidates, and no algebra to get wrong. Verified against
    the closed form for `u` and `y` to ~1e-6.
    """
    g = DEPTHS[CANONICAL_DEPTH if depth == "frustum" else depth]
    x, y = pos[..., 0], pos[..., 1]
    fpos = np.stack([lateral(pos, sim), g(x, y, sim)], axis=-1)
    if vel is None:
        return fpos, None
    h = 1e-5
    p_plus, p_minus = pos + vel * h, pos - vel * h
    f_plus = np.stack([lateral(p_plus, sim), g(p_plus[..., 0], p_plus[..., 1], sim)], -1)
    f_minus = np.stack([lateral(p_minus, sim), g(p_minus[..., 0], p_minus[..., 1], sim)], -1)
    return fpos, (f_plus - f_minus) / (2 * h)
