"""1D observation renderer.

Maps latent world state → 1D observation signal via analytical ray casting.
This is the explicit renderer interface: no pixel rasterisation, no 2D image
intermediate.  The depth effects (closer = larger apparent size, faster
apparent motion) arise naturally from the geometry.

Observer sits at the origin.  Rays fan out with half-FOV = atan(x_far / y_far),
chosen so that the outermost rays graze the far corners of the frustum.
"""

from __future__ import annotations

import numpy as np

from pim.config import SimConfig
from pim.sim import Scene


def _fov_scale(cfg: SimConfig) -> float:
    """tan(half-FOV) — horizontal spread of the ray fan."""
    return cfg.x_far / cfg.y_far


def render_frame(
    positions: np.ndarray,          # (n_objects, 2)
    radii: np.ndarray,              # (n_objects,)
    reflectivities: np.ndarray,     # (n_objects,)
    cfg: SimConfig,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast rays and return the 1D observation for one frame.

    Uses vectorised ray–circle intersection: for each of the ``obs_res`` rays,
    finds the closest circle hit and records its depth, identity, and intensity.

    Returns
    -------
    hit_depth : (obs_res,) float
        Y-coordinate of the first-hit surface.  0 where no circle is hit.
    hit_id : (obs_res,) int
        Index of the first-hit object.  -1 where no circle is hit.
    obs_intensity : (obs_res,) float
        Reflectivity of the first-hit object in [0, 1], plus optional additive
        Gaussian noise clipped to [0, 1].  0 where no circle is hit.
    """
    R = cfg.obs_res
    n = len(radii)

    # Ray directions: s ∈ [-1, 1] uniformly, d = (s·scale, 1) / ‖…‖
    s = np.linspace(-1.0, 1.0, R)          # (R,)
    scale = _fov_scale(cfg)
    dx = s * scale                          # (R,)
    dy = np.ones(R)
    norm = np.hypot(dx, dy)
    dx /= norm
    dy /= norm                              # unit ray directions

    hit_depth = np.zeros(R)
    hit_id = np.full(R, -1, dtype=int)
    obs_intensity = np.zeros(R)

    if n == 0:
        return hit_depth, hit_id, obs_intensity

    cx = positions[:, 0]    # (N,)
    cy = positions[:, 1]    # (N,)

    # Ray–circle intersection (vectorised over R rays × N objects)
    # For unit ray d and circle centre c:  t = (d·c) − sqrt((d·c)² − (|c|² − r²))
    b = dx[:, None] * cx[None, :] + dy[:, None] * cy[None, :]  # (R, N)
    C = cx**2 + cy**2 - radii**2                                 # (N,)
    disc = b**2 - C[None, :]                                     # (R, N)

    t = b - np.sqrt(np.maximum(disc, 0.0))                       # (R, N)
    valid = (disc >= 0) & (t > 1e-9)
    t_masked = np.where(valid, t, np.inf)                        # (R, N)

    best_j = np.argmin(t_masked, axis=1)                         # (R,)
    best_t = t_masked[np.arange(R), best_j]                      # (R,)

    hit_mask = best_t < np.inf
    hit_depth[hit_mask] = dy[hit_mask] * best_t[hit_mask]        # y = t * dy_unit
    hit_id[hit_mask] = best_j[hit_mask]
    obs_intensity[hit_mask] = reflectivities[best_j[hit_mask]]

    if cfg.obs_noise_std > 0 and rng is not None:
        obs_intensity += rng.normal(0.0, cfg.obs_noise_std, R)
        obs_intensity = np.clip(obs_intensity, 0.0, 1.0)

    return hit_depth, hit_id, obs_intensity


def render_scene(scene: Scene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render the full 1D observation sequence for a scene.

    Returns
    -------
    obs_depth     : (n_frames, obs_res) float
    obs_id        : (n_frames, obs_res) int
    obs_intensity : (n_frames, obs_res) float  —  in [0, 1]
    """
    cfg = scene.config
    rng = np.random.default_rng(cfg.seed + 1)  # offset to avoid correlation with sim

    obs_depth = np.zeros((cfg.n_frames, cfg.obs_res))
    obs_id = np.full((cfg.n_frames, cfg.obs_res), -1, dtype=int)
    obs_intensity = np.zeros((cfg.n_frames, cfg.obs_res))

    for f in range(cfg.n_frames):
        obs_depth[f], obs_id[f], obs_intensity[f] = render_frame(
            scene.positions[f], scene.radii, scene.reflectivities, cfg, rng=rng
        )

    return obs_depth, obs_id, obs_intensity
