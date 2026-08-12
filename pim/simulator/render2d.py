"""Omniscient 2D rendering — an **optional extension** of `renderer.py`.

`renderer.py` is untouched and remains the default. Everything here is gated behind
`SimConfig.omni2d`, which defaults to ``False``; with it off this module is never
reached, so existing datasets, checkpoints and results are bit-for-bit unaffected
(pinned by `tests/test_render2d.py::test_defaults_are_bit_identical`).

Why this exists
---------------
The default observation is a **1D perspective scan**: a fan of rays from the origin,
each returning the reflectivity of the *first* surface it hits. That observation is
lossy in two separate ways — it **projects** (a 2D world collapses to a 1D signal)
and it **occludes** (only the nearest surface is reported). Every editability result
in the thread so far was measured through it.

This module supplies the complementary observation: a top-down **orthographic**
raster of the world rectangle, in which nothing is projected away and nothing is
hidden. The world state is the same; only the observation channel changes. It exists
so "readable ≠ grabbable" can be asked of a model that was never denied information
in the first place — if the negative survives full observability, it is not a
consequence of the observation being impoverished.

The grid
--------
Rows span the depth axis ``y ∈ [y_near, y_far]``, columns the lateral axis
``x ∈ [-x_far, x_far]`` — the **bounding rectangle of the frustum**, not the frustum
trapezoid, so the grid is a plain rectangle and the corners outside the trapezoid are
simply never occupied. With the canonical world (12 × 9 units) at the canonical
48 × 64 grid the pixels are exactly square (0.1875 world units each) and a
radius-0.5 disc is ~5.3 px across.

``row 0`` is the **near** plane (``y_near``) and rows increase with depth. Pixel
centres are at ``y_near + (i + 0.5)·dy`` and ``-x_far + (j + 0.5)·dx``.

Flattening
----------
Frames are returned **flattened row-major** (``index = row·W + col``), so the entire
downstream stack — HDF5 schema, dataloaders, the `HiddenStateModel` protocol, every
`pim/eval` metric — continues to see an ``(N, T, R)`` observation with ``R = H·W``
and needs no changes. Only code that *draws* an observation has to know the frame is
2D; use `unflatten` for that.

What it deliberately does NOT do
--------------------------------
* **No occlusion.** Each pixel reports the object containing its centre. Under the
  canonical config objects cannot overlap at all (`collision_margin = 1.6` keeps
  centres ≥ 1.6 diameters apart, versus the 1.0 diameter needed to touch), so the
  tie-break below is unreachable; it exists only so the function is total.
* **No depth falloff and no perspective.** Intensity is the object's reflectivity,
  independent of ``y``. Apparent size is likewise constant with depth — that is the
  whole point of the orthographic view, and it is the sharpest difference from the
  1D scan, where a near object subtends ~4× the rays of a far one.
* **No soft edges.** The disc is a hard indicator, matching `renderer.py`'s
  piecewise-constant silhouette. `soft_render.py`'s knobs are a separate axis and the
  two extensions are mutually exclusive (asserted below).
"""

from __future__ import annotations

import numpy as np

from .config import SimConfig, obs_dim


def omni2d_enabled(cfg: SimConfig) -> bool:
    """True when the omniscient 2D raster replaces the 1D ray-cast."""
    return bool(getattr(cfg, "omni2d", False))


def grid_shape(cfg: SimConfig) -> tuple[int, int]:
    """``(H, W)`` — rows over depth, columns over the lateral axis."""
    return int(cfg.omni2d_h), int(cfg.omni2d_w)


def grid_axes(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinates: ``(y_centres (H,), x_centres (W,))``.

    Row 0 is the near plane. Both axes are uniformly spaced over the frustum's
    bounding rectangle.
    """
    H, W = grid_shape(cfg)
    dy = (cfg.y_far - cfg.y_near) / H
    dx = (2.0 * cfg.x_far) / W
    y = cfg.y_near + (np.arange(H) + 0.5) * dy
    x = -cfg.x_far + (np.arange(W) + 0.5) * dx
    return y, x


def pixel_size(cfg: SimConfig) -> tuple[float, float]:
    """``(dy, dx)`` in world units per pixel."""
    H, W = grid_shape(cfg)
    return (cfg.y_far - cfg.y_near) / H, (2.0 * cfg.x_far) / W


def unflatten(flat: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """``(..., H·W) → (..., H, W)`` — undo the row-major flattening for display."""
    H, W = grid_shape(cfg)
    if flat.shape[-1] != H * W:
        raise ValueError(
            f"last axis is {flat.shape[-1]}, expected H·W = {H}·{W} = {H * W}"
        )
    return flat.reshape(*flat.shape[:-1], H, W)


def world_extent(cfg: SimConfig) -> tuple[float, float, float, float]:
    """``(x_min, x_max, y_min, y_max)`` for `imshow(extent=...)`, in world units."""
    return -cfg.x_far, cfg.x_far, cfg.y_near, cfg.y_far


def render_frame_omni2d(
    positions: np.ndarray,  # (n_objects, 2)
    radii: np.ndarray,  # (n_objects,)
    reflectivities: np.ndarray,  # (n_objects,)
    cfg: SimConfig,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterise one frame of the omniscient top-down view.

    Signature and return contract match `renderer.render_frame`, with every array
    flattened row-major to ``(H·W,)``.

    Returns
    -------
    hit_depth : (H·W,) float
        The **object's** y-coordinate where a pixel is occupied, 0 elsewhere. Note
        this is the disc centre's depth, not the pixel's own y (which is a constant
        function of the index and therefore carries no information). Retained only
        so the stored schema stays meaningful; nothing in the pipeline reads it.
    hit_id : (H·W,) int
        Index of the object containing the pixel centre, -1 for background.
    obs_intensity : (H·W,) float
        That object's reflectivity, plus optional additive Gaussian noise clipped
        to [0, 1]. 0 (before noise) for background.
    """
    H, W = grid_shape(cfg)
    n = len(radii)

    hit_depth = np.zeros(H * W)
    hit_id = np.full(H * W, -1, dtype=int)
    obs_intensity = np.zeros(H * W)

    if n > 0:
        y_c, x_c = grid_axes(cfg)
        # (H, W) pixel-centre coordinates, then (P, 1) against (1, N) objects.
        yy, xx = np.meshgrid(y_c, x_c, indexing="ij")
        px = xx.reshape(-1)[:, None]  # (P, 1)
        py = yy.reshape(-1)[:, None]  # (P, 1)

        d2 = (px - positions[None, :, 0]) ** 2 + (py - positions[None, :, 1]) ** 2
        inside = d2 <= (radii[None, :] ** 2)  # (P, N)

        # Deterministic tie-break: the NEAREST object (smallest y) wins the pixel.
        # Unreachable under the canonical config — objects cannot overlap — but it
        # keeps the function total for configs with a smaller collision margin.
        depth = np.where(inside, positions[None, :, 1], np.inf)  # (P, N)
        best = np.argmin(depth, axis=1)  # (P,)
        occupied = inside[np.arange(H * W), best]

        hit_id[occupied] = best[occupied]
        hit_depth[occupied] = positions[best[occupied], 1]
        obs_intensity[occupied] = reflectivities[best[occupied]]

    if cfg.obs_noise_std > 0 and rng is not None:
        obs_intensity += rng.normal(0.0, cfg.obs_noise_std, H * W)
        obs_intensity = np.clip(obs_intensity, 0.0, 1.0)

    return hit_depth, hit_id, obs_intensity


def validate(cfg: SimConfig) -> None:
    """Raise if the omniscient-2D config is internally inconsistent.

    Two ways to get this wrong, both silent and both expensive:

    * ``obs_res`` out of step with ``omni2d_h · omni2d_w``. The HDF5 schema and every
      downstream consumer size their arrays from ``obs_res``, so a mismatch writes
      correctly-shaped garbage. `obs_dim` is the single source of truth; the dataset
      scripts set ``obs_res`` from it.
    * Soft rendering enabled at the same time. The two extensions modify the same
      `render_frame` dispatch and mean different things; combining them silently
      would give the soft 1D ray-cast, not a soft 2D raster.
    """
    if not omni2d_enabled(cfg):
        return

    from .soft_render import soft_enabled

    H, W = grid_shape(cfg)
    if H <= 0 or W <= 0:
        raise ValueError(f"omni2d grid must be positive, got {H}x{W}")
    if cfg.obs_res != H * W:
        raise ValueError(
            f"omni2d requires obs_res == omni2d_h * omni2d_w, but obs_res={cfg.obs_res} "
            f"and {H}*{W}={H * W}. Set obs_res from `obs_dim(cfg)`."
        )
    if soft_enabled(cfg):
        raise ValueError(
            "omni2d and soft rendering are mutually exclusive — soft_render's knobs "
            "describe the 1D ray-caster. Turn one of them off."
        )
    assert obs_dim(cfg) == H * W  # the invariant the rest of the stack relies on
