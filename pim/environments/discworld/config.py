"""Simulation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class SimConfig:
    """All parameters for one simulation run.

    World geometry
    --------------
    The frustum is a perspective cone: x_max(y) = (x_far / y_far) * y.
    Default values satisfy x_near / y_near == x_far / y_far == 0.50, so the
    frustum is a proper pinhole-camera frustum and the ray-caster's FOV
    naturally covers it exactly.

    The defaults give width=12 / height=9 (≈1.3:1), intentionally close to
    square so the 2D visualisation panel looks balanced.

    Observer is at the origin (0, 0).  The frustum occupies y ∈ [y_near, y_far].
    """

    # ---- reproducibility ------------------------------------------------
    seed: int = 42

    # ---- world geometry -------------------------------------------------
    y_near: float = 3.0  # depth of near plane
    y_far: float = 12.0  # depth of far plane
    x_near: float = 1.5  # frustum half-width at y_near  (= 0.50 * y_near)
    x_far: float = 6.0  # frustum half-width at y_far   (= 0.50 * y_far)

    # ---- objects --------------------------------------------------------
    n_objects: int | None = 3  # fixed count; None → random in [min, max]
    n_objects_min: int = 1
    n_objects_max: int = 5
    radius: float = 0.5
    speed_min: float = 0.05  # world units per frame
    speed_max: float = 0.12

    # ---- simulation -----------------------------------------------------
    n_frames: int = 100
    dt: float = 1.0
    # Trajectory noise — all default to 0 (perfectly linear, constant speed).
    #
    # Velocity noise — perturbs the velocity vector each step.
    #   direction_noise_std: noise on velocity angle (radians/step).
    #     0 = straight lines, ~0.05 = gently curving, ~0.15 = winding.
    #   speed_noise_std: fractional noise on speed (multiplicative, per step).
    #     0 = constant speed, ~0.02 = slight drift, ~0.08 = clearly varying.
    #
    # Position noise — independent Gaussian added directly to position each step
    # (diffusion / Brownian term on top of the velocity drift).
    #   position_noise_std: std in world units per step.
    #     0 = pure drift, ~0.05 = slight jitter, ~0.2 = noticeable diffusion.
    #   With velocity noise = 0 and position_noise_std > 0, motion is exactly
    #   drift (constant velocity) + diffusion.
    direction_noise_std: float = 0.0
    speed_noise_std: float = 0.0
    position_noise_std: float = 0.0

    # ---- 1D observation -------------------------------------------------
    obs_res: int = 128
    # Each object is assigned a reflectivity drawn from Uniform(refl_min, refl_max).
    # Rays return that object's reflectivity as intensity, with no depth falloff.
    refl_min: float = 0.4
    refl_max: float = 0.8
    # Minimum pairwise separation between any two objects' reflectivities.
    # 0.0 = no constraint; e.g. 0.15 ensures objects are distinguishable in
    # the intensity signal.  Requires refl_min_sep * (n_objects-1) ≤ refl_max - refl_min.
    refl_min_sep: float = 0.15
    # If True, reflectivities are uniformly spaced between refl_min and refl_max
    # (deterministic, same for every sample).  Overrides refl_min_sep.
    # If False (default), reflectivities are sampled randomly each scene.
    fixed_reflectivities: bool = False
    # Additive Gaussian noise applied to the full intensity array (incl. background).
    # Set to 0 to disable noise entirely.
    obs_noise_std: float = 0.04  # std in intensity units [0, 1]

    # ---- boundary behaviour ---------------------------------------------
    # "bounce" — objects reflect off the frustum walls (default).
    # "open"   — no boundary; objects drift freely, becoming invisible when
    #            outside the frustum / FOV.  The frustum is just a viewport.
    # "wrap"   — toroidal wrap in the bounding rectangle [-x_far, x_far] ×
    #            [y_near, y_far].  Objects that exit one side reappear on the
    #            opposite side.  Near-boundary ghost collisions are not checked.
    boundary: Literal["bounce", "open", "wrap"] = "bounce"

    # ---- scene generation -----------------------------------------------
    # If True, reject any trajectory where an object's circle is not fully
    # contained inside the frustum at every frame (i.e. never touches an edge).
    # Uses the same rejection-sampling loop as collision avoidance.
    always_in_frustum: bool = False
    max_gen_attempts: int = 300
    # Minimum inter-object separation = collision_margin * 2 * radius.
    # Must be large enough that objects cannot fully overlap between frames.
    collision_margin: float = 1.6

    # ---- soft / differentiable rendering (OPTIONAL) ----------------------
    # All four default to "off"; with the defaults the renderer is bit-for-bit
    # the original hard ray-caster.  See `pim/simulator/soft_render.py` for the
    # full rationale.  Summary:
    #   soft_edge          silhouette softness in world units (0 = hard indicator).
    #                      Makes the render CONTINUOUS in position.
    #   soft_shading       "flat" (constant reflectivity across the disc, the
    #                      original) or "lambert" (x |n.d| = sqrt(1-(perp/r)^2)).
    #                      This is the knob that makes the object's image CURVED,
    #                      so its derivative is nonzero across the whole object
    #                      instead of only at the two silhouette edges.
    #   soft_psf_sigma     Gaussian sensor blur along the ray axis, in rays.
    #   soft_occlusion_temp  temperature of the soft depth ordering; > 0 replaces
    #                      the nearest-hit argmin with a differentiable blend and
    #                      makes the whole renderer autograd-traceable.
    soft_edge: float = 0.0
    soft_shading: Literal["flat", "lambert"] = "flat"
    soft_psf_sigma: float = 0.0
    soft_occlusion_temp: float = 0.0

    # ---- omniscient 2D rendering (OPTIONAL) ------------------------------
    # Off by default; with `omni2d = False` the renderer is bit-for-bit the
    # original 1D ray-caster.  See `pim/simulator/render2d.py` for the full
    # rationale.  Summary: replaces the 1D perspective scan with a top-down
    # ORTHOGRAPHIC raster of the world rectangle -- no projection, no
    # occlusion, no perspective.  The world state is unchanged; only the
    # observation channel differs, so a model trained on it can be compared
    # against a 1D model on the very same scenes.
    #
    # Frames are flattened row-major to (omni2d_h * omni2d_w,), and `obs_res`
    # MUST equal that product -- the whole downstream stack sizes itself from
    # `obs_res`.  Use `obs_dim(cfg)` rather than recomputing it, and call
    # `render2d.validate(cfg)` to catch a mismatch early.
    #
    #   omni2d    enable the omniscient raster.
    #   omni2d_h  rows, spanning depth y in [y_near, y_far]; row 0 = near plane.
    #   omni2d_w  columns, spanning x in [-x_far, x_far].
    # The defaults give exactly square pixels for the default world geometry.
    omni2d: bool = False
    omni2d_h: int = 48
    omni2d_w: int = 64


def obs_dim(cfg: SimConfig) -> int:
    """Flat observation dimensionality — the single source of truth for ``R``.

    The 1D ray-caster gives ``obs_res`` rays; the omniscient 2D raster gives a
    flattened ``omni2d_h x omni2d_w`` grid.  Every array-sizing site should call
    this rather than reading ``obs_res`` directly, so that adding an observation
    channel never means hunting down shape literals.
    """
    if getattr(cfg, "omni2d", False):
        return int(cfg.omni2d_h) * int(cfg.omni2d_w)
    return int(cfg.obs_res)
