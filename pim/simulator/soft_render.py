"""Soft / differentiable rendering — an **optional extension** of `renderer.py`.

`renderer.py` is untouched and remains the default. Everything here is gated behind
four `SimConfig` knobs that all default to "off", and with all four off this module
delegates straight back to `render_frame`, so existing datasets, checkpoints and
results are bit-for-bit unaffected (pinned by `tests/test_soft_render.py`).

Why this exists
---------------
The hard renderer maps positions to observations through a **piecewise-constant**
function: `obs_intensity` is the first-hit object's reflectivity, flat across the
whole disc, with no antialiasing. Two consequences drove
`research/scratch/2026-08-05-observation-space-geometry.md`:

1. the render is *discontinuous* in position (its Jacobian is 0 almost everywhere
   and undefined on the jump set), and
2. because the interior of an object's image is **flat**, moving an object changes
   only the two rays at its silhouette edges.

(2) is the mechanism behind `readable ≠ grabbable`: a linear position probe reads
the *plateau*, while moving the object requires changing the *edges*, and a plateau
is nearly perpendicular to the spikes at its own edges. This module makes both
properties adjustable so the claim can be tested rather than assumed.

The four knobs, and what each one is for
----------------------------------------
``soft_edge`` (world units, 0 = hard)
    Silhouette softness. Coverage becomes ``sigmoid(s / soft_edge)`` where
    ``s = radius − (perpendicular distance from the object centre to the ray)``.
    Makes the render **continuous** in position. Does *not* change the flat
    interior, so on its own it should not move the geometry result.

``soft_shading`` (``"flat"`` | ``"lambert"``)
    ``"lambert"`` multiplies reflectivity by ``|n·d|`` — the cosine between the
    surface normal at the hit point and the ray — which for a circle works out to
    ``sqrt(1 − (perp/r)²)``. This is the knob that **changes the structure**: a
    curved profile has nonzero derivative at *every* ray the object covers, not
    just at its edges.

``soft_psf_sigma`` (rays, 0 = none)
    Gaussian point-spread function, i.e. a sensor blur, applied as a fixed linear
    operator along the ray axis. Note this leaves the interior of a flat plateau
    flat (a convolution of a constant is that constant) and only widens the edge
    transition, so like ``soft_edge`` it is expected to be structurally inert.

``soft_occlusion_temp`` (depth units, 0 = hard nearest-hit)
    The only genuinely non-differentiable step is the nearest-hit ``argmin`` over
    objects. With ``temp > 0`` the "is k in front of j" test becomes
    ``sigmoid((t_j − t_k) / temp)`` and the frame is alpha-composited with soft
    depth ordering, making the whole renderer autograd-traceable. ``temp = 0``
    recovers exact front-to-back compositing.

**"Smoothed but not differentiable"** — the realistic control, and what an ordinary
antialiased simulator does — is ``soft_edge > 0``, ``soft_shading="lambert"``,
``soft_psf_sigma > 0``, ``soft_occlusion_temp = 0``.

Backends
--------
`render_frame_soft` (NumPy) is used for dataset generation. `render_frame_torch`
is the autograd-traceable twin, batched over scenes, used for exact Jacobians
`∂obs/∂position`. They implement identical maths and are pinned to agree.
"""

from __future__ import annotations

import numpy as np

from pim.simulator.config import SimConfig

__all__ = [
    "soft_enabled",
    "blur_matrix",
    "render_frame_soft",
    "render_frame_torch",
]

_EPS = 1e-9
_FAR = 1e9  # stand-in for "no hit"; finite so it survives arithmetic
_SQ_FLOOR = 1e-16  # keeps sqrt's gradient finite without changing its value


def _sigmoid(z):
    """Overflow-free logistic. `1/(1+exp(-z))` overflows for z << 0 at small
    softness; the tanh form saturates gracefully and is identical in exact math."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def soft_enabled(cfg: SimConfig) -> bool:
    """True when any soft-rendering knob is off its default."""
    return (
        getattr(cfg, "soft_edge", 0.0) > 0.0
        or getattr(cfg, "soft_shading", "flat") != "flat"
        or getattr(cfg, "soft_psf_sigma", 0.0) > 0.0
        or getattr(cfg, "soft_occlusion_temp", 0.0) > 0.0
    )


def _ray_dirs(cfg: SimConfig):
    """Unit ray directions, identical to `renderer.render_frame`."""
    from pim.simulator.renderer import _fov_scale

    s = np.linspace(-1.0, 1.0, cfg.obs_res)
    dx = s * _fov_scale(cfg)
    dy = np.ones(cfg.obs_res)
    norm = np.hypot(dx, dy)
    return dx / norm, dy / norm


def blur_matrix(obs_res: int, sigma: float) -> np.ndarray:
    """(R, R) row-normalised Gaussian PSF operator.

    Row normalisation (rather than zero padding) keeps the frame's edges at the
    same brightness as its interior; padding would darken them and manufacture a
    boundary artefact that looks like signal.
    """
    if sigma <= 0:
        return np.eye(obs_res)
    i = np.arange(obs_res)
    K = np.exp(-((i[:, None] - i[None, :]) ** 2) / (2.0 * sigma**2))
    return K / K.sum(axis=1, keepdims=True)


# ── NumPy backend (dataset generation; the "standard simulator" control) ──────


def render_frame_soft(
    positions: np.ndarray,
    radii: np.ndarray,
    reflectivities: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft render of one frame. Signature matches `renderer.render_frame`.

    With every knob at its default this delegates to `render_frame`, so it is
    exactly the original renderer rather than an approximation of it.
    """
    from pim.simulator.renderer import render_frame

    if not soft_enabled(cfg):
        return render_frame(positions, radii, reflectivities, cfg, rng=rng)

    edge = float(getattr(cfg, "soft_edge", 0.0))
    shading = getattr(cfg, "soft_shading", "flat")
    psf = float(getattr(cfg, "soft_psf_sigma", 0.0))
    temp = float(getattr(cfg, "soft_occlusion_temp", 0.0))

    positions = np.asarray(positions, float)
    radii = np.asarray(radii, float)
    dx, dy = _ray_dirs(cfg)

    # Deliberately mirrors `render_frame_torch` line for line; the two are pinned
    # to agree by `tests/test_soft_render.py::test_numpy_and_torch_backends_agree`.
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
    cos_n_d = sq / np.maximum(radii[None, :], _EPS)
    alpha = gate * (_sigmoid(signed / edge) if edge > 0 else (signed > 0))
    shade = reflectivities[None, :] * (cos_n_d if shading == "lambert" else 1.0)

    dt = t_eff[..., None, :] - t_eff[..., :, None]
    front = _sigmoid(dt / temp) if temp > 0 else (dt > 0).astype(float)
    eye = np.eye(alpha.shape[-1])
    keep = 1.0 - alpha[..., None, :] * front * (1.0 - eye)
    intensity = (alpha * shade * keep.prod(-1)).sum(-1)

    if psf > 0:
        intensity = blur_matrix(cfg.obs_res, psf) @ intensity

    # Depth / id use the STRICT hard-hit criterion, not the relaxed one that drives
    # `alpha`. The relaxed gate deliberately keeps `t_eff` finite for rays that merely
    # pass *near* an object (so coverage stays continuous), which would otherwise report
    # a "hit" on essentially every ray -- and `obs_id` is consumed downstream.
    strict = (radii[None, :] ** 2 - perp2 >= 0) & (t_eff < _FAR)
    t_strict = np.where(strict, t_eff, _FAR)
    best = np.argmin(t_strict, axis=-1)
    hit = t_strict[np.arange(cfg.obs_res), best] < _FAR
    hit_depth = np.where(hit, dy * t_strict[np.arange(cfg.obs_res), best], 0.0)
    hit_id = np.where(hit, best, -1).astype(int)

    if cfg.obs_noise_std > 0 and rng is not None:
        intensity = np.clip(
            intensity + rng.normal(0.0, cfg.obs_noise_std, cfg.obs_res), 0.0, 1.0
        )
    else:
        intensity = np.clip(intensity, 0.0, 1.0)
    return hit_depth, hit_id, intensity


# ── Torch backend (autograd-traceable; used for exact ∂obs/∂position) ─────────


def render_frame_torch(positions, radii, reflectivities, cfg: SimConfig):
    """Differentiable render, batched over leading dimensions.

    Parameters
    ----------
    positions : (..., n_obj, 2) tensor, **requires_grad-able**
    radii, reflectivities : (n_obj,) tensors
    cfg : SimConfig; `soft_occlusion_temp > 0` makes the whole map smooth

    Returns
    -------
    intensity : (..., obs_res) tensor, differentiable w.r.t. `positions`

    Noise is never added here: this is the *clean* render, which is the object
    whose Jacobian we want.
    """
    import torch

    edge = float(getattr(cfg, "soft_edge", 0.0))
    shading = getattr(cfg, "soft_shading", "flat")
    psf = float(getattr(cfg, "soft_psf_sigma", 0.0))
    temp = float(getattr(cfg, "soft_occlusion_temp", 0.0))

    dev, dt_ = positions.device, positions.dtype
    dxn, dyn = _ray_dirs(cfg)
    dx = torch.as_tensor(dxn, device=dev, dtype=dt_)
    dy = torch.as_tensor(dyn, device=dev, dtype=dt_)

    cx, cy = positions[..., 0], positions[..., 1]
    b = dx[..., None] * cx[..., None, :] + dy[..., None] * cy[..., None, :]
    cc = cx**2 + cy**2
    perp2 = torch.clamp(cc[..., None, :] - b**2, min=0.0)
    r2 = (radii**2)[..., None, :]
    disc = r2 - perp2
    sq = torch.where(
        disc > 0, torch.sqrt(torch.clamp(disc, min=_SQ_FLOOR)), torch.zeros_like(disc)
    )
    t_front, t_back = b - sq, b + sq
    hy_f, hy_b = dy[..., None] * t_front, dy[..., None] * t_back
    t_at_near = cfg.y_near / dy[..., None]

    visible = (t_front > _EPS) & (hy_f >= cfg.y_near) & (hy_f <= cfg.y_far)
    clamp_near = (hy_f < cfg.y_near) & (hy_b >= cfg.y_near)
    t_eff = torch.where(
        visible,
        t_front,
        torch.where(
            clamp_near, t_at_near.expand_as(t_front), torch.full_like(t_front, _FAR)
        ),
    )
    gate = (t_eff < _FAR).to(dt_)
    signed = radii[..., None, :] - torch.sqrt(perp2 + _EPS)
    alpha = gate * (torch.sigmoid(signed / edge) if edge > 0 else (signed > 0).to(dt_))
    cos_n_d = sq / torch.clamp(radii[..., None, :].expand_as(sq), min=_EPS)
    shade = reflectivities[..., None, :] * (cos_n_d if shading == "lambert" else 1.0)

    d_t = t_eff[..., None, :] - t_eff[..., :, None]
    front = torch.sigmoid(d_t / temp) if temp > 0 else (d_t > 0).to(dt_)
    eye = torch.eye(alpha.shape[-1], device=dev, dtype=dt_)
    keep = 1.0 - alpha[..., None, :] * front * (1.0 - eye)
    intensity = (alpha * shade * keep.prod(-1)).sum(-1)

    if psf > 0:
        B = torch.as_tensor(blur_matrix(cfg.obs_res, psf), device=dev, dtype=dt_)
        intensity = intensity @ B.T
    return torch.clamp(intensity, 0.0, 1.0)
