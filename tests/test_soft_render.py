"""Tests for the optional soft / differentiable renderer.

The load-bearing one is `test_defaults_are_bit_identical`: `soft_render.py` is an
*extension*, and every existing dataset, checkpoint and result depends on the hard
renderer being untouched when the knobs are off.

The rest pin the two properties the module exists to provide: that the soft path
converges to the hard one as the softness goes to zero (so it is the same renderer,
not a different one), and that `lambert` shading actually moves the derivative off
the silhouette and into the object's interior — which is the whole scientific point.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pim.simulator.config import SimConfig
from pim.simulator.renderer import render_frame
from pim.simulator.soft_render import (
    render_frame_soft,
    render_frame_torch,
    soft_enabled,
)

POS = np.array([[-1.0, 6.0], [1.6, 8.5]])
RADII = np.array([0.5, 0.5])
REFL = np.array([0.4, 0.8])


def _cfg(**kw) -> SimConfig:
    return SimConfig(n_objects=2, obs_res=128, obs_noise_std=0.0, **kw)


def test_defaults_are_bit_identical():
    """With every knob off, nothing about the original renderer may change."""
    cfg = _cfg()
    assert not soft_enabled(cfg)
    hard = render_frame(POS, RADII, REFL, cfg)
    soft = render_frame_soft(POS, RADII, REFL, cfg)
    for a, b in zip(hard, soft):
        assert np.array_equal(a, b)


def test_soft_edge_converges_to_hard():
    """As the silhouette softness -> 0 the soft path must reproduce the hard one.

    Measured as MEAN absolute error, not max: at the silhouette one ray always
    disagrees (a sigmoid at its midpoint vs a step), so the max is not monotone
    even though the render is converging.
    """
    hard = render_frame(POS, RADII, REFL, _cfg())[2]
    errs = [
        np.abs(render_frame_soft(POS, RADII, REFL, _cfg(soft_edge=e))[2] - hard).mean()
        for e in (0.05, 0.01, 0.002, 0.0005)
    ]
    assert all(a > b for a, b in zip(errs, errs[1:])), f"not converging: {errs}"
    assert errs[-1] < 1e-3


def test_numpy_and_torch_backends_agree():
    """The two implementations are written out separately; pin them together."""
    cfg = _cfg(soft_edge=0.05, soft_shading="lambert", soft_psf_sigma=1.0)
    npy = render_frame_soft(POS, RADII, REFL, cfg)[2]
    tor = render_frame_torch(
        torch.tensor(POS, dtype=torch.float64),
        torch.tensor(RADII, dtype=torch.float64),
        torch.tensor(REFL, dtype=torch.float64),
        cfg,
    ).numpy()
    assert np.abs(npy - tor).max() < 1e-6


def test_torch_render_is_differentiable():
    """The whole point of the torch backend: exact d(obs)/d(position)."""
    cfg = _cfg(soft_edge=0.05, soft_shading="lambert", soft_occlusion_temp=0.05)
    p = torch.tensor(POS, dtype=torch.float64, requires_grad=True)
    out = render_frame_torch(
        p,
        torch.tensor(RADII, dtype=torch.float64),
        torch.tensor(REFL, dtype=torch.float64),
        cfg,
    )
    (g,) = torch.autograd.grad(out.sum(), p)
    assert torch.isfinite(g).all(), "renderer produced non-finite gradients"
    assert g.abs().max() > 1e-6, "renderer produced no gradient w.r.t. position"


def test_soft_occlusion_gives_gradient_to_the_hidden_object():
    """With hard nearest-hit an occluded object gets exactly zero gradient; the
    soft depth blend is what restores it, and that is why `temp > 0` exists."""
    cfg_hard = _cfg(soft_edge=0.05, soft_shading="lambert", soft_occlusion_temp=0.0)
    cfg_soft = _cfg(soft_edge=0.05, soft_shading="lambert", soft_occlusion_temp=0.2)
    behind = torch.tensor([[0.0, 6.0], [0.02, 9.0]], dtype=torch.float64)

    def grad_on_hidden(cfg):
        p = behind.clone().requires_grad_(True)
        out = render_frame_torch(
            p,
            torch.tensor(RADII, dtype=torch.float64),
            torch.tensor(REFL, dtype=torch.float64),
            cfg,
        )
        (g,) = torch.autograd.grad(out.sum(), p)
        return float(g[1].abs().max())

    assert grad_on_hidden(cfg_soft) > grad_on_hidden(cfg_hard)


def _participation_ratio(**kw):
    """Effective number of rays carrying the change when one object is nudged.

    `N_eff = (sum|d|)^2 / sum(d^2)`.  Threshold-free: ~1 when the change is a
    single ray, ~n when it is spread over n rays.  This is the quantity the whole
    module exists to move -- `readable != grabbable` rests on the change being
    concentrated at the silhouette while position information is spread over the
    object's whole image.
    """
    cfg = SimConfig(n_objects=1, obs_res=128, obs_noise_std=0.0, **kw)
    one_pos, one_r, one_refl = np.array([[0.0, 7.5]]), np.array([0.5]), np.array([0.8])
    a = render_frame_soft(one_pos, one_r, one_refl, cfg)[2]
    moved = one_pos.copy()
    moved[0, 0] += 0.05
    d = np.abs(render_frame_soft(moved, one_r, one_refl, cfg)[2] - a)
    return float(d.sum() ** 2 / max((d**2).sum(), 1e-30))


def test_hard_renderer_change_is_a_single_ray():
    """The premise of the geometry result: the hard render's derivative is a spike."""
    assert _participation_ratio() < 1.5


def test_softening_the_edge_spreads_the_derivative():
    """`soft_edge` is the knob that actually matters -- corrected 2026-08-05.

    Prior expectation (wrong, recorded here so it is not repeated): that shading
    would be the structural knob and antialiasing/blur would be inert. Measured,
    it is the other way round. Softening the silhouette turns the derivative from
    a delta into a band ~1 -> ~10 rays; a Lambertian dome still has its steepest
    slope at the rim and zero slope at its apex, so curving the profile adds
    little on top.
    """
    hard = _participation_ratio()
    edge = _participation_ratio(soft_edge=0.05)
    blur = _participation_ratio(soft_edge=0.05, soft_psf_sigma=1.5)
    assert edge > 5 * hard, f"soft_edge should spread the change: {hard} -> {edge}"
    assert blur > edge, f"psf should spread it further: {edge} -> {blur}"


def test_lambert_curves_the_profile():
    """Shading does change the image shape, even though it barely moves `N_eff`."""
    one_pos, one_r, one_refl = np.array([[0.0, 7.5]]), np.array([0.5]), np.array([0.8])

    def interior_std(shading):
        cfg = SimConfig(
            n_objects=1,
            obs_res=128,
            obs_noise_std=0.0,
            soft_edge=0.02,
            soft_shading=shading,
        )
        o = render_frame_soft(one_pos, one_r, one_refl, cfg)[2]
        lit = np.where(o > 0.5 * o.max())[0]
        return float(o[lit[2] : lit[-2]].std())

    flat, lambert = interior_std("flat"), interior_std("lambert")
    # `flat` is not exactly 0 because the sigmoid coverage has a tail reaching a
    # ray or two inside the silhouette; the point is the ratio, not the absolute.
    assert flat < 1e-3, f"flat shading should be near-flat inside, got {flat:.2e}"
    assert (
        lambert > 50 * flat
    ), f"lambert should curve the profile: {flat:.2e} -> {lambert:.2e}"


@pytest.mark.parametrize("temp", [0.0, 0.05])
def test_soft_occlusion_recovers_hard_ordering(temp):
    """A small temperature must not visibly change the image, only its smoothness."""
    base = render_frame_soft(
        POS, RADII, REFL, _cfg(soft_edge=0.05, soft_shading="lambert")
    )[2]
    out = render_frame_soft(
        POS,
        RADII,
        REFL,
        _cfg(soft_edge=0.05, soft_shading="lambert", soft_occlusion_temp=temp),
    )[2]
    assert np.abs(out - base).max() < 0.05
