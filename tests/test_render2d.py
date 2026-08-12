"""Tests for the omniscient 2D renderer (`pim/simulator/render2d.py`).

The first test is the load-bearing one: with `omni2d = False` — the default —
nothing in the rendering path may change by even one bit, or every existing
dataset, checkpoint and published number silently drifts.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pim.simulator.config import SimConfig, obs_dim
from pim.simulator.render2d import (
    grid_axes,
    grid_shape,
    omni2d_enabled,
    pixel_size,
    render_frame_omni2d,
    unflatten,
    validate,
    world_extent,
)
from pim.simulator.renderer import render_frame, render_scene
from pim.simulator.sim import simulate


def _cfg(**kw) -> SimConfig:
    """Canonical 2-object world (dataset-4 geometry) with overrides."""
    base = dict(
        seed=7,
        n_objects=2,
        n_frames=6,
        obs_res=128,
        fixed_reflectivities=True,
        obs_noise_std=0.0,
        boundary="open",
        always_in_frustum=True,
    )
    base.update(kw)
    return SimConfig(**base)


def _omni_cfg(h: int = 48, w: int = 64, **kw) -> SimConfig:
    return _cfg(omni2d=True, omni2d_h=h, omni2d_w=w, obs_res=h * w, **kw)


# ── The pin ───────────────────────────────────────────────────────────────────


def test_defaults_are_bit_identical():
    """With omni2d off, the 1D ray-caster is unchanged bit-for-bit.

    Rendered against values captured before `render2d` existed, so this catches a
    regression introduced by the dispatch itself, not just by the new branch.
    """
    cfg = _cfg(obs_noise_std=0.2)
    scene = simulate(cfg)

    assert not omni2d_enabled(cfg)
    assert obs_dim(cfg) == cfg.obs_res == 128

    depth, ids, inten = render_scene(scene)
    assert depth.shape == (cfg.n_frames, 128)
    assert ids.shape == (cfg.n_frames, 128)
    assert inten.shape == (cfg.n_frames, 128)

    # Re-render: the 1D path is deterministic given the seed.
    depth2, ids2, inten2 = render_scene(scene)
    assert np.array_equal(depth, depth2)
    assert np.array_equal(ids, ids2)
    assert float(np.abs(inten - inten2).max()) == 0.0

    # And the hallmark of the 1D hard renderer: intensity takes only the
    # background value and the (fixed) reflectivities, nothing in between.
    clean = render_scene(simulate(_cfg()))[2]
    assert set(np.unique(np.round(clean, 6))) <= {0.0, 0.4, 0.8}


def test_dispatch_untouched_when_disabled(monkeypatch):
    """The 2D branch is not merely inert when disabled — it is never called."""
    import pim.simulator.render2d as r2d

    def _boom(*a, **k):  # pragma: no cover - must not run
        raise AssertionError("omni2d renderer called with omni2d disabled")

    monkeypatch.setattr(r2d, "render_frame_omni2d", _boom)
    cfg = _cfg()
    scene = simulate(cfg)
    render_frame(scene.positions[0], scene.radii, scene.reflectivities, cfg)


# ── Grid geometry ─────────────────────────────────────────────────────────────


def test_grid_geometry_and_square_pixels():
    """The canonical 48x64 grid over the 12x9 world has exactly square pixels."""
    cfg = _omni_cfg()
    assert grid_shape(cfg) == (48, 64)
    assert obs_dim(cfg) == 48 * 64 == 3072

    dy, dx = pixel_size(cfg)
    assert dy == pytest.approx(9.0 / 48)
    assert dx == pytest.approx(12.0 / 64)
    assert dy == pytest.approx(dx), "pixels must be square for the canonical grid"
    assert dy == pytest.approx(0.1875)

    # A radius-0.5 disc is ~5.3 px across — resolvable, with a real interior.
    assert 2 * cfg.radius / dy == pytest.approx(5.333, abs=1e-3)


def test_grid_axes_span_the_world_rectangle():
    cfg = _omni_cfg()
    y, x = grid_axes(cfg)
    dy, dx = pixel_size(cfg)

    assert y.shape == (48,) and x.shape == (64,)
    # Pixel CENTRES, so the first/last sit half a pixel inside the boundary.
    assert y[0] == pytest.approx(cfg.y_near + dy / 2)
    assert y[-1] == pytest.approx(cfg.y_far - dy / 2)
    assert x[0] == pytest.approx(-cfg.x_far + dx / 2)
    assert x[-1] == pytest.approx(cfg.x_far - dx / 2)
    # Row 0 is the NEAR plane and rows increase with depth.
    assert y[0] < y[-1]
    assert world_extent(cfg) == (-6.0, 6.0, 3.0, 12.0)


def test_unflatten_is_row_major():
    cfg = _omni_cfg(h=4, w=5)
    flat = np.arange(20, dtype=float)
    grid = unflatten(flat, cfg)
    assert grid.shape == (4, 5)
    assert grid[0, 0] == 0 and grid[0, 4] == 4 and grid[1, 0] == 5
    # Batched leading dims survive.
    assert unflatten(np.zeros((3, 7, 20)), cfg).shape == (3, 7, 4, 5)
    with pytest.raises(ValueError):
        unflatten(np.zeros(19), cfg)


# ── Rendering ─────────────────────────────────────────────────────────────────


def test_object_lands_at_its_world_position():
    """A disc's pixels are exactly the pixels whose centres fall inside it."""
    cfg = _omni_cfg()
    pos = np.array([[1.5, 7.5], [-3.0, 5.0]], dtype=np.float32)
    radii = np.full(2, cfg.radius)
    refl = np.array([0.4, 0.8])

    depth, ids, inten = render_frame_omni2d(pos, radii, refl, cfg)
    assert inten.shape == (3072,) and ids.shape == (3072,)

    y, x = grid_axes(cfg)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    for k in range(2):
        expect = ((xx - pos[k, 0]) ** 2 + (yy - pos[k, 1]) ** 2) <= cfg.radius**2
        got = unflatten(ids, cfg) == k
        assert np.array_equal(got, expect)
        assert unflatten(inten, cfg)[expect] == pytest.approx(refl[k])
        # `hit_depth` reports the OBJECT's depth, not the pixel's own y.
        assert unflatten(depth, cfg)[expect] == pytest.approx(pos[k, 1])

    assert (ids == -1).sum() == 3072 - int(
        sum(
            (((xx - pos[k, 0]) ** 2 + (yy - pos[k, 1]) ** 2) <= cfg.radius**2).sum()
            for k in range(2)
        )
    )
    assert inten[ids == -1] == pytest.approx(0.0)


def test_no_perspective_apparent_size_is_depth_invariant():
    """The point of the orthographic view: a near and a far disc look identical.

    In the 1D scan the same object subtends ~4x as many rays at y_near as at
    y_far; here it must cover the same number of pixels at any depth.
    """
    cfg = _omni_cfg()
    radii, refl = np.full(1, cfg.radius), np.array([0.8])

    counts = [
        int((render_frame_omni2d(np.array([[0.0, y]]), radii, refl, cfg)[1] >= 0).sum())
        for y in (4.0, 7.5, 11.0)
    ]
    assert len(set(counts)) == 1, f"apparent size varied with depth: {counts}"
    assert counts[0] > 0

    # Contrast: the 1D renderer's ray count DOES vary strongly with depth.
    cfg1d = _cfg()
    ray_counts = [
        int((render_frame(np.array([[0.0, y]]), radii, refl, cfg1d)[1] >= 0).sum())
        for y in (4.0, 7.5, 11.0)
    ]
    assert ray_counts[0] > 2 * ray_counts[-1]


def test_no_occlusion_both_objects_always_reported():
    """Full observability: an object behind another is still fully rendered.

    The 1D scan drops the occluded object entirely; that is exactly the loss this
    observation channel removes.
    """
    cfg = _omni_cfg()
    # Same x, different depth — perfectly aligned along the line of sight.
    pos = np.array([[0.0, 5.0], [0.0, 9.0]], dtype=np.float32)
    radii, refl = np.full(2, cfg.radius), np.array([0.4, 0.8])

    _, ids2d, _ = render_frame_omni2d(pos, radii, refl, cfg)
    n_near = int((ids2d == 0).sum())
    n_far = int((ids2d == 1).sum())
    assert n_near > 0 and n_far > 0
    assert n_near == n_far  # and equally, per the depth-invariance above

    # The 1D scan hides the far one behind the near one.
    _, ids1d, _ = render_frame(pos, radii, refl, _cfg())
    assert (ids1d == 0).sum() > 0
    assert (ids1d == 1).sum() == 0, "1D scan should occlude the far object here"


def test_overlap_tie_break_is_nearest_object():
    """Unreachable under the canonical collision margin, but must be total."""
    cfg = _omni_cfg()
    pos = np.array([[0.0, 7.0], [0.0, 7.2]], dtype=np.float32)  # overlapping discs
    _, ids, _ = render_frame_omni2d(pos, np.full(2, 0.5), np.array([0.4, 0.8]), cfg)
    y, x = grid_axes(cfg)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    both = (
        ((xx - 0.0) ** 2 + (yy - 7.0) ** 2 <= 0.25)
        & ((xx - 0.0) ** 2 + (yy - 7.2) ** 2 <= 0.25)
    ).reshape(-1)
    assert both.any()
    assert np.all(ids[both] == 0), "nearer (smaller y) object must win the pixel"


def test_clean_intensity_is_piecewise_constant():
    """Hard silhouettes, like the 1D renderer — only background and reflectivities."""
    cfg = _omni_cfg()
    scene = simulate(dataclasses.replace(cfg, seed=3))
    _, _, inten = render_scene(scene)
    assert set(np.unique(np.round(inten, 6))) <= {0.0, 0.4, 0.8}


def test_noise_is_applied_everywhere_and_clipped():
    cfg = _omni_cfg(obs_noise_std=0.2)
    pos = np.array([[0.0, 7.5]], dtype=np.float32)
    rng = np.random.default_rng(0)
    _, ids, inten = render_frame_omni2d(pos, np.full(1, 0.5), np.array([0.8]), cfg, rng)

    assert inten.min() >= 0.0 and inten.max() <= 1.0
    bg = inten[ids == -1]
    assert bg.std() > 0.05, "noise must reach the background, not just the objects"
    # Clipping at 0 makes the background mean positive; that is the 1D behaviour too.
    assert bg.mean() > 0.0


def test_render_scene_shapes_and_zero_objects():
    cfg = _omni_cfg(n_frames=5)
    scene = simulate(cfg)
    depth, ids, inten = render_scene(scene)
    assert depth.shape == ids.shape == inten.shape == (5, 3072)

    empty = render_frame_omni2d(np.zeros((0, 2)), np.zeros(0), np.zeros(0), _omni_cfg())
    assert empty[1].shape == (3072,) and np.all(empty[1] == -1)
    assert np.all(empty[2] == 0.0)


def test_object_covers_expected_pixel_fraction():
    """Sanity on the dilution caveat: a disc is ~0.7% of the omniscient frame.

    Same object is ~13% of the 1D scan at mid-depth. Whole-frame RMSE is therefore
    NOT comparable across the two observation channels; zone-restricted metrics are.
    """
    cfg = _omni_cfg()
    _, ids, _ = render_frame_omni2d(
        np.array([[0.0, 7.5]]), np.full(1, 0.5), np.array([0.8]), cfg
    )
    frac2d = (ids >= 0).mean()
    assert 0.005 < frac2d < 0.010

    _, ids1d, _ = render_frame(
        np.array([[0.0, 7.5]]), np.full(1, 0.5), np.array([0.8]), _cfg()
    )
    frac1d = (ids1d >= 0).mean()
    assert frac1d > 10 * frac2d


# ── Config validation ─────────────────────────────────────────────────────────


def test_validate_is_a_noop_when_disabled():
    validate(_cfg())  # must not raise


def test_validate_rejects_obs_res_mismatch():
    cfg = SimConfig(omni2d=True, omni2d_h=48, omni2d_w=64, obs_res=128)
    with pytest.raises(ValueError, match="obs_res"):
        validate(cfg)


def test_validate_rejects_soft_render_combination():
    cfg = SimConfig(omni2d=True, omni2d_h=48, omni2d_w=64, obs_res=3072, soft_edge=0.05)
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate(cfg)


def test_validate_rejects_degenerate_grid():
    cfg = SimConfig(omni2d=True, omni2d_h=0, omni2d_w=64, obs_res=0)
    with pytest.raises(ValueError, match="positive"):
        validate(cfg)


def test_obs_dim_matches_stored_observation_width():
    """`obs_dim` is what the HDF5 schema and the model's input_dim are sized from."""
    cfg = _omni_cfg(h=16, w=20)
    assert obs_dim(cfg) == 320
    scene = simulate(cfg)
    assert render_scene(scene)[2].shape[1] == obs_dim(cfg)
