"""Multiple observers on a ring + the circular arena (pim/environments/discworld/observers.py).

The load-bearing test is the first: with the defaults (one observer, frustum region) nothing
about the simulator or the renderer may change. The rest pin the geometry the module claims:
observer 0 is the canonical observer; every observer sits at the pivot's distance and looks at
it; every point of the circle lies in every observer's depth band; rotating the world by one
observer step permutes the views; the flat layout is observer-major.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pim.environments.discworld import observers as ob
from pim.environments.discworld.config import SimConfig, obs_dim
from pim.environments.discworld.renderer import render_frame
from pim.environments.discworld.sim import fully_in_frustum, sample_position, simulate

POS = np.array([[-1.0, 6.0], [1.6, 8.5]])
RADII = np.array([0.5, 0.5])
REFL = np.array([0.4, 0.8])


def _cfg(**kw) -> SimConfig:
    return SimConfig(n_objects=2, obs_res=32, obs_noise_std=0.0, boundary="open", seed=3, n_frames=40, **kw)


def test_defaults_are_bit_identical():
    a, b = _cfg(), _cfg(n_observers=1, region="frustum")
    for x, y in zip(render_frame(POS, RADII, REFL, a), render_frame(POS, RADII, REFL, b)):
        assert np.array_equal(x, y)
    sa, sb = simulate(a), simulate(b)
    assert np.array_equal(sa.positions, sb.positions)
    assert obs_dim(a) == 32


def test_observer_zero_is_canonical_and_layout_is_observer_major():
    cfg = _cfg(n_observers=5)
    poses = ob.observer_poses(cfg)
    assert np.allclose(poses[0], [0.0, 0.0, np.pi / 2])
    d, ids, inten = render_frame(POS, RADII, REFL, cfg)
    assert d.shape == (5 * 32,) and obs_dim(cfg) == 160
    d1, ids1, inten1 = render_frame(POS, RADII, REFL, ob.single_view_config(cfg))
    assert np.array_equal(inten[:32], inten1) and np.array_equal(ids[:32], ids1)
    assert np.array_equal(ob.observer_of_ray(cfg), np.repeat(np.arange(5), 32))


def test_ring_geometry():
    cfg = _cfg(n_observers=7)
    c, d = ob.pivot(cfg), 0.5 * (cfg.y_near + cfg.y_far)
    for pose in ob.observer_poses(cfg):
        assert np.isclose(np.linalg.norm(pose[:2] - c), d)                  # on the ring
        look = np.array([np.cos(pose[2]), np.sin(pose[2])])
        assert np.allclose(pose[:2] + d * look, c)                           # looking at the pivot
        # every point of the circle lies in this observer's depth band [y_near, y_far]
        t = np.linspace(0, 2 * np.pi, 360)
        rim = c + ob.region_radius(cfg) * np.stack([np.cos(t), np.sin(t)], -1)
        y = ob.to_observer_frame(rim, pose)[:, 1]
        assert y.min() >= cfg.y_near - 1e-9 and y.max() <= cfg.y_far + 1e-9


def test_rotating_the_world_permutes_the_views():
    cfg = _cfg(n_observers=4)
    c = ob.pivot(cfg)
    a = 2 * np.pi / 4
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rotated = (POS - c) @ R.T + c                     # world rotated by one observer step
    base = render_frame(POS, RADII, REFL, cfg)[2].reshape(4, 32)
    rot = render_frame(rotated, RADII, REFL, cfg)[2].reshape(4, 32)
    assert np.allclose(np.roll(base, 1, axis=0), rot, atol=1e-9)


def test_circle_region_sampling_and_acceptance():
    cfg = _cfg(region="circle", always_in_frustum=True, n_observers=3)
    rng = np.random.default_rng(0)
    pts = np.array([sample_position(rng, cfg, cfg.radius) for _ in range(2000)])
    dist = np.linalg.norm(pts - ob.pivot(cfg), axis=1)
    assert dist.max() <= ob.region_radius(cfg) - cfg.radius + 1e-9
    assert fully_in_frustum(pts[None], cfg.radius, cfg)
    outside = ob.pivot(cfg) + np.array([ob.region_radius(cfg), 0.0])
    assert not fully_in_frustum(outside[None, None], cfg.radius, cfg)
    scene = simulate(cfg)
    assert ob.inside_region(scene.positions, cfg.radius, cfg)
    with pytest.raises(ValueError):
        simulate(dataclasses.replace(cfg, boundary="bounce"))
