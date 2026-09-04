"""`drop_edge_rays` (2026-09-03, the 8-ray instance): cast obs_res rays, keep the interior.

The kept rays must be bit-identical to rays 1..obs_res-2 of the full render, and every
array-sizing site must follow `obs_dim` (= obs_res - 2).
"""
import numpy as np

from pim.environments.discworld.config import SimConfig, obs_dim
from pim.environments.discworld.renderer import render_frame, render_scene
from pim.environments.discworld.sim import simulate


def _cfg(**kw):
    return SimConfig(n_objects=2, n_frames=6, obs_res=10, radius=1.0, seed=3,
                     fixed_reflectivities=True, always_in_frustum=True, **kw)


def test_obs_dim_drops_two():
    assert obs_dim(_cfg()) == 10
    assert obs_dim(_cfg(drop_edge_rays=True)) == 8


def test_kept_rays_are_the_interior_of_the_full_render():
    pos = np.array([[0.5, 5.0], [-1.0, 9.0]], np.float32)
    rad = np.full(2, 1.0, np.float32)
    refl = np.array([0.4, 0.8], np.float32)
    d0, i0, o0 = render_frame(pos, rad, refl, _cfg())
    d1, i1, o1 = render_frame(pos, rad, refl, _cfg(drop_edge_rays=True))
    assert o0.shape == (10,) and o1.shape == (8,)
    assert np.array_equal(o1, o0[1:-1])
    assert np.array_equal(i1, i0[1:-1])
    assert np.array_equal(d1, d0[1:-1])
    assert (i1 >= 0).any(), "a radius-1 disc at y=5 must light interior rays"


def test_empty_frame_is_also_narrowed():
    d, i, o = render_frame(np.zeros((0, 2), np.float32), np.zeros(0, np.float32),
                           np.zeros(0, np.float32), _cfg(drop_edge_rays=True))
    assert d.shape == i.shape == o.shape == (8,)


def test_render_scene_width_follows_obs_dim():
    cfg = _cfg(drop_edge_rays=True)
    scene = simulate(cfg)
    dep, ids, inten = render_scene(scene)
    assert inten.shape == (cfg.n_frames, 8) and ids.shape == (cfg.n_frames, 8)
    assert dep.shape == (cfg.n_frames, 8)
