import numpy as np
import pytest
from pim.config import SimConfig
from pim.renderer import render_frame, render_scene
from pim.sim import simulate


def _scene(seed=0, n_objects=3, obs_res=64, noise=False):
    cfg = SimConfig(
        n_objects=n_objects, obs_res=obs_res, seed=seed,
        obs_noise_std=0.04 if noise else 0.0,
    )
    return simulate(cfg), cfg


def test_render_frame_shapes():
    scene, cfg = _scene()
    depth, ids, intensity = render_frame(
        scene.positions[0], scene.radii, scene.reflectivities, cfg
    )
    assert depth.shape == (cfg.obs_res,)
    assert ids.shape == (cfg.obs_res,)
    assert intensity.shape == (cfg.obs_res,)


def test_hit_ids_in_range():
    scene, cfg = _scene(n_objects=3)
    _, ids, _ = render_frame(scene.positions[0], scene.radii, scene.reflectivities, cfg)
    valid = ids[ids >= 0]
    assert np.all(valid < cfg.n_objects)
    assert np.all(ids >= -1)


def test_hit_depth_in_scene_range():
    scene, cfg = _scene(n_objects=3, noise=False)
    depth, ids, _ = render_frame(
        scene.positions[0], scene.radii, scene.reflectivities, cfg
    )
    hit_depths = depth[ids >= 0]
    assert np.all(hit_depths >= cfg.y_near - cfg.radius - 1e-6)
    assert np.all(hit_depths <= cfg.y_far + cfg.radius + 1e-6)


def test_no_hit_depth_is_zero():
    scene, cfg = _scene()
    depth, ids, _ = render_frame(
        scene.positions[0], scene.radii, scene.reflectivities, cfg
    )
    assert np.all(depth[ids == -1] == 0.0)


def test_no_hit_intensity_is_zero():
    scene, cfg = _scene(noise=False)
    _, ids, intensity = render_frame(
        scene.positions[0], scene.radii, scene.reflectivities, cfg
    )
    assert np.all(intensity[ids == -1] == 0.0)


def test_hit_intensity_matches_reflectivity():
    """Noise-free: each hit ray's intensity should equal the object's reflectivity."""
    scene, cfg = _scene(noise=False)
    _, ids, intensity = render_frame(
        scene.positions[0], scene.radii, scene.reflectivities, cfg
    )
    hit_mask = ids >= 0
    np.testing.assert_allclose(
        intensity[hit_mask], scene.reflectivities[ids[hit_mask]], atol=1e-6
    )


def test_intensity_in_range():
    """With noise, intensity should stay clipped to [0, 1]."""
    scene, cfg = _scene(noise=True)
    rng = np.random.default_rng(0)
    _, _, intensity = render_frame(
        scene.positions[0], scene.radii, scene.reflectivities, cfg, rng=rng
    )
    assert np.all(intensity >= 0.0)
    assert np.all(intensity <= 1.0)


def test_render_scene_shapes():
    cfg2 = SimConfig(n_objects=2, obs_res=64, seed=0, n_frames=20)
    scene2 = simulate(cfg2)
    depth, ids, intensity = render_scene(scene2)
    assert depth.shape == (20, 64)
    assert ids.shape == (20, 64)
    assert intensity.shape == (20, 64)


def test_closer_object_appears_wider():
    """An object closer to the observer should subtend more scan rays."""
    cfg_near = SimConfig(n_objects=1, obs_res=256, seed=0, obs_noise_std=0.0)
    cfg_far = SimConfig(n_objects=1, obs_res=256, seed=0, obs_noise_std=0.0)

    scene_near = simulate(cfg_near)
    scene_far = simulate(cfg_far)

    r = cfg_near.radius
    pos_near = np.array([[0.0, cfg_near.y_near + r + 0.1]])
    pos_far = np.array([[0.0, cfg_near.y_far - r - 0.1]])
    refl = np.array([0.5])

    _, ids_near, _ = render_frame(pos_near, np.array([r]), refl, cfg_near)
    _, ids_far, _ = render_frame(pos_far, np.array([r]), refl, cfg_far)

    hits_near = np.sum(ids_near >= 0)
    hits_far = np.sum(ids_far >= 0)
    assert hits_near > hits_far, (
        f"Near object should hit more rays: near={hits_near}, far={hits_far}"
    )
