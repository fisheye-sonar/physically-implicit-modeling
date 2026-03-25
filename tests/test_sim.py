import numpy as np
import pytest
from pim.config import SimConfig
from pim.sim import simulate, frustum_half_width


def test_positions_shape():
    cfg = SimConfig(n_objects=3, n_frames=50, seed=0)
    scene = simulate(cfg)
    assert scene.positions.shape == (50, 3, 2)


def test_objects_inside_frustum():
    cfg = SimConfig(n_objects=3, n_frames=100, seed=0)
    scene = simulate(cfg)
    for f in range(cfg.n_frames):
        for i in range(scene.positions.shape[1]):
            x, y = scene.positions[f, i]
            r = scene.radii[i]
            assert y >= cfg.y_near - 1e-6, f"frame {f}, obj {i}: y too small"
            assert y <= cfg.y_far + 1e-6, f"frame {f}, obj {i}: y too large"
            x_bound = frustum_half_width(y, cfg)
            assert abs(x) <= x_bound + 1e-6, f"frame {f}, obj {i}: x out of frustum"


def test_no_collisions():
    cfg = SimConfig(n_objects=3, n_frames=100, seed=2)
    scene = simulate(cfg)
    min_sep = cfg.collision_margin * 2.0 * cfg.radius
    n = scene.positions.shape[1]
    for f in range(cfg.n_frames):
        for a in range(n):
            for b in range(a + 1, n):
                dist = np.linalg.norm(scene.positions[f, a] - scene.positions[f, b])
                assert dist >= min_sep - 1e-6, f"collision at frame {f}, objects {a} and {b}"


def test_random_n_objects():
    cfg = SimConfig(n_objects=None, n_objects_min=2, n_objects_max=3, seed=99)
    scene = simulate(cfg)
    assert 2 <= scene.positions.shape[1] <= 3


def test_colors_shape():
    cfg = SimConfig(n_objects=3, seed=0)
    scene = simulate(cfg)
    assert scene.colors.shape == (3, 3)


def test_deterministic_with_seed():
    cfg = SimConfig(seed=42, n_objects=3)
    s1 = simulate(cfg)
    s2 = simulate(cfg)
    np.testing.assert_array_equal(s1.positions, s2.positions)
