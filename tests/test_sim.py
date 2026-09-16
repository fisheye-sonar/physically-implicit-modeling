import numpy as np
import pytest
from pim.environments.discworld.config import SimConfig
from pim.environments.discworld.sim import simulate, frustum_half_width


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


def test_pair_separation_rigid_pair():
    """dw-pair (2026-09-15): two objects at a fixed centre distance, same velocity, inside the frustum."""
    import numpy as np
    from pim.environments.discworld.config import SimConfig
    from pim.environments.discworld.sim import fully_in_frustum, simulate
    cfg = SimConfig(seed=3, n_objects=2, n_frames=40, boundary="open", always_in_frustum=True,
                    radius=0.5, pair_separation=2.0, position_noise_std=0.0, obs_noise_std=0.0)
    sc = simulate(cfg)
    d = np.linalg.norm(sc.positions[:, 0] - sc.positions[:, 1], axis=-1)
    assert np.allclose(d, 2.0, atol=1e-9)
    assert np.allclose(sc.velocities[:, 0], sc.velocities[:, 1])
    assert fully_in_frustum(sc.positions, cfg.radius, cfg)
    for bad in (dict(n_objects=3), dict(boundary="bounce"), dict(pair_separation=1.0)):
        try:
            simulate(SimConfig(**{**dict(seed=3, n_objects=2, boundary="open", radius=0.5, pair_separation=2.0), **bad}))
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")
