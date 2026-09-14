"""The appearance partition and its factorisation on a MULTI-OBSERVER instance (2026-09-13).

Pins: one observer is unchanged (labels bit-identical to the single-view code path); with V
observers the factors are (centre, length) per view, view-major, each with an extra "unseen"
class; view k's factors equal the single-view factors of the positions transformed into
observer k's frame; a disc outside a view's fan gets that view's unseen classes; edit moves
carry 2·V tiles per case; the cell-indexed target refuses.
"""
from __future__ import annotations

import numpy as np
import pytest

from pim.environments.discworld import observers as ob
from pim.environments.discworld.config import SimConfig
from pim.environments.discworld.grid_target import AppearanceTarget, categorical_target
from pim.environments.discworld.sim import simulate

SIM8 = {"radius": 1.0, "y_near": 3.0, "y_far": 12.0, "x_far": 6.0, "obs_res": 10, "drop_edge_rays": True}
SIM5 = {**SIM8, "n_observers": 5, "region": "circle"}
CFG5 = SimConfig(n_objects=2, n_frames=40, obs_res=10, drop_edge_rays=True, radius=1.0, boundary="open",
                 obs_noise_std=0.0, fixed_reflectivities=True, always_in_frustum=True, region="circle",
                 n_observers=5, max_gen_attempts=5000, seed=11)
T = categorical_target("appearance-fac")
A = AppearanceTarget(1)


def _positions(n_scenes=6):
    return np.concatenate([simulate(SimConfig(**{**CFG5.__dict__, "seed": s})).positions for s in range(n_scenes)])  # (F, 2, 2)


def test_single_view_path_is_unchanged():
    pos = _positions()
    inside = np.array([ob.inside_region(p[None, None], 1.0, CFG5) for p in pos.reshape(-1, 2)])
    p1 = pos.reshape(-1, 2)[inside]
    # the single-view code must not know about views at all: same labels with and without the key
    assert np.array_equal(T.factor_labels(p1[:, None], SIM8), T.factor_labels(p1[:, None], {**SIM8, "n_observers": 1}))
    assert T.cat.factor_sizes(SIM8) == (15, 5) and T.n_tiles_on(SIM8) == 4 and T.n_factors_on(SIM8) == 2


def test_shapes_and_sizes_with_five_views():
    assert T.cat.factor_sizes(SIM5) == (16, 6) * 5
    assert T.n_factors_on(SIM5) == 10 and T.n_tiles_on(SIM5) == 20 and T.n_classes_on(SIM5) == 110
    assert A.n_cells(SIM5) == 31          # per view (30 runs + unseen); the joint product is never enumerated
    pos = _positions(2)
    lab = T.factor_labels(pos, SIM5)
    assert lab.shape == (pos.shape[0], 2 * 10) and lab.min() >= 0 and lab.max() < 110


def test_view_k_factors_are_single_view_factors_in_that_frame():
    pos = _positions()                                            # (F, 2, 2) world
    poses = ob.observer_poses(CFG5)
    fac5 = T.cat.factors_of(A.cell_of(pos, SIM5), SIM5)          # (F, 2, 10)
    for k, pose in enumerate(poses):
        q = ob.to_observer_frame(pos, pose)
        seen = A.cell_of.__self__ is A and A._view_cells(pos, SIM5)[..., k] < 30
        f1 = np.full(fac5.shape[:-1] + (2,), -1)
        f1[seen] = T.cat.factors_of(A.cell_of(q[seen], SIM8), SIM8)
        assert np.array_equal(fac5[..., 2 * k: 2 * k + 2][seen], f1[seen])
        assert (fac5[..., 2 * k][~seen] == 15).all() and (fac5[..., 2 * k + 1][~seen] == 5).all()
    # geometry fact (scanned 2026-09-13): every arena position lights >= 1 kept ray in EVERY view,
    # so the "unseen" class is a guard that never fires on dw-8ray-obs5 (invisibility = occlusion)
    assert (A._view_cells(pos, SIM5) < 30).all()


def test_edit_moves_and_refusals():
    sc = simulate(CFG5)
    pos = sc.positions[None].repeat(3, 0)
    mv = T.edit_moves(pos, np.array([0, 1, 0]), 20, SIM5)
    assert mv["tile"].shape == (3, 10) and mv["old"].shape == (3, 10)
    with pytest.raises(ValueError):
        A.labels_from_cells(np.zeros(3, int), np.zeros(3), SIM5)
    with pytest.raises(ValueError):
        categorical_target("appearance-d2").cell_of(pos[0], SIM5)
