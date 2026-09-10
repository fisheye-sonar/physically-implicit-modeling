"""The grid probe target (discworld state as Othello-shaped cells) and the pieces the
canonical pipeline gained with it (2026-09-09): the shared class-logit swap, the
kind-agnostic Probe Skill accessor and tripwire, and the bench's categorical branch."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pim.editors.pinv import swap_class_logits
from pim.environments.discworld.grid_target import CANONICAL, N_CLASSES, GridTarget
from pim.metrics.decodability import insample_gap_from_stats, probe_skill_from_stats
from pim.probes.mlp import check_probe_sanity

SIM = {"radius": 0.5, "y_near": 2.0, "y_far": 20.0, "x_far": 10.0}


def test_name_round_trip_and_canonical():
    g = GridTarget.parse("grid-16x8")
    assert g == CANONICAL and g.name == "grid-16x8" and g.g == 128 and g.n_classes == 3
    assert GridTarget.parse("grid-8x4") == GridTarget(8, 4)
    for bad in ("full", "pos", "grid", "grid16x8", "frustum"):
        assert GridTarget.parse(bad) is None


def test_every_reachable_position_falls_in_a_cell_and_labels_agree_with_cells():
    g = CANONICAL
    rng = np.random.default_rng(0)
    r = SIM["radius"]
    y = rng.uniform(SIM["y_near"] + r, SIM["y_far"] - r, size=(500, 2))
    scale = SIM["x_far"] / SIM["y_far"]
    x = rng.uniform(-1, 1, size=(500, 2)) * (scale * y - r)          # inside the reach
    pos = np.stack([x, y], -1)                                        # (500, N_OBJ, 2)
    cells = g.cell_of(pos, SIM)
    assert cells.min() >= 0 and cells.max() < g.g
    lab, conflicts = g.label_frames(pos, SIM)
    assert lab.shape == (500, g.g) and lab.dtype == np.uint8
    ar = np.arange(500)
    for j in (0, 1):
        own = cells[:, j]
        other = cells[:, 1 - j]
        nearer = y[:, j] <= y[:, 1 - j]
        # where the object has its own cell, or shares one and is nearer, its label wins
        mask = (own != other) | nearer
        assert (lab[ar[mask], own[mask]] == j + 1).all()
    assert conflicts == int((cells[:, 0] == cells[:, 1]).sum())
    # exactly two non-empty cells unless the objects share one
    assert ((lab > 0).sum(1) == np.where(cells[:, 0] == cells[:, 1], 1, 2)).all()


def test_cell_of_frustum_matches_cell_of_world():
    from pim.environments.discworld.frustum import lateral

    g = CANONICAL
    rng = np.random.default_rng(1)
    y = rng.uniform(SIM["y_near"] + 0.5, SIM["y_far"] - 0.5, size=300)
    x = rng.uniform(-1, 1, size=300) * (SIM["x_far"] / SIM["y_far"] * y - 0.5)
    pos = np.stack([x, y], -1)
    assert (g.cell_of_frustum(lateral(pos, SIM), 1.0 / y, SIM) == g.cell_of(pos, SIM)).all()


def test_edit_cells_reads_the_teleport():
    g = CANONICAL
    pos = np.zeros((3, 4, 2, 2))
    pos[..., 1] = 10.0                          # every object at depth 10
    pos[:, 3, 0, 0] = 3.0                       # object 0 teleports laterally at frame 3
    mv = g.edit_cells(pos, np.array([0, 0, 1]), ef=3, sim=SIM)
    assert (mv["cls"] == np.array([1, 1, 2])).all()
    assert (mv["A"][:2] != mv["B"][:2]).all()   # object 0 moved cell
    assert mv["A"][2] == mv["B"][2]             # object 1 did not move


def test_swap_class_logits_is_a_per_sample_two_class_exchange():
    lg = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    tile = torch.tensor([1, 3])
    a, b = torch.tensor([0, 2]), torch.tensor([2, 1])
    out = swap_class_logits(lg, tile, a, b)
    assert out is not lg and torch.equal(lg, torch.arange(24.).reshape(2, 4, 3))  # input untouched
    for i in range(2):
        for t in range(4):
            if t != tile[i]:
                assert torch.equal(out[i, t], lg[i, t])
        assert out[i, tile[i], a[i]] == lg[i, tile[i], b[i]]
        assert out[i, tile[i], b[i]] == lg[i, tile[i], a[i]]
    # swapping twice is the identity
    assert torch.equal(swap_class_logits(out, tile, a, b), lg)


def test_probe_skill_accessor_selects_the_right_kind():
    assert probe_skill_from_stats({"r2": 0.9, "r2_insample": 0.95}) == pytest.approx(0.9)
    assert insample_gap_from_stats({"r2": 0.9, "r2_insample": 0.95}) == pytest.approx(0.05)
    cls = {"error_rate": 10.0, "error_rate_insample": 8.0, "majority_class_error_rate": 40.0}
    assert probe_skill_from_stats(cls) == pytest.approx(0.75)
    assert insample_gap_from_stats(cls) == pytest.approx(0.05)
    assert np.isnan(insample_gap_from_stats({"error_rate": 10.0, "majority_class_error_rate": 40.0}))


def test_tripwire_runs_on_classification_stats():
    lin = {0: (None, {"error_rate": 20.0, "error_rate_insample": 19.0, "majority_class_error_rate": 40.0})}
    good = {0: (None, {"error_rate": 10.0, "error_rate_insample": 9.0, "majority_class_error_rate": 40.0})}
    bad = {0: (None, {"error_rate": 30.0, "error_rate_insample": 5.0, "majority_class_error_rate": 40.0})}
    rep = check_probe_sanity(lin, good, strict=True, log=None)
    assert rep["n_violations"] == 0 and rep["rows"][0]["r2_mlp"] == pytest.approx(0.75)
    with pytest.raises(AssertionError):
        check_probe_sanity(lin, bad, strict=True, log=None)


def test_bench_arrays_grid_branch_shapes():
    """Data-dependent: skipped where the canonical instance is absent."""
    from pim.environments import layout
    from pim.environments.discworld import bench as dwb

    if not layout.edits_file("discworld", "dw-noiseless").exists():
        pytest.skip("dw-noiseless edit bench not present")
    a = dwb.bench_arrays(n=8, target="grid-16x8", basis_name="frustum", instance="dw-noiseless")
    assert a["kind"] == "classification" and a["y"].shape == (8, 128) and a["y"].dtype == np.int64
    assert (a["change_mask"].sum(1) == 2).all()
    ar = np.arange(8)
    assert (a["y"][ar, a["cells"]["A"]] == 0).all()
    assert (a["y"][ar, a["cells"]["B"]] == a["cells"]["cls"]).all()
    assert (a["cells"]["A"] != a["cells"]["B"]).all()             # no same-cell no-ops
    assert a["selection"]["n"] == 8 and a["y"].max() < N_CLASSES
