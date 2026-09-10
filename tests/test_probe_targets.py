"""The 2026-09-09 probe targets: the discworld APPEARANCE partition (cells = runs of lit
rays, the observation-exact partition), the Othello signed mine/theirs REGRESSION target,
and the token bench's categorical branch."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from pim.environments.discworld.grid_target import (
    AppearanceTarget, GridTarget, categorical_target, covered_rays)

SIM8 = {"radius": 1.0, "y_near": 3.0, "y_far": 12.0, "x_far": 6.0, "obs_res": 10,
        "drop_edge_rays": True}


def _reachable(sim, n, seed=0):
    rng = np.random.default_rng(seed)
    r, scale = sim["radius"], sim["x_far"] / sim["y_far"]
    y = rng.uniform(sim["y_near"] + r, sim["y_far"] - r, size=n)
    x = rng.uniform(-1, 1, size=n) * (scale * y - r)
    return np.stack([x, y], -1).astype(np.float32)


def test_target_names_resolve():
    assert isinstance(categorical_target("grid-16x8"), GridTarget)
    assert categorical_target("appearance") == AppearanceTarget(1)
    assert categorical_target("appearance-d3") == AppearanceTarget(3)
    assert categorical_target("appearance-lat") == AppearanceTarget(lateral_only=True)
    assert categorical_target("appearance").name == "appearance"
    assert categorical_target("appearance-d2").name == "appearance-d2"
    assert categorical_target("appearance-lat").name == "appearance-lat"
    for bad in ("full", "pos", "appearance-x", "grid"):
        assert categorical_target(bad) is None


def test_covered_rays_matches_the_renderer():
    import h5py

    from pim.environments.discworld.renderer import render_frame
    from pim.metrics.zone_editability import object_constants, sim_config_from

    from pim.environments import layout

    p = layout.edits_file("discworld", "dw-8ray")
    if not p.exists():
        pytest.skip("dw-8ray edit bench not present")
    with h5py.File(p) as f:                    # the instance's own renderer config
        sim = json.loads(f.attrs["config_json"])["dataset"]["sim"]
    cfg = sim_config_from(sim, 2)
    rad, refl = object_constants(sim, 2)
    P = _reachable(sim, 3000)
    hit = covered_rays(P, sim)
    ref = np.stack([render_frame(P[i:i + 1], rad[:1], refl[:1], cfg)[2] > 0 for i in range(len(P))])
    assert hit.shape == (3000, 8) and np.array_equal(hit, ref)


def test_appearance_partition_on_dw8ray_has_30_contiguous_runs():
    A = AppearanceTarget()
    runs = A.runs(SIM8)
    assert len(runs) == 30 and A.n_cells(SIM8) == 30
    assert all(0 <= f <= la < 8 for f, la in runs)
    assert {la - f + 1 for f, la in runs} == {1, 2, 3, 4, 5}      # far discs light 1 ray, near 5
    P = _reachable(SIM8, 5000)
    c = A.cell_of(P, SIM8)
    assert c.min() >= 0 and c.max() < 30
    # the cell IS the run: same cell ⇔ same lit rays
    hit = covered_rays(P, SIM8)
    codes = {}
    for ci, h in zip(c, map(tuple, hit)):
        codes.setdefault(int(ci), set()).add(h)
    assert all(len(v) == 1 for v in codes.values())
    assert len({next(iter(v)) for v in codes.values()}) == len(codes)


def test_appearance_variants_refine_or_merge_the_runs():
    A, D2, LAT = AppearanceTarget(), AppearanceTarget(2), AppearanceTarget(lateral_only=True)
    P = _reachable(SIM8, 4000)
    c, c2, cl = A.cell_of(P, SIM8), D2.cell_of(P, SIM8), LAT.cell_of(P, SIM8)
    assert D2.n_cells(SIM8) == 60 and (c2 // 2 == c).all()          # d2 refines each run
    assert LAT.n_cells(SIM8) == 15                                   # 15 distinct centres
    runs = A.runs(SIM8)
    cen = np.array([f + la for f, la in runs])
    assert all(len(set(cl[c == i])) == 1 for i in range(30))         # lat merges by centre
    assert len(set(cen)) == 15


def test_appearance_labels_and_edit_cells():
    A = AppearanceTarget()
    pos = np.stack([_reachable(SIM8, 200, 1), _reachable(SIM8, 200, 2)], 1)   # (200, N_OBJ, 2)
    lab, conflicts = A.label_frames(pos, SIM8)
    assert lab.shape == (200, 30) and lab.max() <= 2
    cells = A.cell_of(pos, SIM8)
    assert conflicts == int((cells[:, 0] == cells[:, 1]).sum())
    seq = np.repeat(pos[:, None], 3, axis=1)                          # (200, 3, N_OBJ, 2)
    seq[:, 2, 0] = _reachable(SIM8, 200, 3)                          # object 0 teleports at frame 2
    mv = A.edit_cells(seq, np.zeros(200, int), ef=2, sim=SIM8)
    assert (mv["A"] == cells[:, 0]).all() and (mv["cls"] == 1).all()


# ── Othello: the signed mine/theirs regression target ─────────────────────────


def test_signed_mine_target():
    from pim.environments.othello.data import BLANK, MINE, THEIRS, signed_mine

    m = np.array([[BLANK, MINE, THEIRS, MINE]], np.int8)
    assert np.array_equal(signed_mine(m), np.array([[0.0, 1.0, -1.0, 1.0]], np.float32))
    assert signed_mine(m).dtype == np.float32


def test_othello_regression_arms_run_on_a_tiny_model():
    """PI / ND / GS through 64-output REGRESSION probes on a few shipped cases."""
    from pim.environments.othello import arms as oa
    from pim.environments.othello.bench import benchmark_from_cases, BENCHMARK_PKL, case_targets
    from pim.models import build
    from pim.probes.base import WorldStateProbe
    import pickle

    if not BENCHMARK_PKL.exists():
        pytest.skip("Li et al.'s benchmark pkl not vendored here")
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build("transformer_l_tokens", {"vocab": 61, "block_size": 59, "n_layer": 2,
                                           "n_head": 2, "n_embd": 32}).to(dev).eval()
    cases = pickle.load(open(BENCHMARK_PKL, "rb"))[:12]
    bench = benchmark_from_cases(cases)
    cur, tgt = case_targets(bench)
    lin = {e: WorldStateProbe(32, 64, None).to(dev).eval() for e in range(3)}
    mlp = {e: WorldStateProbe(32, 64, 16).to(dev).eval() for e in range(3)}
    for m in (*lin.values(), *mlp.values()):
        for p_ in m.parameters():
            p_.requires_grad_(False)
    for mode in ("add_sub", "pinv"):
        probs, card = oa.linear_arm(model, bench, lin, tgt, cur, mode=mode, alpha=0.5, points={1})
        assert probs.shape == (12, 64) and "edit_index_union" in card
    probs, card = oa.grad_steer_arm(model, bench, mlp, 1, alpha=0.05, n_steps=2, beta=0.2,
                                    target_labels=tgt)
    assert probs.shape == (12, 64) and np.isfinite(card["edit_index_union"])


# ── the token bench's categorical branch (data-dependent) ─────────────────────


def test_token_bench_categorical_branch():
    from pim.environments.discworld import token_bench as tkb
    from pim.environments.discworld.tokens import FrameVocab
    from pim.models import build

    from pim.environments import layout

    vocab_p = layout.tokens_dir("dw-8ray") / "vocab.npz"
    if not layout.edits_file("discworld", "dw-8ray").exists() or not vocab_p.exists():
        pytest.skip("dw-8ray tokens not present")
    vocab = FrameVocab.load(vocab_p)
    tb = tkb.load_token_bench(vocab, n=6, target="appearance", basis_name="frustum",
                              instance="dw-8ray")
    assert tb.kind == "classification" and tb.tgt.dtype == torch.long and tb.tgt.shape == (6, 30)
    assert (tb.change_mask.sum(1) == 2).all() and tb.selection["n"] == 6
    torch.manual_seed(0)
    dev = tkb.DEV
    model = build("transformer_l_tokens", {"vocab": int(vocab.size), "block_size": 39, "n_layer": 2,
                                           "n_head": 2, "n_embd": 32}).to(dev).eval()
    from pim.probes.base import WorldStateProbe
    lin = {e: (WorldStateProbe(32, 30, None, n_classes=3).to(dev).eval(), {}) for e in range(3)}
    mlp = {e: (WorldStateProbe(32, 30, 16, n_classes=3).to(dev).eval(), {}) for e in range(3)}
    for pr, _ in (*lin.values(), *mlp.values()):
        for p_ in pr.parameters():
            p_.requires_grad_(False)
    uns, u = tkb.unsteered(model, tb)
    recs = tkb.pinv_arm(model, tb, lin, (1.0,), uns) + tkb.nanda_arm(model, tb, lin[1][0], 1, (1.0,), uns) \
        + tkb.grad_steer_arm(model, tb, mlp, [1], (0.05,), uns, n_steps=2)
    assert len(recs) == 5 and all("edit_index" in r for r in recs)
    assert "readout_landed" in recs[0]
    with pytest.raises(ValueError):
        tkb.pinv_arm(model, tb, lin, (1.0,), uns, dims="pos")


def test_appearance_cell_of_is_chunk_invariant(monkeypatch):
    """Labelling the probe corpus is chunked (the one-shot ray–disc test on 128 rays OOM-killed
    a unit); every chunk size must give the same cells, on every variant and any leading shape."""
    P = _reachable(SIM8, 3000).reshape(500, 3, 2, 2)
    for T in (AppearanceTarget(), AppearanceTarget(3), AppearanceTarget(lateral_only=True)):
        ref = T.cell_of(P, SIM8)
        assert ref.shape == (500, 3, 2)
        monkeypatch.setattr(AppearanceTarget, "CHUNK", 7)
        assert (T.cell_of(P, SIM8) == ref).all()
        monkeypatch.setattr(AppearanceTarget, "CHUNK", 1 << 18)


def test_unseen_run_snaps_to_the_nearest_realisable_run():
    from pim.environments.discworld.grid_target import _nearest_run
    runs = ((3, 4), (2, 5), (7, 8), (1, 6))
    assert _nearest_run(runs, 3, 5) == 0        # centre 8: (3,4) and (2,5) tie on centre and L1 → the earlier run
    assert _nearest_run(runs, 7, 9) == 2        # one extra grazing ray → the run it grazes
    assert _nearest_run(runs, 0, 7) == 3


# ── the SNAPPED regression target (2026-09-10) ────────────────────────────────────


def test_snapped_target_names_resolve():
    from pim.environments.discworld.grid_target import (
        SnappedTarget, selection_target, snapped_target, target_cells)

    sn = snapped_target("pos@appearance")
    assert isinstance(sn, SnappedTarget) and sn.base == "pos" and sn.cat == AppearanceTarget(1)
    assert sn.name == "pos@appearance"
    assert snapped_target("full@grid-16x8") == SnappedTarget("full", GridTarget(16, 8))
    for bad in ("pos", "full", "appearance", "pos@", "vel@appearance", "pos@grid", "pos@nothing"):
        assert snapped_target(bad) is None
    # a snapped target is NOT categorical — the pipeline takes the regression branch …
    assert categorical_target("pos@appearance") is None
    # … but its partition defines what a genuine edit is
    assert selection_target("pos@appearance") == AppearanceTarget(1)
    assert selection_target("grid-4x2") == GridTarget(4, 2)
    assert selection_target("pos") is None
    assert target_cells("pos@appearance", SIM8) == 30 and target_cells("full", SIM8) is None


def test_cell_centres_lie_in_their_own_cell_and_snap_is_constant_per_cell():
    from pim.environments.discworld.frustum import basis
    from pim.environments.discworld.grid_target import frustum_to_world, snapped_target

    P = _reachable(SIM8, 4000)
    # the basis inversion the centres rely on
    F = basis(P, None, SIM8)[0]
    assert np.allclose(frustum_to_world(F, SIM8), P, atol=1e-4)
    for name in ("appearance", "appearance-lat", "appearance-d2", "grid-4x2", "grid-16x8"):
        cat = categorical_target(name)
        C = cat.centroids(SIM8)
        assert C.shape == (cat.n_cells(SIM8), 2)
        assert np.array_equal(cat.cell_of(frustum_to_world(C, SIM8), SIM8), np.arange(len(C)))
        assert np.array_equal(cat.centroids(SIM8), C)           # cached, deterministic
    sn = snapped_target("pos@appearance")
    S = sn.snap(P, SIM8)
    cells = sn.cat.cell_of(P, SIM8)
    assert S.shape == P.shape
    for c in np.unique(cells):                 # one value per cell, the cell's centre
        assert np.allclose(S[cells == c], sn.cat.centroids(SIM8)[c])
    # the snapped value sits INSIDE the cell of the position it replaces
    assert np.array_equal(sn.cat.cell_of(frustum_to_world(S, SIM8), SIM8), cells)
    # and is close to the true frustum coordinates on average (a 30-cell partition of the region)
    assert np.abs(S - F).mean(0).max() < 0.1 * F.std(0).max() + 0.05


def test_probe_targets_snapped_branch_is_regression_shaped():
    from pim.environments.discworld import arms as dwa

    rng = np.random.default_rng(1)
    pos = _reachable(SIM8, 20 * 7).reshape(20, 7, 1, 2)
    pos = np.concatenate([pos, _reachable(SIM8, 20 * 7, seed=2).reshape(20, 7, 1, 2)], 2)
    vel = rng.normal(size=pos.shape).astype(np.float32) * 0.01
    y, nc = dwa._targets("pos@appearance", pos, vel, SIM8, "frustum")
    assert nc is None and y.shape == (20, 7, 4) and y.dtype == np.float32
    yf, _ = dwa._targets("full@appearance", pos, vel, SIM8, "frustum")
    assert yf.shape == (20, 7, 8)
    yr, _ = dwa._targets("full", pos, vel, SIM8, "frustum")
    assert np.allclose(yf[..., 4:], yr[..., 4:])            # velocities untouched by the snap
    C = categorical_target("appearance").centroids(SIM8)
    assert all(any(np.allclose(v, c, atol=1e-6) for c in C) for v in y[..., :2].reshape(-1, 2)[:50])
    with pytest.raises(ValueError):
        dwa._targets("pos@appearance", pos, vel, SIM8, "cartesian")
    assert dwa.probe_recipe("pos@appearance", "dw-8ray")["n_seq"] == 30_000     # the regression recipe


def test_bench_arrays_snapped_branch_on_dw8ray():
    """Data-dependent: the snapped bench is the regression bench on the appearance
    partition's filtered cases, with the edit asking for the new cell's centre."""
    from pim.environments import layout
    from pim.environments.discworld import bench as dwb
    from pim.environments.discworld.frustum import basis
    from pim.environments.discworld.grid_target import snapped_target

    if not layout.edits_file("discworld", "dw-8ray").exists():
        pytest.skip("dw-8ray edit bench not present")
    a = dwb.bench_arrays(n=12, target="pos@appearance", basis_name="frustum", instance="dw-8ray")
    assert a["kind"] == "regression" and a["cells"] is None
    assert a["y"].shape == (12, 4) and a["y"].dtype == np.float32
    assert (a["change_mask"].sum(1) == 2).all() and a["out_dims"] == [0, 1, 2, 3]
    assert a["selection"]["n"] == 12 and "appearance" in a["selection"]["rule"]
    sn = snapped_target("pos@appearance")
    sim = a["sim"]
    # y IS the snapped post-edit state …
    assert np.allclose(a["y"], sn.snap(a["pos"][:, dwb.EF], sim).reshape(12, -1), atol=1e-6)
    # … every case asks the edited object to move to a DIFFERENT cell centre
    pre = sn.snap(a["pos"][:, dwb.EF - 1], sim).reshape(12, -1)
    moved = np.abs(a["y"] - pre).reshape(12, 2, 2)[np.arange(12), a["edit_object"]]
    assert (moved.max(-1) > 1e-6).all()
    # … and differs from the unsnapped target while the zones (world space) are the same
    c = dwb.bench_arrays(n=12, target="pos", basis_name="frustum", instance="dw-8ray",
                         select=np.arange(12))
    assert not np.allclose(a["y"], basis(a["pos"][:, dwb.EF], None, sim)[0].reshape(12, -1))
    assert np.array_equal(a["zones"].target, c["zones"].target)
    with pytest.raises(ValueError):
        dwb.bench_arrays(n=4, target="pos@appearance", basis_name="cartesian", instance="dw-8ray")
