"""The 2026-09-09 probe targets: the discworld APPEARANCE partition (cells = runs of lit
rays, the observation-exact partition), the Othello signed mine/theirs REGRESSION target,
and the token bench's categorical branch."""
from __future__ import annotations

import json
from pathlib import Path

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

    p = Path("datasets/discworld/dw-8ray/eval/edits.h5")
    if not p.exists():
        pytest.skip("dw-8ray eval split not present")
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
    assert all(0 <= f <= l < 8 for f, l in runs)
    assert {l - f + 1 for f, l in runs} == {1, 2, 3, 4, 5}      # far discs light 1 ray, near 5
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
    cen = np.array([f + l for f, l in runs])
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

    root = Path("datasets/discworld/dw-8ray")
    if not (root / "eval/edits.h5").exists() or not (root / "tokens/vocab.npz").exists():
        pytest.skip("dw-8ray tokens not present")
    vocab = FrameVocab.load(root / "tokens/vocab.npz")
    tb = tkb.load_token_bench(vocab, n=6, target="appearance", basis_name="frustum",
                              data_dir=root / "eval")
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
