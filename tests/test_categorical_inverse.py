"""The categorical inverse map (2026-09-20): g inverts the state the block's OWN probes read — one-hot
labels + Cartesian velocity — fitted streamed; a categorical bench is never written through the
continuous map; the driver adds it to scored runs only in scope and only on request; and the script
that removes the old continuous-state arms touches categorical discworld blocks and nothing else."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pim.probes.inverse import (CATEGORICAL_STATE, CategoricalState, encode_categorical_state,
                                fit_inverse_map, fit_inverse_map_stream)
from pim.scoring import blocks, driver

REPO = Path(__file__).resolve().parents[1]


def test_encoding_is_one_hot_per_tile_then_the_raw_continuous_tail():
    lab = torch.tensor([[0, 2, 1], [1, 1, 0]])
    vel = torch.tensor([[0.5, -1.0], [2.0, 3.0]])
    x = encode_categorical_state(lab, vel, n_classes=3)
    assert x.shape == (2, 3 * 3 + 2)
    assert x[0, :9].tolist() == [1, 0, 0, 0, 0, 1, 0, 1, 0] and x[0, 9:].tolist() == [0.5, -1.0]
    assert x[:, :9].sum(1).tolist() == [3.0, 3.0]                      # exactly one class per tile
    assert encode_categorical_state(lab, None, 3).shape == (2, 9)


def test_only_the_continuous_tail_is_standardised():
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 4, size=(20, 5, 2))
    vel = rng.normal(3.0, 2.0, size=(20, 5, 4)).astype(np.float32)
    st = CategoricalState(lab, 4, extra=vel, device="cpu")
    assert (st.n, st.T, st.n_tiles, st.dim) == (20, 5, 2, 2 * 4 + 4)
    tr = torch.arange(16)
    xm, xs = st.moments(tr)
    assert torch.all(xm[:8] == 0) and torch.all(xs[:8] == 1)            # a rare class keeps gain 1, not 1/sd
    assert torch.allclose(xm[8:], torch.from_numpy(vel[:16].reshape(-1, 4).mean(0)), atol=1e-5)
    s, f = torch.tensor([0, 3]), torch.tensor([1, 4])
    assert torch.equal(st.build(s, f), encode_categorical_state(torch.from_numpy(lab[[0, 3], [1, 4]]),
                                                                torch.from_numpy(vel[[0, 3], [1, 4]]), 4))


class _Rows:
    """An in-memory stand-in for ``baselines.MemmapRows`` (same surface)."""
    def __init__(self, arr):
        self.mm = torch.from_numpy(arr)
        self.n, self.T, self.dim = arr.shape
        self.device = torch.device("cpu")

    def build(self, seq, frame):
        return self.mm[seq, frame]


def test_the_streamed_fit_agrees_with_the_dense_fit():
    """Same probe body, optimiser and loss; only the delivery of rows (and hence the minibatch order)
    differs — the two must explain the same share of a residual that IS a function of the state."""
    rng = np.random.default_rng(1)
    N, T, tiles, C, d = 240, 8, 3, 5, 12
    lab = rng.integers(0, C, size=(N, T, tiles))
    vel = rng.normal(size=(N, T, 2)).astype(np.float32)
    emb = rng.normal(size=(tiles, C, d)).astype(np.float32)
    W = rng.normal(size=(2, d)).astype(np.float32)
    H = sum(emb[t][lab[..., t]] for t in range(tiles)) + vel @ W + 0.3 * rng.normal(size=(N, T, d)).astype(np.float32)
    H = H.astype(np.float32)
    perm = np.random.default_rng(0).permutation(N)
    tr, te = perm[:192], perm[192:]
    state = CategoricalState(lab, C, extra=vel, device="cpu")
    # like for like: the dense fitter's batch is fixed at FIT_BATCH (4096 rows > this whole train split, so
    # ONE step per epoch) — give the streamed fit the same, and both enough epochs to converge
    g, st = fit_inverse_map_stream(state, _Rows(H), tr, te, hidden=32, epochs=800, seed=0)
    X = encode_categorical_state(torch.from_numpy(lab.reshape(-1, tiles)), torch.from_numpy(vel.reshape(-1, 2)), C).numpy()
    rows = lambda seqs: (np.repeat(seqs, T) * T + np.tile(np.arange(T), len(seqs)))      # noqa: E731
    gd, sd = fit_inverse_map(X[rows(tr)], H.reshape(-1, d)[rows(tr)], X[rows(te)], H.reshape(-1, d)[rows(te)],
                             hidden=32, epochs=800, device="cpu")
    assert st["state"] == CATEGORICAL_STATE and st["d_in"] == tiles * C + 2 and st["rows_train"] == 192 * T
    assert st["r2"] > 0.8 and abs(st["r2"] - sd["r2"]) < 0.05, (st["r2"], sd["r2"])
    assert st["r2_insample"] >= st["r2"] - 0.02
    assert not any(p.requires_grad for p in g.parameters())
    # the frozen map returns residuals in RAW units for an encoded target state
    out = g(encode_categorical_state(torch.from_numpy(lab[te[:4], 0]), torch.from_numpy(vel[te[:4], 0]), C))
    assert out.shape == (4, d) and float(((out - torch.from_numpy(H[te[:4], 0])) ** 2).mean()) < 1.0


def test_a_categorical_bench_is_never_written_through_the_continuous_map():
    from pim.environments.discworld import arms as dwa

    class _B:
        kind = "classification"
    with pytest.raises(ValueError, match="state the block's own probes read"):
        dwa.inverse_arms(None, {"appearance-fac": _B()}, basis_name="frustum", target="full")
    _B.kind = "regression"
    with pytest.raises(ValueError, match="state the block's own probes read"):
        dwa.inverse_arms(None, {"frustum": _B()}, basis_name="frustum", target="appearance-fac")


def test_scope_and_the_add_gate(monkeypatch):
    S = {"dw_cat_im": {"instances": ("dw-8ray",), "targets": ("appearance-fac",)}}
    assert blocks.cat_inverse_in_scope("dw-8ray", "appearance-fac", S)
    assert not blocks.cat_inverse_in_scope("dw-noiseless", "appearance-fac", S)
    assert not blocks.cat_inverse_in_scope("dw-8ray", "grid-16x8", S)
    assert not blocks.cat_inverse_in_scope("dw-8ray", "appearance-fac", {})          # no setting: nowhere
    pi, im = [{"editor": "PI"}], [{"editor": "PI"}, {"editor": "IM"}]
    prev = {"bases": {"frustum": {"kind": "regression", "arms": pi},
                      "cartesian": {"kind": "regression", "arms": im},
                      "appearance-fac": {"kind": "classification", "target": "appearance-fac", "arms": pi},
                      "grid-16x8": {"kind": "classification", "target": "grid-16x8", "arms": pi}}}
    r = {"env": "discworld", "instance": "dw-8ray", "topic": "t", "run": "x"}
    monkeypatch.delenv("PIM_ADD_CAT_IM", raising=False)
    assert driver.missing_inverse(r, prev, S) == ["frustum"]            # a categorical block is not "missing" IM
    monkeypatch.setenv("PIM_ADD_CAT_IM", "1")
    assert driver.missing_inverse(r, prev, S) == ["frustum", "appearance-fac"]       # in scope, on request
    assert driver.missing_inverse({**r, "instance": "dw-blink"}, prev, S) == ["frustum"]   # out of scope: never
    assert driver.missing_inverse(r, prev, {}) == ["frustum"]


def test_the_clear_script_touches_categorical_discworld_blocks_only():
    spec = importlib.util.spec_from_file_location(
        "clear_continuous_im", REPO / "experiments" / "categorical_inverse" / "scripts" / "clear_continuous_im.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.CATEGORICAL_STATE == CATEGORICAL_STATE
    arms = [{"editor": "PI[zspace]", "edit_index": 0.4}, {"editor": "IM", "edit_index": 0.7}, {"editor": "IM-NN", "edit_index": 0.5}]
    blk = lambda kind, **kw: {"kind": kind, "arms": [dict(a) for a in arms],                     # noqa: E731
                              "best": {"PI": arms[0], "IM": arms[1], "IM-NN": arms[2]},
                              "best_by_dims": {"all": {"PI": arms[0], "IM": arms[1], "IM-NN": arms[2]}},
                              "inverse_map": {"g_r2": [0.5], **kw}}
    s = {"env": "discworld", "bases": {"cartesian": blk("regression"), "appearance-fac": blk("classification"),
                                      "done": blk("classification", state=CATEGORICAL_STATE)}}
    before = json.dumps(s["bases"]["cartesian"]), json.dumps(s["bases"]["done"])
    assert mod.clear(s, "2026-09-20") == ["appearance-fac"]
    fac = s["bases"]["appearance-fac"]
    assert [a["editor"] for a in fac["arms"]] == ["PI[zspace]"] and fac["best"]["PI"] == arms[0]
    assert fac["best"]["IM"] is None and fac["best"]["IM-NN"] is None and fac["best_by_dims"]["all"]["IM"] is None
    assert "inverse_map" not in fac and fac["inverse_cleared"]["date"] == "2026-09-20"
    assert (json.dumps(s["bases"]["cartesian"]), json.dumps(s["bases"]["done"])) == before
    assert mod.clear(s, "2026-09-20") == []                                  # idempotent
    assert mod.clear({"env": "othello", "bases": {"mine_signed": blk("classification")}}, "2026-09-20") == []
