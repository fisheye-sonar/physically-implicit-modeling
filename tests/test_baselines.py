"""The decodability baselines: causal masking, and the gate against the canonical fit.

``pim.probes.baselines`` carries its own fit loop only because the causal-history
feature matrix is too large to materialise (19.7 GB on discworld). That is a licence to
stream, NOT a licence to compute something different — so the streamed linear fit is
asserted here to reproduce ``pim.probes.linear.fit_linear`` on a problem small enough to
run densely, and the MLP path to land in the same place.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pim.probes.baselines import CausalHistory, _row_index, fit_baseline_probe
from pim.probes.linear import fit_linear

N, T, R, D = 240, 5, 6, 3


def _dense(hist, seq):
    """The same rows fit_baseline_probe sees, materialised — test-only."""
    s, f = _row_index(torch.as_tensor(seq), hist.T, hist.src.device)
    return hist.build(s, f).cpu().numpy()


@pytest.fixture()
def setup():
    torch.manual_seed(0)
    obs = torch.rand(N, T, R)
    hist = CausalHistory(obs)
    w = torch.randn(hist.dim, D) * 0.3
    s, f = _row_index(torch.arange(N), T, obs.device)
    y = (hist.build(s, f) @ w).reshape(N, T, D) + 0.01 * torch.randn(N, T, D)
    return hist, y, np.arange(0, 180), np.arange(180, N)


def test_history_is_causal_and_left_padded():
    """Row (s,t) sees frames 0..t and NOTHING after — the property that makes this a
    fair floor rather than a leak."""
    torch.manual_seed(1)
    obs = torch.rand(4, T, R)
    hist = CausalHistory(obs)
    x = hist.build(torch.tensor([2, 2]), torch.tensor([0, 3])).reshape(2, T, R)
    assert torch.allclose(x[0, 0], obs[2, 0]) and x[0, 1:].abs().max() == 0
    assert torch.allclose(x[1, :4], obs[2, :4]) and x[1, 4:].abs().max() == 0


def test_one_hot_history_matches_dense_equivalent():
    tok = torch.randint(0, 7, (3, T))
    got = CausalHistory(tok, kind="one_hot", vocab=7).build(
        torch.tensor([1]), torch.tensor([2])).reshape(T, 7)
    assert got[:3].argmax(-1).tolist() == tok[1, :3].tolist()
    assert got[:3].sum() == 3 and got[3:].sum() == 0     # one-hot, then zero-padded


def test_streamed_linear_reproduces_the_canonical_dense_fit(setup):
    """THE GATE: same probe, to lstsq precision."""
    hist, y, tr, te = setup
    probe, stats = fit_baseline_probe(hist, y, tr, te, hidden=None, seed=0)

    x_tr, x_te = _dense(hist, tr), _dense(hist, te)
    y_tr = y[tr].reshape(-1, D).numpy()
    y_te = y[te].reshape(-1, D).numpy()
    ref, ref_stats = fit_linear(x_tr, y_tr, x_te, y_te, device="cpu", seed=0)

    assert np.allclose(probe.net.weight.detach().numpy(),
                       ref.net.weight.detach().numpy(), atol=1e-3)
    assert abs(stats["r2"] - ref_stats["r2"]) < 1e-4
    assert stats["n_train_rows"] == len(x_tr) == len(tr) * T


def test_streamed_mlp_lands_where_the_dense_fit_does(setup):
    """The SGD path consumes randomness in its own order, so this is a same-place check,
    not a bit-equality one — an MLP that cannot match a linear map on linear data is a
    broken loop, and that is what this catches."""
    hist, y, tr, te = setup
    _, stats = fit_baseline_probe(hist, y, tr, te, hidden=8, seed=0,
                                  epochs=600, batch=128)
    _, lin = fit_baseline_probe(hist, y, tr, te, hidden=None, seed=0)
    assert stats["r2"] > lin["r2"] - 0.05        # the probe-sanity relation
    assert stats["r2_insample"] >= stats["r2"] - 0.02


def test_stats_carry_the_overfit_check(setup):
    """r2_insample − r2 is the overfitting test Table 3 reports; it must always be there."""
    hist, y, tr, te = setup
    _, stats = fit_baseline_probe(hist, y, tr, te, hidden=None, seed=0)
    assert {"r2", "r2_insample", "per_dim_r2", "d_in"} <= set(stats)
    assert stats["d_in"] == T * R


def test_row_mask_drops_invalid_rows(setup):
    """Othello games are 9-60 moves long; fitting on the zero-padding past a game's end
    would report an easy constant as decodability."""
    hist, y, tr, te = setup
    mask = torch.ones(N, T, dtype=torch.bool)
    mask[:, 3:] = False
    _, stats = fit_baseline_probe(hist, y, tr, te, hidden=None, row_mask=mask, seed=0)
    assert stats["n_train_rows"] == len(tr) * 3
    assert stats["n_test_rows"] == len(te) * 3


def test_observation_cache_key_separates_spans(tmp_path):
    """The observation floor is ARCHITECTURE-dependent through `state_span`: the probe is
    given the history the model actually consumes. Two architectures with different spans
    must therefore get different fits, and two with the SAME span must share one — both of
    which come down to the span being in the cache key. (Keying the floor on the instance
    alone would have compared a Transformer-S run to a Transformer-L floor.)"""
    from pim.probes.cache import ProbeCache

    c = ProbeCache(tmp_path)

    def k(span):
        return c.key(None, kind="observation", basis="frustum", span=span)[0]

    assert k(39) != k(61)
    assert k(39) == k(39)


def test_model_free_cache_key_is_distinct_from_any_model(tmp_path):
    """model=None is the observation baseline; it must never collide with a real fit."""
    import torch.nn as nn

    from pim.probes.cache import ProbeCache

    c = ProbeCache(tmp_path)
    m = nn.Linear(4, 4)
    assert c.key(None, kind="observation")[1]["model"] == "none"
    assert c.key(m, kind="observation")[0] != c.key(None, kind="observation")[0]


def test_memmap_source_reproduces_the_dense_linear_fit(tmp_path):
    """A residual stack on disk, streamed, gives the SAME probe as the dense fit."""
    from pim.probes.baselines import MemmapRows, fit_probe_stream

    rng = np.random.default_rng(3)
    arr = np.lib.format.open_memmap(tmp_path / "r.npy", mode="w+", dtype=np.float32,
                                    shape=(120, 4, 12))
    arr[:] = rng.standard_normal((120, 4, 12)).astype(np.float32)
    arr.flush()
    w = rng.standard_normal((12, 3)).astype(np.float32)
    y = torch.from_numpy(arr @ w + 0.01 * rng.standard_normal((120, 4, 3)).astype(np.float32))
    tr, te = np.arange(0, 90), np.arange(90, 120)
    probe, st = fit_probe_stream(MemmapRows(arr, device="cpu"), y, tr, te, hidden=None)
    ref, rst = fit_linear(arr[tr].reshape(-1, 12), y[tr].reshape(-1, 3).numpy(),
                          arr[te].reshape(-1, 12), y[te].reshape(-1, 3).numpy(),
                          device="cpu", seed=0)
    assert np.allclose(probe.net.weight.detach().numpy(), ref.net.weight.detach().numpy(), atol=1e-3)
    assert abs(st["r2"] - rst["r2"]) < 1e-4


def test_collect_residuals_points_subset():
    from pim.models.recurrent import RecurrentConfig, RecurrentL
    from pim.probes.base import collect_residuals

    torch.manual_seed(0)
    m = RecurrentL(RecurrentConfig(input_dim=8, d_model=16, n_layers=2, dropout=0.0)).eval()
    obs = np.random.default_rng(0).standard_normal((6, 5, 8)).astype(np.float32)
    full = collect_residuals(m, obs, batch=4)
    one = collect_residuals(m, obs, batch=4, points=[1])
    assert one.shape == (1, 6, 5, 16) and np.array_equal(one[0], full[1])
