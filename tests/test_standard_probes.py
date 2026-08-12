"""Tests for the standardised readability probes.

The load-bearing one is `test_mlp_default_is_unchanged`: the MLP Grad Steering editor
writes through a frozen `MLPExtractor` with the original 1x128 defaults, and its
published results are tied to that exact architecture.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pim.extractors import MLPExtractor, StateDefinition, fit_readability_probes


def _sdef(d=4):
    return StateDefinition(name="p", state_shape=(d,), extract_fn=lambda x: x)


def _data(n_seq=120, T=8, H=16, D=3, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((n_seq, T, H)).astype(np.float32)
    Y = (S @ rng.standard_normal((H, D))).astype(np.float32)
    Y += noise * rng.standard_normal((n_seq, T, D)).astype(np.float32)
    return S, Y


def test_mlp_default_is_unchanged() -> None:
    """Depth 1 must reproduce the original 1x128 net, keys and forward alike."""
    torch.manual_seed(0)
    a = MLPExtractor(16, _sdef())
    assert sorted(a.state_dict()) == [
        "net.0.bias",
        "net.0.weight",
        "net.2.bias",
        "net.2.weight",
    ]
    assert len(a.net) == 3 and a.net[0].out_features == 128
    torch.manual_seed(0)
    b = MLPExtractor(16, _sdef(), n_hidden_layers=1)
    x = torch.randn(5, 16)
    assert torch.equal(a(x), b(x))


def test_depth_adds_layers() -> None:
    m = MLPExtractor(16, _sdef(), mlp_hidden=32, n_hidden_layers=3)
    assert len([mod for mod in m.net if isinstance(mod, torch.nn.Linear)]) == 4
    assert m(torch.randn(5, 16)).shape == (5, 4)


def test_zero_hidden_layers_rejected() -> None:
    with pytest.raises(ValueError, match="n_hidden_layers"):
        MLPExtractor(16, _sdef(), n_hidden_layers=0)


def test_held_out_is_not_in_sample() -> None:
    """The whole point: the reported R2 must be held-out, and below the in-sample one."""
    S, Y = _data()
    r = fit_readability_probes(S, Y, n_epochs=5)
    assert r["linear_r2"] <= r["linear_r2_insample"] + 1e-9
    assert r["n_train_seq"] + r["n_heldout_seq"] == S.shape[0]
    assert r["n_heldout_seq"] == 24  # 20% of 120 sequences


def test_split_is_by_sequence_not_by_row() -> None:
    """Held-out rows must come from sequences the probes never saw."""
    S, Y = _data(n_seq=10, T=4)
    r = fit_readability_probes(S, Y, n_epochs=2)
    assert r["n_train_seq"] == 8 and r["n_heldout_seq"] == 2


def test_mask_is_applied() -> None:
    S, Y = _data(n_seq=40)
    mask = np.zeros(S.shape[:2], bool)
    mask[:, :4] = True
    r = fit_readability_probes(S, Y, mask=mask, n_epochs=2)
    assert np.isfinite(r["linear_r2"]) and np.isfinite(r["mlp_r2"])


def test_rejects_wrong_rank() -> None:
    S, Y = _data()
    with pytest.raises(ValueError, match=r"\(N, T"):
        fit_readability_probes(S.reshape(-1, S.shape[-1]), Y.reshape(-1, Y.shape[-1]))
