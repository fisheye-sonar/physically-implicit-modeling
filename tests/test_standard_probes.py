"""Tests for the standardised readability probes.

The load-bearing one is `test_mlp_default_is_unchanged`: the MLP Grad Steering editor
writes through a frozen `MLPExtractor` with the original 1x128 defaults, and its
published results are tied to that exact architecture.
"""

from __future__ import annotations

import warnings

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


# ── the MLP-below-linear tripwire (added 2026-08-24) ─────────────────────────────────────────
# Guards the failure that produced the 2026-08-22 velocity-decodability numbers: 262k probe
# parameters fit on 48k rows scored 0.954-0.959 in-sample against -0.073/-0.090 held-out, and
# were read as "velocity is barely decodable" in two findings before the cause was spotted.

def _fit(n_seq, *, noise_target, seed=0, n_t=20, h=16, d=3):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(n_seq, n_t, h)).astype(np.float32)
    y = (rng.normal(size=(n_seq, n_t, d)) if noise_target
         else (s.reshape(-1, h) @ rng.normal(size=(h, d))).reshape(n_seq, n_t, d)).astype(np.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = fit_readability_probes(s, y, device="cpu", seed=seed)
    return r, [x for x in w if "scored BELOW" in str(x.message)]


def test_tripwire_silent_when_mlp_matches_linear():
    """A well-posed fit must not warn: the MLP ties the linear probe within tolerance."""
    r, trip = _fit(120, noise_target=False)
    assert r["gap"] > -0.01, r["gap"]
    assert not trip, trip[0].message if trip else None


def test_tripwire_fires_on_a_memorising_probe():
    """Starved MLP: perfect in-sample, negative held-out, below the linear probe."""
    r, trip = _fit(12, noise_target=True)
    assert r["mlp_r2"] < r["linear_r2"] - 0.01
    assert r["mlp_r2_insample"] > r["mlp_r2"] + 0.10
    assert len(trip) == 1
    assert "memorising the probe training set" in str(trip[0].message)


def test_tripwire_names_undertrained_separately_from_overfit():
    """Low in-sample AND low held-out is an under-trained probe — the opposite fix."""
    rng = np.random.default_rng(3)
    s = rng.normal(size=(40, 20, 16)).astype(np.float32)
    y = (s.reshape(-1, 16) @ rng.normal(size=(16, 3))).reshape(40, 20, 3).astype(np.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = fit_readability_probes(s, y, device="cpu", seed=0, n_epochs=1, lr=1e-6)
        trip = [x for x in w if "scored BELOW" in str(x.message)]
    assert trip, "a probe trained for 1 epoch at lr=1e-6 should trip the wire"
    m = str(trip[0].message)
    assert "under-trained probe, not an overfit one" in m
    assert "memorising" not in m
    assert r["mlp_r2_insample"] < r["mlp_r2"] + 0.10


def test_mlp_insample_is_never_the_heldout_value_in_disguise():
    """The in-sample field must be a genuine train-split score, not a copy of the held-out one."""
    r, _ = _fit(12, noise_target=True)
    assert r["mlp_r2_insample"] != r["mlp_r2"]
    assert r["mlp_r2_insample"] > 0.9   # it memorised the train split
