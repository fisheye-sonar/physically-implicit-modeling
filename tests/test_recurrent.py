"""Recurrent-L: protocol surface, per-step == full-sequence, edit semantics, registry."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pim.models import build, n_points
from pim.models.recurrent import RecurrentConfig, RecurrentL, RecurrentState
from pim.probes.base import collect_residuals

CFG = RecurrentConfig(input_dim=16, d_model=32, n_layers=2, dropout=0.0)


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return RecurrentL(CFG).eval()


def test_protocol_surface_and_points(model):
    for attr in ("embed", "_run", "residual_stack", "decode", "norm_out", "n_layers",
                 "probe_layer", "state_span", "state_from_obs", "advance", "flat_state",
                 "predict_step", "rollout_with_edit"):
        assert hasattr(model, attr), attr
    assert n_points(model) == 3
    assert model.state_span > 40


def test_forward_follows_transformer_l_alignment(model):
    obs = torch.randn(3, 9, 16)
    with torch.no_grad():
        pred = model(obs[:, :-1])
    assert pred.shape == (3, 8, 16)          # one prediction per given position


def test_parameter_count_matches_transformer_l():
    """4 x 1024 was chosen to match the canonical Transformer-L (25,371,776)."""
    m = RecurrentL(RecurrentConfig())
    n = sum(p.numel() for p in m.parameters())
    assert abs(n - 25_371_776) / 25_371_776 < 0.02, n


def test_edit_none_is_bit_identical(model):
    obs = torch.randn(3, 6, 16)
    with torch.no_grad():
        s = model.state_from_obs(obs)
        assert torch.equal(model.decode(s), model.decode(s, edit=None))


def test_per_step_path_equals_full_sequence(model):
    """THE correctness gate: advancing one frame at a time reproduces the one-pass
    forward — predictions AND every residual point, at every step."""
    obs = torch.randn(2, 7, 16)
    with torch.no_grad():
        full_pred = model(obs)                              # (2, 7, 16)
        full_res = model.residual_stack(obs)                # (3, 2, 7, 32)
        s = model.state_from_obs(obs[:, :1])
        for t in range(7):
            assert torch.allclose(model.decode(s), full_pred[:, t], atol=1e-5), t
            for ell in range(3):
                model.probe_layer = ell
                assert torch.allclose(model.flat_state(s), full_res[ell][:, t], atol=1e-5)
            if t < 6:
                s = model.advance(s, obs[:, t + 1])
        # and state_from_obs of a longer prefix lands on the same state
        s7 = model.state_from_obs(obs)
        assert torch.allclose(s7.h_prev, s.h_prev, atol=1e-5)


def test_edit_at_every_point_changes_the_immediate_prediction(model):
    obs = torch.randn(4, 5, 16)
    with torch.no_grad():
        s = model.state_from_obs(obs)
        base = model.decode(s)
        for ell in range(3):
            model.probe_layer = ell
            h = model.flat_state(s)
            out = model.decode_with_edit(s, ell, h + 3.0)
            assert not torch.allclose(out, base), ell


def test_hidden_state_is_only_the_hiddens(model):
    """The pending frame is NOT part of the hidden state: two states with the same h_prev
    and different obs_t decode differently, and the same h_t is carried forward from
    either only through what the step computes."""
    obs = torch.randn(2, 4, 16)
    with torch.no_grad():
        s = model.state_from_obs(obs)
        assert s.h_prev.shape == (2, 2, 32) and s.obs_t.shape == (2, 16)
        s2 = RecurrentState(s.h_prev, s.obs_t + 1.0, s.length)
        assert not torch.allclose(model.decode(s), model.decode(s2))


def test_carried_edit_persists_and_transient_does_not(model):
    obs = torch.randn(3, 5, 16)
    with torch.no_grad():
        s = model.state_from_obs(obs)
        model.probe_layer = 1
        h = model.flat_state(s)
        model.carry_edits = True
        carried = model.rollout_with_edit(s, 1, h + 2.0, 4)
        model.carry_edits = False
        transient = model.rollout_with_edit(s, 1, h + 2.0, 4)
        model.carry_edits = True
        plain = torch.stack([p for p, _ in [model.predict_step(s)]], 1)
    assert torch.allclose(carried[:, 0], transient[:, 0])         # same first prediction
    assert not torch.allclose(carried[:, 1], transient[:, 1])     # then they diverge
    assert not torch.allclose(carried[:, 0], plain[:, 0])         # and both moved step 0


def test_hook_matches_tuple_edit_and_fires_everywhere(model):
    obs = torch.randn(2, 5, 16)
    seen = []
    with torch.no_grad():
        s = model.state_from_obs(obs)
        model.probe_layer = 1
        v = model.flat_state(s) + 1.5

        def hook(layer, x):
            seen.append((layer, tuple(x.shape)))
            if layer == 1:
                x = x.clone()
                x[:, -1] = v
            return x
        a = model.decode(s, edit=hook)
        b = model.decode(s, edit=(1, v))
    assert torch.allclose(a, b)
    assert [lay for lay, _ in seen] == [0, 1, 2]
    assert all(sh == (2, 1, 32) for _, sh in seen)


def test_rollout_with_hook_carries_like_the_tuple_form(model):
    obs = torch.randn(2, 5, 16)
    with torch.no_grad():
        s = model.state_from_obs(obs)
        model.probe_layer = 1
        v = model.flat_state(s) + 1.5
        def hook(layer, x):
            if layer != 1:
                return x
            x = x.clone()
            x[:, -1] = v
            return x
        assert torch.allclose(model.rollout_with_hook(s, hook, 3),
                              model.rollout_with_edit(s, 1, v, 3))


def test_collect_residuals_shape(model):
    obs = np.random.default_rng(0).standard_normal((5, 6, 16)).astype(np.float32)
    R = collect_residuals(model, obs, batch=2)
    assert R.shape == (3, 5, 6, 32)


def test_registry_builds_and_infers():
    m = build("recurrent_l", {"input_dim": 16, "d_model": 32, "n_layers": 2, "dropout": 0.0})
    assert isinstance(m, RecurrentL) and n_points(m) == 3


def test_collect_residuals_memmap_equals_ram(model, tmp_path):
    """The disk-backed stack is the SAME numbers — it exists only to survive a memory cap."""
    obs = np.random.default_rng(1).standard_normal((5, 6, 16)).astype(np.float32)
    ram = collect_residuals(model, obs, batch=2)
    mm = collect_residuals(model, obs, batch=2, memmap=tmp_path / "r.npy")
    assert isinstance(mm, np.memmap) and np.array_equal(ram, np.asarray(mm))
