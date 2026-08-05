"""Tests for the causal transformer world model.

The load-bearing one is `test_buffer_rollout_matches_full_sequence`: it pins the
fact that the carried state spans `n_layers*(window-1)+1` frames, not `window`.
Sizing the buffer by `window` silently diverges from exactly `t = window` onward
and would mis-state how much history an edit has to overwrite.
"""

from __future__ import annotations

import torch

from pim.world_models.transformer import ModelConfig, TransformerModel


def _model(**kw):
    torch.manual_seed(0)
    return TransformerModel(
        ModelConfig(input_dim=32, d_model=64, n_heads=4, **kw)
    ).eval()


def test_shapes_and_protocol():
    m = _model(window=4, n_layers=3)
    obs = torch.randn(2, 12, 32)
    pred, state = m(obs)
    assert pred.shape == (2, 11, 32)
    assert state.obs_buffer.shape == (2, m.state_span, 32)
    assert m.state_span == 3 * (4 - 1) + 1
    for view, dim in [
        ("obs_window", m.state_span * 32),
        ("activations", 64),
        ("kv_cache", 3 * 2 * m.state_span * 64),
    ]:
        m.state_view = view
        assert m.hidden_size == dim


@torch.no_grad()
def test_buffer_rollout_matches_full_sequence():
    """One-pass banded forward == step-by-step buffer rollout, for every window."""
    for window in (2, 3, 4, 8):
        m = _model(window=window, n_layers=3)
        obs = torch.randn(2, 20, 32)
        seq, _ = m(obs)
        state, steps = None, []
        for t in range(obs.shape[1] - 1):
            p, state = m.step(obs[:, t], state)
            steps.append(p)
        assert torch.allclose(
            seq, torch.stack(steps, 1), atol=1e-4
        ), f"window={window}: buffer rollout diverges from the full-sequence pass"


@torch.no_grad()
def test_state_from_flat_roundtrips_obs_window():
    m = _model(window=3, n_layers=2)
    m.state_view = "obs_window"
    obs = torch.randn(2, 10, 32)
    _, state = m(obs)
    flat = m.flat_state(state)
    rebuilt = m.state_from_flat(flat)
    assert torch.allclose(rebuilt.obs_buffer, state.obs_buffer)
    assert torch.allclose(m.decode(rebuilt), m.decode(state))


@torch.no_grad()
def test_activation_view_is_read_only():
    m = _model(window=3, n_layers=2)
    m.state_view = "activations"
    obs = torch.randn(2, 10, 32)
    _, state = m(obs)
    assert m.flat_state(state).shape == (2, 64)
    try:
        m.state_from_flat(torch.zeros(2, 64))
    except ValueError:
        return
    raise AssertionError("state_from_flat must refuse the activations view")


@torch.no_grad()
def test_activation_edit_changes_prediction_at_every_layer():
    """An edit at any residual point must move the current prediction."""
    m = _model(window=4, n_layers=3)
    obs = torch.randn(2, 12, 32)
    _, state = m(obs)
    base = m.decode(state)
    for layer in range(m.cfg.n_layers + 1):
        m.probe_layer = layer
        h = m._activations(state)
        edited = m.decode_with_edit(state, layer, h + 3.0 * torch.randn_like(h))
        assert (edited - base).abs().max() > 1e-4, f"edit at layer {layer} was inert"


@torch.no_grad()
def test_probe_layer_zero_is_the_encoder_port():
    """Residual point 0 must equal relu(Linear(obs)) at the current frame."""
    m = _model(window=4, n_layers=2)
    m.state_view = "activations"
    m.probe_layer = 0
    obs = torch.randn(2, 9, 32)
    _, state = m(obs)
    assert torch.allclose(
        m.flat_state(state), m.embed(state.obs_buffer[:, -1]), atol=1e-5
    )
