"""Tests for the causal transformer world model.

The load-bearing one is `test_buffer_rollout_matches_full_sequence`: it pins the
fact that the carried state spans `n_layers*(window-1)+1` frames, not `window`.
Sizing the buffer by `window` silently diverges from exactly `t = window` onward
and would mis-state how much history an edit has to overwrite.
"""

from __future__ import annotations

import torch

from pim.models.transformer_s import ModelConfig, TransformerS as TransformerModel


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


# ── the callable edit hook (multi-position / multi-layer writes) ──────────────
#
# `_run`'s original `edit=(layer, vector)` writes the LAST position only, which cannot express a
# *history* edit: the residual stream at position t represents frame t, so rewriting the history
# means writing every position, at every layer (the stream is recomputed by each block). The hook
# form exists for that. These tests pin (a) that adding it left the default path bit-identical,
# and (b) that the two forms agree where they overlap.


@torch.no_grad()
def test_edit_hook_absent_is_bit_identical():
    """A no-op hook must reproduce the unedited forward pass EXACTLY, not approximately."""
    m = _model(window=4, n_layers=3)
    obs = torch.randn(2, 9, 32)
    state = m.state_from_obs(obs)
    base = m.decode(state)
    identity = m.decode(state, edit=lambda i, x: x)
    assert torch.equal(base, identity)


@torch.no_grad()
def test_edit_hook_matches_tuple_form_at_the_last_position():
    """Writing only the last position through the hook == the (layer, vector) form."""
    m = _model(window=4, n_layers=3)
    obs = torch.randn(2, 9, 32)
    state = m.state_from_obs(obs)
    torch.manual_seed(1)
    vec = torch.randn(2, 64)

    def hook(i, x):
        if i != 2:
            return x
        x = x.clone()
        x[:, -1] = vec
        return x

    assert torch.equal(m.decode(state, edit=(2, vec)), m.decode(state, edit=hook))


@torch.no_grad()
def test_edit_hook_writes_every_position_and_every_layer():
    """A write at earlier positions must reach the prediction — that is the whole point."""
    m = _model(window=4, n_layers=3)
    obs = torch.randn(2, 9, 32)
    state = m.state_from_obs(obs)
    base = m.decode(state)
    seen = []

    def perturb(i, x, cols=slice(None)):
        g = torch.Generator().manual_seed(i)
        x = x.clone()
        x[:, cols] = x[:, cols] + torch.randn(x[:, cols].shape, generator=g)
        return x

    def hook(i, x):
        seen.append((i, tuple(x.shape)))
        return perturb(i, x)

    edited = m.decode(state, edit=hook)
    assert [i for i, _ in seen] == [0, 1, 2, 3]           # n_layers residual points + the final one
    assert all(shape == (2, m.state_span, 64) for _, shape in seen)
    assert not torch.allclose(base, edited)

    # writing ONLY the earliest positions (never the last) must still reach the prediction —
    # that is what distinguishes a history edit from the last-position write.
    early = m.decode(state, edit=lambda i, x: perturb(i, x, cols=slice(None, -1)))
    assert (early - base).abs().max() > 1e-2


@torch.no_grad()
def test_a_constant_offset_is_invisible_to_a_pre_norm_transformer():
    """Adding the SAME value to every channel changes nothing — it is LayerNorm's null space.

    Worth pinning because it is a live trap for anyone writing an activation edit: a naive
    "shift the residual stream by c" write reads as a null result from the editor rather than
    from the model. Only a direction that survives the norm can carry an edit.
    """
    m = _model(window=4, n_layers=3)
    obs = torch.randn(2, 9, 32)
    state = m.state_from_obs(obs)
    base = m.decode(state)
    shifted = m.decode(state, edit=lambda i, x: x + 0.5)
    assert (shifted - base).abs().max() < 1e-4


@torch.no_grad()
def test_edit_hook_recorded_residuals_stay_consistent():
    """`want_resid` must report the stream the blocks actually consumed."""
    m = _model(window=4, n_layers=2)
    m.state_view = "activations"
    m.probe_layer = 1
    obs = torch.randn(2, 9, 32)
    state = m.state_from_obs(obs)
    plain = m._activations(state)
    shifted = m._activations(state, edit=lambda i, x: x + 1.0 if i == 1 else x)
    assert torch.allclose(shifted, plain + 1.0, atol=1e-5)


@torch.no_grad()
def test_residual_stack_shape_and_agreement_with_activations():
    """The (layer x window position) grid must agree with `_activations` where they overlap."""
    m = _model(window=4, n_layers=3)
    obs = torch.randn(2, 9, 32)
    state = m.state_from_obs(obs)
    stack = m.residual_stack(state)
    assert stack.shape == (4, 2, m.state_span, 64)      # n_layers + 1 residual points
    m.state_view = "activations"
    for layer in range(4):
        m.probe_layer = layer
        assert torch.equal(stack[layer, :, -1], m._activations(state))
