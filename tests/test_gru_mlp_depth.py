"""Tests for the optional MLP depth on the GRU encoder / decoder.

The load-bearing one is `test_depth_zero_is_bit_identical`: the depth knobs are an
*extension*, and every existing GRU checkpoint and published editability result
depends on the depth-0 architecture being untouched.

The rest pin the property the knobs exist to provide. A single-`nn.Linear` decoder
is **affine**, so

    decode(h0 + d1 + d2) == decode(h0 + d1) + decode(h0 + d2) - decode(h0)

holds identically for *any* vectors, which means an "edits superpose" result read
off the decoded observation is forced by algebra rather than by structure in the
latent. `dec_hidden_layers >= 1` must break that identity.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from pim.world_models.gru import GRUModel, ModelConfig


def _model(**kw) -> GRUModel:
    torch.manual_seed(0)
    return GRUModel(ModelConfig(input_dim=16, hidden_size=24, **kw)).eval()


def _affine_violation(model: GRUModel) -> float:
    """max |decode(h0+d1+d2) - (decode(h0+d1) + decode(h0+d2) - decode(h0))|."""
    g = torch.Generator().manual_seed(1)
    h0, d1, d2 = (torch.randn(6, model.hidden_size, generator=g) for _ in range(3))
    s = model.state_from_flat
    with torch.no_grad():
        lhs = model.decode(s(h0 + d1 + d2))
        rhs = model.decode(s(h0 + d1)) + model.decode(s(h0 + d2)) - model.decode(s(h0))
    return float((lhs - rhs).abs().max())


def test_depth_zero_is_bit_identical() -> None:
    """Depth 0 must reproduce the original forward/step/decode exactly."""
    m = _model()
    assert m.enc_trunk is None and m.dec_trunk is None
    assert m.has_affine_decoder

    g = torch.Generator().manual_seed(0)
    obs = torch.rand(3, 7, 16, generator=g)

    with torch.no_grad():
        pred, h_n = m(obs)
        x = F.relu(m.encoder(obs[:, :-1, :]))
        h_ref, h_n_ref = m.gru(x, None)
        assert torch.equal(pred, m.decoder(h_ref))
        assert torch.equal(h_n, h_n_ref)

        state = ref_state = None
        for t in range(4):
            pred_t, state = m.step(obs[:, t], state)
            xx = F.relu(m.encoder(obs[:, t])).unsqueeze(1)
            h_out, ref_state = m.gru(xx, ref_state)
            assert torch.equal(pred_t, m.decoder(h_out.squeeze(1)))
        assert torch.equal(m.decode(state), m.decoder(state[-1]))
        assert torch.equal(m.get_hidden_states(obs), m.gru(x)[0])


def test_depth_zero_state_dict_has_no_trunk_keys() -> None:
    """Old checkpoints have no trunk keys, so depth-0 must not introduce any."""
    keys = set(_model().state_dict())
    assert not any(k.startswith(("enc_trunk", "dec_trunk")) for k in keys)
    # ...and a deep model loads into a deep model of the same config.
    deep = _model(enc_hidden_layers=2, dec_hidden_layers=2)
    reloaded = GRUModel(ModelConfig(**dataclasses.asdict(deep.cfg)))
    reloaded.load_state_dict(deep.state_dict())  # strict=True: exact key match


def test_affine_decoder_identity_holds_at_depth_zero() -> None:
    assert _affine_violation(_model()) < 1e-5


@pytest.mark.parametrize("act", ["relu", "elu", "silu", "gelu"])
def test_nonlinear_decoder_breaks_the_affine_identity(act: str) -> None:
    m = _model(dec_hidden_layers=2, mlp_activation=act)
    assert not m.has_affine_decoder
    assert _affine_violation(m) > 1e-2


def test_encoder_depth_alone_leaves_the_decoder_affine() -> None:
    """The superposition artifact is a *decoder* property; encoder depth must not fix it."""
    m = _model(enc_hidden_layers=2, dec_hidden_layers=0)
    assert m.has_affine_decoder
    assert _affine_violation(m) < 1e-5


def test_depth_changes_shapes_nowhere() -> None:
    """Protocol surface is unchanged by depth — eval code must not need to branch."""
    g = torch.Generator().manual_seed(0)
    obs = torch.rand(3, 7, 16, generator=g)
    for kw in ({}, {"enc_hidden_layers": 2, "dec_hidden_layers": 2}):
        m = _model(**kw)
        pred, h_n = m(obs)
        assert pred.shape == (3, 6, 16) and h_n.shape == (1, 3, 24)
        p_seq, h_seq = m.observe_sequence(obs)
        assert p_seq.shape == (3, 6, 16) and h_seq.shape == (3, 6, 24)
        flat = m.flat_state(h_n)
        assert flat.shape == (3, 24)
        assert m.decode(m.state_from_flat(flat)).shape == (3, 16)
        p_next, s_next = m.predict_step(m.state_from_flat(flat))
        assert p_next.shape == (3, 16) and s_next.shape == (1, 3, 24)


def test_unknown_activation_rejected() -> None:
    with pytest.raises(ValueError, match="unknown mlp_activation"):
        _model(dec_hidden_layers=1, mlp_activation="tanh")
