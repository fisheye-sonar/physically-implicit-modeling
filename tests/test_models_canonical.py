"""The canonical architectures: protocol surface, edit-hook neutrality, registry rules.

Tiny configs throughout — no checkpoints, no GPU required. The bit-identity of the new
classes against the four retired bridge classes was gated on the real canonical
checkpoints at port time (2026-08-31); what these tests keep permanent is the surface
and the invariants that gate relied on.
"""

from __future__ import annotations

import pytest
import torch

from pim.models import (
    ModelConfig,
    TransformerL,
    TransformerLTokens,
    TransformerS,
    TransformerSTokens,
    build,
    n_points,
)
from pim.models.registry import _infer_arch

S_CFG = ModelConfig(input_dim=16, d_model=32, n_layers=2, n_heads=2, window=4)


def _all_models():
    return [
        ("transformer_s", TransformerS(S_CFG)),
        ("transformer_s_tokens", TransformerSTokens(S_CFG, vocab=13)),
        ("transformer_l", TransformerL(obs_res=16, block_size=10, n_layer=2, n_head=2,
                                       n_embd=32)),
        ("transformer_l_tokens", TransformerLTokens(vocab=13, block_size=10, n_layer=2,
                                                    n_head=2, n_embd=32)),
    ]


@pytest.mark.parametrize("name,model", _all_models())
def test_protocol_surface(name, model):
    for attr in ("embed", "_run", "residual_stack", "decode", "norm_out", "n_layers",
                 "probe_layer", "state_span"):
        assert hasattr(model, attr), f"{name} missing {attr}"
    assert n_points(model) == model.n_layers + 1


@pytest.mark.parametrize("name,model", _all_models())
def test_edit_none_is_bit_identical(name, model):
    """edit=None must take NEITHER branch — the load-bearing neutrality."""
    torch.manual_seed(0)
    model.eval()
    inp = (torch.randint(0, 13, (3, 8)) if "tokens" in name
           else torch.randn(3, 8, 16))
    with torch.no_grad():
        a = model.decode(model.state_from_obs(inp) if name == "transformer_s" else inp)
        b = model.decode(model.state_from_obs(inp) if name == "transformer_s" else inp,
                         edit=None)
    assert torch.equal(a, b)


@pytest.mark.parametrize("name,model", _all_models())
def test_residual_stack_has_n_points(name, model):
    model.eval()
    inp = (torch.randint(0, 13, (2, 6)) if "tokens" in name else torch.randn(2, 6, 16))
    if name == "transformer_s":
        inp = model.state_from_obs(inp)
    with torch.no_grad():
        rs = model.residual_stack(inp)
    assert rs.shape[0] == model.n_layers + 1


def test_single_site_edit_moves_only_last_position():
    torch.manual_seed(0)
    m = TransformerLTokens(vocab=13, block_size=10, n_layer=2, n_head=2, n_embd=32).eval()
    idx = torch.randint(1, 13, (2, 6))
    v = torch.randn(2, 32)
    with torch.no_grad():
        rs0 = m.residual_stack(idx)
        rs1 = m.residual_stack(idx, edit=(1, v))
    # points before the edit are untouched; the edited point differs at the last
    # position only (later points differ wherever attention carries it)
    assert torch.equal(rs0[0], rs1[0])
    assert torch.equal(rs1[1][:, :-1], rs0[1][:, :-1])
    assert torch.allclose(rs1[1][:, -1], v)


def test_state_span_formulas():
    assert TransformerS(S_CFG).state_span == 2 * (4 - 1) + 1
    assert TransformerL(obs_res=16, block_size=10, n_layer=2, n_head=2,
                        n_embd=32).state_span == 10


def test_tokens_models_refuse_rollout():
    m = TransformerSTokens(S_CFG, vocab=13)
    with pytest.raises(NotImplementedError):
        m.predict_step(None)


def test_registry_builds_all_four():
    assert isinstance(build("transformer_s", {"input_dim": 16, "d_model": 32,
                                              "n_layers": 2, "n_heads": 2,
                                              "window": 4}), TransformerS)
    assert isinstance(build("transformer_l", {"obs_res": 16, "block_size": 10,
                                              "n_layer": 2, "n_head": 2,
                                              "n_embd": 32}), TransformerL)


def test_registry_legacy_rules():
    """The documented recognition rules for the two legacy checkpoint formats."""
    assert _infer_arch({"model_config": {"obs_res": 128, "block_size": 39}}) == "transformer_l"
    assert _infer_arch({"vocab": 61, "model_config": {"window": 16}}) == "transformer_s_tokens"
    assert _infer_arch({"model_config": {"window": 16, "input_dim": 128}}) == "transformer_s"
    assert _infer_arch({"model_config": {},
                        "model_state": {"gpt.tok_emb.weight": 0,
                                        "gpt.blocks.0.x": 0}}) == "transformer_l_tokens"
    with pytest.raises(ValueError):
        _infer_arch({"model_config": {"det_size": 200}})  # RSSM: out of scope, must refuse


def test_registry_explicit_arch_wins():
    assert _infer_arch({"arch": "transformer_l",
                        "model_config": {"window": 16, "input_dim": 128}}) == "transformer_l"
