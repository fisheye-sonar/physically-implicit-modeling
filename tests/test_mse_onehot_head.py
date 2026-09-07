"""The MSE-on-one-hot Othello head (2026-09-04): loss and the output-kind mapping.

* `mse_next_move_onehot` equals a hand-computed Brier score over the kept positions.
* `move_probs`: "logits" is the softmax that every canonical CE run used (unchanged),
  "raw" is the identity on the 60 move outputs, "clipnorm" clips at 0 and renormalises
  (all-zero rows become uniform). `board_probs` lays each of them on the board.
* The token model carries `output_kind` and the registry round-trips it.
"""
import numpy as np
import torch
import torch.nn.functional as F

from pim.environments.othello.data import board_probs, move_probs
from pim.models.registry import build
from pim.training.train import IGNORE, mse_next_move_onehot, xy_tokens


def _tiny(output_kind="raw"):
    torch.manual_seed(0)
    return build("transformer_l_tokens", {"vocab": 61, "block_size": 59, "n_layer": 1,
                                          "n_head": 2, "n_embd": 16, "dropout": 0.0,
                                          "output_kind": output_kind}).eval()


def test_loss_is_the_brier_score_over_kept_positions():
    m = _tiny()
    g = torch.Generator().manual_seed(1)
    tok = torch.randint(1, 61, (4, 60), generator=g)
    ln = torch.tensor([60, 30, 12, 60])
    loss = mse_next_move_onehot(m, (tok, ln))
    x, y = xy_tokens(tok, ln, 59)
    out = m.logits(x)
    keep = y != IGNORE
    onehot = F.one_hot(y.clamp_min(0), 61).float()
    ref = ((out[keep] - onehot[keep]) ** 2).mean()
    assert torch.allclose(loss, ref)
    assert keep.sum() == (60 - 1) + (30 - 1) + (12 - 1) + (60 - 1)


def test_move_probs_kinds():
    torch.manual_seed(0)
    out = torch.randn(5, 61)
    lg = move_probs(out, "logits")
    assert torch.allclose(lg, torch.softmax(out[:, 1:], -1))
    raw = move_probs(out, "raw")
    assert torch.equal(raw, out[:, 1:])
    cn = move_probs(out, "clipnorm")
    assert (cn >= 0).all() and torch.allclose(cn.sum(-1), torch.ones(5))
    zero = torch.zeros(2, 61) - 1.0                         # all negative -> all clipped
    assert torch.allclose(move_probs(zero, "clipnorm"), torch.full((2, 60), 1 / 60))
    b = board_probs(out, "raw")
    assert b.shape == (5, 64) and np.allclose(b[:, [27, 28, 35, 36]], 0.0)
    assert np.allclose(b[:, :27], out[:, 1:28].numpy())


def test_output_kind_round_trips_through_the_registry():
    assert _tiny("raw").output_kind == "raw"
    assert _tiny("logits").output_kind == "logits"
    assert build("transformer_l_tokens", {"vocab": 61, "block_size": 59, "n_layer": 1,
                                          "n_head": 2, "n_embd": 16}).output_kind == "logits"
