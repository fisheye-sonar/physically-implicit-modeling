"""Token-model editability scored the Othello way (2026-09-05): shapes, sets, arms.

Uses the dw-8ray instance's tokens if present (skipped otherwise) and a tiny random
token model, so it checks the wiring — cases, hooks, scorecards — not any number.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
TOK = REPO / "datasets" / "discworld" / "dw-8ray" / "tokens"
pytestmark = pytest.mark.skipif(not (TOK / "vocab.npz").exists(), reason="dw-8ray tokens absent")


def _tiny(vocab_size):
    from pim.models.registry import build
    torch.manual_seed(0)
    return build("transformer_l_tokens", {"vocab": vocab_size, "block_size": 39, "n_layer": 2,
                                          "n_head": 2, "n_embd": 16, "dropout": 0.0}).eval().to(
        "cuda" if torch.cuda.is_available() else "cpu")


def test_token_bench_cases_and_arms(tmp_path):
    from pim.environments.discworld import arms as dwa
    from pim.environments.discworld import bench as dwb
    from pim.environments.discworld import token_bench as tkb
    from pim.environments.discworld.tokens import FrameVocab, decode
    vocab = FrameVocab.load(TOK / "vocab.npz")
    inst = REPO / "datasets" / "discworld" / "dw-8ray"
    tb = tkb.load_token_bench(vocab, n=16, target="full", basis_name="cartesian",
                              data_dir=inst / "eval")
    assert tb.tokens.shape == (16, 20) and tb.tgt.shape == (16, 8)
    # the context tokens are the stored edits tokens; post frames decode to the clean frame
    stored = np.load(TOK / "edits.npy")[:16, :20]
    assert np.array_equal(tb.tokens, stored)
    a = dwb.bench_arrays(16, "full", "cartesian", inst / "eval")
    assert np.array_equal(decode(tb.post_tok, vocab), a["clean"][:, 20])
    assert np.array_equal(decode(tb.pre_tok, vocab), a["zones"].gt_unedited)
    assert tb.keep.any()
    model = _tiny(vocab.size)
    uns, uc = tkb.unsteered(model, tb)
    assert uns.shape == (16, vocab.size) and np.allclose(uns.sum(1), 1, atol=1e-5)
    assert -1 <= uc["edit_index"] <= 1 and uc["n_scored"] == int(tb.keep.sum())
    enc, tag = tkb.token_encoder(vocab)
    lin = dwa.fit_probes(model, target="full", n_seq=64, family="linear", basis_name="cartesian",
                         data_dir=inst / "probe", cache_dir=tmp_path, encoder=enc, encoder_tag=tag,
                         log=None)
    assert set(lin) == set(range(3)) and lin[0][0].d_in == 16
    recs = tkb.pinv_arm(model, tb, lin, [1.0], uns, dims="pos")
    assert len(recs) == 3 and all(-1 <= r["edit_index"] <= 1 for r in recs)
    assert all("fidelity_ratio" in r and "zone_edit_index_expected" in r for r in recs)
    mlp = dwa.fit_probes(model, target="full", n_seq=64, family="mlp", basis_name="cartesian",
                         data_dir=inst / "probe", cache_dir=tmp_path, encoder=enc, encoder_tag=tag,
                         log=None)
    gs = tkb.grad_steer_arm(model, tb, mlp, [0, 2], [0.05], uns, n_steps=2)
    assert len(gs) == 2 and gs[0]["editor"] == "GS@L0"
    # the cache key carries the encoder: a refit is a hit, a frames key is not the same file
    hit = dwa.fit_probes(model, target="full", n_seq=64, family="linear", basis_name="cartesian",
                         data_dir=inst / "probe", cache_dir=tmp_path, encoder=enc, encoder_tag=tag,
                         log=None)
    assert torch.allclose(hit[0][0].net.weight, lin[0][0].net.weight)
