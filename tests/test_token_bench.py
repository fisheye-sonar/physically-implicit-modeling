"""Token-model editability scored the Othello way (2026-09-05): shapes, sets, arms.

Uses the dw-8ray instance's tokens if present (skipped otherwise) and a tiny random
token model, so it checks the wiring — cases, hooks, scorecards — not any number.
"""
from pathlib import Path

import json

import numpy as np
import pytest
import torch

from pim.environments import layout

REPO = layout.REPO
TOK = layout.tokens_dir("dw-8ray")
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
    INST = "dw-8ray"
    tb = tkb.load_token_bench(vocab, n=16, target="full", basis_name="cartesian",
                              instance=INST)
    assert tb.tokens.shape == (16, 20) and tb.tgt.shape == (16, 8)
    # the context tokens are the stored edits tokens for THE CASES THE BENCH SELECTED
    # (dw-8ray carries edits_selection.json since 2026-09-08 — 20% of its teleports render
    # an identical frame, so the bench is a filtered case list, not the first n).
    sp = layout.edits_selection("discworld", INST)
    sel = (json.loads(sp.read_text())["select"][:16] if sp.exists() else list(range(16)))
    stored = np.load(TOK / "edits.npy")[sel][:, :20]
    assert np.array_equal(tb.tokens, stored)
    # every selected case is scoreable, which is the point of the selection
    assert bool(tb.keep.all()), f"{int(tb.keep.sum())}/16 scoreable on the selected cases"
    # and the unfiltered bench is still reachable, and still equals the first n
    tb0 = tkb.load_token_bench(vocab, n=16, target="full", basis_name="cartesian",
                               instance=INST, use_selection=False)
    assert np.array_equal(tb0.tokens, np.load(TOK / "edits.npy")[:16, :20])
    a = dwb.bench_arrays(16, "full", "cartesian", instance=INST)
    assert np.array_equal(decode(tb.post_tok, vocab), a["clean"][:, 20])
    assert np.array_equal(decode(tb.pre_tok, vocab), a["zones"].gt_unedited)
    assert tb.keep.any()
    model = _tiny(vocab.size)
    uns, uc = tkb.unsteered(model, tb)
    assert uns.shape == (16, vocab.size) and np.allclose(uns.sum(1), 1, atol=1e-5)
    assert -1 <= uc["edit_index"] <= 1 and uc["n_scored"] == int(tb.keep.sum())
    enc, tag = tkb.token_encoder(vocab)
    lin = dwa.fit_probes(model, target="full", n_seq=64, family="linear", basis_name="cartesian",
                         probe={"instance": INST, "size": "120k"}, cache_dir=tmp_path, encoder=enc, encoder_tag=tag,
                         log=None)
    assert set(lin) == set(range(3)) and lin[0][0].d_in == 16
    recs = tkb.pinv_arm(model, tb, lin, [1.0], uns, dims="pos")
    assert len(recs) == 3 and all(-1 <= r["edit_index"] <= 1 for r in recs)
    assert all("fidelity_ratio" in r and "zone_edit_index_expected" in r for r in recs)
    mlp = dwa.fit_probes(model, target="full", n_seq=64, family="mlp", basis_name="cartesian",
                         probe={"instance": INST, "size": "120k"}, cache_dir=tmp_path, encoder=enc, encoder_tag=tag,
                         log=None)
    gs = tkb.grad_steer_arm(model, tb, mlp, [0, 2], [0.05], uns, n_steps=2)
    assert len(gs) == 2 and gs[0]["editor"] == "GS@L0"
    # the cache key carries the encoder: a refit is a hit, a frames key is not the same file
    hit = dwa.fit_probes(model, target="full", n_seq=64, family="linear", basis_name="cartesian",
                         probe={"instance": INST, "size": "120k"}, cache_dir=tmp_path, encoder=enc, encoder_tag=tag,
                         log=None)
    assert torch.allclose(hit[0][0].net.weight, lin[0][0].net.weight)
