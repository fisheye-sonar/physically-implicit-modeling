"""Frames as tokens (2026-09-05): coding, vocabulary, round trips, the instance writer."""
import json

import h5py
import numpy as np
import pytest

from pim.environments.discworld import tokens as tk

LV = np.array([0.0, 0.4, 0.8], np.float32)


def _frames(rng, n, T=5, R=8):
    return LV[rng.integers(0, 3, size=(n, T, R))]


def test_codes_are_a_bijection_on_the_grid_and_reject_off_grid_values():
    rng = np.random.default_rng(0)
    x = _frames(rng, 50)
    c = tk.frame_codes(x, LV)
    assert c.shape == (50, 5) and c.min() >= 0 and c.max() < 3 ** 8
    # distinct frames ↔ distinct codes
    flat = x.reshape(-1, 8)
    assert len(np.unique(c)) == len(np.unique(flat, axis=0))
    with pytest.raises(ValueError):
        tk.frame_codes(np.array([[0.0, 0.5, 0.8, 0, 0, 0, 0, 0]], np.float32), LV)


def test_vocab_ids_are_deterministic_and_unseen_patterns_map_to_unk(tmp_path):
    rng = np.random.default_rng(1)
    x = _frames(rng, 40)
    counts = np.bincount(tk.frame_codes(x, LV).ravel(), minlength=3 ** 8)
    v = tk.vocab_from_counts(counts, LV, 8)
    assert v.size == int((counts > 0).sum()) + 1 and v.frames.shape == (v.size, 8)
    assert np.isnan(v.frames[tk.UNK]).all()
    ids = tk.encode(x, v)
    assert ids.dtype == np.int16 and (ids > 0).all()
    assert np.array_equal(tk.decode(ids, v), x)                 # round trip
    # ids follow ascending pattern code, independent of frequency
    codes = tk.frame_codes(v.frames[1:], LV)
    assert np.all(np.diff(codes) > 0)
    unseen = np.full((1, 8), 0.8, np.float32)                   # all rays on: never drawn above?
    if counts[tk.frame_codes(unseen, LV)[0]] == 0:
        assert tk.encode(unseen, v)[0] == tk.UNK
    # save / load
    v.save(tmp_path / "vocab.npz")
    w = tk.FrameVocab.load(tmp_path / "vocab.npz")
    assert np.array_equal(w.code_to_id, v.code_to_id) and np.allclose(w.frames[1:], v.frames[1:])


def test_tokenize_instance_writes_every_file(tmp_path):
    rng = np.random.default_rng(2)
    N, T, R = 37, 6, 8
    inst = tmp_path / "dw-tiny"
    (inst / "train").mkdir(parents=True)
    (inst / "probe").mkdir()
    (inst / "eval").mkdir()
    train = _frames(rng, N, T, R)
    mm = np.memmap(inst / "train" / "obs.f32", np.float32, mode="w+", shape=(N, T, R))
    mm[:] = train
    mm.flush()
    del mm
    eval_only = LV[np.array([[2, 2, 2, 2, 2, 2, 2, 2]])].astype(np.float32)   # a frame not in train
    assert not (np.abs(train.reshape(-1, R) - eval_only).sum(1) == 0).any()
    for name, p in tk.h5_splits(inst):        # a synthetic instance: layout-v1 relative names
        x = _frames(rng, 5, T, R)
        if name == "edits":
            x[0, 0] = eval_only[0]
        with h5py.File(p, "w") as h:
            h.create_dataset("obs_intensity", data=x)
    (inst / "train" / "corpus.json").write_text(json.dumps(          # the machine-written contract
        {"instance": "dw-tiny", "n": N, "n_frames": T, "obs_dim": R}))
    meta = tk.tokenize_instance(inst, chunk=10, log=lambda *a, **k: None)
    tok, ln, v, meta2 = tk.load_tokens(inst / "tokens")
    assert tok.shape == (N, T) and tok.dtype == np.int16 and (ln == T).all()
    assert np.array_equal(tk.decode(np.asarray(tok), v), train)   # train round-trips through the file
    assert meta["frames_only_outside_train"] >= 1 and meta2["vocab_size"] == v.size
    ed = np.load(inst / "tokens" / "edits.npy")
    assert ed.shape == (5, T) and (ed > 0).all()                  # the eval-only frame IS in the vocab
    assert np.array_equal(tk.decode(ed[0, 0], v), eval_only[0])
