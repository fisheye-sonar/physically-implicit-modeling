"""``corpus.verify_splits`` — the Othello split provenance check (2026-09-07).

Replaces the row-hash ``assert_disjoint``, which (a) ran on a 200k slice of the pool and
(b) used content identity as its criterion, so it FAILED on a correct corpus (short games
recur by chance). The check that matters is index provenance: regenerate, compare.
"""

from __future__ import annotations

import numpy as np
import pytest

from pim.environments.othello import corpus as oc
from pim.environments.othello import data as od


def _write_split(path, lo, n, seed=0, flip=True, tamper_lo=None):
    stoi = od.canonical_vocab()
    tok = np.zeros((n, oc.MAXLEN), np.int8)
    ln = np.zeros(n, np.int8)
    for j in range(n):
        g = od._one_game((lo + j, seed, flip))[: oc.MAXLEN]
        ln[j] = len(g)
        tok[j, : len(g)] = [stoi[s] for s in g]
    np.savez(path, tokens=tok, lengths=ln, lo=lo if tamper_lo is None else tamper_lo,
             seed=seed, flip=flip)
    return path


def test_verify_splits_passes_on_index_seeded_disjoint_files(tmp_path):
    paths = {"train": _write_split(tmp_path / "train_6.npz", 100, 6),
             "test": _write_split(tmp_path / "test_4.npz", 200, 4)}
    assert oc.verify_splits(paths, n_check=3, log=None) == {"train": (100, 106), "test": (200, 204)}


def test_verify_splits_catches_a_wrong_recorded_lo(tmp_path):
    paths = {"train": _write_split(tmp_path / "train_6.npz", 100, 6, tamper_lo=101)}
    with pytest.raises(AssertionError, match="not the game at index"):
        oc.verify_splits(paths, n_check=3, log=None)


def test_verify_splits_catches_overlapping_ranges(tmp_path):
    paths = {"train": _write_split(tmp_path / "train_6.npz", 100, 6),
             "test": _write_split(tmp_path / "test_4.npz", 103, 4)}
    with pytest.raises(AssertionError, match="index ranges overlap"):
        oc.verify_splits(paths, n_check=3, log=None)


def test_verify_splits_honours_the_no_flip_rules(tmp_path):
    paths = {"train": _write_split(tmp_path / "train_4.npz", 50, 4, flip=False)}
    oc.verify_splits(paths, n_check=2, log=None)
