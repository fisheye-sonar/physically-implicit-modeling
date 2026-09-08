"""`benchmark_from_cases` must reproduce the shipped benchmark exactly.

The refactor (2026-09-02) that made the Benchmark constructible from any case list — so
synthesised cases at every move number go through the identical construction — moved the
target computation (`case_targets`) off the pkl and onto the dataclass. This pins both to
the previous, pkl-reading implementation.
"""
import pickle

import numpy as np

from pim.environments.othello.bench import (
    BENCHMARK_PKL, MINE, THEIRS, benchmark_from_cases, case_targets, load_benchmark)
from pim.environments.othello.vendor.othello import OthelloBoardState


def _legacy_case_targets(cases):
    cur = np.zeros(len(cases), np.int64)
    for i, c in enumerate(cases):
        b = OthelloBoardState()
        b.update(c["history"], prt=False)
        nxt = 2 if b.next_hand_color > 0 else 0
        cur[i] = MINE if c["ori_color"] == nxt else THEIRS
    return cur, np.where(cur == MINE, THEIRS, MINE)


def test_load_benchmark_is_benchmark_from_cases_on_the_pkl():
    with open(BENCHMARK_PKL, "rb") as f:
        cases = pickle.load(f)
    a, b = load_benchmark(), benchmark_from_cases(cases)
    assert a.n_cases == b.n_cases == 1001
    assert np.array_equal(a.pos_int, b.pos_int) and np.array_equal(a.new_class, b.new_class)
    assert a.legal_pre == b.legal_pre and a.legal_post == b.legal_post
    assert all(np.array_equal(x, y) for x, y in zip(a.tokens, b.tokens))
    assert all(np.array_equal(x, y) for x, y in zip(a.case_ids, b.case_ids))
    assert sum(len(i) for i in a.case_ids) == 1001


def test_case_targets_match_legacy_pkl_implementation():
    with open(BENCHMARK_PKL, "rb") as f:
        cases = pickle.load(f)
    bench = load_benchmark()
    cur, tgt = case_targets(bench)
    cur0, tgt0 = _legacy_case_targets(cases)
    assert np.array_equal(cur, cur0) and np.array_equal(tgt, tgt0)
    assert set(np.unique(cur)) <= {MINE, THEIRS} and np.all(tgt != cur)


def test_benchmark_from_synthetic_case_buckets_by_length():
    with open(BENCHMARK_PKL, "rb") as f:
        cases = pickle.load(f)
    sub = [c for c in cases if len(c["history"]) in (7, 12)][:20]
    b = benchmark_from_cases(sub)
    assert [t.shape[1] for t in b.tokens] == [7, 12]
    assert b.n_cases == len(sub) == sum(len(i) for i in b.case_ids)
    assert len(b.cur_lab) == len(b.tgt_lab) == len(sub)
