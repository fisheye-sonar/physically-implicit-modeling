"""The Othello editability bench: Li et al.'s shipped 1001 intervention cases.

Each case is a real game history plus one board square whose colour the editor must flip
(their §4.1 benchmark, ``vendor/intervention_benchmark.pkl``). Ported 2026-08-31 from
``othello_transfer/othello_data.py`` (Benchmark/load_benchmark) and
``othello_transfer/linear_intervention.py`` (case_targets) — the latter previously
existed twice, restated verbatim in ``ours_on_othello/evaluate.py``; this is now the one
copy.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pim.environments.othello.data import canonical_vocab
from pim.environments.othello.vendor.othello import OthelloBoardState

BENCHMARK_PKL = Path(__file__).parent / "vendor" / "intervention_benchmark.pkl"

# mine/theirs label encoding, shared with `data.tokens_and_labels`
BLANK, MINE, THEIRS = 0, 1, 2


@dataclass
class Benchmark:
    tokens: list[np.ndarray]  # per bucket, (B, L) int64
    case_ids: list[np.ndarray]  # per bucket, indices into the 1001
    pos_int: np.ndarray  # (1001,) intervened square
    new_class: np.ndarray  # (1001,) requested class, their `2 - ori_color`
    legal_pre: list[list[int]]
    legal_post: list[list[int]]

    @property
    def n_cases(self) -> int:
        return len(self.pos_int)


def load_benchmark() -> Benchmark:
    """The 1001 cases, grouped into equal-length buckets.

    Buckets exist for one reason: the gradient-steering hook writes ``x[:, -1]``, so
    every row in a batch must have its last real move at the same index. Grouping by
    history length keeps that true without padding and without touching the hook.
    """
    with open(BENCHMARK_PKL, "rb") as f:
        ds = pickle.load(f)
    stoi = canonical_vocab()

    pos_int = np.array([c["pos_int"] for c in ds], int)
    new_class = np.array([int(2 - c["ori_color"]) for c in ds], int)
    legal_pre, legal_post = [], []
    for c, sq, new in zip(ds, pos_int, new_class):
        pre = OthelloBoardState()
        pre.update(c["history"], prt=False)
        legal_pre.append(sorted(pre.get_valid_moves()))
        post = OthelloBoardState()
        post.update(c["history"], prt=False)
        post.state[sq // 8, sq % 8] = new - 1
        legal_post.append(sorted(post.get_valid_moves()))

    by_len: dict[int, list[int]] = {}
    for i, c in enumerate(ds):
        by_len.setdefault(len(c["history"]), []).append(i)
    toks, ids = [], []
    for L in sorted(by_len):
        members = np.array(by_len[L], int)
        ids.append(members)
        toks.append(np.array([[stoi[s] for s in ds[i]["history"]] for i in members], np.int64))
    return Benchmark(toks, ids, pos_int, new_class, legal_pre, legal_post)


def case_targets(bench: Benchmark) -> tuple[np.ndarray, np.ndarray]:
    """Per case: the intervened tile's CURRENT and TARGET label in mine/theirs coordinates.

    The benchmark flips absolute colour. The player to move does not change, so flipping
    colour flips MINE↔THEIRS exactly.
    """
    with open(BENCHMARK_PKL, "rb") as f:
        ds = pickle.load(f)
    cur = np.zeros(bench.n_cases, np.int64)
    for i, c in enumerate(ds):
        b = OthelloBoardState()
        b.update(c["history"], prt=False)
        nxt = 2 if b.next_hand_color > 0 else 0
        cur[i] = MINE if c["ori_color"] == nxt else THEIRS
    tgt = np.where(cur == MINE, THEIRS, MINE)
    return cur, tgt
