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
    cur_lab: np.ndarray  # (N,) the intervened tile's CURRENT label, mine/theirs frame
    tgt_lab: np.ndarray  # (N,) the label the edit asks for (the flip of cur_lab)

    @property
    def n_cases(self) -> int:
        return len(self.pos_int)


def benchmark_from_cases(cases: list[dict]) -> Benchmark:
    """A Benchmark from ``{history, pos_int, ori_color}`` cases — the pkl's own format.

    ``load_benchmark`` is this applied to the shipped 1001; a synthesised case set
    (``experiments/othello_edit_by_step``, every move number 1–59) goes through the same
    construction, so bucketing, legal sets and targets cannot differ between the two.

    Buckets exist for one reason: the gradient-steering hook writes ``x[:, -1]``, so
    every row in a batch must have its last real move at the same index. Grouping by
    history length keeps that true without padding and without touching the hook.
    """
    stoi = canonical_vocab()

    pos_int = np.array([c["pos_int"] for c in cases], int)
    new_class = np.array([int(2 - c["ori_color"]) for c in cases], int)
    legal_pre, legal_post = [], []
    cur = np.zeros(len(cases), np.int64)
    for i, (c, sq, new) in enumerate(zip(cases, pos_int, new_class)):
        pre = OthelloBoardState()
        pre.update(c["history"], prt=False)
        legal_pre.append(sorted(pre.get_valid_moves()))
        # The benchmark flips absolute colour. The player to move does not change, so
        # in the mover's frame the tile reads MINE iff its colour is the next hand's,
        # and the flip is exactly MINE<->THEIRS.
        nxt = 2 if pre.next_hand_color > 0 else 0
        cur[i] = MINE if c["ori_color"] == nxt else THEIRS
        post = OthelloBoardState()
        post.update(c["history"], prt=False)
        post.state[sq // 8, sq % 8] = new - 1
        legal_post.append(sorted(post.get_valid_moves()))
    tgt = np.where(cur == MINE, THEIRS, MINE)

    by_len: dict[int, list[int]] = {}
    for i, c in enumerate(cases):
        by_len.setdefault(len(c["history"]), []).append(i)
    toks, ids = [], []
    for L in sorted(by_len):
        members = np.array(by_len[L], int)
        ids.append(members)
        toks.append(np.array([[stoi[s] for s in cases[i]["history"]] for i in members],
                             np.int64))
    return Benchmark(toks, ids, pos_int, new_class, legal_pre, legal_post, cur, tgt)


def load_benchmark() -> Benchmark:
    """Li et al.'s shipped 1001 cases, grouped into equal-length buckets."""
    with open(BENCHMARK_PKL, "rb") as f:
        return benchmark_from_cases(pickle.load(f))


def case_targets(bench: Benchmark) -> tuple[np.ndarray, np.ndarray]:
    """Per case: the intervened tile's CURRENT and TARGET label in mine/theirs coordinates.

    Computed once, in ``benchmark_from_cases``; kept as a function so every caller reads
    the same two arrays (and so a Benchmark built from any case set serves them).
    """
    return bench.cur_lab, bench.tgt_lab
