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

from pim.environments.othello.data import BLANK, CENTRE, MINE, THEIRS, canonical_vocab  # noqa: F401
from pim.environments.othello.vendor.othello import OthelloBoardState

BENCHMARK_PKL = Path(__file__).parent / "vendor" / "intervention_benchmark.pkl"
_REPO = Path(__file__).resolve().parents[3]


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


def benchmark_from_cases(cases: list[dict], flip: bool = True, placement: str = "enclosure") -> Benchmark:
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
        pre = OthelloBoardState(flip=flip, placement=placement)
        pre.update(c["history"], prt=False)
        legal_pre.append(sorted(pre.get_valid_moves()))
        # The benchmark flips absolute colour. The player to move does not change, so
        # in the mover's frame the tile reads MINE iff its colour is the next hand's,
        # and the flip is exactly MINE<->THEIRS.
        nxt = 2 if pre.next_hand_color > 0 else 0
        cur[i] = MINE if c["ori_color"] == nxt else THEIRS
        post = OthelloBoardState(flip=flip, placement=placement)
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


def cases_path(instance: str) -> Path:
    """Where an instance's intervention cases live: ``edits/v1/cases_1001.pkl`` under the
    instance's dataset dir for EVERY instance (layout v2, 2026-09-10 — Li's shipped set is
    copied there for oth-uniform, cmp-verified). Before the copy exists (a fresh clone
    without datasets/, or layout v1) oth-uniform falls back to the vendored pkl in git."""
    from pim.environments.layout import othello_cases_file

    p = othello_cases_file(instance)
    if instance == "oth-uniform" and not p.exists():
        return BENCHMARK_PKL
    return p


def load_benchmark(instance: str = "oth-uniform") -> Benchmark:
    """The instance's 1001 intervention cases, grouped into equal-length buckets: Li et
    al.'s shipped set for oth-uniform, the synthesised set for any other instance, both
    replayed with the instance's rules."""
    from pim.environments.othello.corpus import rules_of

    with open(cases_path(instance), "rb") as f:
        return benchmark_from_cases(pickle.load(f), **rules_of(instance))


def shipped_length_distribution() -> dict[int, int]:
    """History length -> count over Li et al.'s 1001 cases (prefixes of 5-30 moves)."""
    with open(BENCHMARK_PKL, "rb") as f:
        cases = pickle.load(f)
    out: dict[int, int] = {}
    for c in cases:
        out[len(c["history"])] = out.get(len(c["history"]), 0) + 1
    return out


# (the four CENTRE squares are never flipped in the shipped benchmark: 0/1001)


def synthesise_cases(histories: list[list[int]], n: int, length_counts: dict[int, int],
                     seed: int = 0, flip: bool = True, placement: str = "enclosure",
                     log=print) -> tuple[list[dict], dict]:
    """Li-style intervention cases from held-out games, matching a prefix-length distribution.

    The recipe measured on the shipped 1001 (2026-09-02) and used by
    ``experiments/othello_by_step/edit``: a real game prefix plus ONE occupied, non-centre
    square flipped to the opposite colour, rejected if the flip leaves the legal set
    unchanged or empties it. ``length_counts`` (e.g. ``shipped_length_distribution()``) fixes
    how many cases each prefix length gets, so a synthesised bench has Li's own position
    mix; ``n`` rescales it. Returns the cases in the pkl's format plus a manifest.
    """
    rng = np.random.default_rng(seed)
    tot = sum(length_counts.values())
    quota = {L: int(round(n * c / tot)) for L, c in sorted(length_counts.items())}
    cases, stats = [], {}
    for L, want in quota.items():
        pool = np.array([i for i, h in enumerate(histories) if len(h) > L])
        got = tried = rej_same = rej_empty = 0
        for g in rng.permutation(pool):
            if got >= want:
                break
            tried += 1
            h = list(histories[g][:L])
            board = OthelloBoardState(flip=flip, placement=placement)
            board.update(h, prt=False)
            pre = sorted(board.get_valid_moves())
            if not pre:
                continue
            occ = [sq for sq in range(64) if board.state[sq // 8, sq % 8] != 0 and sq not in CENTRE]
            for sq in rng.permutation(occ):
                sq = int(sq)
                ori = 0.0 if board.state[sq // 8, sq % 8] < 0 else 2.0
                post = OthelloBoardState(flip=flip, placement=placement)
                post.update(h, prt=False)
                post.state[sq // 8, sq % 8] = int(2 - ori) - 1
                legal_post = sorted(post.get_valid_moves())
                if not legal_post:
                    rej_empty += 1
                    continue
                if legal_post == pre:
                    rej_same += 1
                    continue
                cases.append({"history": h, "pos_int": sq, "ori_color": ori, "game": int(g)})
                got += 1
                break
        stats[int(L)] = {"want": want, "n": got, "games_tried": tried, "pool": int(len(pool)),
                         "rejected_same_legal": rej_same, "rejected_empty_legal": rej_empty}
        if log:
            log(f"  prefix {L:2d}: {got}/{want} cases from {tried} games "
                f"(rejected same-legal {rej_same}, empty {rej_empty})", flush=True)
    manifest = {"n_cases": len(cases), "seed": seed, "flip": flip, "placement": placement,
                "length_quota": quota,
                "recipe": "prefix of a held-out game; one uniformly random occupied non-centre "
                          "square flipped to the opposite colour; rejected if the legal set is "
                          "unchanged or empty; prefix lengths follow the shipped 1001",
                "stats": stats}
    return cases, manifest


def case_targets(bench: Benchmark) -> tuple[np.ndarray, np.ndarray]:
    """Per case: the intervened tile's CURRENT and TARGET label in mine/theirs coordinates.

    Computed once, in ``benchmark_from_cases``; kept as a function so every caller reads
    the same two arrays (and so a Benchmark built from any case set serves them).
    """
    return bench.cur_lab, bench.tgt_lab
