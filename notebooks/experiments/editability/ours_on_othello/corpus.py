"""Synthetic Othello corpora, generated once and sliced into a NESTED scale ladder.

Every game comes from **their** generator (`data.othello.get_ood_game`, reached through
`othello_transfer/othello_data._one_game`), which picks `random.choice(possible_next_steps)`
— uniform over the legal set. That is what makes `edit_index`'s uniform-over-legal reference
the true conditional distribution of this data rather than an approximation of it.

Determinism and disjointness
----------------------------
`_one_game((i, seed))` seeds itself with `seed * 1_000_003 + i`, so a game is a pure
function of its **index**. Splits are therefore carved out as disjoint index ranges of one
seed rather than as different seeds, and the ladder rungs are strict prefixes of one
another — `L1 ⊂ L2 ⊂ D`, not four independent draws:

    [0, 20_000_000)                     training pool (M/L1/L2/D/F take prefixes)
    [90_000_000, 90_010_000)            held-out test — the OOD generalisation gate
    [91_000_000, 91_020_000)            probe harvest
    their `intervention_benchmark.pkl`  the 1001 intervention cases (checked into their repo)

⛔ The analogous mistake is already pinned in `directions/othello-architecture-on-discworld.md`:
generating an eval split from an index range the training corpus also covers silently turns
held-out data into training data. `assert_disjoint` hashes the actual token rows and fails
hard rather than trusting the arithmetic.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
for _p in (str(_HERE), str(_HERE.parent / "othello_transfer"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import othello_data as od  # noqa: E402

SEED = 0
BLOCK = 59
MAXLEN = 60
CACHE = _REPO / "runs" / "ours_on_othello" / "corpus"

TRAIN_LO = 0
TEST_LO, TEST_N = 90_000_000, 10_000
PROBE_LO, PROBE_N = 91_000_000, 20_000

# The ladder. Every rung runs the SAME number of optimiser steps; only the pool differs.
LADDER = {"M": 90_000, "L1": 1_000_000, "L2": 5_000_000, "D": 20_000_000}


def _games(lo: int, n: int, n_workers: int | None = None) -> list[list[int]]:
    """`n` games at indices [lo, lo+n), through their generator. Small `n` only."""
    n_workers = n_workers or multiprocessing.cpu_count()
    args = [(i, SEED) for i in range(lo, lo + n)]
    with multiprocessing.Pool(n_workers) as pool:
        return list(pool.imap(od._one_game, args, chunksize=256))


def _generate(lo: int, n: int, chunk: int = 500_000, n_workers: int | None = None,
              log=print) -> tuple[np.ndarray, np.ndarray]:
    """Tokenise as we go, in chunks, straight into a preallocated array.

    ⛔ Do not materialise the games first. A Python `list[list[int]]` of 20M games is
    ~33 GB (60 ints x 28 bytes each, plus list overhead) and will OOM a 59 GB box. Held as
    int8 tokens the same corpus is **1.2 GB**, so the only requirement is that the
    conversion happens chunk by chunk rather than at the end.
    """
    n_workers = n_workers or multiprocessing.cpu_count()
    stoi = od.canonical_vocab()
    tok = np.zeros((n, MAXLEN), np.int8)
    ln = np.zeros(n, np.int8)
    t0 = time.time()
    with multiprocessing.Pool(n_workers) as pool:
        for c0 in range(0, n, chunk):
            c1 = min(c0 + chunk, n)
            args = [(i, SEED) for i in range(lo + c0, lo + c1)]
            for j, g in enumerate(pool.imap(od._one_game, args, chunksize=256)):
                m = g[:MAXLEN]
                ln[c0 + j] = len(m)
                tok[c0 + j, : len(m)] = [stoi[s] for s in m]
            el = time.time() - t0
            log(f"    {c1:>10,}/{n:,}  {c1 / el:>8,.0f} games/s  "
                f"eta {(n - c1) / max(c1 / el, 1) / 60:5.1f} min")
    return tok, ln


def to_tokens(games: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """(N, 60) int8 tokens padded with 0, and (N,) int8 lengths.

    int8 is safe: the vocabulary is 0…60. 20M games is 1.2 GB.
    """
    stoi = od.canonical_vocab()
    n = len(games)
    tok = np.zeros((n, MAXLEN), np.int8)
    ln = np.zeros(n, np.int8)
    for i, g in enumerate(games):
        m = g[:MAXLEN]
        ln[i] = len(m)
        tok[i, : len(m)] = [stoi[s] for s in m]
    return tok, ln


def _row_hashes(tok: np.ndarray) -> set[bytes]:
    return {hashlib.blake2b(r.tobytes(), digest_size=8).digest() for r in tok}


def assert_disjoint(**splits: np.ndarray) -> None:
    """Hard-fail if any two named splits share an identical game."""
    hashes = {k: _row_hashes(v) for k, v in splits.items()}
    names = list(hashes)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = hashes[a] & hashes[b]
            if common:
                raise AssertionError(
                    f"{len(common)} identical games shared between {a!r} and {b!r} — "
                    "the index ranges overlap, or the generator is not index-seeded."
                )
    print("  ✓ splits are pairwise disjoint (" + ", ".join(f"{k} {len(v):,}"
                                                           for k, v in hashes.items()) + ")")


def build(n_train: int = LADDER["D"], log=print, only: tuple[str, ...] | None = None
          ) -> dict[str, Path]:
    """Generate (or reuse) the named splits. Returns their paths.

    Measured throughput is **~4.7k games/s on 32 cores**, so 20M takes ~70 min — not the
    ~11 min first estimated from `othello_data.synthetic_games`, whose 1,487 games/s was
    *already* a 32-core number rather than a single-core one (my error, 2026-08-21).
    Generation is CPU-only and training is GPU-only, so the two are overlapped rather than
    serialised: each rung generates exactly the pool it needs (`train_<n>.npz`, all prefixes
    of index range [0, n), so nesting is preserved), and the 20M pool is built in the
    background while the smaller rungs train.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    plan = [("train", TRAIN_LO, n_train), ("test", TEST_LO, TEST_N), ("probe", PROBE_LO, PROBE_N)]
    if only is not None:
        plan = [x for x in plan if x[0] in only]
    for name, lo, n in plan:
        p = CACHE / f"{name}_{n}.npz"
        out[name] = p
        if p.exists():
            log(f"  {name:<6} {n:>10,} games — cached")
            continue
        # A larger pool at the same `lo` already contains this one as a prefix, so reuse it
        # rather than regenerating. This is what keeps the ladder nested AND cheap: once the
        # 20M pool exists, every smaller rung slices it instead of paying generation again.
        bigger = sorted((q for q in CACHE.glob(f"{name}_*.npz")
                         if q.stem.split("_")[-1].isdigit()
                         and int(q.stem.split("_")[-1]) >= n),
                        key=lambda q: int(q.stem.split("_")[-1]))
        if bigger:
            out[name] = bigger[0]
            log(f"  {name:<6} {n:>10,} games — prefix of {bigger[0].name}")
            continue
        t0 = time.time()
        tok, ln = _generate(lo, n, log=log)
        np.savez(p, tokens=tok, lengths=ln, lo=lo, seed=SEED)
        log(f"  {name:<6} {n:>10,} games in {time.time() - t0:6.1f}s  "
            f"({n / (time.time() - t0):,.0f}/s, {p.stat().st_size / 1e6:.0f} MB)  "
            f"mean length {ln.mean():.1f}")
    return out


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["tokens"], z["lengths"]


def rung(train_path: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """The nested prefix for one ladder rung — `L1 ⊂ L2 ⊂ D` by construction."""
    tok, ln = load(train_path)
    n = LADDER[name]
    if n > len(tok):
        raise ValueError(f"rung {name} wants {n:,} games, corpus has {len(tok):,}")
    return tok[:n], ln[:n]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else LADDER["D"]
    only = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    print(f"generating corpora (train pool {n:,}"
          f"{', splits ' + ','.join(only) if only else ''})", flush=True)
    paths = build(n, only=only)
    if {"train", "test", "probe"} <= set(paths):
        tr, _ = load(paths["train"])
        te, _ = load(paths["test"])
        pr, _ = load(paths["probe"])
        assert_disjoint(train=tr[: min(len(tr), 200_000)], test=te, probe=pr)
