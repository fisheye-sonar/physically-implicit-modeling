"""Synthetic Othello corpora, generated once and sliced into a NESTED scale ladder.

Ported 2026-08-31 from ``ours_on_othello/corpus.py``; the only change is where games
come from (``pim.environments.othello.data``, backed by the vendored generator) and
where the cache lives (inside the environment instance's dataset directory).

Every game comes from **their** generator, which picks uniformly over the legal set —
that is what makes the Edit Index's uniform-over-legal reference the true conditional
distribution of this data rather than an approximation of it.

Determinism and disjointness
----------------------------
``data._one_game((i, seed))`` seeds itself with ``seed * 1_000_003 + i``, so a game is a
pure function of its **index**. Splits are therefore carved out as disjoint index ranges
of one seed rather than as different seeds, and ladder rungs are strict prefixes of one
another — anything that moves across the ladder is data diversity, not compute:

    [0, 20_000_000)              training pool (rungs take prefixes)
    [90_000_000, 90_010_000)     held-out test — the OOD generalisation gate
    [91_000_000, 91_020_000)     probe harvest
    vendor/intervention_benchmark.pkl   the 1001 intervention cases (theirs, shipped)

⛔ Generating an eval split from an index range the training corpus also covers silently
turns held-out data into training data. ``verify_splits`` checks the two things that can
actually go wrong: that the declared ``[lo, lo+n)`` ranges are disjoint, and that games
REGENERATED at sampled indices are bit-identical to the stored rows (so the recorded
``lo``/``seed``/``flip`` describe the file and the generator is a pure function of its
index — the 2026-08-20 pid-seeding bug is exactly what that catches). Row-content identity
is deliberately NOT the criterion: 9-move games recur by chance, so 6 of the probe split's
20k games also appear somewhere in the 20M train pool without any index overlap.
"""

from __future__ import annotations

import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

from pim.environments.othello import data as od

SEED = 0
BLOCK, MAXLEN = od.T_MODEL, od.MAXLEN      # 59 / 60, defined once in data.py
# The environment instances. `datasets/` is resolved against the repo root (= CWD by repo
# convention, same as every loader here). `flip` is the ONE rule that differs: oth-noflip
# (2026-09-06) never recolours enclosed discs — same legality, passes, game end, index law.
# `placement` (2026-09-08) is the second rule: "enclosure" = Othello's; "adjacent" =
# oth-adjacent, a move must touch one of the mover's own discs (8-neighbourhood), nothing is
# recoloured — colour is causally relevant WITHOUT the enclosure geometry.
INSTANCES = {
    "oth-uniform": {"dir": Path("datasets/othello/oth-uniform/corpus"), "flip": True, "placement": "enclosure"},
    "oth-noflip": {"dir": Path("datasets/othello/oth-noflip/corpus"), "flip": False, "placement": "enclosure"},
    "oth-adjacent": {"dir": Path("datasets/othello/oth-adjacent/corpus"), "flip": False, "placement": "adjacent"},
}
CACHE = INSTANCES["oth-uniform"]["dir"]          # the canonical instance, unchanged callers


def corpus_dir(instance: str = "oth-uniform") -> Path:
    return INSTANCES[instance]["dir"]


def flip_of(instance: str = "oth-uniform") -> bool:
    """The recolouring rule of an instance — every replay (labels, legal sets, bench) must use it."""
    return INSTANCES[instance]["flip"]


def placement_of(instance: str = "oth-uniform") -> str:
    """The placement rule of an instance: "enclosure" | "adjacent"."""
    return INSTANCES[instance].get("placement", "enclosure")


def rules_of(instance: str = "oth-uniform") -> dict:
    """Both rules as keyword arguments — ``OthelloBoardState(**rules_of(inst))`` and every
    replaying helper (``tokens_and_labels``, ``legal_sets``, ``synthesise_cases``, …)."""
    return {"flip": flip_of(instance), "placement": placement_of(instance)}

TRAIN_LO = 0
TEST_LO, TEST_N = 90_000_000, 10_000
PROBE_LO, PROBE_N = 91_000_000, 20_000
# probe_large: the probe-CAPACITY sweep corpus (2026-09-02) — 170k games ≈ 4.7M positions,
# 5x the canonical probe rows, so wide probes are data-limited by width, not memorisation.
# A fresh disjoint index range; the canonical probe split is untouched.
PROBE_LARGE_LO, PROBE_LARGE_N = 92_000_000, 170_000

# The ladder. Every rung runs the SAME number of optimiser steps; only the pool differs.
LADDER = {"M": 90_000, "L1": 1_000_000, "L2": 5_000_000, "D": 20_000_000}


def _generate(lo: int, n: int, chunk: int = 500_000, n_workers: int | None = None,
              log=print, flip: bool = True, placement: str = "enclosure") -> tuple[np.ndarray, np.ndarray]:
    """Tokenise as we go, in chunks, straight into a preallocated array.

    ⛔ Do not materialise the games first. A Python ``list[list[int]]`` of 20M games is
    ~33 GB and will OOM a 59 GB box. Held as int8 tokens the same corpus is **1.2 GB**,
    so the only requirement is that the conversion happens chunk by chunk.
    """
    n_workers = n_workers or multiprocessing.cpu_count()
    stoi = od.canonical_vocab()
    tok = np.zeros((n, MAXLEN), np.int8)
    ln = np.zeros(n, np.int8)
    t0 = time.time()
    with multiprocessing.Pool(n_workers) as pool:
        for c0 in range(0, n, chunk):
            c1 = min(c0 + chunk, n)
            args = [(i, SEED, flip, placement) for i in range(lo + c0, lo + c1)]
            for j, g in enumerate(pool.imap(od._one_game, args, chunksize=256)):
                m = g[:MAXLEN]
                ln[c0 + j] = len(m)
                tok[c0 + j, : len(m)] = [stoi[s] for s in m]
            el = time.time() - t0
            log(f"    {c1:>10,}/{n:,}  {c1 / el:>8,.0f} games/s  "
                f"eta {(n - c1) / max(c1 / el, 1) / 60:5.1f} min")
    return tok, ln


def _regen_row(index: int, seed: int, flip: bool, stoi: dict,
               placement: str = "enclosure") -> tuple[np.ndarray, int]:
    g = od._one_game((index, seed, flip, placement))[:MAXLEN]
    row = np.zeros(MAXLEN, np.int8)
    row[: len(g)] = [stoi[s] for s in g]
    return row, len(g)


def verify_splits(paths: dict[str, Path], n_check: int = 8, log=print) -> dict[str, tuple[int, int]]:
    """THE provenance check for a set of split files (see the module docstring).

    1. The declared index ranges ``[lo, lo + n)`` are pairwise disjoint — the property that
       makes held-out data held out.
    2. At ``n_check`` sampled indices per split (always including the first and last row),
       the game regenerated from the recorded ``lo``/``seed``/``flip`` equals the stored
       row bit for bit — so the metadata describes the file and the generator is
       index-seeded. Raises ``AssertionError`` on either failure; returns the ranges.
    """
    stoi = od.canonical_vocab()
    rng = np.random.default_rng(0)
    ranges: dict[str, tuple[int, int]] = {}
    for name, p in paths.items():
        z = np.load(p)
        tok, ln, lo, seed = z["tokens"], z["lengths"], int(z["lo"]), int(z["seed"])
        flip = bool(z["flip"]) if "flip" in z.files else True   # pre-2026-09-06 files
        placement = str(z["placement"]) if "placement" in z.files else "enclosure"   # pre-2026-09-08
        n = len(tok)
        ranges[name] = (lo, lo + n)
        idx = sorted({0, n - 1, *rng.integers(0, n, n_check).tolist()})
        for j in idx:
            row, length = _regen_row(lo + j, seed, flip, stoi, placement)
            assert np.array_equal(row, tok[j]) and int(ln[j]) == length, (
                f"{name}: stored row {j} is not the game at index {lo + j} (seed {seed}, "
                f"flip {flip}) — the recorded lo/seed/flip do not describe {p.name}")
    names = list(ranges)
    for i, a_ in enumerate(names):
        for b_ in names[i + 1:]:
            (a0, a1), (b0, b1) = ranges[a_], ranges[b_]
            assert a1 <= b0 or b1 <= a0, (
                f"index ranges overlap: {a_} [{a0:,}, {a1:,}) vs {b_} [{b0:,}, {b1:,}) — "
                f"held-out data would be training data")
    if log:
        log("  ✓ splits verified: " + ", ".join(f"{k} [{lo:,}, {hi:,})" for k, (lo, hi) in ranges.items())
            + f"; {n_check}+2 regenerated rows per split bit-identical")
    return ranges


def build(n_train: int = LADDER["D"], log=print, only: tuple[str, ...] | None = None,
          instance: str = "oth-uniform") -> dict[str, Path]:
    """Generate (or reuse) the named splits. Returns their paths.

    Measured throughput is **~4.7k games/s on 32 cores**, so 20M takes ~70 min.
    Generation is CPU-only, so it can overlap GPU training rather than serialise it.
    """
    cache, flip, placement = corpus_dir(instance), flip_of(instance), placement_of(instance)
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    plan = [("train", TRAIN_LO, n_train), ("test", TEST_LO, TEST_N), ("probe", PROBE_LO, PROBE_N),
            ("probe_large", PROBE_LARGE_LO, PROBE_LARGE_N)]
    if only is not None:
        plan = [x for x in plan if x[0] in only]
    for name, lo, n in plan:
        p = cache / f"{name}_{n}.npz"
        out[name] = p
        if p.exists():
            log(f"  {name:<6} {n:>10,} games — cached")
            continue
        # A larger pool at the same `lo` already contains this one as a prefix, so reuse
        # it rather than regenerating — that keeps the ladder nested AND cheap.
        bigger = sorted((q for q in cache.glob(f"{name}_*.npz")
                         if q.stem.split("_")[-1].isdigit()
                         and int(q.stem.split("_")[-1]) >= n),
                        key=lambda q: int(q.stem.split("_")[-1]))
        if bigger:
            out[name] = bigger[0]
            log(f"  {name:<6} {n:>10,} games — prefix of {bigger[0].name}")
            continue
        t0 = time.time()
        tok, ln = _generate(lo, n, log=log, flip=flip, placement=placement)
        np.savez(p, tokens=tok, lengths=ln, lo=lo, seed=SEED, flip=flip, placement=placement,
                 instance=instance)
        log(f"  {name:<6} {n:>10,} games in {time.time() - t0:6.1f}s  "
            f"({n / (time.time() - t0):,.0f}/s, {p.stat().st_size / 1e6:.0f} MB)  "
            f"mean length {ln.mean():.1f}")
    return out


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["tokens"], z["lengths"]


def rung(train_path: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """The nested prefix for one ladder rung — ``L1 ⊂ L2 ⊂ D`` by construction."""
    tok, ln = load(train_path)
    n = LADDER[name]
    if n > len(tok):
        raise ValueError(f"rung {name} wants {n:,} games, corpus has {len(tok):,}")
    return tok[:n], ln[:n]


def probe_data(path: Path, n: int | None = None, flip: bool = True, placement: str = "enclosure"):
    """``tokens_and_labels`` for a corpus split, CACHED beside it as ``<stem>_labels.npz``.

    Labelling replays every game through the board simulator (~10 min for 170k games), so
    the result is a corpus artefact, not something to recompute per caller.
    """
    import dataclasses

    from pim.environments.othello.data import ProbeData, canonical_vocab, tokens_and_labels

    path = Path(path)
    tok, ln = load(path)
    n = len(tok) if n is None else min(n, len(tok))
    cache = path.with_name(f"{path.stem}_labels_{n}.npz")
    if cache.exists():
        z = np.load(cache)
        return ProbeData(**{k: z[k] for k in z.files})
    itos = {v: k for k, v in canonical_vocab().items()}
    data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(tok[:n], ln[:n])],
                             flip=flip, placement=placement)
    np.savez(cache, **{f.name: getattr(data, f.name) for f in dataclasses.fields(data)})
    return data


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else LADDER["D"]
    only = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else None
    instance = sys.argv[3] if len(sys.argv) > 3 else "oth-uniform"
    print(f"generating corpora for {instance} (train pool {n:,}"
          f"{', splits ' + ','.join(only) if only else ''})", flush=True)
    paths = build(n, only=only, instance=instance)
    verify_splits(paths)
