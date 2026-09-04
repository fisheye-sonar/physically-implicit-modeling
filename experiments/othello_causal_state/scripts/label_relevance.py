#!/usr/bin/env python
"""Which discs does the game still depend on? Per-square relevance flags for Othello.

For every position of N held-out (TEST-split) games, and every occupied square q, two
exact flags computed with the vendored rules (`OthelloBoardState`):

  dead        q lies on NO ray (row, column, either diagonal, any distance) from ANY
              empty square. Legality and flips at an empty square e depend only on the
              discs along the 8 rays from e; empties only ever shrink and rays are fixed,
              so such a disc can never influence any future legal set or flip, whatever
              is played. Sufficient, not necessary — conservative.
  irrelevant  flipping q's colour leaves the EFFECTIVE legal set (the generator's
              `get_valid_moves`, pass handling included) unchanged. Under uniform-random
              legal play the next-move distribution depends only on that set, so q has
              exactly zero influence on the model's target at this position. dead ⊂
              irrelevant.

Plus controls per square: `frontier` (8-adjacent to an empty), `age` (moves since
placed), `since_flip` (moves since the colour last changed), `n_flips`, `dist_empty`
(Chebyshev distance to the nearest empty), and the probe target `mine` (blank 0 / mine 1
/ theirs 2, exactly `tokens_and_labels`' convention: "mine" = the player about to move).

Position t = the board after move t (0-based), the position the model reads at token t
and the probes are labelled on. Output: scores/relevance_<name>.npz (int8/int16 arrays
of shape (N, 59, 64) + tokens, mask, lengths, game ids) and a JSON manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.data import T_MODEL, canonical_vocab  # noqa: E402
from pim.environments.othello.vendor.othello import OthelloBoardState  # noqa: E402

EXP = REPO / "experiments" / "othello_causal_state"
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _collinear() -> np.ndarray:
    """(64, 64) bool — C[e, q]: q lies on one of the 8 rays from e, at any distance."""
    C = np.zeros((64, 64), bool)
    for e in range(64):
        r, c = divmod(e, 8)
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            while 0 <= rr < 8 and 0 <= cc < 8:
                C[e, rr * 8 + cc] = True
                rr, cc = rr + dr, cc + dc
    return C


def _cheb() -> np.ndarray:
    r, c = np.divmod(np.arange(64), 8)
    return np.maximum(np.abs(r[:, None] - r[None]), np.abs(c[:, None] - c[None]))


COLLINEAR, CHEB = _collinear(), _cheb()
NEIGH = (CHEB == 1)


def _rays() -> np.ndarray:
    """(512, 7) squares along each (origin e, direction d) ray in order, -1 padded."""
    R = -np.ones((64 * 8, 7), int)
    for e in range(64):
        r, c = divmod(e, 8)
        for k, (dr, dc) in enumerate(DIRS):
            rr, cc, n = r + dr, c + dc, 0
            while 0 <= rr < 8 and 0 <= cc < 8:
                R[e * 8 + k, n] = rr * 8 + cc
                n, rr, cc = n + 1, rr + dr, cc + dc
    return R


RAYS, RAY_ORIGIN = _rays(), np.repeat(np.arange(64), 8)


def traversable(occ: np.ndarray) -> np.ndarray:
    """Occupied squares on a GAP-FREE run from some empty square: the squares a
    legality/flip computation at this position can read. Colour-blind, so it does not
    suffer the single-flip test's redundancy problem (two discs that each bracket the same
    move both look 'irrelevant' to a flip). A disc that is NOT traversable cannot affect
    the current legal set for any colouring of the board."""
    occ_pad = np.append(occ, False)                    # index -1 -> False
    reached = np.cumprod(occ_pad[RAYS], axis=1).astype(bool)   # consecutive occupied
    reached &= (~occ)[RAY_ORIGIN][:, None]             # only rays FROM empty squares
    out = np.zeros(65, bool)
    out[RAYS[reached]] = True
    return out[:64]


FIELDS = ("occupied", "mine", "dead", "irrelevant", "traversable", "frontier", "age",
          "since_flip", "n_flips", "dist_empty")


def label_game(moves: list[int]) -> dict:
    """All flags for one game, positions 0..len-1 (padded to T_MODEL)."""
    T = min(len(moves), T_MODEL)
    out = {f: np.zeros((T_MODEL, 64), np.int16 if f in ("age", "since_flip", "n_flips",
                                                       "dist_empty") else np.int8)
           for f in FIELDS}
    out["mask"] = np.zeros(T_MODEL, bool)
    b = OthelloBoardState()
    placed = np.full(64, -1, int)
    last_change = np.full(64, -1, int)
    n_flips = np.zeros(64, int)
    prev = b.state.flatten().copy()
    for t in range(T):
        b.umpire(moves[t])
        st = b.state.flatten()
        changed = st != prev
        placed[changed & (prev == 0)] = t
        n_flips[changed & (prev != 0)] += 1
        last_change[changed] = t
        prev = st.copy()
        occ, empt = st != 0, st == 0
        nxt = 1 if b.next_hand_color > 0 else -1
        seen = COLLINEAR[empt].any(0) if empt.any() else np.zeros(64, bool)
        out["occupied"][t] = occ
        out["mine"][t] = np.where(empt, 0, np.where(st == nxt, 1, 2))
        out["dead"][t] = occ & ~seen
        out["traversable"][t] = traversable(occ)
        out["frontier"][t] = occ & (NEIGH[empt].any(0) if empt.any() else False)
        out["age"][t] = np.where(occ, t - placed, -1)
        out["since_flip"][t] = np.where(occ, t - last_change, -1)
        out["n_flips"][t] = np.where(occ, n_flips, -1)
        out["dist_empty"][t] = CHEB[:, empt].min(1) if empt.any() else 99
        legal0 = b.get_valid_moves()
        irr = np.zeros(64, bool)
        for q in np.where(occ)[0]:
            r, c = divmod(int(q), 8)
            b.state[r, c] *= -1
            irr[q] = b.get_valid_moves() == legal0
            b.state[r, c] *= -1
        out["irrelevant"][t] = irr
        out["mask"][t] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-games", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"test{a.n_games}"
    (EXP / "scores").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("test",))["test"])
    itos = {v: k for k, v in canonical_vocab().items()}
    games = [[int(itos[int(t)]) for t in row[:L]] for row, L in zip(tok[:a.n_games], ln[:a.n_games])]
    print(f"{len(games)} test games; labelling with {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        res = pool.map(label_game, games, chunksize=8)
    arrays = {f: np.stack([r[f] for r in res]) for f in FIELDS}
    mask = np.stack([r["mask"] for r in res])
    occ = arrays["occupied"].astype(bool) & mask[:, :, None]
    dead, irr = arrays["dead"].astype(bool), arrays["irrelevant"].astype(bool)
    trav = arrays["traversable"].astype(bool)
    assert not (dead & ~irr).any(), "a dead disc changed the legal set — flags inconsistent"
    assert not (dead & trav).any(), "a dead disc is traversable — flags inconsistent"
    assert not (occ & ~trav & ~irr).any(), "an unread disc changed the legal set — impossible"
    unread = occ & ~trav
    per_move_dead = dead.sum((0, 2)) / np.maximum(mask.sum(0), 1)
    np.savez_compressed(EXP / "scores" / f"relevance_{name}.npz", tokens=tok[:a.n_games],
                        lengths=ln[:a.n_games], game_ids=np.arange(a.n_games), mask=mask,
                        **arrays)
    manifest = {
        "name": name, "n_games": a.n_games, "positions": int(mask.sum()),
        "occupied_squares": int(occ.sum()), "dead_squares": int(dead.sum()),
        "irrelevant_squares": int(irr.sum()),
        "unread_squares": int(unread.sum()),
        "unread_not_dead_squares": int((unread & ~dead).sum()),
        "flip_irrelevant_but_read": int((irr & trav).sum()),
        "share_of_flip_irrelevant_that_is_read": float((irr & trav).sum() / max(irr.sum(), 1)),
        "positions_with_a_dead_square": int(dead.any(2).sum()),
        "dead_per_position_by_move": [round(float(v), 3) for v in per_move_dead],
        "irrelevant_share_of_occupied_by_move": [
            round(float(v), 3) for v in (irr.sum((0, 2)) / np.maximum(occ.sum((0, 2)), 1))],
        "first_move_with_dead": int(np.argmax(per_move_dead > 0)) + 1 if dead.any() else None,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (EXP / "scores" / f"relevance_{name}.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if "by_move" not in k}, indent=1))
    print("irrelevant share of occupied, moves 1,10,20,30,40,50,58:",
          [manifest["irrelevant_share_of_occupied_by_move"][i - 1] for i in (1, 10, 20, 30, 40, 50, 58)])
    print("dead per position, moves 40..58:", manifest["dead_per_position_by_move"][39:58])


if __name__ == "__main__":
    main()
