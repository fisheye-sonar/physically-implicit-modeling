"""True counterfactual GAMES for an Othello edit case (2026-09-11; in ``pim`` since 2026-09-19).

An edit case asks for one tile's colour flipped. ``search_cf`` looks for a REAL game — same
length, same player to move — whose board IS that flipped board, so the model can be run on a
history that genuinely produces the edited world. Two uses, both reference points for the Edit
Index rather than editors: the index CEILING (``scripts/index_ceiling.py``: what the measure
gives the model's own prediction on the true counterfactual), and the split of edit cases into
REACHABLE (a legal board) and UNREACHABLE (no game produces it).

Moved verbatim from ``experiments/edit_direction_alignment/scripts/othello_alignment.py``; the
candidate order is seeded by the tile, so a case's counterfactual is reproducible.
"""
from __future__ import annotations

import numpy as np

from pim.environments.othello.vendor.othello import OthelloBoardState


def replay(h, rules):
    """The board after the move list ``h`` under ``rules`` (``corpus.rules_of``), or None if illegal."""
    b = OthelloBoardState(**rules)
    try:
        b.update(h, prt=False)
    except AssertionError:
        return None
    return b


def mine_board(b) -> np.ndarray:
    """(64,) blank 0 / mine 1 / theirs 2, relative to the player to move."""
    st = (b.state + 1).flatten()                          # white 0 / blank 1 / black 2
    nxt = 2 if b.next_hand_color > 0 else 0
    return np.where(st == 1, 0, np.where(st == nxt, 1, 2))


def search_cf(h, s, rules, max_sub=400, subs_only=False):
    """Closest real history (same length, same mover) whose board has tile s flipped.

    Returns ((history, board) | None, distance in tiles from the wanted board, the original board);
    distance 0 = an EXACT counterfactual.

    ⛔ Move SWAPS produce legal histories the model handles BADLY (legal mass 0.845, against
    0.994 for single-move substitutions and 0.998 for ordinary held-out prefixes), and they
    contaminated the first run of this analysis. They are kept because substitutions alone
    never reach the flipped board exactly (0/900 cases); the caller MUST therefore filter on
    the model's own error on the returned history (``scripts/index_ceiling.py``: "ordinary"
    = within the 95th percentile of held-out prefixes of that length), which screens them."""
    b0 = replay(h, rules)
    m0 = mine_board(b0)
    want = m0.copy()
    want[s] = 3 - want[s]                                  # flip mine<->theirs at s
    best, best_d = None, 99
    cands = []
    for k in range(len(h)):
        bk = replay(h[:k], rules)
        for mv in bk.get_valid_moves():
            if mv != h[k]:
                cands.append(h[:k] + [mv] + h[k + 1:])
    if not subs_only:
        for i in range(len(h)):
            for k in range(i + 1, len(h)):
                hh = list(h)
                hh[i], hh[k] = hh[k], hh[i]
                cands.append(hh)
    rng = np.random.default_rng(s)
    for hh in [cands[j] for j in rng.permutation(len(cands))[:max_sub]]:
        b = replay(hh, rules)
        if b is None or b.next_hand_color != b0.next_hand_color:
            continue
        m = mine_board(b)
        if m[s] != want[s]:
            continue
        d = int((m != want).sum())
        if d < best_d:
            best, best_d = (hh, m), d
            if d == 0:
                break
    return best, best_d, m0


def flips_per_move(tokens: np.ndarray, lengths: np.ndarray, rules: dict) -> dict:
    """How often the rules RECOLOUR a disc: discs flipped per move / per game over a set of games
    (the corpus statistic that separates the flipping variants: standard Othello ≈ 2.2 per
    move, adjacent-flip ≈ 0.27, the no-flip variants exactly 0)."""
    from pim.environments.othello.data import canonical_vocab

    itos = {v: k for k, v in canonical_vocab().items()}
    n_flipped = n_moves = 0
    for row, L in zip(tokens, lengths):
        b = OthelloBoardState(**rules)
        for t in range(int(L)):
            before = b.state.copy()
            b.umpire(itos[int(row[t])])
            n_flipped += int(((before != 0) & (before != b.state)).sum())
            n_moves += 1
    return {"n_games": int(len(tokens)), "n_moves": int(n_moves), "n_flipped": int(n_flipped),
            "flips_per_move": n_flipped / n_moves, "flips_per_game": n_flipped / len(tokens)}
