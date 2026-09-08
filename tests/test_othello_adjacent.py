"""The adjacency-placement Othello variant (oth-adjacent, 2026-09-08): a move must touch one
of the mover's own discs (8-neighbourhood); nothing is recoloured; passes/game end as before.
Also pins that the refactored legality code reproduces the enclosure games byte-for-byte."""
import random

import numpy as np

from pim.environments.othello import corpus as oc
from pim.environments.othello.data import _one_game, synthetic_games, tokens_and_labels
from pim.environments.othello.vendor.othello import OthelloBoardState, eights, get_ood_game

# the first 12 moves of the enclosure games at (seed 0, i 0/7) before the 2026-09-08 refactor
_ENCLOSURE_PREFIX = {0: _one_game((0, 0))[:12], 7: _one_game((7, 0))[:12]}


def _touches_own(board, sq, color):
    r, c = sq // 8, sq % 8
    return any(0 <= r + d[0] < 8 and 0 <= c + d[1] < 8 and board.state[r + d[0], c + d[1]] == color
               for d in eights)


def test_adjacent_rule_holds_move_by_move():
    random.seed(5)
    g = get_ood_game(0, flip=False, placement="adjacent")
    assert 4 <= len(g) <= 60
    b = OthelloBoardState(flip=False, placement="adjacent")
    placed = {}
    for mv in g:
        legal = b.get_valid_moves()
        assert mv in legal
        # every legal square touches a disc of the player who will actually place there
        for sq in legal:
            col = b.next_hand_color if b.tentative_move(sq) == 1 else -b.next_hand_color
            assert _touches_own(b, sq, col)
        colour = b.next_hand_color if b.tentative_move(mv) == 1 else -b.next_hand_color
        b.umpire(mv)
        placed[mv] = colour
        for sq, c in placed.items():
            assert b.state[sq // 8, sq % 8] == c          # never recoloured
    assert int((b.state != 0).sum()) == 4 + len(g)


def test_enclosure_games_unchanged_by_refactor():
    # the vendored generator's enclosure games must be identical before/after `_legal`
    for i, pre in _ENCLOSURE_PREFIX.items():
        assert _one_game((i, 0))[:12] == pre
        assert _one_game((i, 0, True, "enclosure"))[:12] == pre
    # the enclosure legal set equals the original two-scan computation on random boards
    random.seed(11)
    b = OthelloBoardState()
    for mv in _one_game((3, 0))[:20]:
        b.umpire(mv)
    legal = set(b.get_valid_moves())
    brute = {sq for sq in range(64) if b.state[sq // 8, sq % 8] == 0
             and len(b._captures(sq // 8, sq % 8, b.next_hand_color)) > 0}
    assert legal == brute


def test_rules_travel_with_the_instance():
    assert oc.rules_of("oth-adjacent") == {"flip": False, "placement": "adjacent"}
    assert oc.rules_of("oth-uniform") == {"flip": True, "placement": "enclosure"}
    games = synthetic_games(6, seed=1, n_workers=2, **oc.rules_of("oth-adjacent"))
    d = tokens_and_labels(games, **oc.rules_of("oth-adjacent"))
    for i, g in enumerate(games):
        b = OthelloBoardState(**oc.rules_of("oth-adjacent"))
        for t, mv in enumerate(g[:59]):
            b.umpire(mv)
            assert np.array_equal(d.labels[i, t], (b.state + 1).flatten().astype(np.int8))
    # colour is not forced by position: some disc sits on a square of the "wrong" parity
    parity_ok = [((sq // 8 + sq % 8) % 2 == 0) == (c > 0) for sq, c in
                 ((sq, b.state[sq // 8, sq % 8]) for sq in range(64)) if c != 0]
    assert not all(parity_ok) and any(parity_ok)
