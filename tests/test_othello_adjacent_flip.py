"""oth-adjacent-flip (2026-09-09): adjacency decides legality, the enclosure scan decides what
the placed disc recolours. Pins that recolouring actually happens, that it equals the
enclosure captures, and that oth-adjacent (flip off) is untouched by the change."""
import random

import numpy as np

from pim.environments.othello import corpus as oc
from pim.environments.othello.data import synthetic_games, tokens_and_labels
from pim.environments.othello.vendor.othello import OthelloBoardState, eights, get_ood_game


def _touches_own(board, sq, color):
    r, c = sq // 8, sq % 8
    return any(0 <= r + d[0] < 8 and 0 <= c + d[1] < 8 and board.state[r + d[0], c + d[1]] == color
               for d in eights)


def test_instance_row():
    assert oc.rules_of("oth-adjacent-flip") == {"flip": True, "placement": "adjacent"}
    assert oc.rules_of("oth-adjacent") == {"flip": False, "placement": "adjacent"}
    assert oc.corpus_dir("oth-adjacent-flip").parts[-3:] == ("othello", "oth-adjacent-flip", "corpus")


def test_adjacent_flip_recolours_exactly_the_enclosed_discs():
    random.seed(5)
    g = get_ood_game(0, flip=True, placement="adjacent")
    assert 4 <= len(g) <= 60
    b = OthelloBoardState(flip=True, placement="adjacent")
    n_flipped = 0
    for mv in g:
        legal = b.get_valid_moves()
        assert mv in legal
        for sq in legal:                      # legality is adjacency, as in oth-adjacent
            col = b.next_hand_color if b.tentative_move(sq) == 1 else -b.next_hand_color
            assert _touches_own(b, sq, col)
        colour = b.next_hand_color if b.tentative_move(mv) == 1 else -b.next_hand_color
        expect = {(r, c) for r, c in b._captures(mv // 8, mv % 8, colour)}
        before = b.state.copy()
        b.umpire(mv)
        changed = {(r, c) for r in range(8) for c in range(8)
                   if before[r, c] != 0 and before[r, c] != b.state[r, c]}
        assert changed == expect             # recolouring == the enclosure captures
        n_flipped += len(changed)
    assert n_flipped > 0                     # the variant is NOT oth-adjacent
    assert int((b.state != 0).sum()) == 4 + len(g)


def test_adjacent_without_flip_is_unchanged():
    # flip off: the adjacency branch must still hand the umpire nothing to recolour
    b = OthelloBoardState(flip=False, placement="adjacent")
    for sq in range(64):
        if b.state[sq // 8, sq % 8] == 0:
            ok, tbf = b._legal(sq // 8, sq % 8, b.next_hand_color)
            assert tbf == []
    random.seed(5)
    g = get_ood_game(0, flip=False, placement="adjacent")
    b = OthelloBoardState(flip=False, placement="adjacent")
    placed = {}
    for mv in g:
        colour = b.next_hand_color if b.tentative_move(mv) == 1 else -b.next_hand_color
        b.umpire(mv)
        placed[mv] = colour
        assert all(b.state[s // 8, s % 8] == c for s, c in placed.items())


def test_labels_follow_the_adjacent_flip_rules():
    rules = oc.rules_of("oth-adjacent-flip")
    games = synthetic_games(6, seed=1, n_workers=2, **rules)
    d = tokens_and_labels(games, **rules)
    for i, g in enumerate(games):
        b = OthelloBoardState(**rules)
        for t, mv in enumerate(g[:59]):
            b.umpire(mv)
            assert np.array_equal(d.labels[i, t], (b.state + 1).flatten().astype(np.int8))
