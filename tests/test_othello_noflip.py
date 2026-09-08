"""The no-flip Othello variant (2026-09-06): same legality and passes, no recolouring."""
import random

import numpy as np

from pim.environments.othello.data import _one_game, synthetic_games, tokens_and_labels
from pim.environments.othello.vendor.othello import OthelloBoardState, get_ood_game


def test_noflip_never_recolours_and_keeps_legality():
    random.seed(123)
    g = get_ood_game(0, flip=False)
    assert 4 <= len(g) <= 60
    b = OthelloBoardState(flip=False)
    placed = {}
    for mv in g:
        legal = b.get_valid_moves()
        assert mv in legal                       # every move was legal under the enclosure rule
        colour = b.next_hand_color if b.tentative_move(mv) == 1 else -b.next_hand_color
        b.umpire(mv)
        placed[mv] = colour
        for sq, c in placed.items():             # nothing ever changes colour
            assert b.state[sq // 8, sq % 8] == c
    # disc count = 4 + moves; in flip Othello it would be the same count but colours differ
    assert int((b.state != 0).sum()) == 4 + len(g)


def test_flip_default_is_unchanged_and_index_seeding_holds():
    a = _one_game((7, 0))
    b = _one_game((7, 0, True))
    c = _one_game((7, 0, False))
    assert a == b
    assert c != a or len(c) != len(a)           # different rules, different game (overwhelmingly)
    assert _one_game((7, 0, False)) == c        # deterministic


def test_labels_follow_the_noflip_rules():
    games = synthetic_games(6, seed=1, n_workers=2, flip=False)
    d = tokens_and_labels(games, flip=False)
    for i, g in enumerate(games):
        b = OthelloBoardState(flip=False)
        for t, mv in enumerate(g[:59]):
            b.umpire(mv)
            assert np.array_equal(d.labels[i, t], (b.state + 1).flatten().astype(np.int8))
