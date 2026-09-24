"""Counterfactual games + the flip statistic (pim.environments.othello.counterfactual, 2026-09-19)."""
import numpy as np
import pytest

from pim.environments.othello import corpus as oc
from pim.environments.othello.counterfactual import flips_per_move, mine_board, replay, search_cf


def _test_games(instance, n):
    try:
        tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("test",), instance=instance)["test"])
    except Exception:
        pytest.skip(f"{instance} test split absent")
    return tok[:n], ln[:n]


def test_flips_follow_the_rules():
    tok, ln = _test_games("oth-uniform", 20)
    f = flips_per_move(tok, ln, oc.rules_of("oth-uniform"))
    assert f["n_moves"] == int(ln.sum()) and 1.5 < f["flips_per_move"] < 3.0       # standard Othello ≈ 2.2
    tok, ln = _test_games("oth-noflip", 20)
    assert flips_per_move(tok, ln, oc.rules_of("oth-noflip"))["n_flipped"] == 0       # nothing is ever recoloured


def test_a_found_counterfactual_is_a_real_game_with_the_tile_flipped():
    from pim.environments.othello.data import canonical_vocab

    tok, ln = _test_games("oth-uniform", 40)
    itos = {v: k for k, v in canonical_vocab().items()}
    rules, hits = oc.rules_of("oth-uniform"), 0
    for row in tok:
        h = [int(itos[int(t)]) for t in row[:16]]
        m0 = mine_board(replay(h, rules))
        s = int(np.flatnonzero(m0 > 0)[-1])                       # an occupied tile
        best, d, m_orig = search_cf(h, s, rules)
        assert np.array_equal(m_orig, m0)
        if best is None:
            continue
        hh, m = best
        b = replay(hh, rules)
        assert b is not None and len(hh) == len(h)                # a legal game of the same length
        assert b.next_hand_color == replay(h, rules).next_hand_color
        assert m[s] == 3 - m0[s] and np.array_equal(mine_board(b), m)   # the tile IS flipped on its board
        want = m0.copy()
        want[s] = 3 - want[s]
        assert d == int((m != want).sum())
        hits += 1
    assert hits > 0
