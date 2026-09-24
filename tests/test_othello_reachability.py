"""The exact reachability decision (``pim.environments.othello.reachability``).

1. The bitboard engine equals the vendored ``OthelloBoardState`` move by move — board, player to
   move, and the legal move set (passes included) — on random games under all four rule sets.
2. Every verdict equals brute force: all games of a short length are enumerated with the VENDORED
   engine, and a flipped board is reachable exactly when it is in that set. This checks the
   search and both prunes against an answer that does not depend on this module.
3. On the real bench: witnesses replay to the target, every case the heuristic ``search_cf``
   solves is found reachable, and ``oth-noflip``'s flipped boards are unreachable (colour equals
   square parity there, so a single flip can never be produced).
"""
import copy
import pickle

import numpy as np
import pytest

from pim.environments.othello import corpus as oc
from pim.environments.othello.bench import cases_path
from pim.environments.othello.counterfactual import search_cf
from pim.environments.othello.reachability import (START, decide, play, search, state_of,
                                                   valid_moves)
from pim.environments.othello.vendor.othello import OthelloBoardState

RULES = {"standard": {"flip": True, "placement": "enclosure"},
         "noflip": {"flip": False, "placement": "enclosure"},
         "adjacent": {"flip": False, "placement": "adjacent"},
         "adjacent-flip": {"flip": True, "placement": "adjacent"}}


@pytest.mark.parametrize("name", list(RULES))
def test_engine_matches_vendor(name):
    rules = RULES[name]
    rng = np.random.default_rng(0)
    for _ in range(40):
        vb, st = OthelloBoardState(**rules), START
        for _ply in range(60):
            vm = sorted(vb.get_valid_moves())
            assert sorted(valid_moves(st, rules)) == vm
            if not vm:
                break
            m = int(rng.choice(vm))
            vb.umpire(m)
            st = play(st, m, rules)
            assert st == state_of(vb)


def _all_games(rules, length):
    """Every state reachable in exactly ``length`` moves, by enumeration with the vendored engine."""
    out, stack = set(), [(OthelloBoardState(**rules), 0)]
    while stack:
        b, d = stack.pop()
        if d == length:
            out.add(state_of(b))
            continue
        for m in b.get_valid_moves():
            nb = copy.deepcopy(b)
            nb.umpire(m)
            stack.append((nb, d + 1))
    return out


@pytest.mark.parametrize("name,length", [("standard", 6), ("noflip", 6), ("adjacent", 4), ("adjacent-flip", 4)])
def test_verdicts_match_brute_force(name, length):
    rules = RULES[name]
    reach = _all_games(rules, length)
    rng = np.random.default_rng(1)
    pool = sorted(reach)
    n_yes = n_no = 0
    for i in rng.permutation(len(pool))[:25]:
        b, w, mover = pool[i]
        occ = b | w
        for sq in [s for s in range(64) if (occ >> s) & 1]:
            target = (b ^ (1 << sq), w ^ (1 << sq), mover)
            v = search(target, length, rules, budget=10**7)
            assert v.status != "undecided"
            truth = target in reach
            assert (v.status == "reachable") == truth, (name, target, v.status)
            n_yes += truth
            n_no += not truth
    assert n_no > 0
    if name in ("standard", "adjacent-flip"):
        assert n_yes > 0                   # flip rules do produce some single-flip boards


def _cases(inst, n):
    return pickle.load(open(cases_path(inst), "rb"))[:n]


def test_bench_witnesses_and_heuristic_agreement():
    rules = oc.rules_of("oth-uniform")
    n_heur = 0
    for c in _cases("oth-uniform", 12):
        h, sq = [int(x) for x in c["history"]], int(c["pos_int"])
        v = decide(h, sq, rules, budget=3_000_000)       # a reachable verdict's witness is replayed inside
        best, dist, _ = search_cf(h, sq, rules)
        if best is not None and dist == 0:
            n_heur += 1
            assert v.status == "reachable"
    assert n_heur > 0


def test_noflip_flips_are_unreachable():
    rules = oc.rules_of("oth-noflip")
    for c in _cases("oth-noflip", 10):
        v = decide([int(x) for x in c["history"]], int(c["pos_int"]), rules, budget=3_000_000)
        assert v.status == "unreachable"
