"""Othello instances (2026-09-06): corpus paths and rules per instance; the case synthesiser."""
import pytest

from pim.environments.othello import corpus as oc
from pim.environments.othello.bench import benchmark_from_cases, synthesise_cases
from pim.environments.othello.data import synthetic_games


def test_instance_table():
    assert oc.flip_of("oth-uniform") is True and oc.flip_of("oth-noflip") is False
    # paths come from pim.environments.layout: REPO-anchored, the split's role directory
    # under layout v2 (corpus/ under v1)
    d = oc.corpus_dir("oth-noflip", "train")
    assert d.is_absolute() and d.parts[-3:-1] == ("othello", "oth-noflip") and d.name in ("train", "corpus")
    assert oc.corpus_dir("oth-noflip", "test").name in ("eval", "corpus")
    with pytest.raises(KeyError):
        oc.corpus_dir("oth-nope")


def test_synthesised_cases_change_the_legal_set_and_build_a_benchmark():
    games = synthetic_games(120, seed=3, n_workers=4, flip=False)
    cases, man = synthesise_cases(games, 30, {10: 1, 20: 1, 30: 1}, seed=0, flip=False, log=None)
    assert len(cases) == 30 and man["n_cases"] == 30
    assert all(len(c["history"]) in (10, 20, 30) for c in cases)
    b = benchmark_from_cases(cases, flip=False)
    assert all(p != q and q for p, q in zip(b.legal_pre, b.legal_post))   # every flip changes the legal set
    assert set(b.cur_lab) <= {1, 2} and all(c != t for c, t in zip(b.cur_lab, b.tgt_lab))
