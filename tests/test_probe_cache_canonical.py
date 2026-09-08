"""The probe cache must never serve one model's probes for another model.

Guards the 2026-08-21 failure directly: a cache key that covered the probe settings
and the data but not the *model* served a random-init control the trained model's
probes, and both reported an identical error rate — destroying the only baseline that
makes an absolute probe error interpretable.

Successor to ``tests/test_probe_cache.py`` (which guarded the old
``othello_arch/editability.py`` cache through ``importorskip`` — i.e. silently skipped
if that tree was absent). This one imports ``pim.probes.cache`` directly: if the cache
is missing, the suite FAILS, it does not skip.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pim.probes.cache import ProbeCache

KW = dict(target="pos", n_seq=1500, split="test", family="mlp", basis="cartesian",
          seed=0, data="datasets/x")


class _Fake(nn.Module):
    def __init__(self, v: float = 1.0, span: int = 39):
        super().__init__()
        self.w = nn.Parameter(torch.full((4, 4), v))
        self.state_span = span


@pytest.fixture()
def cache(tmp_path):
    return ProbeCache(tmp_path)


def test_identical_weights_share_a_key(cache):
    assert cache.key(_Fake(1.0), **KW)[0] == cache.key(_Fake(1.0), **KW)[0]


def test_different_weights_never_share_a_key(cache):
    """The 2026-08-21 bug, as an assertion."""
    assert cache.key(_Fake(1.0), **KW)[0] != cache.key(_Fake(2.0), **KW)[0]


def test_state_span_is_in_the_key(cache):
    assert cache.key(_Fake(1.0, 39), **KW)[0] != cache.key(_Fake(1.0, 61), **KW)[0]


@pytest.mark.parametrize("field,value", [("target", "full"), ("n_seq", 10_000),
                                         ("split", "val"), ("family", "linear"),
                                         ("basis", "width"), ("data", "datasets/y")])
def test_every_fitting_argument_moves_the_key(cache, field, value):
    m = _Fake()
    assert cache.key(m, **{**KW, field: value})[0] != cache.key(m, **KW)[0]


def test_version_bump_moves_the_key(cache, monkeypatch):
    m = _Fake()
    a = cache.key(m, **KW)[0]
    monkeypatch.setattr(ProbeCache, "VERSION", ProbeCache.VERSION + 1)
    assert cache.key(m, **KW)[0] != a


def test_store_then_load_round_trips(cache):
    m = _Fake()
    name, prov = cache.key(m, **KW)
    cache.store(name, prov, {0: ("probe", {"r2": 0.9})})
    got = cache.load(name, prov)
    assert got[0][1]["r2"] == 0.9


def test_miss_returns_none(cache):
    name, prov = cache.key(_Fake(), **KW)
    assert cache.load(name, prov) is None


def test_tampered_provenance_raises(cache):
    m = _Fake()
    name, prov = cache.key(m, **KW)
    cache.store(name, prov, {0: "p"})
    bad = {**prov, "model": "deadbeef"}
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        cache.load(name, bad)


def test_no_partial_file_after_store(cache):
    m = _Fake()
    name, prov = cache.key(m, **KW)
    cache.store(name, prov, {0: "p"})
    assert not list(cache.dir.glob("*.partial"))
