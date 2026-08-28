"""The probe cache must never serve one model's probes for another model.

Guards the 2026-08-21 failure directly: `fit_probe_grid`'s key covered the probe settings and
the data but not the *model*, so a random-init control was served the trained model's probes and
both reported an identical error rate — which would have destroyed the only baseline that makes
an absolute probe error interpretable.

Key logic only — no GPU, no dataset, no fitting.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch
import torch.nn as nn

_REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (_REPO / "notebooks/experiments/editability/othello_arch",
           _REPO / "notebooks/experiments/editability/othello_gpt",
           _REPO, _REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

E = pytest.importorskip("editability", reason="editability.py needs the experiment deps")

KW = dict(target="pos", n_seq=1500, split="test", hidden=512, basis_name="cartesian")


class _Fake(nn.Module):
    def __init__(self, v: float = 1.0, span: int = 39):
        super().__init__()
        self.w = nn.Parameter(torch.full((4, 4), v))
        self.state_span = span


def test_identical_weights_share_a_key():
    assert E._probe_cache_key(_Fake(1.0), **KW)[0] == E._probe_cache_key(_Fake(1.0), **KW)[0]


def test_different_weights_never_share_a_key():
    """The 2026-08-21 bug, as an assertion."""
    assert E._probe_cache_key(_Fake(1.0), **KW)[0] != E._probe_cache_key(_Fake(2.0), **KW)[0]


def test_state_span_is_in_the_key():
    assert E._probe_cache_key(_Fake(1.0, 39), **KW)[0] != E._probe_cache_key(_Fake(1.0, 61), **KW)[0]


@pytest.mark.parametrize("field,value", [("target", "full"), ("n_seq", 10_000),
                                         ("split", "val"), ("hidden", None),
                                         ("basis_name", "width")])
def test_every_fitting_argument_moves_the_key(field, value):
    m = _Fake()
    assert E._probe_cache_key(m, **{**KW, field: value})[0] != E._probe_cache_key(m, **KW)[0]


def test_linear_and_mlp_are_distinguishable_in_the_provenance():
    m = _Fake()
    assert E._probe_cache_key(m, **KW)[1]["hidden"] == 512
    assert E._probe_cache_key(m, **{**KW, "hidden": None})[1]["hidden"] == "linear"


def test_provenance_round_trips_and_detects_tampering(tmp_path):
    m = _Fake()
    name, prov = E._probe_cache_key(m, **KW)
    fp = tmp_path / name
    torch.save({"provenance": prov, "probes": {0: ("probe", {"r2": 0.9})}}, fp)
    blob = torch.load(fp, weights_only=False)
    assert blob["provenance"] == prov
    blob["provenance"]["model"] = "deadbeef"
    assert blob["provenance"] != prov
