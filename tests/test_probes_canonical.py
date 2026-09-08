"""The canonical probes: linear (closed form), MLP-128, the tripwire, decodability.

CPU-only, synthetic data. The bit-identity of ``fit_linear``/``fit_mlp`` against the
retired ``othello_probe.fit_probe`` was gated at port time (2026-08-31); these keep the
behavioural invariants permanent.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pim.metrics.decodability import (
    probe_skill_classification,
    probe_skill_regression,
    r2,
    trivial_error_rate,
)
from pim.probes import (
    CANONICAL_HIDDEN,
    ProbeSanityError,
    check_probe_sanity,
    fit_linear,
    fit_mlp,
)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    H, D, N = 32, 3, 2000
    W = rng.standard_normal((H, D)) * 0.3
    X = rng.standard_normal((N, H)).astype(np.float32)
    Y = (X @ W + 0.05 * rng.standard_normal((N, D))).astype(np.float32)
    return X[:1600], Y[:1600], X[1600:], Y[1600:]


def test_canonical_mlp_width_is_128():
    assert CANONICAL_HIDDEN == 128


def test_linear_probe_is_closed_form(data):
    """Two fits with different seeds must be IDENTICAL — lstsq has no RNG."""
    xtr, ytr, xte, yte = data
    _, s0 = fit_linear(xtr, ytr, xte, yte, device="cpu", seed=0)
    _, s1 = fit_linear(xtr, ytr, xte, yte, device="cpu", seed=99)
    assert s0["r2"] == s1["r2"]
    assert s0["r2"] > 0.95


def test_linear_probe_forward_matches_stats(data):
    xtr, ytr, xte, yte = data
    probe, s = fit_linear(xtr, ytr, xte, yte, device="cpu", seed=0)
    with torch.no_grad():
        pred = probe(torch.tensor(xte)).numpy()
    assert abs(r2(pred, yte, ytr.mean(0)) - s["r2"]) < 1e-6


def test_mlp_default_hidden_is_canonical(data):
    xtr, ytr, xte, yte = data
    probe, s = fit_mlp(xtr, ytr, xte, yte, epochs=2, device="cpu", seed=0)
    assert probe.hidden == CANONICAL_HIDDEN
    assert s["kind"] == f"mlp-{CANONICAL_HIDDEN}"


def test_probe_is_pure_function_of_raw_activation(data):
    """Standardisation moments are buffers: probe(h) must not depend on caller scaling."""
    xtr, ytr, xte, yte = data
    probe, _ = fit_linear(xtr, ytr, xte, yte, device="cpu", seed=0)
    h = torch.tensor(xte[:5])
    with torch.no_grad():
        a, b = probe(h), probe(h.clone())
    assert torch.equal(a, b)
    assert not probe.x_std.requires_grad


def test_tripwire_fires_on_mlp_below_linear():
    lin = {0: (None, {"r2": 0.90, "r2_insample": 0.91})}
    mlp = {0: (None, {"r2": 0.70, "r2_insample": 0.99})}  # memorising
    with pytest.raises(ProbeSanityError, match="beaten by linear"):
        check_probe_sanity(lin, mlp, strict=True, log=None)
    rep = check_probe_sanity(lin, mlp, strict=False, log=None)
    assert rep["n_violations"] == 1


def test_tripwire_passes_when_mlp_wins():
    lin = {0: (None, {"r2": 0.80, "r2_insample": 0.81})}
    mlp = {0: (None, {"r2": 0.85, "r2_insample": 0.88})}
    rep = check_probe_sanity(lin, mlp, strict=True, log=None)
    assert rep["n_violations"] == 0


# ── decodability (the cross-environment axis) ────────────────────────────────


def test_probe_skill_equals_r2_on_regression():
    rng = np.random.default_rng(1)
    y, p = rng.standard_normal((100, 4)), rng.standard_normal((100, 4))
    m = np.zeros(4)
    assert abs(probe_skill_regression(p, y, m) - r2(p, y, m)) < 1e-12


def test_probe_skill_classification_anchors():
    """1 at perfect, 0 at the majority baseline — the axis both environments share."""
    train = np.array([[0], [0], [0], [1]])
    y = np.array([[0], [0], [1], [1]])
    assert probe_skill_classification(y, y, train) == 1.0
    maj = np.zeros_like(y)
    base = trivial_error_rate(y, train)
    assert abs(probe_skill_classification(maj, y, train)
               - (1 - (maj != y).mean() / base)) < 1e-12
