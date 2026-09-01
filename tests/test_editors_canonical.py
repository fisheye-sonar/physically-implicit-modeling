"""The canonical editors: PI (with the y-affine regression test), ND, GS, nullspace.

The PI landing test is the permanent form of the 2026-08-31 y-affine finding: the
canonical decompositions must ROUND-TRIP THROUGH ``probe.forward`` — the probe's own
output, not a hand-derived read-out, decides whether the write landed. The legacy
decomposition is asserted to MISS on a probe with a non-trivial y-affine, so the bug
can never again hide behind a decomposition that agrees with itself.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pim.editors.grad_steer import build_edit_spec, make_intervention_hook
from pim.editors.nanda import addition_delta, probe_direction
from pim.editors.nullspace import multiprobe_delta
from pim.editors.pinv import decompose_hidden, inject_state, pinv_maps, pinv_step, readout_error
from pim.probes.base import WorldStateProbe
from pim.probes.nullspace import fit_nullspace_cascade

H, D, B = 64, 4, 16


@pytest.fixture()
def probe():
    """A linear regression probe with NON-TRIVIAL affines on both ends — the exact
    configuration the y-affine bug required to be visible."""
    torch.manual_seed(0)
    p = WorldStateProbe(
        H, D, None,
        x_mean=torch.randn(H) * 0.3, x_std=torch.rand(H) * 2 + 0.1,
        y_mean=torch.tensor([0.0, 7.9, 0.0, 7.9]),
        y_std=torch.tensor([1.7, 1.9, 1.7, 1.9]),
    )
    with torch.no_grad():
        p.net.weight.copy_(torch.randn(D, H) * 0.2)
        p.net.bias.copy_(torch.randn(D) * 0.1)
    return p.eval()


# ── PI ───────────────────────────────────────────────────────────────────────


def test_inject_state_lands_exactly():
    torch.manual_seed(1)
    A = torch.randn(D, H)
    Ap = torch.linalg.pinv(A)
    b = torch.randn(D)
    h = torch.randn(B, H)
    t = torch.randn(B, D)
    h2 = inject_state(h, t, A, Ap, b)
    assert torch.allclose(h2 @ A.T + b, t, atol=1e-4)


def test_inject_state_preserves_nullspace():
    torch.manual_seed(2)
    A = torch.randn(D, H)
    Ap = torch.linalg.pinv(A)
    h = torch.randn(B, H)
    h2 = inject_state(h, torch.randn(B, D), A, Ap, torch.zeros(D))
    _, perp0 = decompose_hidden(h, A, Ap)
    _, perp2 = decompose_hidden(h2, A, Ap)
    assert torch.allclose(perp0, perp2, atol=1e-4)


@pytest.mark.parametrize("space", ["zspace", "raw"])
def test_pinv_step_lands_through_probe_forward(probe, space):
    """The y-affine regression test: the α=1 write must make PROBE.FORWARD read the
    raw-units target. This is the check the 2026-08-31 bug could not have survived."""
    torch.manual_seed(3)
    h0 = torch.randn(B, H)
    tgt = torch.randn(B, D) * 2 + torch.tensor([0.0, 8.0, 0.0, 8.0])
    h1 = h0 + pinv_step(h0, tgt, probe, space=space)
    assert readout_error(h1, tgt, probe) < 1e-3


def test_legacy_space_documents_the_bug(probe):
    """Legacy must MISS on a probe with a non-trivial y-affine (that IS the bug)."""
    torch.manual_seed(3)
    h0 = torch.randn(B, H)
    tgt = torch.randn(B, D) * 2 + torch.tensor([0.0, 8.0, 0.0, 8.0])
    h1 = h0 + pinv_step(h0, tgt, probe, space="legacy")
    assert readout_error(h1, tgt, probe) > 1.0


def test_all_maps_agree_on_what_the_probe_reads(probe):
    """The three decompositions describe ONE probe: their read-outs of the same h are
    the same quantity, up to the y-affine (legacy/zspace are standardised-y)."""
    torch.manual_seed(4)
    h = torch.randn(B, H)
    m = pinv_maps(probe)
    raw_read = h @ m["raw"].A.T + m["raw"].b
    assert torch.allclose(raw_read, probe(h), atol=1e-4)
    leg_read = h @ m["legacy"].A.T + m["legacy"].b
    assert torch.allclose(leg_read * probe.y_std + probe.y_mean, probe(h), atol=1e-4)


# ── ND ───────────────────────────────────────────────────────────────────────


def test_nanda_direction_is_unit_and_standardised(probe):
    d = probe_direction(probe, [0, 1])
    assert abs(float(d.norm()) - 1.0) < 1e-6
    W = probe.net.weight.detach()
    ref = (W[[0, 1]] / probe.x_std).sum(0)
    assert torch.allclose(d, ref / ref.norm())


def test_nanda_delta_scales_with_activation_norm(probe):
    d = probe_direction(probe, [0])
    x = torch.randn(B, H)
    delta = addition_delta(x, d, 0.5)
    assert torch.allclose(delta.norm(dim=1), 0.5 * x.norm(dim=1), atol=1e-5)


# ── GS ───────────────────────────────────────────────────────────────────────


def test_grad_steer_hook_moves_only_from_start_layer(probe):
    """The sequential schedule: residual points before L_s are untouched."""
    x0 = torch.randn(B, H)
    cm = torch.zeros(B, D, dtype=torch.bool)
    cm[:, 0] = True
    tv = torch.zeros(B, D)
    spec = build_edit_spec(probe, x0, cm, tv, beta=0.2)
    hook = make_intervention_hook({2: probe, 3: probe}, {2: spec, 3: spec},
                                  start_layer=2, alpha=1e-3, n_steps=2)
    stream = torch.randn(B, 5, H)
    assert torch.equal(hook(1, stream.clone()), stream)  # before L_s: untouched
    out = hook(2, stream.clone())
    assert not torch.equal(out, stream)  # at L_s: written
    assert torch.equal(out[:, :-1], stream[:, :-1])  # only the LAST position


# ── nullspace ────────────────────────────────────────────────────────────────


def test_multiprobe_oracle_targets_equal_subspace_projection():
    """Σ_k A_k⁺(A_k Δh) == P_S(Δh) — the identity the notebook verified numerically."""
    rng = np.random.default_rng(0)
    S_true = np.linalg.qr(rng.standard_normal((32, 8)))[0]
    Y = (rng.standard_normal((2000, 32)) @ S_true) @ rng.standard_normal((8, 3))
    Hs = rng.standard_normal((2000, 32))
    Y = Hs @ S_true @ rng.standard_normal((8, 3)) + 0.05 * rng.standard_normal((2000, 3))
    casc = fit_nullspace_cascade(Hs, Y, slice(0, 1600), slice(1600, None),
                                 max_iter=10, r2_stop=0.02, log=None)
    S = casc.subspace()
    h0, dh = Hs[:8], rng.standard_normal((8, 32))
    per = [casc.read(k, h0 + dh) for k in range(casc.n_probes)]
    got = multiprobe_delta(casc, h0, None, per_probe_targets=per)
    assert np.abs(got - dh @ S @ S.T).max() < 1e-8


# ── dims-restricted PI (added 2026-09-01) ────────────────────────────────────
#
# Lets a probe fitted on the FULL state drive only part of it, so "one probe set
# instead of two" can be tested rather than assumed.


def test_dims_restricted_solve_lands_only_those_dims(probe):
    torch.manual_seed(7)
    h0 = torch.randn(B, H)
    tgt = torch.randn(B, D) * 2 + torch.tensor([0.0, 8.0, 0.0, 8.0])
    read = probe(h0 + pinv_step(h0, tgt, probe, dims=[0, 1]))
    assert (read[:, :2] - tgt[:, :2]).abs().max() < 1e-3      # driven dims land
    assert (read[:, 2:] - tgt[:, 2:]).abs().max() > 1e-2      # the rest stay free


def test_dims_restricted_write_is_smaller():
    """A rank-k solve has a larger null space than the full-rank one, so it writes less.
    That is the whole reason the restriction might beat a separately-fitted probe."""
    torch.manual_seed(8)
    p = WorldStateProbe(H, D, None, x_std=torch.rand(H) + 0.2,
                        y_std=torch.rand(D) + 0.5, y_mean=torch.randn(D))
    with torch.no_grad():
        p.net.weight.copy_(torch.randn(D, H) * 0.2)
    p.eval()
    h0, tgt = torch.randn(B, H), torch.randn(B, D)
    full = pinv_step(h0, tgt, p).norm(dim=1).mean()
    sub = pinv_step(h0, tgt, p, dims=[0, 1]).norm(dim=1).mean()
    assert sub < full


def test_dims_none_is_the_unrestricted_solve(probe):
    torch.manual_seed(9)
    h0, tgt = torch.randn(B, H), torch.randn(B, D)
    assert torch.equal(pinv_step(h0, tgt, probe), pinv_step(h0, tgt, probe, dims=None))
