"""Tests for pim.extractors and pim.editors.probe_steering."""

import numpy as np
import pytest
import torch

from pim.extractors.base import StateDefinition
from pim.extractors.linear import LinearExtractor
from pim.extractors.mlp import MLPExtractor
from pim.extractors.matching import identity_mse, hungarian_mse
from pim.extractors.training import train_extractor, fit_lstsq
from pim.editors.probe_steering import probe_decomposition, decompose_hidden, inject_state


# ── Fixtures ──────────────────────────────────────────────────────────────────

HIDDEN_SIZE = 32
STATE_SHAPE = (3, 2)  # 3 objects, 2D positions
N, T = 20, 10

def make_state_def():
    return StateDefinition(
        name="positions",
        state_shape=STATE_SHAPE,
        extract_fn=lambda b: b["positions"],
    )

def make_random_data(seed=0):
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((N, T, HIDDEN_SIZE)).astype(np.float32)
    gt = rng.standard_normal((N, T, *STATE_SHAPE)).astype(np.float32)
    return h, gt


# ── StateDefinition ───────────────────────────────────────────────────────────

def test_state_def_output_dim():
    sd = make_state_def()
    assert sd.output_dim == 3 * 2

def test_state_def_scalar():
    sd = StateDefinition(name="refl", state_shape=(4,), extract_fn=lambda b: b["r"])
    assert sd.output_dim == 4


# ── LinearExtractor ───────────────────────────────────────────────────────────

def test_linear_extractor_shape():
    sd = make_state_def()
    ext = LinearExtractor(HIDDEN_SIZE, sd)
    h = torch.randn(N, T, HIDDEN_SIZE)
    out = ext(h)
    assert out.shape == (N, T, *STATE_SHAPE)

def test_linear_extractor_unbatched():
    sd = make_state_def()
    ext = LinearExtractor(HIDDEN_SIZE, sd)
    h = torch.randn(HIDDEN_SIZE)
    out = ext(h)
    assert out.shape == STATE_SHAPE


# ── MLPExtractor ──────────────────────────────────────────────────────────────

def test_mlp_extractor_shape():
    sd = make_state_def()
    ext = MLPExtractor(HIDDEN_SIZE, sd, mlp_hidden=64)
    h = torch.randn(N, T, HIDDEN_SIZE)
    out = ext(h)
    assert out.shape == (N, T, *STATE_SHAPE)


# ── Matching losses ───────────────────────────────────────────────────────────

def test_identity_mse_zero():
    x = torch.randn(4, 3, 2)
    assert identity_mse(x, x).item() == pytest.approx(0.0)

def test_hungarian_mse_symmetry():
    """hungarian_mse should be <= identity_mse (it can only do better)."""
    pred = torch.randn(8, 2, 2)
    tgt = torch.randn(8, 2, 2)
    assert hungarian_mse(pred, tgt).item() <= identity_mse(pred, tgt).item() + 1e-5

def test_hungarian_mse_correct_swap():
    """If pred has objects swapped vs target, hungarian should recover zero loss."""
    tgt = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])   # (1, 2, 2)
    pred = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])  # swapped
    assert hungarian_mse(pred, tgt).item() == pytest.approx(0.0, abs=1e-6)


# ── Training ──────────────────────────────────────────────────────────────────

def test_train_extractor_runs():
    sd = make_state_def()
    ext = LinearExtractor(HIDDEN_SIZE, sd)
    h, gt = make_random_data()
    losses = train_extractor(ext, h, gt, n_epochs=3, lr=1e-2, batch_size=8)
    assert len(losses) == 3
    assert all(isinstance(l, float) for l in losses)

def test_fit_lstsq_returns_mse():
    sd = make_state_def()
    ext = LinearExtractor(HIDDEN_SIZE, sd)
    h, gt = make_random_data()
    mse = fit_lstsq(ext, h, gt)
    assert isinstance(mse, float)
    assert mse >= 0.0

def test_fit_lstsq_exact_on_linear_data():
    """fit_lstsq should achieve near-zero MSE when data is exactly linear."""
    H, D = 16, 6
    sd = StateDefinition(name="test", state_shape=(D,), extract_fn=lambda b: b["x"])
    ext = LinearExtractor(H, sd)

    # Generate exactly linear data: gt = h @ W.T + b
    rng = np.random.default_rng(42)
    W = rng.standard_normal((D, H)).astype(np.float32)
    b_true = rng.standard_normal(D).astype(np.float32)
    h = rng.standard_normal((50, 10, H)).astype(np.float32)
    gt = (h @ W.T + b_true).reshape(50, 10, D)

    mse = fit_lstsq(ext, h, gt)
    assert mse < 1e-5


# ── Probe steering ────────────────────────────────────────────────────────────

def test_probe_decomposition_shapes():
    sd = make_state_def()
    ext = LinearExtractor(HIDDEN_SIZE, sd)
    A, b, A_pinv = probe_decomposition(ext)
    assert A.shape == (sd.output_dim, HIDDEN_SIZE)
    assert b.shape == (sd.output_dim,)
    assert A_pinv.shape == (HIDDEN_SIZE, sd.output_dim)

def test_probe_decomposition_pinv_property():
    """A @ A_pinv ≈ I_D (pseudoinverse property)."""
    sd = make_state_def()
    ext = LinearExtractor(HIDDEN_SIZE, sd)
    A, b, A_pinv = probe_decomposition(ext)
    eye_approx = A @ A_pinv
    torch.testing.assert_close(eye_approx, torch.eye(sd.output_dim), atol=1e-5, rtol=0)

def test_decompose_hidden_sums_to_h():
    """h_parallel + h_perp should equal h exactly."""
    sd = make_state_def()
    ext = LinearExtractor(HIDDEN_SIZE, sd)
    A, b, A_pinv = probe_decomposition(ext)
    h = torch.randn(5, HIDDEN_SIZE)
    h_par, h_perp = decompose_hidden(h, A, A_pinv)
    torch.testing.assert_close(h_par + h_perp, h, atol=1e-5, rtol=0)

def test_inject_state_round_trip():
    """After inject_state, the probe should read back the target exactly."""
    sd = make_state_def()
    ext = LinearExtractor(HIDDEN_SIZE, sd)
    A, b, A_pinv = probe_decomposition(ext)

    h = torch.randn(5, HIDDEN_SIZE)
    target = torch.randn(5, sd.output_dim)

    h_edited = inject_state(h, target, A, A_pinv, b)
    readback = h_edited @ A.T + b   # what the probe would read
    torch.testing.assert_close(readback, target, atol=1e-4, rtol=0)

def test_inject_state_preserves_null_space():
    """inject_state should not change the null-space (perp) component of h."""
    sd = make_state_def()
    ext = LinearExtractor(HIDDEN_SIZE, sd)
    A, b, A_pinv = probe_decomposition(ext)

    h = torch.randn(5, HIDDEN_SIZE)
    target = torch.randn(5, sd.output_dim)

    _, h_perp_orig = decompose_hidden(h, A, A_pinv)
    h_edited = inject_state(h, target, A, A_pinv, b)
    _, h_perp_new = decompose_hidden(h_edited, A, A_pinv)

    torch.testing.assert_close(h_perp_new, h_perp_orig, atol=1e-5, rtol=0)
