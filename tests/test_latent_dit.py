"""Tests for pim.world_models.latent_dit — VAE freezing, protocol, state views, latent rollout."""

import dataclasses

import pytest
import torch

from pim.world_models.latent_dit import LatentDiTConfig, LatentDiTModel, LatentDiTState
from pim.world_models.protocol import HiddenStateModel, WorldModel
from pim.world_models.vae import ObsVAE, VAEConfig, fit_latent_scale

R = 16  # obs_res
Z = 4  # latent_dim
W = 3  # window (latent frames carried)
D = 32  # d_model
B, T = 3, 10


def make_model(seed: int = 0) -> LatentDiTModel:
    torch.manual_seed(seed)
    cfg = LatentDiTConfig(
        vae=dict(input_dim=R, latent_dim=Z, hidden=32, n_layers=1, latent_scale=1.5),
        core=dict(
            d_model=D, n_layers=2, n_heads=2, window=W, n_mean_eps=4, n_sample_steps=3
        ),
    )
    model = LatentDiTModel(cfg)
    with torch.no_grad():  # move off the zero-init identity start
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    model.eval()
    return model


def make_obs(seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, T, R, generator=g)


# ── VAE ───────────────────────────────────────────────────────────────────────


def test_vae_shapes_and_normalization_roundtrip():
    vae = ObsVAE(VAEConfig(input_dim=R, latent_dim=Z, hidden=32, n_layers=1))
    obs = torch.rand(5, 7, R)
    mu, logvar = vae.encode(obs)
    assert mu.shape == logvar.shape == (5, 7, Z)
    assert vae.decode(mu).shape == (5, 7, R)
    vae.cfg.latent_scale = 2.0
    zn = vae.encode_normalized(obs)
    assert torch.allclose(zn * 2.0, mu, atol=1e-6)
    assert torch.allclose(vae.decode_normalized(zn), vae.decode(mu), atol=1e-6)


def test_vae_decode_is_clamped_to_observation_range():
    vae = ObsVAE(VAEConfig(input_dim=R, latent_dim=Z, hidden=32, n_layers=1))
    out = vae.decode(torch.randn(4, Z) * 50)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_fit_latent_scale_matches_posterior_std():
    vae = ObsVAE(VAEConfig(input_dim=R, latent_dim=Z, hidden=32, n_layers=1))
    obs = torch.rand(256, R)
    scale = fit_latent_scale(vae, obs)
    assert scale == pytest.approx(float(vae.encode(obs)[0].std()), rel=1e-5)


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_implements_protocols():
    model = make_model()
    assert isinstance(model, WorldModel)
    assert isinstance(model, HiddenStateModel)


def test_forward_shapes():
    model = make_model()
    pred, state = model(make_obs())
    assert pred.shape == (B, T - 1, R)
    assert isinstance(state, LatentDiTState)
    assert state.latent_buffer.shape == (B, W, Z)
    assert (state.length == W).all()


def test_step_and_predict_step_shapes():
    model = make_model()
    obs = make_obs()
    state = None
    for t in range(3):
        pred, state = model.step(obs[:, t], state)
        assert pred.shape == (B, R)
        assert int(state.length[0]) == min(t + 1, W)
    pred_next, state_next = model.predict_step(state)
    assert pred_next.shape == (B, R)


def test_step_matches_observe_sequence():
    """Sequential step() reproduces the batched teacher-forcing pass."""
    model = make_model()
    obs = make_obs()
    pred_seq, _ = model.observe_sequence(obs)
    state = None
    with torch.no_grad():
        for t in range(T - 1):
            pred_t, state = model.step(obs[:, t], state)
            assert torch.allclose(pred_t, pred_seq[:, t], atol=1e-5), f"t={t}"


def test_predictions_are_causal_and_window_bounded():
    model = make_model()
    obs = make_obs()
    t_cut = 5
    mod = obs.clone()
    mod[:, t_cut + 1 :] = torch.rand_like(mod[:, t_cut + 1 :])
    pred, _ = model.observe_sequence(obs)
    pred_mod, _ = model.observe_sequence(mod)
    assert torch.allclose(pred[:, : t_cut + 1], pred_mod[:, : t_cut + 1], atol=1e-5)
    # frames older than the window cannot affect a late prediction
    t = 8
    mod2 = obs.clone()
    mod2[:, : t - W + 1] = torch.rand_like(mod2[:, : t - W + 1])
    pred_mod2, _ = model.observe_sequence(mod2)
    assert torch.allclose(pred[:, t], pred_mod2[:, t], atol=1e-5)


# ── VAE stays frozen ──────────────────────────────────────────────────────────


def test_vae_is_frozen_and_gets_no_gradient():
    model = make_model()
    model.train()
    assert not model.vae.training, "VAE must stay in eval mode even when training"
    loss = model.diffusion_loss(make_obs())
    loss.backward()
    assert all(p.grad is None for p in model.vae.parameters())
    assert any(p.grad is not None for p in model.core.parameters())


def test_diffusion_loss_scalar_and_finite():
    model = make_model()
    loss = model.diffusion_loss(make_obs())
    assert loss.ndim == 0 and torch.isfinite(loss)


# ── State views ───────────────────────────────────────────────────────────────


def test_state_views_shapes_and_roundtrip():
    model = make_model()
    obs = make_obs()
    expected = {
        "latent_window": W * Z,
        "activations": D,
        "kv_cache": 2 * 2 * (W - 1) * D,
    }
    for view, size in expected.items():
        model.state_view = view
        assert model.hidden_size == size
        assert model.get_hidden_states(obs).shape == (B, T - 1, size)
    model.state_view = "latent_window"
    _, state = model.step(obs[:, 0])
    flat = model.flat_state(state)
    assert torch.allclose(
        model.state_from_flat(flat).latent_buffer, state.latent_buffer
    )


def test_readonly_views_reject_state_from_flat():
    model = make_model()
    model.state_view = "activations"
    with pytest.raises(ValueError, match="latent_window"):
        model.state_from_flat(torch.zeros(1, model.hidden_size))


def test_latent_window_view_is_the_encoded_frames():
    """The default flat state is literally the last W normalised latents."""
    model = make_model()
    obs = make_obs()
    h = model.get_hidden_states(obs)
    t = 5  # fully warmed
    with torch.no_grad():
        z = model.encode(obs[:, t - W + 1 : t + 1]).reshape(B, -1)
    assert torch.allclose(h[:, t], z, atol=1e-6)


def test_state_from_obs_matches_stepwise_state():
    model = model_ = make_model()
    obs = make_obs()
    state = None
    with torch.no_grad():
        for t in range(6):
            _, state = model.step(obs[:, t], state)
        direct = model_.state_from_obs(obs[:, :6])
    assert torch.allclose(direct.latent_buffer, state.latent_buffer, atol=1e-6)


# ── Prediction modes ──────────────────────────────────────────────────────────


def test_predict_modes_and_passthrough():
    model = make_model()
    obs = make_obs()
    model.predict_mode = "mean"
    assert model.core.predict_mode == "mean"
    p_mean, _ = model.step(obs[:, 0])
    model.predict_mode = "sample"
    p_sample, _ = model.step(obs[:, 0])
    assert not torch.allclose(p_mean, p_sample, atol=1e-5)
    model.predict_mode = "sample_fresh"
    model.noise_gen = torch.Generator().manual_seed(3)
    a, _ = model.step(obs[:, 0])
    model.noise_gen = torch.Generator().manual_seed(3)
    b, _ = model.step(obs[:, 0])
    assert torch.equal(a, b)


def test_core_uses_identity_data_transform():
    """Latents are pre-normalised, so the core must not rescale them."""
    model = make_model()
    assert model.core.cfg.data_transform == "identity"
    x = torch.randn(2, Z)
    assert torch.equal(model.core._to_diff(x), x)


# ── Checkpoint loader dispatch ────────────────────────────────────────────────


def test_load_checkpoint_dispatches_latent_dit(tmp_path):
    from pim.world_models.loader import load_checkpoint

    model = make_model()
    ckpt = {
        "epoch": 1,
        "model_state": model.state_dict(),
        "model_config": dataclasses.asdict(model.cfg),
        "val_loss": 0.0,
    }
    path = tmp_path / "best_model.pt"
    torch.save(ckpt, path)
    loaded, _ = load_checkpoint(path)
    assert isinstance(loaded, LatentDiTModel)
    p1, _ = model.observe_sequence(make_obs())
    p2, _ = loaded.observe_sequence(make_obs())
    assert torch.allclose(p1, p2, atol=1e-6)
