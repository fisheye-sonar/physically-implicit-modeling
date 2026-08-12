"""Tests for pim.world_models.dit — protocol, determinism, causality, window."""

import pytest
import torch

from pim.world_models.dit import DiTModel, DiTState, ModelConfig
from pim.world_models.protocol import HiddenStateModel, WorldModel

# ── Fixtures ──────────────────────────────────────────────────────────────────

R = 16  # obs_res
W = 4  # window
B, T = 3, 12


def make_model(seed: int = 0, **overrides) -> DiTModel:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        input_dim=R,
        d_model=32,
        n_layers=2,
        n_heads=2,
        window=W,
        n_sample_steps=3,
        **overrides,
    )
    model = DiTModel(cfg)
    # Perturb away from the zero-initialised (identity) start so the tests
    # exercise a non-trivial function of the inputs.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    model.eval()
    return model


def make_obs(seed: int = 1, batch: int = B, t: int = T) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(batch, t, R, generator=g)


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_implements_protocols():
    model = make_model()
    assert isinstance(model, WorldModel)
    assert isinstance(model, HiddenStateModel)


def test_forward_shapes():
    model = make_model()
    pred, state = model(make_obs())
    assert pred.shape == (B, T - 1, R)
    assert isinstance(state, DiTState)
    assert state.obs_buffer.shape == (B, W, R)
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
    assert int(state_next.length[0]) == min(4, W)


def test_observe_sequence_and_hidden_state_shapes():
    model = make_model()
    obs = make_obs()
    pred, h = model.observe_sequence(obs)
    assert pred.shape == (B, T - 1, R)
    assert h.shape == (B, T - 1, model.hidden_size)
    h2 = model.get_hidden_states(obs)
    assert torch.allclose(h, h2)


def test_flat_state_roundtrip():
    model = make_model()
    obs = make_obs()
    _, state = model.step(obs[:, 0])
    flat = model.flat_state(state)
    assert flat.shape == (B, model.hidden_size)
    state2 = model.state_from_flat(flat)
    assert torch.allclose(state.obs_buffer, state2.obs_buffer)
    # Injected states are assumed fully warmed
    assert (state2.length == W).all()


def test_decode_matches_step_prediction():
    model = make_model()
    obs = make_obs()
    pred, state = model.step(obs[:, 0])
    assert torch.allclose(model.decode(state), pred, atol=1e-6)


# ── Determinism ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["mean", "sample"])
def test_prediction_is_deterministic(mode):
    model = make_model()
    model.predict_mode = mode
    obs = make_obs()
    pred1, _ = model.observe_sequence(obs)
    pred2, _ = model.observe_sequence(obs)
    assert torch.equal(pred1, pred2)
    p1, s1 = model.step(obs[:, 0])
    p2, s2 = model.step(obs[:, 0])
    assert torch.equal(p1, p2)
    r1, _ = model.predict_step(s1)
    r2, _ = model.predict_step(s2)
    assert torch.equal(r1, r2)


def test_predict_modes_differ():
    """Mean readout and ODE sample are different deterministic functions."""
    model = make_model()
    obs = make_obs()
    model.predict_mode = "mean"
    pred_mean, _ = model.step(obs[:, 0])
    model.predict_mode = "sample"
    pred_sample, _ = model.step(obs[:, 0])
    assert not torch.allclose(pred_mean, pred_sample, atol=1e-5)


# ── Causality and window ──────────────────────────────────────────────────────


def test_predictions_are_causal():
    """pred[:, t] must not change when observations after t change."""
    model = make_model()
    obs = make_obs()
    t_cut = 6
    obs_mod = obs.clone()
    obs_mod[:, t_cut + 1 :] = torch.rand_like(obs_mod[:, t_cut + 1 :])
    pred, _ = model.observe_sequence(obs)
    pred_mod, _ = model.observe_sequence(obs_mod)
    assert torch.allclose(pred[:, : t_cut + 1], pred_mod[:, : t_cut + 1], atol=1e-5)
    # sanity: the change DOES affect later predictions
    assert not torch.allclose(pred[:, t_cut + 1 :], pred_mod[:, t_cut + 1 :], atol=1e-5)


def test_state_is_bounded_by_window():
    """pred[:, t] must not change when frames older than the window change."""
    model = make_model()
    obs = make_obs()
    t = 9  # window covers frames t-W+1..t = 6..9
    obs_mod = obs.clone()
    obs_mod[:, : t - W + 1] = torch.rand_like(obs_mod[:, : t - W + 1])
    pred, _ = model.observe_sequence(obs)
    pred_mod, _ = model.observe_sequence(obs_mod)
    assert torch.allclose(pred[:, t], pred_mod[:, t], atol=1e-5)


def test_step_matches_observe_sequence():
    """Sequential step() must reproduce the batched teacher-forcing pass."""
    model = make_model()
    obs = make_obs()
    pred_seq, _ = model.observe_sequence(obs)
    state = None
    with torch.no_grad():
        for t in range(T - 1):
            pred_t, state = model.step(obs[:, t], state)
            assert torch.allclose(pred_t, pred_seq[:, t], atol=1e-5), f"t={t}"


# ── Training objective ────────────────────────────────────────────────────────


def test_diffusion_loss_scalar_and_finite():
    model = make_model()
    loss = model.diffusion_loss(make_obs())
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_diffusion_loss_backward():
    model = make_model()
    model.train()
    loss = model.diffusion_loss(make_obs())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


# ── State views ───────────────────────────────────────────────────────────────


def test_state_views_shapes():
    model = make_model()
    obs = make_obs()
    cfg = model.cfg
    expected = {
        "obs_window": W * R,
        "activations": cfg.d_model,
        "kv_cache": cfg.n_layers * 2 * (W - 1) * cfg.d_model,
    }
    for view, size in expected.items():
        model.state_view = view
        assert model.hidden_size == size
        h = model.get_hidden_states(obs)
        assert h.shape == (B, T - 1, size)


def test_readonly_views_reject_state_from_flat():
    model = make_model()
    model.state_view = "activations"
    with pytest.raises(ValueError, match="obs_window"):
        model.state_from_flat(torch.zeros(1, model.hidden_size))


def test_obs_window_view_is_raw_observations():
    """The default flat state is literally the last W observations."""
    model = make_model()
    obs = make_obs()
    h = model.get_hidden_states(obs)
    t = 7  # fully-warmed position
    window = obs[:, t - W + 1 : t + 1].reshape(B, -1)
    assert torch.allclose(h[:, t], window)


# ── Checkpoint loader dispatch ────────────────────────────────────────────────


def test_load_checkpoint_dispatches_dit(tmp_path):
    import dataclasses

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
    loaded, info = load_checkpoint(path)
    assert isinstance(loaded, DiTModel)
    pred1, _ = model.observe_sequence(make_obs())
    pred2, _ = loaded.observe_sequence(make_obs())
    assert torch.allclose(pred1, pred2, atol=1e-6)


def test_trunk_resid_sink_collects_all_points():
    """resid_sink returns n_layers+1 residual points; the last equals feats."""
    model = make_model()
    obs = make_obs()
    cur = model._to_diff(obs[:, :-1])
    nxt = model._to_diff(obs[:, 1:])
    tau = torch.zeros(B, T - 1)
    from pim.world_models.dit.blocks import band_causal_mask

    mask = band_causal_mask(T - 1, model.cfg.window, obs.device)
    sink: list = []
    with torch.no_grad():
        feats, _ = model._trunk(cur, nxt, tau, mask, resid_sink=sink)
    assert len(sink) == model.cfg.n_layers + 1
    for x in sink:
        assert x.shape == (B, T - 1, model.cfg.d_model)
    assert torch.equal(sink[-1], feats)


def test_sample_fresh_is_stochastic_and_seedable():
    """sample_fresh draws per-row noise; a seeded generator makes it reproducible."""
    model = make_model()
    model.predict_mode = "sample_fresh"
    obs = make_obs()
    p1, _ = model.step(obs[:, 0])
    p2, _ = model.step(obs[:, 0])
    assert not torch.allclose(p1, p2, atol=1e-6), "fresh noise should vary between calls"
    # rows differ from each other (per-sample noise, not one shared vector)
    model.noise_gen = torch.Generator().manual_seed(0)
    a, _ = model.step(obs[:, 0])
    model.noise_gen = torch.Generator().manual_seed(0)
    b, _ = model.step(obs[:, 0])
    assert torch.equal(a, b), "same seed must reproduce the sample"


def test_identity_data_transform_is_a_no_op():
    """data_transform='identity' leaves latents unscaled in both directions."""
    model = make_model(data_transform="identity")
    x = torch.randn(4, R)
    assert torch.equal(model._to_diff(x), x)
    assert torch.equal(model._from_diff(x), x)
    # and the default is still the [0,1] → [-1,1] map
    d = make_model()
    assert torch.allclose(d._to_diff(torch.zeros(2, R)), -torch.ones(2, R))
