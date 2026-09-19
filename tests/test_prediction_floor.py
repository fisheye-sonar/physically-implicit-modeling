"""Predictive loss + Bayes floor (2026-09-19): the metric arithmetic, the sampler's renderer and
prior against the canonical simulator, the blink marker model against ``blink_schedule``, the
exact Othello floor, and a tiny end-to-end sampler run. CPU-sized; nothing here needs a GPU."""
import dataclasses
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pim.environments import layout
from pim.metrics import prediction as mp

SIM8 = {"y_near": 3.0, "y_far": 12.0, "x_near": 1.5, "x_far": 6.0, "radius": 1.0, "dt": 1.0, "obs_res": 10,
        "drop_edge_rays": True, "refl_min": 0.4, "refl_max": 0.8, "speed_min": 0.05, "speed_max": 0.12,
        "collision_margin": 1.6, "n_objects": 2, "fixed_reflectivities": True, "boundary": "open",
        "always_in_frustum": True, "obs_noise_std": 0.0, "position_noise_std": 0.0}


def test_metric_arithmetic_matches_the_training_objectives():
    g = torch.Generator().manual_seed(0)
    lg, y = torch.randn(5, 7, 11, generator=g), torch.randint(0, 11, (5, 7), generator=g)
    ce = mp.next_token_ce(lg.numpy(), y.numpy())
    assert ce.shape == (5,) and math.isclose(ce.mean(), float(F.cross_entropy(lg.reshape(-1, 11), y.reshape(-1))), rel_tol=1e-6)
    y2 = y.clone()
    y2[:, -2:] = -100                                           # ce_next_move's IGNORE padding
    ce2 = mp.next_token_ce(lg.numpy(), y2.numpy(), ignore_index=-100)
    assert math.isclose(ce2.mean(), float(F.cross_entropy(lg.reshape(-1, 11), y2.reshape(-1), ignore_index=-100)), rel_tol=1e-6)
    a, b = torch.rand(4, 6, 8, generator=g), torch.rand(4, 6, 8, generator=g)
    assert math.isclose(mp.next_frame_mse(a.numpy(), b.numpy()).mean(), float(F.mse_loss(a, b)), rel_tol=1e-6)


def test_expected_frame_drops_unk_and_is_the_mean_frame():
    frames = np.array([[np.nan, np.nan], [0.0, 0.4], [0.8, 0.8]])           # id 0 = UNK
    m, dropped = mp.expected_frame(np.array([[0.0, 1.0, 0.0], [0.5, 0.25, 0.25]]), frames)
    assert np.allclose(m[0], [0.0, 0.4]) and np.allclose(m[1], [0.4, 0.6]) and np.allclose(dropped, [0.0, 0.5])


def test_bracket_and_excess():
    assert mp.floor_bracket({"ce": {"exact": 2.0}}, "ce") == (2.0, 2.0)
    assert mp.floor_bracket({"mse": {"lo": 0.1, "hi": 0.3}}, "mse") == (0.1, 0.3) and mp.floor_bracket({}, "mse") is None
    # the DISPLAY form: an exact floor has no ±; a sampled one is midpoint ± (half-width + the larger SE)
    assert mp.floor_estimate({"ce": {"exact": 2.0}}, "ce") == (2.0, 0.0) and mp.floor_estimate(None, "ce") is None
    v, pm = mp.floor_estimate({"mse": {"lo": 0.1, "hi": 0.3, "lo_se": 0.01, "hi_se": 0.02}}, "mse")
    assert math.isclose(v, 0.2) and math.isclose(pm, 0.12)
    e, epm, rel = mp.excess_estimate(0.25, (v, pm))
    assert math.isclose(e, 0.05) and math.isclose(epm, 0.12) and math.isclose(rel, 0.25)
    assert math.isclose(mp.gap_closed(0.25, 1.2, 0.2), 0.95) and math.isnan(mp.gap_closed(0.25, None, 0.2))
    lo, hi = mp.excess(0.35, (0.1, 0.3))
    assert math.isclose(lo, 0.05) and math.isclose(hi, 0.25) and mp.excess(0.3, None) is None


def test_sampler_renderer_matches_render_frame_and_prior_matches_the_generator():
    from pim.environments.discworld.bayes import World
    from pim.environments.discworld.renderer import render_frame
    from pim.metrics.zone_editability import sim_config_from

    W = World(SIM8, 40, "cpu")
    gen = torch.Generator().manual_seed(0)
    p0, v = W.sample_prior(400, gen)
    cfg = sim_config_from(SIM8, 2)
    rad, refl = np.array([1.0, 1.0]), np.array([0.4, 0.8])
    vis = torch.rand(400, 2, generator=gen) > 0.3
    ours = W.render(p0, vis).numpy()
    for i in range(400):
        ref = render_frame(p0[i].numpy(), rad, refl, cfg, visible=vis[i].numpy())[2]
        assert np.array_equal(np.rint(ref / 0.4).astype(int), ours[i]), i
    # the prior's support is the generator's: inside the frustum, speeds in range
    assert bool((W.xlim(p0[..., 1]) >= p0[..., 0].abs()).all())
    sp = v.norm(dim=-1)
    assert float(sp.min()) >= 0.05 and float(sp.max()) <= 0.12
    # acceptance = sim.simulate's rule on a straight trajectory
    from pim.environments.discworld.sim import fully_in_frustum
    full = dataclasses.replace(cfg, always_in_frustum=True)
    tr = W.traj(p0, v)
    for i in range(60):
        t = tr[i].numpy()
        ok = fully_in_frustum(t, 1.0, full) and bool((np.linalg.norm(t[:, 0] - t[:, 1], axis=-1) >= 3.2).all())
        assert ok == bool(W.accepted(tr[i])), i


def test_marker_process_is_calibrated_against_blink_schedule():
    from pim.environments.discworld.bayes import marker_process
    from pim.environments.discworld.blink import blink_schedule
    from pim.environments.discworld.sim import SimConfig

    sim = {"blink_prob": 0.05, "blink_mean": 7.0, "blink_max": 12, "blink_warmup": 3}
    vis = np.stack([blink_schedule(SimConfig(seed=k, n_frames=40, n_objects=2, **sim), 2) for k in range(4000)])
    mk = marker_process(vis, sim)                                  # raises if any realised config has p = 0
    m = vis[:, :-1] != vis[:, 1:]                                  # the markers that occurred, frames 0..T-2
    exp, got = mk["pi"][:, :-1].sum((0, 1)), m.sum((0, 1))
    assert np.all(np.abs(exp - got) < 4 * np.sqrt(exp)), (exp, got)
    # the mean log-probability of what happened ≈ minus the mean entropy (a calibrated predictive)
    assert abs(-mk["logp_true"].mean() - mk["entropy"].mean()) < 0.1 * mk["entropy"].mean()
    assert not mk["pi"][:, -1].any() and not marker_process(np.ones((3, 40, 2), bool), {})["pi"].any()


def test_othello_exact_floor_is_the_gates_floor():
    from pim.environments.othello import arms as oa
    from pim.environments.othello import corpus as oc
    from pim.environments.othello.bayes import exact_ce_floor
    from pim.models.registry import build

    try:
        tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("test",), instance="oth-adjacent-flip")["test"])
    except Exception:
        pytest.skip("oth-adjacent-flip test split absent")
    tok, ln = tok[:6], ln[:6]
    rules = oc.rules_of("oth-adjacent-flip")
    ce, top1, n = exact_ce_floor(tok, ln, **rules)
    torch.manual_seed(0)
    model = build("transformer_l_tokens", {"vocab": 61, "block_size": 59, "n_layer": 1, "n_head": 2, "n_embd": 16,
                                           "dropout": 0.0}).eval().to(oa.DEV)
    g = oa.gates(model, tok, ln, log=None, **rules)
    assert math.isclose(ce, g["bayes_ce"], rel_tol=1e-12) and math.isclose(top1, g["bayes_top1"], rel_tol=1e-12)
    assert n == g["n_positions"]


@pytest.mark.skipif(not layout.eval_file("discworld", "dw-8ray").exists(), reason="dw-8ray eval split absent")
def test_sampler_end_to_end_tiny():
    from pim.environments.discworld.bayes import bayes_floor

    r = bayes_floor("dw-8ray", n_seq=6, particles=48, sweeps=2, init_sweeps=20, device="cpu", exact_draws=0, log=lambda *a, **k: None)
    assert r["diagnostics"]["parity"] == 1.0 and r["n_sequences"] == 6
    for obj in ("mse", "ce"):
        assert 0 < r[obj]["lo"] and 0 < r[obj]["hi"] and len(r[obj]["by_position"]["lo"]) == 39
    assert r["mse"]["lo"] < 0.05                                   # far below the unconditional frame variance (0.108)
    json.dumps(r)


@pytest.mark.skipif(not layout.eval_file("discworld", "dw-8ray").exists(), reason="dw-8ray eval split absent")
def test_trivial_predictors_bound_the_problem():
    from pim.environments.discworld.bayes import trivial_predictors

    r = trivial_predictors("dw-8ray", n_seq=200)
    # the constant frame's loss is the observation's per-ray variance; copying the last frame is far better
    assert 0.05 < r["mse"]["value"] < 0.2 and 0 < r["persistence_mse"]["value"] < r["mse"]["value"]
    assert 0 < r["ce"]["value"] < math.log(3 ** 8)                 # below the uniform distribution over all patterns
