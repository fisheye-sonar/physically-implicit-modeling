"""Parity tests: BatchedInteractiveWorld must reproduce InteractiveWorld exactly.

The contract (see the module docstring of ``pim/simulator/interactive_batched.py``):

* With noise OFF (``drift_force_std=0``, ``obs_noise_std=0``) the batched world matches the
  scalar world **bit-for-bit in float64** — positions, velocities, observations, and the
  collision / wall / death / rebirth event flags — given the same initial state and actions.
* With noise ON, bit-parity is impossible (per-world numpy generators vs one batched tensor),
  so parity is statistical: matched noise sigma and matched death rates.

These tests are the reason we can trust the fast path; a silently different simulator would
invalidate every downstream result.
"""

import numpy as np
import pytest
import torch

from pim.simulator.config import SimConfig
from pim.simulator.interactive import InteractiveConfig, InteractiveWorld
from pim.simulator.interactive_batched import BatchedInteractiveWorld

TOL = 1e-10  # float64 bit-parity tolerance


def make_sim(n=2, obs_noise=0.0, obs_res=64):
    return SimConfig(
        n_objects=n,
        radius=0.5,
        obs_res=obs_res,
        obs_noise_std=obs_noise,
        fixed_reflectivities=True,
        boundary="bounce",
    )


def deterministic_cfg(**kw):
    """Noise-free config so the two implementations are directly comparable."""
    base = dict(dynamics="force", drift_force_std=0.0, friction=0.02, init_speed=0.28)
    base.update(kw)
    return InteractiveConfig(**base)


def build_pair(sim, cfg, B, seed=0):
    """Scalar worlds + a batched world seeded to the SAME initial conditions."""
    scal = [InteractiveWorld(sim, cfg, seed=seed + b) for b in range(B)]
    batch = BatchedInteractiveWorld(
        sim, cfg, batch=B, seed=seed, device="cpu", dtype=torch.float64
    )
    # copy the scalar initial conditions into the batched world so only the DYNAMICS are compared
    batch._pos = torch.tensor(
        np.stack([w.positions for w in scal]), dtype=torch.float64
    )
    batch._vel = torch.tensor(
        np.stack([w.velocities for w in scal]), dtype=torch.float64
    )
    return scal, batch


# ── 1. bit-exact dynamics parity, force mode ──────────────────────────────────
@pytest.mark.parametrize("dynamics", ["force", "shift"])
def test_bitexact_dynamics_parity(dynamics):
    """Positions, velocities and observations must match to float64 precision."""
    sim = make_sim(obs_noise=0.0)
    cfg = deterministic_cfg(dynamics=dynamics, base_drift_speed=0.06)
    B, T = 12, 40
    scal, batch = build_pair(sim, cfg, B)
    rng = np.random.default_rng(0)
    for t in range(T):
        a = rng.integers(-1, 2, size=(B, 2, 2)).astype(np.float64)
        for b, w in enumerate(scal):
            w.step(a[b])
        batch.step(torch.tensor(a, dtype=torch.float64))
        p_s = np.stack([w.positions for w in scal])
        v_s = np.stack([w.velocities for w in scal])
        assert (
            np.abs(p_s - batch.positions.numpy()).max() < TOL
        ), f"positions diverged at t={t}"
        assert (
            np.abs(v_s - batch.velocities.numpy()).max() < TOL
        ), f"velocities diverged at t={t}"


def test_bitexact_observation_parity():
    """The batched ray-caster must reproduce renderer.render_frame exactly."""
    sim = make_sim(obs_noise=0.0, obs_res=128)
    cfg = deterministic_cfg()
    B, T = 8, 30
    scal, batch = build_pair(sim, cfg, B)
    rng = np.random.default_rng(1)
    # re-render so the scalar worlds' cached obs matches their injected positions
    for t in range(T):
        a = rng.integers(-1, 2, size=(B, 2, 2)).astype(np.float64)
        o_s = np.stack(
            [w.step(a[b])[0] for b, w in enumerate(scal)]
        )  # float32 (scalar casts)
        o_b, _ = batch.step(torch.tensor(a, dtype=torch.float64))
        # InteractiveWorld._render() returns float32, so compare at that precision — and
        # require EXACT equality after the same cast, which is stronger than a tolerance.
        assert (o_s == o_b.float().numpy()).all(), (
            f"observations diverged at t={t}; max |diff| = "
            f"{np.abs(o_s - o_b.float().numpy()).max():.3e}"
        )


# ── 2. event-flag parity (collision / wall / death / rebirth sequencing) ──────
# NOTE: after a *rebirth* the two implementations legitimately diverge, because each
# resamples fresh initial conditions from its own RNG stream (per-world numpy vs one batched
# torch generator).  So event parity is tested with `reset_on_death=False` (the world freezes
# on death, nothing is resampled), and the death→dying→rebirth *timing* is tested separately
# without requiring post-rebirth states to agree.
def test_event_flag_parity_no_respawn():
    """collision / wall / died / alive must fire on exactly the same frames."""
    sim = make_sim(obs_noise=0.0)
    cfg = deterministic_cfg(
        death_on_collision=True,
        death_on_wall=True,
        reset_on_death=False,  # freeze on death => no RNG-driven divergence
        max_speed=1.0,
    )
    B, T = 16, 80
    scal, batch = build_pair(sim, cfg, B, seed=5)
    rng = np.random.default_rng(2)
    n_died = 0
    for t in range(T):
        a = rng.integers(-1, 2, size=(B, 2, 2)).astype(np.float64)
        infos = [w.step(a[b])[1] for b, w in enumerate(scal)]
        _, ib = batch.step(torch.tensor(a, dtype=torch.float64))
        for key in ["died", "collision"]:
            s_ = np.array([bool(i[key]) for i in infos])
            assert (
                s_ == ib[key].numpy()
            ).all(), (
                f"'{key}' flags differ at t={t}: scalar={s_}, batched={ib[key].numpy()}"
            )
        s_wall = np.stack([np.asarray(i["wall"]) for i in infos])
        assert (s_wall == ib["wall"].numpy()).all(), f"wall flags differ at t={t}"
        s_alive = np.array([bool(i["alive"]) for i in infos])
        assert (s_alive == ib["alive"].numpy()).all(), f"alive differs at t={t}"
        n_died += int(np.array([bool(i["died"]) for i in infos]).sum())
    d_s = np.array([w.deaths for w in scal])
    assert (d_s == batch.deaths.numpy()).all(), "cumulative death counts differ"
    assert n_died > 0, "test is vacuous — no deaths occurred"


def test_death_rebirth_timing_parity():
    """The death → N noise frames → rebirth state machine must have identical TIMING.

    Only compared up to each world's first rebirth; afterwards the two implementations
    legitimately differ because each resamples its own initial conditions.
    """
    sim = make_sim(obs_noise=0.0)
    NOISE = 3
    cfg = deterministic_cfg(
        death_on_collision=True,
        death_on_wall=True,
        reset_on_death=True,
        reset_noise_frames=NOISE,
        max_speed=1.0,
    )
    B, T = 16, 40
    scal, batch = build_pair(sim, cfg, B, seed=5)
    rng = np.random.default_rng(2)
    seq_s = [[] for _ in range(B)]
    seq_b = [[] for _ in range(B)]
    for t in range(T):
        a = rng.integers(-1, 2, size=(B, 2, 2)).astype(np.float64)
        infos = [w.step(a[b])[1] for b, w in enumerate(scal)]
        _, ib = batch.step(torch.tensor(a, dtype=torch.float64))
        for b in range(B):
            seq_s[b].append(
                (
                    "died"
                    if infos[b]["died"]
                    else (
                        "dying"
                        if infos[b]["dying"]
                        else "rebirth" if infos[b]["rebirth"] else "-"
                    )
                )
            )
            seq_b[b].append(
                (
                    "died"
                    if bool(ib["died"][b])
                    else (
                        "dying"
                        if bool(ib["dying"][b])
                        else "rebirth" if bool(ib["rebirth"][b]) else "-"
                    )
                )
            )
    checked = 0
    for b in range(B):
        # compare only up to (and including) the first rebirth
        cut = seq_s[b].index("rebirth") + 1 if "rebirth" in seq_s[b] else len(seq_s[b])
        assert (
            seq_s[b][:cut] == seq_b[b][:cut]
        ), f"world {b} event timeline differs:\n  scalar  {seq_s[b][:cut]}\n  batched {seq_b[b][:cut]}"
        if "rebirth" in seq_s[b]:
            checked += 1
            i = seq_s[b].index("died")
            assert (
                seq_s[b][i + 1 : i + 1 + NOISE] == ["dying"] * NOISE
            ), "bad scalar dying run"
            assert (
                seq_b[b][i + 1 : i + 1 + NOISE] == ["dying"] * NOISE
            ), "bad batched dying run"
            assert (
                seq_b[b][i + 1 + NOISE] == "rebirth"
            ), "rebirth did not follow the noise run"
    assert checked > 0, "test is vacuous — no world completed a death→rebirth cycle"


def test_shift_guard_parity():
    """Shift-mode's sequential accept-guard (and its 'blocked' flags) must match."""
    sim = make_sim(obs_noise=0.0)
    cfg = deterministic_cfg(dynamics="shift", base_drift_speed=0.0, shift_scale=0.3)
    B, T = 12, 40
    scal, batch = build_pair(sim, cfg, B, seed=7)
    for t in range(T):
        # drive both objects toward the frustum centre so the guard actually engages
        a = np.zeros((B, 2, 2))
        mid = np.array([0.0, (sim.y_near + sim.y_far) / 2])
        for b, w in enumerate(scal):
            for i in range(2):
                a[b, i] = np.clip(mid - w.positions[i], -1, 1)
        infos = [w.step(a[b])[1] for b, w in enumerate(scal)]
        _, ib = batch.step(torch.tensor(a, dtype=torch.float64))
        s_blocked = np.stack([np.asarray(i["blocked"]) for i in infos])
        assert (
            s_blocked == ib["blocked"].numpy()
        ).all(), f"blocked flags differ at t={t}"
        p_s = np.stack([w.positions for w in scal])
        assert (
            np.abs(p_s - batch.positions.numpy()).max() < TOL
        ), f"positions diverged at t={t}"
    assert s_blocked.sum() > 0, "test is vacuous — the guard never engaged"


# ── 3. statistical parity with noise ON ───────────────────────────────────────
def test_noise_statistics_match():
    """With observation noise on, the added noise must have the right sigma."""
    std = 0.2
    sim_noisy, sim_clean = make_sim(obs_noise=std), make_sim(obs_noise=0.0)
    cfg = deterministic_cfg()
    B = 64
    wn = BatchedInteractiveWorld(sim_noisy, cfg, batch=B, seed=3, dtype=torch.float64)
    wc = BatchedInteractiveWorld(sim_clean, cfg, batch=B, seed=3, dtype=torch.float64)
    wc._pos, wc._vel = wn.positions, wn.velocities
    a = torch.zeros(B, 2, 2, dtype=torch.float64)
    resid = []
    for _ in range(20):
        on, _ = wn.step(a)
        oc, _ = wc.step(a)
        # only unclipped interior pixels carry the raw noise
        interior = (oc > 0.05) & (oc < 0.95)
        resid.append((on - oc)[interior].numpy())
    r = np.concatenate(resid)
    assert abs(r.std() - std) < 0.02, f"noise sigma {r.std():.3f} != {std}"
    assert abs(r.mean()) < 0.02, f"noise mean {r.mean():.3f} != 0"


def test_death_rate_matches_scalar_statistically():
    """With full noise on, the two implementations must die at the same rate."""
    sim = make_sim(obs_noise=0.2)
    cfg = InteractiveConfig(
        dynamics="force",
        death_on_collision=True,
        death_on_wall=True,
        reset_on_death=True,
        reset_noise_frames=4,
        init_speed=0.28,
    )
    B, T = 48, 120
    scal = [InteractiveWorld(sim, cfg, seed=1000 + b) for b in range(B)]
    for w in scal:
        for _ in range(T):
            w.step(np.zeros((2, 2)))
    rate_s = sum(w.deaths for w in scal) / (B * T)
    wb = BatchedInteractiveWorld(sim, cfg, batch=B, seed=1000)
    for _ in range(T):
        wb.step(torch.zeros(B, 2, 2))
    rate_b = float(wb.deaths.sum()) / (B * T)
    assert abs(rate_s - rate_b) < 0.35 * max(
        rate_s, 1e-9
    ), f"death rates differ too much: scalar {rate_s:.4f} vs batched {rate_b:.4f}"


# ── 4. mechanics of the batched class itself ──────────────────────────────────
def test_action_schema_and_clipping():
    sim = make_sim(obs_noise=0.0)
    w = BatchedInteractiveWorld(
        sim, deterministic_cfg(), batch=4, seed=0, dtype=torch.float64
    )
    _, i3 = w.step(torch.zeros(4, 2, 3, dtype=torch.float64))  # (n,3) schema, active=0
    assert torch.allclose(i3["action"], torch.zeros(4, 2, 2, dtype=torch.float64))
    _, ic = w.step(torch.full((4, 2, 2), 5.0, dtype=torch.float64))  # must clamp to 1
    assert float(ic["action"].max()) == 1.0


def test_bounce_keeps_objects_in_frustum_batched():
    sim = make_sim(obs_noise=0.0)
    w = BatchedInteractiveWorld(
        sim, deterministic_cfg(max_speed=1.0), batch=32, seed=4, dtype=torch.float64
    )
    g = torch.Generator().manual_seed(0)
    r = sim.radius
    for _ in range(150):
        w.step(torch.rand(32, 2, 2, generator=g, dtype=torch.float64) * 2 - 1)
        p = w.positions
        y, x = p[..., 1], p[..., 0]
        assert bool(((y - r) >= sim.y_near - 1e-9).all()) and bool(
            ((y + r) <= sim.y_far + 1e-9).all()
        )
        t = (y - sim.y_near) / (sim.y_far - sim.y_near)
        x_lim = sim.x_near + (sim.x_far - sim.x_near) * t - r
        assert bool((x.abs() <= x_lim + 1e-9).all()), "object escaped the frustum"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_matches_cpu():
    """The GPU path must agree with the CPU path (float64, noise off)."""
    sim = make_sim(obs_noise=0.0)
    cfg = deterministic_cfg()
    B, T = 16, 25
    wc = BatchedInteractiveWorld(
        sim, cfg, batch=B, seed=11, device="cpu", dtype=torch.float64
    )
    wg = BatchedInteractiveWorld(
        sim, cfg, batch=B, seed=11, device="cuda", dtype=torch.float64
    )
    wg._pos, wg._vel = wc.positions.cuda(), wc.velocities.cuda()
    g = torch.Generator().manual_seed(0)
    for t in range(T):
        a = torch.rand(B, 2, 2, generator=g, dtype=torch.float64) * 2 - 1
        oc, _ = wc.step(a)
        og, _ = wg.step(a.cuda())
        assert (oc - og.cpu()).abs().max() < 1e-9, f"cuda/cpu diverged at t={t}"
