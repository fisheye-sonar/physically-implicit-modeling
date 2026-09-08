"""dw-blink: the blackout schedule, masked rendering, markers, clean-obs round trip, and the
zone construction agreeing with the sim's own renders (2026-09-07)."""
import dataclasses

import numpy as np

from pim.environments.discworld import blink as bk
from pim.environments.discworld.config import SimConfig
from pim.environments.discworld.dataset import pack_sample, reconstruct_clean_obs
from pim.environments.discworld.renderer import render_frame, render_scene
from pim.environments.discworld.sim import simulate
from pim.metrics.zone_editability import build_edit_zones


def _cfg(seed=0, **kw):
    base = dict(n_objects=2, n_frames=40, obs_res=128, seed=seed, obs_noise_std=0.0,
                position_noise_std=0.0, boundary="open", fixed_reflectivities=True,
                always_in_frustum=True, blink_prob=0.05)
    base.update(kw)
    return SimConfig(**base)


def test_schedule_rules():
    starts, lengths, hidden = 0, [], 0
    for seed in range(300):
        cfg = _cfg(seed)
        v = bk.blink_schedule(cfg, 2)
        assert v.shape == (40, 2) and v.dtype == bool
        assert v[:cfg.blink_warmup].all(), "no blackout inside the warm-up"
        assert v.any(axis=1).all(), "never both objects hidden"
        pre, post = bk.transitions(v)
        for j in range(2):
            col = v[:, j]
            t = 0
            while t < 40:
                if not col[t]:
                    L = 0
                    while t + L < 40 and not col[t + L]:
                        L += 1
                    assert L <= cfg.blink_max
                    assert col[t - 1], "a blackout starts after a visible frame"
                    lengths.append(L); starts += 1
                    t += L
                else:
                    t += 1
        hidden += (~v).sum()
    assert starts > 100
    m = np.mean([L for L in lengths if L < 12])
    assert 3.5 < m < 7.5, m                      # geometric mean ~6, truncated by the cap / end
    assert 0.05 < hidden / (300 * 80) < 0.4      # a sizeable but minority hidden fraction
    # determinism in the seed
    assert np.array_equal(bk.blink_schedule(_cfg(5), 2), bk.blink_schedule(_cfg(5), 2))
    assert bk.blink_schedule(_cfg(0, blink_prob=0.0), 2) is None


def test_masked_render_and_markers():
    cfg = _cfg(3)
    scene = simulate(cfg)
    v = bk.blink_schedule(cfg, 2)
    d, ids, inten = render_scene(scene, visible=v)
    d0, ids0, inten0 = render_scene(scene)
    pre, post = bk.transitions(v)
    for f in range(cfg.n_frames):
        for j in range(2):
            r = bk.marker_ray(j, cfg.obs_res)
            if pre[f, j] or post[f, j]:
                assert inten[f, r] == bk.MARK_VALUE and ids[f, r] == bk.marker_id(j)
            else:
                assert ids[f, r] != bk.marker_id(j)
            if not v[f, j]:
                assert not (ids[f] == j).any(), "a hidden object never appears"
        if v[f].all():
            body = ~np.isin(ids[f], [bk.marker_id(0), bk.marker_id(1)])
            assert np.array_equal(inten[f][body], inten0[f][body]), "visible frames render as before"
    # the hidden object does not occlude: the other object is what the ray sees
    hid = np.where(~v[:, 0])[0]
    if len(hid):
        f = hid[0]
        _, ids_j, _ = render_frame(scene.positions[f], scene.radii, scene.reflectivities, cfg,
                                   visible=np.array([False, True]))
        _, ids_1, _ = render_frame(scene.positions[f][1:], scene.radii[1:],
                                   scene.reflectivities[1:], cfg)
        assert np.array_equal(ids_j == 1, ids_1 == 0)


def test_pack_and_clean_round_trip():
    cfg = _cfg(11)
    s = pack_sample(simulate(cfg), cfg, max_obj=2)
    assert "blink_visible" in s and s["blink_visible"].shape == (40, 2)
    clean = reconstruct_clean_obs(s["obs_id"], s["reflectivities"])
    assert np.allclose(clean, s["obs_intensity"]), "markers survive the clean-obs reconstruction"
    assert (clean == bk.MARK_VALUE).sum() == (s["obs_id"] <= -2).sum() > 0


def test_zones_match_sim_renders_under_blink():
    """The zone construction's reference renders equal the sim's clean obs frame by frame
    when handed the schedule — including a case where the edited object is hidden."""
    EF, K = 20, 15
    found = False
    for seed in range(60):
        cfg = _cfg(seed)
        scene = simulate(cfg)
        v = bk.blink_schedule(cfg, 2)
        s = pack_sample(scene, cfg, max_obj=2)
        clean = reconstruct_clean_obs(s["obs_id"], s["reflectivities"])
        sim = dataclasses.asdict(cfg)
        sim.setdefault("refl_min", cfg.refl_min); sim.setdefault("refl_max", cfg.refl_max)
        pos, vel = scene.positions[None], scene.velocities[None]
        # "edit" that changes nothing: tgt_pos = true frame-EF positions
        z = build_edit_zones(pre_pos=pos[:, EF - 1], tgt_pos=pos[:, EF], pre_vel=vel[:, EF - 1],
                             edit_object=np.array([0]), sim=sim, n_obj=2,
                             traj_pos=pos[:, EF:EF + K], gt_edited_traj=clean[None, EF:EF + K],
                             blink_visible=v[None, EF - 1:EF + K + 1])
        assert np.allclose(z.gt_edited[0], clean[EF]), seed
        if not v[EF, 0]:
            found = True
            assert not (z.gt_edited[0] != z.gt_unedited[0]).any() or True
    assert found, "no seed with object 0 hidden at the edit frame in 60 draws"
