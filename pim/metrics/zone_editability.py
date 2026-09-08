"""Editability of a predicted FRAME against ray zones — discworld's construction.

(``editability.py`` until 2026-09-07; renamed beside ``set_editability.py`` so the two
constructions read as the pair they are. The Edit Index and fidelity FORMULAS live once in
``edit_index.py``; this module supplies the ingredients: the two clean renders and the
ray masks.)

**One implementation** (moved here 2026-08-31 from ``scripts/editability_metrics.py``,
where it lived under the same no-re-derivation rule). The registry row for each metric is
in ``research/REGISTRY.md``; this file is the code those rows refer to. Do not re-derive
these formulas in a notebook — the master notebook calls in here, and that is what makes
its numbers reviewable.

The Othello / token-model analogue (uniform-over-set references instead of ray zones)
is ``pim/metrics/set_editability.py`` — a separate module with distinct names, because
the two constructions share the FORMULA (``edit_index.py``) but not the ingredients.

Replaces (2026-07-30) the old ratio-style `reach % of swap` / `collateral % of swap` /
`selectivity` / `ghost ratio`. Those measured **change away from the unsteered rollout**,
normalised by the true-state swap — so an editor that merely *scrambled* the observation
scored a large "reach" (400%+ was observed), and the denominator was a soft, model-dependent
reference that varied a lot across models, making cross-model sweeps incomparable.

The replacement has two layers.

**Layer 1 — absolute error against ground truth, decomposed by ray zone.** Every number is
an RMSE against the simulator's true post-edit observation at the edit frame, restricted to
a zone of rays. Same units (observation intensity in [0,1]), lower is better everywhere, no
normalisation, no soft reference:

    edit_frame_rmse   all rays
    target_rmse       rays the edited object must occupy after the edit
    ghost_rmse        rays it occupied before and must now vacate
    collateral_rmse   rays the OTHER object occupies (it must not move)

**Layer 2 — the Edit Index, a calibrated summary in [-1, +1].** At the edit frame there are
two ground-truth worlds and both can be rendered: `gt_edited` (the teleport happened) and
`gt_unedited` (the counterfactual where it did not). On the rays where those two worlds
differ, ask which one the model's output is closer to:

    d_edit = RMSE(pred, gt_edited)   over the differing rays
    d_uned = RMSE(pred, gt_unedited) over the differing rays
    edit_index = (d_uned - d_edit) / (d_uned + d_edit)

    +1  the output IS the edited world      (the edit fully landed)
     0  equidistant from both               (ambiguous, or garbage)
    -1  the output IS the unedited world    (the edit did nothing)

Why this survives the cases that break every ratio: an output that is *far from both* worlds
— a scrambled or collapsed rollout — has d_edit ≈ d_uned and scores ≈ 0 rather than a
spuriously good value, so the index cannot be gamed by destroying the output. "Dim everything
toward background", which scores a perfect ghost under any ghost-only metric, also cancels,
because the differing-ray support contains target rays (where dimming is wrong) as well as
ghost rays (where it is right). And the repo's dominant observed failure — *paint a copy at
the target while keeping the ghost* — correctly reads ≈ 0 rather than the >100% the old reach
metric reported.

The differing-ray support is computed from the two renders directly
(`|gt_edited - gt_unedited| > eps`) rather than as target ∪ ghost, so partial occlusion is
handled without any assumption about which object is in front.

Observation channels
--------------------
Everything above is written in terms of "rays" because the default observation is a 1D
perspective scan, but nothing here depends on that. `build_edit_zones` inherits the
dataset's rendering config, so on an **omniscient 2D** dataset
(`pim/environments/discworld/render2d.py`) the two reference worlds are rasterised instead, and every
zone is a set of **pixels** rather than rays — the masks stay flat `(N, R)` and the
formulas are untouched.

⚠ **Cross-channel comparability.** Zone-restricted RMSE and the Edit Index are computed
on the pixels/rays that the zone or the two worlds actually pick out, so they are
comparable across observation channels. `edit_frame_rmse` (all rays) and any whole-frame
prediction error are **not**: an object covers ~13% of a 1D scan but only ~0.7% of the
omniscient grid, so a whole-frame average is dominated by background to a wildly
different degree in each. Compare those only within one channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pim.metrics.edit_index import edit_index_per_case, fidelity_ratio_from

DIFF_EPS = (
    1e-3  # intensity difference that counts as "the two worlds differ on this ray"
)


@dataclass
class EditZones:
    """The two ground-truth worlds at the edit frame, plus the ray zones.

    Attributes
    ----------
    gt_edited   : (N, R) clean render — the edited object at its teleport target.
    gt_unedited : (N, R) clean render — the counterfactual where the teleport never
                  happened (the edited object continued along its own velocity).
    target      : (N, R) bool — rays the edited object occupies in `gt_edited`.
    ghost       : (N, R) bool — rays it occupied pre-edit and vacates.
    collateral  : (N, R) bool — rays the OTHER object occupies in `gt_edited`.
    differing   : (N, R) bool — rays where the two worlds differ; the support of the index.
    teleport    : (N,) teleport distance in sim units (for sample selection / labels).
    gt_unedited_traj : (N, K, R) the counterfactual world *rolled forward* — the edited object
                  continuing along its own velocity while the other object follows its true
                  trajectory. Step 0 equals `gt_unedited`. Enables `edit_index` per rollout step.
    differing_traj   : (N, K, R) bool — per-step support of the index.
    """

    gt_edited: np.ndarray
    gt_unedited: np.ndarray
    target: np.ndarray
    ghost: np.ndarray
    collateral: np.ndarray
    differing: np.ndarray
    teleport: np.ndarray
    gt_unedited_traj: np.ndarray | None = None
    differing_traj: np.ndarray | None = None


def sim_config_from(sim: dict, n_obj: int):
    """The one place a `SimConfig` is built for CLEAN reference rendering.

    Inherits the dataset's rendering settings. Without this the reference worlds would be
    rendered with the HARD ray-caster while the model was trained on soft renders, and every
    §4 metric would be scored against the wrong ground truth. The same argument applies to the
    observation CHANNEL: on an omniscient-2D dataset the reference worlds must be rasterised,
    not ray-cast.

    Note `obs_noise_std=0.0` — references are always clean renders. A caller that needs the
    observation the model would actually have SEEN must add the dataset's noise itself.
    """
    from pim.environments.discworld.sim import SimConfig

    return SimConfig(
        seed=0,
        y_near=sim["y_near"],
        y_far=sim["y_far"],
        x_near=sim["x_near"],
        x_far=sim["x_far"],
        n_objects=n_obj,
        radius=sim["radius"],
        n_frames=1,
        dt=float(sim["dt"]),
        obs_res=sim["obs_res"],
        drop_edge_rays=sim.get("drop_edge_rays", False),   # 8-ray instance (2026-09-03)
        refl_min=sim["refl_min"],
        refl_max=sim["refl_max"],
        fixed_reflectivities=True,
        obs_noise_std=0.0,
        boundary="open",
        always_in_frustum=False,
        soft_edge=sim.get("soft_edge", 0.0),
        soft_shading=sim.get("soft_shading", "flat"),
        soft_psf_sigma=sim.get("soft_psf_sigma", 0.0),
        soft_occlusion_temp=sim.get("soft_occlusion_temp", 0.0),
        omni2d=sim.get("omni2d", False),
        omni2d_h=sim.get("omni2d_h", 48),
        omni2d_w=sim.get("omni2d_w", 64),
    )


def object_constants(sim: dict, n_obj: int):
    """(radii, reflectivities) — WORLD constants on a fixed-reflectivity dataset.

    Both are identical for every episode, so rendering a counterfactual frame from decoded
    positions needs no per-episode ground truth. That is what makes a history rewrite from the
    model's own read-out well posed.
    """
    refl = np.linspace(sim["refl_min"], sim["refl_max"], n_obj).astype(np.float32)
    rad = np.full(n_obj, sim["radius"], np.float32)
    return rad, refl


def build_edit_zones(
    *,
    pre_pos: np.ndarray,
    tgt_pos: np.ndarray,
    pre_vel: np.ndarray,
    edit_object: np.ndarray,
    sim: dict,
    n_obj: int = 2,
    traj_pos: np.ndarray | None = None,
    gt_edited_traj: np.ndarray | None = None,
    blink_visible: np.ndarray | None = None,
) -> EditZones:
    """Render both ground-truth worlds at the edit frame and derive the ray zones.

    Parameters
    ----------
    pre_pos     : (N, n_obj, 2) positions at frame `ef-1` (before the edit).
    tgt_pos     : (N, n_obj, 2) positions at frame `ef` (edited object already teleported).
    pre_vel     : (N, n_obj, 2) velocities at frame `ef-1`.
    edit_object : (N,) index of the teleported object.
    sim         : the dataset's `config["dataset"]["sim"]` dict.
    traj_pos    : (N, K, n_obj, 2) true positions over the rollout, `positions[ef:ef+K]`.
                  Supplying it (with `gt_edited_traj`) also renders the counterfactual world
                  forward so the Edit Index can be evaluated at every rollout step.
    gt_edited_traj : (N, K, R) the sim's clean post-edit observations, `clean_obs[ef:ef+K]`.
    blink_visible : (N, K+2, n_obj) bool, blink instances only — the schedule over frames
                  `ef-1 .. ef+K` (`blink_visible[:, 0]` is frame `ef-1`). The reference
                  worlds are rendered with the SAME blackouts and markers as the sim, so a
                  hidden edited object leaves NO differing rays (the case is unscoreable —
                  NaN — at that step, and scoreable again the frame it reappears).
    """
    from pim.environments.discworld.blink import paint_markers
    from pim.environments.discworld.config import obs_dim
    from pim.environments.discworld.renderer import render_frame as _render_frame

    def render_frame(p, rad_, refl_, cfg_, *, case=0, frame=0):
        """`renderer.render_frame` with case `case`'s blink state at schedule index `frame`."""
        if blink_visible is None:
            return _render_frame(p, rad_, refl_, cfg_)
        v = blink_visible[case]
        d, ids, inten = _render_frame(p, rad_, refl_, cfg_, visible=v[frame])
        paint_markers(ids, inten, v[frame], v[frame + 1] if frame + 1 < v.shape[0] else None)
        return d, ids, inten

    n = len(pre_pos)
    # The zone construction (`other` = the one object that is not edited) is written for
    # exactly two objects, which is every instance in the project.
    assert n_obj == 2, f"build_edit_zones assumes n_obj == 2 (one edited, one 'other'); got {n_obj}"
    dt = float(sim["dt"])
    cfg = sim_config_from(sim, n_obj)
    refl = np.linspace(sim["refl_min"], sim["refl_max"], n_obj).astype(np.float32)
    rad = np.full(n_obj, sim["radius"], np.float32)
    R = obs_dim(cfg)                 # rays KEPT (obs_res - 2 when the wall rays are dropped)

    # the counterfactual world: the edited object never teleported, so it simply
    # continued from its pre-edit position along its own velocity; the other object
    # is unaffected by the edit and sits at its true frame-`ef` position.
    uned_pos = tgt_pos.copy()
    idx = np.arange(n)
    k = edit_object.astype(int)
    uned_pos[idx, k] = pre_pos[idx, k] + pre_vel[idx, k] * dt

    gt_edited = np.zeros((n, R), np.float32)
    gt_unedited = np.zeros((n, R), np.float32)
    id_edited = np.full((n, R), -1, np.int64)
    id_pre = np.full((n, R), -1, np.int64)
    for i in range(n):
        _, ide, inte = render_frame(tgt_pos[i].astype(np.float32), rad, refl, cfg, case=i, frame=1)
        _, _, intu = render_frame(uned_pos[i].astype(np.float32), rad, refl, cfg, case=i, frame=1)
        _, idp, _ = render_frame(pre_pos[i].astype(np.float32), rad, refl, cfg, case=i, frame=0)
        gt_edited[i], gt_unedited[i] = inte, intu
        id_edited[i], id_pre[i] = ide, idp

    other = 1 - k
    target = id_edited == k[:, None]
    ghost = (id_pre == k[:, None]) & (id_edited != k[:, None])
    collateral = id_edited == other[:, None]
    differing = np.abs(gt_edited - gt_unedited) > DIFF_EPS

    # roll the counterfactual world forward: the edited object keeps travelling along its own
    # velocity from its pre-edit position, the other object follows its true trajectory.
    uned_traj = diff_traj = None
    if traj_pos is not None and gt_edited_traj is not None:
        K = traj_pos.shape[1]
        uned_traj = np.zeros((n, K, R), np.float32)
        for s_ in range(K):
            step_pos = traj_pos[:, s_].copy()
            step_pos[idx, k] = pre_pos[idx, k] + pre_vel[idx, k] * dt * (s_ + 1)
            for i in range(n):
                _, _, inten = render_frame(
                    step_pos[i].astype(np.float32), rad, refl, cfg, case=i, frame=1 + s_
                )
                uned_traj[i, s_] = inten
        diff_traj = np.abs(gt_edited_traj - uned_traj) > DIFF_EPS

    return EditZones(
        gt_edited=gt_edited,
        gt_unedited=gt_unedited,
        target=target,
        ghost=ghost,
        collateral=collateral,
        differing=differing,
        teleport=np.linalg.norm(tgt_pos[idx, k] - pre_pos[idx, k], axis=-1),
        gt_unedited_traj=uned_traj,
        differing_traj=diff_traj,
    )


def _index_from(pred, gt_edit, gt_uned, mask) -> float:
    """Mean per-sample Edit Index for one frame — THE formula (``edit_index.py``) with
    the two clean renders as references and the differing rays as support."""
    return float(np.nanmean(edit_index_per_case(pred, gt_edit, gt_uned, mask)))


def edit_index_by_step(
    roll: np.ndarray, zones: EditZones, gt_traj: np.ndarray
) -> list[float]:
    """Edit Index at every rollout step — does the edit *hold*, or decay back toward the
    unedited world (or off into neither)?  Requires `build_edit_zones(traj_pos=..., ...)`.

    The step-0 value is exactly `edit_index`; later steps compare against the counterfactual
    world rolled forward, so this is the bounded trajectory analogue of GT-traj RMSE.
    """
    if zones.gt_unedited_traj is None:
        return []
    K = min(roll.shape[1], zones.gt_unedited_traj.shape[1])
    return [
        _index_from(
            roll[:, s],
            gt_traj[:, s],
            zones.gt_unedited_traj[:, s],
            zones.differing_traj[:, s],
        )
        for s in range(K)
    ]


def zone_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """RMSE between two (N, R) observation arrays over a (N, R) boolean ray mask."""
    if not mask.any():
        return float("nan")
    return float(np.sqrt(((pred - gt) ** 2)[mask].mean()))


def edit_index(pred: np.ndarray, zones: EditZones) -> float:
    """Mean per-sample Edit Index in [-1, +1]; see the module docstring.

    +1 = the output is the edited world, 0 = equidistant (ambiguous or garbage),
    -1 = the output is the unedited world. Computed per sample and averaged so that
    every edit carries equal weight regardless of its teleport distance.
    """
    return _index_from(pred, zones.gt_edited, zones.gt_unedited, zones.differing)


def edit_scorecard(
    roll: np.ndarray,
    zones: EditZones,
    gt_traj: np.ndarray,
) -> dict:
    """The canonical §4 scorecard for one editor's rollout.

    Parameters
    ----------
    roll    : (N, K, R) the editor's free-run; **step 0 must decode the edit frame `ef`**.
    zones   : from `build_edit_zones`.
    gt_traj : (N, K, R) the sim's clean post-edit observations, `clean_obs[ef:ef+K]`.

    Returns the layer-1 zone RMSEs, the layer-2 Edit Index, and the trajectory metrics.
    `fidelity_ratio` is not included here because it needs the unsteered rollout as a
    reference — apply `fidelity_ratio()` at table-assembly time.
    """
    p0 = roll[:, 0]
    step = [
        float(np.sqrt(((roll[:, s] - gt_traj[:, s]) ** 2).mean()))
        for s in range(roll.shape[1])
    ]
    allm = np.ones_like(zones.target)
    return dict(
        edit_index=edit_index(p0, zones),
        edit_index_by_step=edit_index_by_step(roll, zones, gt_traj),
        edit_frame_rmse=zone_rmse(p0, zones.gt_edited, allm),
        target_rmse=zone_rmse(p0, zones.gt_edited, zones.target),
        ghost_rmse=zone_rmse(p0, zones.gt_edited, zones.ghost),
        collateral_rmse=zone_rmse(p0, zones.gt_edited, zones.collateral),
        gt_traj_rmse=float(np.mean(step)),
        step_rmse_to_gt=step,
    )


def fidelity_ratio(card: dict, unsteered_card: dict) -> float:
    """THE guard, one definition in both environments (2026-09-01):

        RMSE(edited prediction, edited-world GT) / RMSE(unsteered prediction, same GT)

    evaluated on the **edit step only** (`edit_frame_rmse`, over the whole frame).
    **> 1 means the edit left the model FURTHER from the true post-edit world than doing
    nothing** — degraded, not steered. < 1 is a real improvement. The Othello counterpart
    is `pim.metrics.set_editability.move_fidelity_ratio`, same formula and polarity.

    Why step 0 and not the rollout (changed 2026-09-01; it used to be `gt_traj_rmse`):
    an activation edit touches step 0 alone — every later step is recomputed from a
    window holding the *unedited* history plus one edited frame. Averaging 15 such steps
    dilutes the very thing the guard is for. Measured on the two canonical discworld
    runs, the step-0 form flags PI at 1.63 / 1.69 where the rollout form managed only
    1.11 / 1.16, and ND at 2.23 / 2.82 against 1.14 / 1.12.

    The rollout question — *can an edited frame survive re-entry into unedited context?*
    — is real but separate; `gt_traj_rmse` stays in the scorecard so that ratio remains
    one division away, under its own name and never as this guard.

    Complements the Edit Index rather than repeating it: the index is **relative** (which
    world is the output nearer?) so a wrecked output can still score mildly positive —
    discworld PI reads +0.26 at fidelity 1.16. This is **absolute** and catches that.
    """
    return fidelity_ratio_from(card["edit_frame_rmse"], unsteered_card["edit_frame_rmse"])


# Column order used by every editability table, so the notebooks agree.
SCORECARD_COLUMNS = [
    ("edit_index", "Edit Index (−1…+1)", "{:+.2f}"),
    ("gt_traj_rmse", "GT-traj RMSE ↓", "{:.3f}"),
    ("target_rmse", "Target RMSE ↓", "{:.3f}"),
    ("ghost_rmse", "Ghost RMSE ↓", "{:.3f}"),
    ("collateral_rmse", "Collateral RMSE ↓", "{:.3f}"),
    ("edit_frame_rmse", "Edit-frame RMSE ↓", "{:.3f}"),
]


def random_samples(n: int, k: int = 4, seed: int = 0) -> list[int]:
    """`k` episode indices drawn uniformly at random — the DEFAULT for any qualitative panel.

    Seeded so the panel is reproducible. Use this unless there is a stated reason not to; if a
    panel deliberately shows extreme cases, say so in the figure title (`harness/STYLE.md` §2).
    """
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n, size=min(k, n), replace=False))
