#!/usr/bin/env python3
"""Canonical §4 editability metric set for the editability thread.

**One implementation, imported by every editability notebook and eval script** —
`scripts/eval_controls.py`, `notebooks/experiments/editability/00_master_editability.ipynb`
and the `controls/` notebooks. The prose definitions live in
`notebooks/experiments/editability/METRICS_AND_EDITORS.md`; this file is the code
they refer to. Do not re-derive these formulas in a notebook.

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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    """
    from pim.simulator.renderer import render_frame
    from pim.simulator.sim import SimConfig

    n = len(pre_pos)
    dt = float(sim["dt"])
    cfg = SimConfig(
        seed=0,
        y_near=sim["y_near"],
        y_far=sim["y_far"],
        x_near=sim["x_near"],
        x_far=sim["x_far"],
        n_objects=n_obj,
        radius=sim["radius"],
        n_frames=1,
        dt=dt,
        obs_res=sim["obs_res"],
        refl_min=sim["refl_min"],
        refl_max=sim["refl_max"],
        fixed_reflectivities=True,
        obs_noise_std=0.0,  # both references are CLEAN renders
        boundary="open",
        always_in_frustum=False,
    )
    refl = np.linspace(sim["refl_min"], sim["refl_max"], n_obj).astype(np.float32)
    rad = np.full(n_obj, sim["radius"], np.float32)
    R = int(sim["obs_res"])

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
        _, ide, inte = render_frame(tgt_pos[i].astype(np.float32), rad, refl, cfg)
        _, _, intu = render_frame(uned_pos[i].astype(np.float32), rad, refl, cfg)
        _, idp, _ = render_frame(pre_pos[i].astype(np.float32), rad, refl, cfg)
        gt_edited[i], gt_unedited[i] = inte, intu
        id_edited[i], id_pre[i] = ide, idp

    other = 1 - k if n_obj == 2 else None
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
                    step_pos[i].astype(np.float32), rad, refl, cfg
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
    """Mean per-sample Edit Index for one frame, given both references and a ray mask."""
    out = np.full(len(pred), np.nan)
    for i in range(len(pred)):
        m = mask[i]
        if not m.any():
            continue
        d_e = np.sqrt(((pred[i, m] - gt_edit[i, m]) ** 2).mean())
        d_u = np.sqrt(((pred[i, m] - gt_uned[i, m]) ** 2).mean())
        if d_u + d_e > 1e-12:
            out[i] = (d_u - d_e) / (d_u + d_e)
    return float(np.nanmean(out))


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
    """`GT-traj RMSE(editor) / GT-traj RMSE(unsteered)`.

    > 1 means the edited rollout ended up FURTHER from the true post-edit world than
    doing nothing at all — the edit degraded the model rather than steering it.
    """
    return card["gt_traj_rmse"] / max(unsteered_card["gt_traj_rmse"], 1e-9)


# Column order used by every editability table, so the notebooks agree.
SCORECARD_COLUMNS = [
    ("edit_index", "Edit Index (−1…+1)", "{:+.2f}"),
    ("gt_traj_rmse", "GT-traj RMSE ↓", "{:.3f}"),
    ("target_rmse", "Target RMSE ↓", "{:.3f}"),
    ("ghost_rmse", "Ghost RMSE ↓", "{:.3f}"),
    ("collateral_rmse", "Collateral RMSE ↓", "{:.3f}"),
    ("edit_frame_rmse", "Edit-frame RMSE ↓", "{:.3f}"),
]
