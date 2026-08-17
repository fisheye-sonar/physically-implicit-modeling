#!/usr/bin/env python3
"""Hidden-size ablation on ACTION models — one evaluation pass per checkpoint.

Mirrors `scripts/eval_controls.py` (the plain-GRU hidden-size sweep) so the four findings of
that sweep can be checked for replication on action-conditioned world models:

  F1 predictive quality  — next-step / free-run RMSE against that model's own baselines
  F2 recoverability      — position & velocity R², linear and MLP, held out by SEQUENCE
  F3 canonicality        — fiber residual ‖h − g(pos,vel)‖/‖h‖, linear and MLP
  F4 editability         — the canonical §4 scorecard from `scripts/editability_metrics.py`

⚠ This script deliberately does NOT reuse `scripts/eval_editability_endogenous.py`'s scorecard.
That one still computes the metric set **retired on 2026-07-30** (`reach` / `collat` / `ghost` /
`select`), which scored *change* rather than *correctness* and normalised by a model-dependent
soft reference. `CLAUDE.md` forbids reintroducing them. Everything here goes through
`editability_metrics.py`: Edit Index, Target / Ghost / Collateral / Edit-frame / GT-traj RMSE,
and the fidelity ratio.

Two families, each with a *built-in* action channel that is the natural oracle:

  **exogenous** (`XG_A_H*` / `XG_C_H*`) — GRU conditioned on continuous teleport-to-absolute-
  coordinate actions (`datasets/7_cont_teleport`). The edit is a teleport the world actually
  performs, so "issue the action" is a ground-truth handle the model was trained to obey.
  `XG_C_*` is the identical recipe with actions **withheld**, isolating action-knowledge from
  capacity.

  **endogenous** (`EN_H*`) — `EndogenousActorGRU` at level 3: force dynamics, death on
  object/wall collision, and a REINFORCE survival objective trained *alongside* prediction.
  The edit teleports object 0 to a random in-frustum target.

Writes `runs/action_sweep/eval/<code>.json` (+ `<code>_rollouts.npz` for the waterfalls) so the
notebook only loads, plots and tabulates.

Usage
-----
    python scripts/eval_action_sweep.py --family exogenous --runs XG_A_H8 XG_A_H256
    python scripts/eval_action_sweep.py --family endogenous --runs EN_H256 --n-edits 64
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch

from editability_metrics import build_edit_zones, edit_scorecard, fidelity_ratio
from pim.editors import (
    fit_state_subspace,
    inject_state,
    manifold_steer,
    probe_decomposition,
)
from pim.extractors import (
    LinearExtractor,
    MLPExtractor,
    StateDefinition,
    fit_readability_probes,
)
from pim.simulator.config import SimConfig
from pim.simulator.renderer import render_frame

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("runs/action_sweep")
EF, K, N_OBJ, N_CTX = 20, 15, 2, 6
XG_EVAL = Path("datasets/15_teleport_eval_single/eval.h5")

EDITORS = [
    "Unsteered",
    "Readout injection",
    "Global-PCA projection",
    "MLP-probe gradient",
    "Decoder gradient (oracle)",
    "Action interface (oracle)",
]


# ── shared numerics ───────────────────────────────────────────────────────────
def sim_from_dataset4() -> SimConfig:
    """The world settings both families share (dataset 4's, which dataset 7 was matched to)."""
    d4 = json.load(open("datasets/4_fixed_refl_inview/dataset.json"))
    stored = {k: v for k, v in d4["sim"].items() if k in SimConfig.__dataclass_fields__}
    return SimConfig(**stored)


def clean_render(pos: np.ndarray, sim: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    cfg = SimConfig(**{**sim.__dict__, "obs_noise_std": 0.0})
    refl = np.linspace(sim.refl_min, sim.refl_max, N_OBJ).astype(np.float32)
    radii = np.full(N_OBJ, sim.radius, np.float32)
    _, hid, inten = render_frame(pos.astype(np.float32), radii, refl, cfg)
    return inten.astype(np.float32), hid


def probe_block(
    states: np.ndarray, pos: np.ndarray, vel: np.ndarray, seed: int = 0
) -> dict:
    """F2 + F3 on one model: readability and canonicality, both linear and MLP.

    `states` (N, T, H), `pos`/`vel` (N, T, n_obj*2). Readability uses the STANDARD probes
    (`fit_readability_probes`: linear lstsq + 2x256 MLP, both fit on the same 80% of
    SEQUENCES and scored on the held-out 20%) — never a hand-rolled probe.
    """
    out = {}
    for name, tgt in (("pos", pos), ("vel", vel)):
        r = fit_readability_probes(states, tgt, device=DEVICE, seed=seed)
        out[f"{name}_r2_linear"] = r["linear_r2"]
        out[f"{name}_r2_mlp"] = r["mlp_r2"]
    # LATE-t (t >= 15) is the headline for velocity: before then the belief has not converged and
    # an all-t number badly under-reads it (H256: all-t MLP 0.784 vs late-t 0.877). The registry
    # requires the split; position is reported both ways for consistency.
    LATE_T = 15
    if states.shape[1] > LATE_T + 2:
        for name, tgt in (("pos", pos), ("vel", vel)):
            r = fit_readability_probes(states[:, LATE_T:], tgt[:, LATE_T:], device=DEVICE, seed=seed)
            out[f"late_{name}_r2_linear"] = r["linear_r2"]
            out[f"late_{name}_r2_mlp"] = r["mlp_r2"]
    # fiber residual: how much of h is NOT a function of the physical state (pos, vel)
    pv = np.concatenate([pos, vel], -1)
    n_tr = int(0.8 * states.shape[0])
    X = states.reshape(-1, states.shape[-1]).astype(np.float64)
    P = pv.reshape(-1, pv.shape[-1]).astype(np.float64)
    ntr_rows = n_tr * states.shape[1]
    A = np.linalg.lstsq(
        np.c_[P[:ntr_rows], np.ones(ntr_rows)], X[:ntr_rows], rcond=None
    )[0]
    pred = np.c_[P[ntr_rows:], np.ones(len(P) - ntr_rows)] @ A
    Xte = X[ntr_rows:]
    out["fiber_resid_linear"] = float(np.linalg.norm(Xte - pred) / np.linalg.norm(Xte))
    rm = fit_readability_probes(pv, states, device=DEVICE, seed=seed)
    with torch.no_grad():
        pm = (
            rm["mlp"](torch.tensor(P[ntr_rows:], dtype=torch.float32, device=DEVICE))
            .cpu()
            .numpy()
        )
    out["fiber_resid_mlp"] = float(np.linalg.norm(Xte - pm) / np.linalg.norm(Xte))
    return out


def latent_editors(
    model, h0, target, states_bank, pos_bank, tgt_obs, decode_fn
) -> dict:
    """The canonical latent write-mechanisms, all aimed at the same position readout."""
    H = h0.shape[1]
    sdef = StateDefinition(
        name="positions_flat",
        state_shape=(N_OBJ * 2,),
        extract_fn=lambda b: b["positions"],
    )
    lin = LinearExtractor(H, sdef, use_lstsq=True)
    lin.fit(states_bank[None], pos_bank[None], device=DEVICE)
    lin = lin.to(DEVICE).eval()
    A, b, A_pinv = probe_decomposition(lin)

    def inject(h, t):
        return inject_state(h, t, A, A_pinv, b)

    out = {"Readout injection": inject(h0, target)}

    bank = torch.from_numpy(states_bank).float().to(DEVICE)
    sub = fit_state_subspace(bank, var_threshold=0.99)
    from dataclasses import replace

    sub = replace(
        sub,
        mean=sub.mean.to(DEVICE),
        basis=sub.basis.to(DEVICE),
        explained_variance_ratio=sub.explained_variance_ratio.to(DEVICE),
    )
    out["Global-PCA projection"] = manifold_steer(h0, target, inject, sub, n_iters=25)

    # MLP Grad Steering writes through a FROZEN 1x128 MLPExtractor — a different object from
    # the reporting probe above, and it must stay on its published defaults.
    probe = MLPExtractor(H, sdef).to(DEVICE)
    probe.fit(states_bank[None], pos_bank[None], device=DEVICE)
    probe = probe.to(DEVICE).eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    h = h0.clone().requires_grad_(True)
    opt = torch.optim.Adam([h], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        ((probe(h) - target) ** 2).mean().backward()
        opt.step()
    out["MLP-probe gradient"] = h.detach()

    h = h0.clone().requires_grad_(True)
    opt = torch.optim.Adam([h], lr=0.05)
    for _ in range(250):
        opt.zero_grad()
        ((decode_fn(h) - tgt_obs) ** 2).mean().backward()
        opt.step()
    out["Decoder gradient (oracle)"] = h.detach()
    return out


def score_all(ROLL: dict, zones, gt_roll: np.ndarray) -> dict:
    cards = {k: edit_scorecard(v, zones, gt_roll) for k, v in ROLL.items()}
    for c in cards.values():
        c["fidelity_ratio"] = fidelity_ratio(c, cards["Unsteered"])
    return cards


# ── exogenous family ──────────────────────────────────────────────────────────
def xg_load(code: str):
    from pim.world_models.action_gru_continuous import (
        ActionContinuousModelConfig,
        ActionGRUContinuousModel,
    )

    cfg_json = json.load(open(ROOT / code / "config.json"))
    mc = cfg_json["model_config"]
    use_actions = cfg_json["use_actions"]
    if use_actions:
        model = ActionGRUContinuousModel(
            ActionContinuousModelConfig(
                input_dim=mc["input_dim"],
                hidden_size=mc["hidden_size"],
                n_obj=mc["n_obj"],
            )
        )
    else:
        from pim.world_models.gru.model import GRUModel, ModelConfig

        model = GRUModel(
            ModelConfig(input_dim=mc["input_dim"], hidden_size=mc["hidden_size"])
        )
    sd = torch.load(
        ROOT / code / "best_model.pt", map_location=DEVICE, weights_only=False
    )
    model.load_state_dict(sd["model_state"] if "model_state" in sd else sd)
    return model.to(DEVICE).eval(), use_actions, mc["hidden_size"]


def xg_data(
    n_edits: int,
    n_probe: int,
    seed: int = 0,
    h5_path: Path | None = None,
    n_gt_steps: int = K,
):
    """Held-out teleport episodes with the edit **synthesised**, not harvested.

    Harvesting naturally-occurring teleports does not work: with `p_action = 0.30` the chance of
    a quiet scored window is ~1e-5 (measured: 9 usable episodes in 4000). So instead each episode
    keeps its realistic pre-edit context from the dataset and the edit is *constructed*, exactly
    as the dataset-4 edits split does:

      * keep only sequences with **no action at `ef-1`**, so `positions[ef]` really is the
        un-teleported world and `pre_pos + pre_vel·dt` is the correct counterfactual;
      * keep only sequences where the **edited object** takes no further action in the scored
        window, so its two worlds stay separated for all K steps (the *other* object may act —
        that happens identically in both worlds, so every zone metric stays valid);
      * sample a target in-frustum, clear of the other object, and encode the teleport with the
        generator's own `normalize_action`, so the action is one the model was trained on.

    Alignment (from `pim/simulator/actions_continuous.py`): `actions[s]` drives the transition
    **into** frame `s+1`, so a teleport visible at `ef` is commanded at `ef-1`.
    """
    from pim.simulator.actions_continuous import normalize_action
    from pim.simulator.edits_dataset import _sample_in_frustum

    sim = sim_from_dataset4()
    with h5py.File(h5_path or XG_EVAL, "r") as f:
        obs = f["obs_intensity"][:].astype(np.float32)
        act = f["actions"][:, :, :N_OBJ, :].astype(np.float32)
        pos = f["positions"][:, :, :N_OBJ, :].astype(np.float32)
        vel = f["velocities"][:, :, :N_OBJ, :].astype(np.float32)

    rng = np.random.default_rng(seed)
    # The eval world is generated with `p_action = 0`, so it contains **no teleports at all** —
    # not before the edit frame and not after it. The single teleport under test is synthesised
    # below. That makes these episodes structurally identical to the canonical dataset-4 edits
    # split, so the exogenous families and the dataset-4 control are actually comparable.
    assert not (act[..., 0] > 0.5).any(), (
        f"{h5_path or XG_EVAL} contains teleports of its own; an edit set must carry exactly one "
        "intervention — the one under test. Regenerate it with --p-action 0.0."
    )
    chosen = list(range(min(n_edits, len(obs))))
    edit_obj = list(rng.integers(N_OBJ, size=len(chosen)))
    if len(chosen) < n_edits:
        raise RuntimeError(f"only {len(chosen)} usable edit episodes, need {n_edits}")
    sel = np.array(chosen)
    edit_obj = np.array(edit_obj, int)
    ix = np.arange(len(sel))

    # --- sample the teleport target and encode the action the world would accept
    tgt = np.zeros((len(sel), 2), np.float32)
    for i, s in enumerate(sel):
        other = pos[s, EF, 1 - edit_obj[i]]
        for _ in range(200):
            c = _sample_in_frustum(rng, sim, margin=sim.radius).astype(np.float32)
            if np.linalg.norm(c - other) > 2.2 * sim.radius:
                tgt[i] = c
                break
        else:
            tgt[i] = c
    a1, a2 = zip(
        *[normalize_action("teleport", float(t[0]), float(t[1]), sim) for t in tgt]
    )
    act_edit = act[sel].copy()
    act_edit[:, EF - 1] = 0.0
    act_edit[ix, EF - 1, edit_obj] = np.stack(
        [np.ones(len(sel)), np.array(a1), np.array(a2)], 1
    ).astype(np.float32)
    act_noop = act[sel].copy()
    act_noop[:, EF - 1] = 0.0  # latent arms are never told about the teleport

    # --- the two ground-truth worlds over the scored horizon.
    # ⛔ CONSTRUCTED, never read out of the dataset. This world fires its own random teleports on
    # ~30% of transitions; any that land inside the scored horizon would put events into the ground
    # truth that the free-running model was never told about and cannot predict, contaminating
    # GT-traj RMSE, the fidelity ratio and the by-step Edit Index — and breaking comparability with
    # every edits split that carries a single teleport. Filtering the EDITED object's later actions
    # is not enough: the other object's teleports land in the same window.
    # So roll the frame-`ef` state forward under the world's passive (ballistic) dynamics for both
    # worlds. The model free-runs without actions, so the passive continuation IS the fair target,
    # and it is exactly what `build_edit_zones` constructs on the canonical splits.
    KG = n_gt_steps
    dt = float(sim.dt)
    p0 = pos[sel, EF].astype(np.float64)                      # (n, n_obj, 2)
    v0 = vel[sel, EF].astype(np.float64)
    kk = np.arange(KG, dtype=np.float64)[None, :, None, None]
    pos_uned = (p0[:, None] + kk * dt * v0[:, None]).astype(np.float32)
    pos_edit = pos_uned.copy()
    tgt_traj = (tgt[:, None, :].astype(np.float64)
                + np.arange(KG, dtype=np.float64)[None, :, None] * dt * v0[ix, edit_obj][:, None, :])
    pos_edit[ix[:, None], np.arange(KG)[None, :], edit_obj[:, None]] = tgt_traj.astype(np.float32)

    R = obs.shape[-1]
    gt_edit = np.zeros((len(sel), KG, R), np.float32)
    gt_uned = np.zeros((len(sel), KG, R), np.float32)
    for i in range(len(sel)):
        for k in range(KG):
            gt_edit[i, k] = clean_render(pos_edit[i, k], sim)[0]
            gt_uned[i, k] = clean_render(pos_uned[i, k], sim)[0]

    tgt_pos = pos[sel, EF].copy()
    tgt_pos[ix, edit_obj] = tgt
    zones = build_edit_zones(
        pre_pos=pos[sel, EF - 1],
        tgt_pos=tgt_pos,
        pre_vel=vel[sel, EF - 1],
        edit_object=edit_obj,
        sim=json.load(open("datasets/4_fixed_refl_inview/dataset.json"))["sim"],
        n_obj=N_OBJ,
        traj_pos=pos_edit,
        gt_edited_traj=gt_edit,
    )
    zones.gt_unedited_traj = gt_uned
    zones.differing_traj = np.abs(gt_edit - gt_uned) > 1e-3
    # step 0 of the constructed pair IS the frame-`ef` pair, so use it for the step-0 zones too
    # rather than `build_edit_zones`' own ballistic estimate (they differ by the position noise).
    zones.gt_unedited = gt_uned[:, 0]
    zones.differing = np.abs(zones.gt_edited - zones.gt_unedited) > 1e-3

    print(
        f"  [xg] {len(sel)} single-edit episodes (teleport-free world, one synthesised "
        f"teleport at ef); "
        f"mean teleport {zones.teleport.mean():.2f} sim units"
    )
    return dict(
        sim=sim,
        sel=sel,
        obs=obs[sel],
        act_edit=act_edit,
        act_noop=act_noop,
        pos=pos[sel],
        vel=vel[sel],
        edit_obj=edit_obj,
        tgt=tgt,
        zones=zones,
        gt_roll=gt_edit,
        tgt_pos=tgt_pos,
        probe_obs=obs[:n_probe],
        probe_act=act[:n_probe],
        probe_pos=pos[:n_probe],
        probe_vel=vel[:n_probe],
    )


@torch.no_grad()
def xg_warm(model, obs, act, use_actions, upto=EF):
    """Teacher-force steps 0..upto-1 on `obs` with the supplied action sequence.

    Pass `act_noop` for the latent arms (they are never told about the teleport) and `act_edit`
    for the action-interface oracle (it is). The resulting state decodes frame `upto`.
    """
    o = torch.from_numpy(obs).float().to(DEVICE)
    a = torch.from_numpy(act).float().to(DEVICE) if use_actions else None
    state = None
    for t in range(upto):
        state = (
            model.step(o[:, t], state, action=a[:, t])[1]
            if use_actions
            else model.step(o[:, t], state)[1]
        )
    return state


@torch.no_grad()
def xg_rollout(model, h_flat, use_actions, steps=K):
    state = model.state_from_flat(h_flat)
    out = [model.decode(state)]
    for _ in range(steps - 1):
        p, state = model.predict_step(state)
        out.append(p)
    return torch.stack(out, 1).cpu().numpy()


def xg_eval(code: str, n_edits: int, n_probe: int, E: dict) -> dict:
    model, use_actions, H = xg_load(code)
    sim = E["sim"]
    res = {
        "code": code,
        "hidden_size": H,
        "use_actions": use_actions,
        "family": "exogenous",
    }

    # F1 + F2 + F3 -------------------------------------------------------------
    with torch.no_grad():
        po = torch.from_numpy(E["probe_obs"]).float().to(DEVICE)
        pa = (
            torch.from_numpy(E["probe_act"]).float().to(DEVICE) if use_actions else None
        )
        pred, hs = (
            model.observe_sequence(po, actions=pa)
            if use_actions
            else model.observe_sequence(po)
        )
        pred, hs = pred.cpu().numpy(), hs.cpu().numpy()
    gt_next = np.zeros_like(pred)
    for i in range(len(E["probe_obs"])):
        for t in range(pred.shape[1]):
            gt_next[i, t] = clean_render(E["probe_pos"][i, t + 1], sim)[0]
    res["nextstep_rmse_vs_clean"] = float(np.sqrt(((pred - gt_next) ** 2).mean()))
    res["noise_floor_rmse"] = float(
        np.sqrt(((E["probe_obs"][:, 1:] - gt_next) ** 2).mean())
    )
    T = hs.shape[1]
    res.update(
        probe_block(
            hs,
            E["probe_pos"][:, :T].reshape(len(hs), T, -1),
            E["probe_vel"][:, :T].reshape(len(hs), T, -1),
        )
    )

    # F4 -----------------------------------------------------------------------
    zones, gt_roll = E["zones"], E["gt_roll"]
    n = len(E["obs"])
    # latent arms never learn about the teleport: the commanding action is a no-op for them
    state0 = xg_warm(model, E["obs"], E["act_noop"], use_actions)
    h0 = model.flat_state(state0).float()
    target = torch.from_numpy(E["tgt_pos"].reshape(n, N_OBJ * 2)).float().to(DEVICE)
    tgt_obs = torch.from_numpy(gt_roll[:, 0]).float().to(DEVICE)

    with torch.no_grad():
        ob = torch.from_numpy(E["probe_obs"][:600]).float().to(DEVICE)
        ab = (
            torch.from_numpy(E["probe_act"][:600]).float().to(DEVICE)
            if use_actions
            else None
        )
        hb = (
            (
                model.get_hidden_states(ob, actions=ab)
                if use_actions
                else model.get_hidden_states(ob)
            )
            .cpu()
            .numpy()
        )
    states_bank = hb.reshape(-1, hb.shape[-1])
    pos_bank = E["probe_pos"][:600, : hb.shape[1]].reshape(-1, N_OBJ * 2)

    Hs = {"Unsteered": h0}
    Hs.update(
        latent_editors(
            model,
            h0,
            target,
            states_bank,
            pos_bank,
            tgt_obs,
            lambda h: model.decode(model.state_from_flat(h)),
        )
    )
    ROLL = {k: xg_rollout(model, v, use_actions) for k, v in Hs.items()}

    # the built-in handle: command the teleport through the ACTION channel instead of editing h
    if use_actions:
        state_a = xg_warm(model, E["obs"], E["act_edit"], use_actions)
        ROLL["Action interface (oracle)"] = xg_rollout(
            model, model.flat_state(state_a).float(), use_actions
        )

    cards = score_all(ROLL, zones, gt_roll)
    # keep the per-step curves: `METRICS_AND_EDITORS.md` requires the by-step Edit Index
    # wherever the step-0 index is reported — landing an edit and holding it are different
    # things, and stripping the lists here made that impossible to plot.
    res["editability"] = dict(cards)
    res["n_edits"] = n
    np.savez_compressed(
        ROOT / "eval" / f"{code}_rollouts.npz",
        **{f"roll::{k}": v for k, v in ROLL.items()},
        ctx=E["obs"][:, EF - N_CTX : EF],
        gt_roll=gt_roll,
        tgt_mask=zones.target,
        ghost_mask=zones.ghost,
        teleport=zones.teleport,
    )
    return res


# ── endogenous family ─────────────────────────────────────────────────────────
def en_load(code: str, root: Path | None = None):
    from pim.world_models.actor_gru import EndogenousActorConfig, EndogenousActorGRU

    ck = torch.load(
        (root or ROOT) / code / "ckpt_final.pt", map_location=DEVICE, weights_only=False
    )
    cfg = EndogenousActorConfig(**ck["model_cfg"])
    model = EndogenousActorGRU(cfg).to(DEVICE)
    model.load_state_dict(ck["actor"])
    return model.eval(), cfg


@torch.no_grad()
def en_data(model, n_edits: int, seed: int = 0):
    """Roll the actor in a death-free interactive world, then teleport object 0 at `ef`.

    The two ground-truth worlds are produced by FORKING the simulator at `ef` and stepping both
    forks under the SAME action sequence — the honest counterfactual for a force world, where
    `pos(t+1) = pos(t) + v·dt` does not hold and the ballistic construction inside
    `build_edit_zones` would be wrong.
    """
    from pim.simulator.edits_dataset import _sample_in_frustum
    from pim.simulator.interactive import InteractiveConfig, InteractiveWorld

    sim = sim_from_dataset4()
    icfg = InteractiveConfig(
        dynamics="force", death_on_collision=False, death_on_wall=False, init_speed=0.28
    )
    worlds = [InteractiveWorld(sim, icfg, seed=seed + 5000 + b) for b in range(n_edits)]
    cur = np.stack([w.reset(seed=seed + 5000 + b) for b, w in enumerate(worlds)])
    R = worlds[0].obs_res
    obs = np.zeros((n_edits, EF + K + 1, R), np.float32)
    pos = np.zeros((n_edits, EF + K + 1, N_OBJ, 2), np.float32)
    vel = np.zeros((n_edits, EF + K + 1, N_OBJ, 2), np.float32)
    acts = np.zeros((n_edits, EF + K, model.cfg.n_obj, model.cfg.n_axes), np.float32)
    obs[:, 0] = cur
    for b, w in enumerate(worlds):
        pos[b, 0], vel[b, 0] = w.positions, w.velocities
    state = None
    prev_a = torch.zeros(n_edits, model.cfg.n_obj, model.cfg.n_axes, device=DEVICE)
    for t in range(EF + K):
        h, state = model.gru_step(
            torch.from_numpy(cur).float().to(DEVICE), state, prev_action=prev_a
        )
        a, *_ = model.act(h, deterministic=True)
        prev_a = a
        a_np = a.cpu().numpy()
        acts[:, t] = a_np
        for b, w in enumerate(worlds):
            o, info = w.step(a_np[b])
            obs[b, t + 1] = o
            pos[b, t + 1], vel[b, t + 1] = info["positions"], w.velocities
        cur = obs[:, t + 1]

    rng = np.random.default_rng(seed)
    tgt = np.stack(
        [_sample_in_frustum(rng, sim, margin=sim.radius) for _ in range(n_edits)]
    ).astype(np.float32)

    # --- the two GT worlds, rolled forward by forking the simulator at `ef`
    gt_edit = np.zeros((n_edits, K, R), np.float32)
    gt_uned = np.zeros((n_edits, K, R), np.float32)
    pos_edit = np.zeros((n_edits, K, N_OBJ, 2), np.float32)
    for b in range(n_edits):
        w_u = InteractiveWorld(sim, icfg, seed=seed + 5000 + b)
        w_u.reset(seed=seed + 5000 + b)
        w_u._pos[:] = pos[b, EF]
        w_u._vel[:] = vel[b, EF]
        w_e = copy.deepcopy(w_u)
        w_e._pos[0] = tgt[b]
        gt_uned[b, 0] = clean_render(w_u.positions, sim)[0]
        gt_edit[b, 0] = clean_render(w_e.positions, sim)[0]
        pos_edit[b, 0] = w_e.positions
        for k in range(1, K):
            w_u.step(acts[b, EF + k - 1])
            w_e.step(acts[b, EF + k - 1])
            gt_uned[b, k] = clean_render(w_u.positions, sim)[0]
            gt_edit[b, k] = clean_render(w_e.positions, sim)[0]
            pos_edit[b, k] = w_e.positions
    return dict(
        sim=sim,
        obs=obs,
        acts=acts,
        pos=pos,
        vel=vel,
        tgt=tgt,
        gt_edit=gt_edit,
        gt_uned=gt_uned,
        pos_edit=pos_edit,
    )


def en_zones(E):
    """Canonical zones, but with the counterfactual supplied from the SIMULATOR forks.

    `pre_vel = 0` and `pre_pos = pos[ef]` makes `build_edit_zones` use the true unedited frame-`ef`
    world as its counterfactual instead of a ballistic extrapolation, which does not hold here.
    """
    n = len(E["pos"])
    pre = E["pos"][:, EF].copy()
    post = pre.copy()
    post[:, 0] = E["tgt"]
    z = build_edit_zones(
        pre_pos=pre,
        tgt_pos=post,
        pre_vel=np.zeros_like(pre),
        edit_object=np.zeros(n, int),
        sim=json.load(open("datasets/4_fixed_refl_inview/dataset.json"))["sim"],
        n_obj=N_OBJ,
        traj_pos=E["pos_edit"],
        gt_edited_traj=E["gt_edit"],
    )
    # replace the ballistic roll-forward with the simulator's own counterfactual
    z.gt_unedited_traj = E["gt_uned"]
    z.differing_traj = np.abs(E["gt_edit"] - E["gt_uned"]) > 1e-3
    return z


@torch.no_grad()
def en_warm(model, obs, acts, upto=EF):
    o = torch.from_numpy(obs).float().to(DEVICE)
    a = torch.from_numpy(acts).float().to(DEVICE)
    state = None
    for t in range(upto):
        _, state = model.gru_step(
            o[:, t], state, prev_action=(None if t == 0 else a[:, t - 1])
        )
    return model.flat_state(state), state


@torch.no_grad()
def en_rollout(model, h_flat, mode="self", steps=K):
    state = model.state_from_flat(h_flat)
    out = []
    for _ in range(steps):
        h = model.flat_state(state)
        a = (
            model.act(h, deterministic=True)[0]
            if mode == "self"
            else model._noop(h.shape[0], h.device)
        )
        p = model.decode_action(h, a)
        out.append(p)
        _, state = model.gru_step(p, state, prev_action=a)
    return torch.stack(out, 1).cpu().numpy()


def en_eval(code: str, n_edits: int, mode: str = "self") -> dict:
    model, cfg = en_load(code)
    H = cfg.hidden_size if hasattr(cfg, "hidden_size") else cfg.hidden
    res = {
        "code": code,
        "hidden_size": int(H),
        "family": "endogenous",
        "rollout_mode": mode,
    }
    E = en_data(model, n_edits)
    zones = en_zones(E)
    gt_roll = E["gt_edit"]

    h0, _ = en_warm(model, E["obs"], E["acts"])
    h0 = h0.float()
    n = len(E["obs"])
    # the probe target: the frame-`ef` world with object 0 moved to the teleport target
    target = E["pos"][:, EF].reshape(n, N_OBJ * 2).copy()
    target[:, 0:2] = E["tgt"]
    target_t = torch.from_numpy(target).float().to(DEVICE)

    T = EF + K
    with torch.no_grad():
        o = torch.from_numpy(E["obs"]).float().to(DEVICE)
        a = torch.from_numpy(E["acts"]).float().to(DEVICE)
        st, hs = None, []
        for t in range(T):
            h, st = model.gru_step(
                o[:, t], st, prev_action=(None if t == 0 else a[:, t - 1])
            )
            hs.append(model.flat_state(st))
        hs = torch.stack(hs, 1).cpu().numpy()
    res.update(
        probe_block(
            hs, E["pos"][:, :T].reshape(n, T, -1), E["vel"][:, :T].reshape(n, T, -1)
        )
    )
    with torch.no_grad():
        preds = (
            torch.stack(
                [
                    model.decode_action(
                        torch.from_numpy(hs[:, t]).float().to(DEVICE), a[:, t]
                    )
                    for t in range(T - 1)
                ],
                1,
            )
            .cpu()
            .numpy()
        )
    gt_next = np.stack(
        [
            [clean_render(E["pos"][i, t + 1], E["sim"])[0] for t in range(T - 1)]
            for i in range(n)
        ]
    )
    res["nextstep_rmse_vs_clean"] = float(np.sqrt(((preds - gt_next) ** 2).mean()))
    res["noise_floor_rmse"] = float(np.sqrt(((E["obs"][:, 1:T] - gt_next) ** 2).mean()))

    states_bank = hs[:, :EF].reshape(-1, hs.shape[-1])
    pos_bank = E["pos"][:, :EF].reshape(-1, N_OBJ * 2)
    tgt_obs = torch.from_numpy(gt_roll[:, 0]).float().to(DEVICE)

    Hs = {"Unsteered": h0}
    Hs.update(
        latent_editors(
            model,
            h0,
            target_t,
            states_bank,
            pos_bank,
            tgt_obs,
            lambda h: model.decode_action(h, model._noop(h.shape[0], h.device)),
        )
    )
    ROLL = {k: en_rollout(model, v, mode) for k, v in Hs.items()}
    cards = score_all(ROLL, zones, gt_roll)
    # keep the per-step curves: `METRICS_AND_EDITORS.md` requires the by-step Edit Index
    # wherever the step-0 index is reported — landing an edit and holding it are different
    # things, and stripping the lists here made that impossible to plot.
    res["editability"] = dict(cards)
    res["n_edits"] = n
    np.savez_compressed(
        ROOT / "eval" / f"{code}_rollouts.npz",
        **{f"roll::{k}": v for k, v in ROLL.items()},
        ctx=E["obs"][:, EF - N_CTX : EF],
        gt_roll=gt_roll,
        tgt_mask=zones.target,
        ghost_mask=zones.ghost,
        teleport=zones.teleport,
    )
    return res


# ── passive reference (the published plain-GRU sweep), recomputed with THIS estimator ────
def passive_eval(code: str, n_probe: int) -> dict:
    """F1/F2/F3 for `runs/controls/H*` using the same probes as the action families.

    The published `runs/controls/eval/<code>.json` numbers cannot be plotted beside this sweep:
    `eval_controls.py` splits the probe data **by row** (leaking near-duplicate neighbouring
    frames) and uses a 1x128 MLP, while everything here uses `fit_readability_probes`
    (2x256, split by SEQUENCE). Its **editability** block is on the canonical §4 metric set and
    IS comparable — the notebook takes that straight from the published JSON.
    """
    from pim.world_models import load_checkpoint, load_dataset

    model, _ = load_checkpoint(
        Path("runs/controls") / code / "best_model.pt", device=DEVICE
    )
    model.eval()
    bundle = load_dataset(Path("datasets/4_fixed_refl_inview"), n_obj_keep=N_OBJ)
    test = bundle.test
    with torch.no_grad():
        o = torch.from_numpy(test.obs[:n_probe]).float().to(DEVICE)
        pred, hs = model.observe_sequence(o)
        pred, hs = pred.cpu().numpy(), hs.cpu().numpy()
    T = hs.shape[1]
    clean = test.clean_obs[:n_probe, 1 : T + 1]
    res = {
        "code": code,
        "hidden_size": int(model.hidden_size),
        "family": "passive",
        "nextstep_rmse_vs_clean": float(np.sqrt(((pred - clean) ** 2).mean())),
        "noise_floor_rmse": float(
            np.sqrt(((test.obs[:n_probe, 1 : T + 1] - clean) ** 2).mean())
        ),
    }
    pos = test.positions[:n_probe, :T, :N_OBJ, :].reshape(n_probe, T, -1)
    with h5py.File(test.h5_path, "r") as f:
        vel = f["velocities"][:n_probe, :T, :N_OBJ, :].reshape(n_probe, T, -1)
    res.update(probe_block(hs, pos, vel))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--family", choices=["exogenous", "endogenous", "passive"], required=True
    )
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n-edits", type=int, default=128)
    ap.add_argument("--n-probe", type=int, default=800)
    a = ap.parse_args()
    (ROOT / "eval").mkdir(parents=True, exist_ok=True)

    E = xg_data(a.n_edits, a.n_probe) if a.family == "exogenous" else None
    for code in a.runs:
        print(f"\n=== {code} ===")
        if a.family == "exogenous":
            res = xg_eval(code, a.n_edits, a.n_probe, E)
        elif a.family == "endogenous":
            res = en_eval(code, a.n_edits)
        else:
            res = passive_eval(code, a.n_probe)
        name = f"passive_{code}" if a.family == "passive" else code
        (ROOT / "eval" / f"{name}.json").write_text(json.dumps(res, indent=1))
        print(
            f"H={res['hidden_size']}  next-step {res['nextstep_rmse_vs_clean']:.4f}  "
            f"posR2 lin {res['pos_r2_linear']:.3f} mlp {res['pos_r2_mlp']:.3f}  "
            f"velR2 lin {res['vel_r2_linear']:.3f}  fiber lin {res['fiber_resid_linear']:.3f} "
            f"mlp {res['fiber_resid_mlp']:.3f}"
        )
        for k, c in res.get("editability", {}).items():
            print(
                f"    {k:<28} index {c['edit_index']:+.3f}  fidelity {c['fidelity_ratio']:.2f}"
            )


if __name__ == "__main__":
    main()
