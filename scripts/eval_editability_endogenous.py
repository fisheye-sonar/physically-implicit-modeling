#!/usr/bin/env python3
"""§4 grabbability test (v2): is the actor's latent an editable object handle?

Rebuilt after Sevan's review of v1, which had three problems:
  1. **Waterfall bug** — v1 injected the TRUE target obs row into *every* column and
     dropped each editor's own step-0 decode, so every column looked teacher-forced on the
     edit frame and the exact frame the metrics score was hidden.  Now each column shows
     **its own rollout from step 0** (the immediate post-edit decode); the ground truth is
     its own separate column.  Only "True-swap" ever sees the target obs (by construction).
  2. **Off-distribution rollout** — v1 rolled out with no-op actions, but the actor *always*
     acts, so those rollouts were off-policy for it.  Now rollouts run in two modes:
     ``self`` (the model's own policy acts on its imagined world — in-distribution) and
     ``noop`` (the passive convention of the earlier object-individuation work).
  3. **Predictor quality unmeasured** — a blurry model fails editing for the wrong reason.
     Every model now reports an open-loop **quality gate**: free-run RMSE vs the true future
     (given the true actions) and a **sharpness (total-variation) ratio** vs GT.

Editors: Unsteered / True-swap (soft ref) / Readout injection / Global-PCA projection /
PCA geodesic / MLP-probe gradient / Decoder gradient (oracle).
Writes runs/endogenous/editability_metrics_v2.json + waterfalls to runs/endogenous/edit_figs_v2/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from pim.editors.manifold_steering import (
    fit_state_subspace,
    manifold_steer,
    manifold_steer_local,
)
from pim.simulator.config import SimConfig
from pim.simulator.edits_dataset import _sample_in_frustum
from pim.simulator.interactive import InteractiveConfig, InteractiveWorld
from pim.simulator.renderer import render_frame
from pim.world_models.actor_gru import EndogenousActorConfig, EndogenousActorGRU

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("runs/endogenous")
EF, K, N_CTX = 20, 15, 6

LABEL = {
    "L3": "L3 force+goal · weak 256h · seed 0",
    "L3b": "L3 force+goal · weak 256h · seed 1",
    "L3s0": "L3 force+goal · STRONG 512h · seed 0",
    "L3s1": "L3 force+goal · STRONG 512h · seed 1",
    "L2s0": "L2 force · no goal · STRONG 512h",
}

EDITORS = [
    "Unsteered",
    "True-swap",
    "Readout injection",
    "Global-PCA projection",
    "PCA geodesic",
    "MLP-probe gradient",
    "Decoder gradient",
]


def render_clean(pos, refl, sim):
    cfg = SimConfig(**{**sim.__dict__, "obs_noise_std": 0.0})
    _, hid, inten = render_frame(
        pos.astype(np.float32),
        np.full(len(pos), sim.radius, np.float32),
        refl.astype(np.float32),
        cfg,
    )
    return inten.astype(np.float32), hid


# ── edits set: the actor drives a (death-free) context, then object 0 is teleported ──
@torch.no_grad()
def build_edits(actor, sim, refl, n, seed):
    icfg = InteractiveConfig(
        dynamics="force", death_on_collision=False, death_on_wall=False, init_speed=0.28
    )
    worlds = [InteractiveWorld(sim, icfg, seed=seed + 5000 + b) for b in range(n)]
    cur = np.stack([w.reset(seed=seed + 5000 + b) for b, w in enumerate(worlds)])
    R = worlds[0].obs_res
    T = EF + K + 2
    obs = np.zeros((n, T, R), np.float32)
    act = np.zeros((n, T - 1, 2, 2), np.float32)
    pos = np.zeros((n, T, 2, 2), np.float32)
    obs[:, 0] = cur
    state = None
    c = actor.cfg
    prev_a = torch.zeros(n, c.n_obj, c.n_axes, device=DEVICE)
    for t in range(T - 1):
        h, state = actor.gru_step(
            torch.from_numpy(cur).float().to(DEVICE), state, prev_action=prev_a
        )
        a, *_ = actor.act(h, deterministic=True)
        prev_a = a
        a_np = a.cpu().numpy()
        act[:, t] = a_np
        for b, w in enumerate(worlds):
            o, info = w.step(a_np[b])
            obs[b, t + 1] = o
            pos[b, t + 1] = info["positions"]
        cur = obs[:, t + 1]

    rng = np.random.default_rng(seed)
    tgt = np.stack(
        [_sample_in_frustum(rng, sim, margin=sim.radius) for _ in range(n)]
    ).astype(np.float32)
    tgt_obs = np.zeros((n, R), np.float32)
    kmask = np.zeros((n, R), bool)
    ghostmask = np.zeros((n, R), bool)
    othermask = np.zeros((n, R), bool)
    for b in range(n):
        pre = pos[b, EF].copy()
        post = pre.copy()
        post[0] = tgt[b]
        ti, tid = render_clean(post, refl, sim)
        _, pid = render_clean(pre, refl, sim)
        tgt_obs[b] = ti
        kmask[b] = tid == 0
        othermask[b] = tid == 1
        ghostmask[b] = (pid == 0) & (tid != 0)
    return dict(
        obs=obs,
        act=act,
        pos=pos,
        tgt=tgt,
        tgt_obs=tgt_obs,
        kmask=kmask,
        ghostmask=ghostmask,
        othermask=othermask,
    )


# ── model ops ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def warm(model, obs_t, act_t=None):
    """Teacher-force obs[0..EF-1].  When the model puts the action in its transition, a_{t-1}
    must be supplied too (``act_t``); otherwise it is ignored."""
    state = None
    for t in range(EF):
        prev = None if (act_t is None or t == 0) else act_t[:, t - 1]
        _, state = model.gru_step(obs_t[:, t, :], state, prev_action=prev)
    return model.flat_state(state), state


@torch.no_grad()
def rollout(model, h_flat, mode, steps=K):
    """Free-run from an (edited) state. Step 0 = the immediate post-edit decode.

    mode "self" : the model's own policy acts on its imagined world (in-distribution).
    mode "noop" : no action (the passive convention of the earlier object-individuation work).
    """
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
    return torch.stack(out, 1)


@torch.no_grad()
def quality_gate(model, E):
    """Open-loop free-run from the edit frame using the TRUE actions vs the TRUE future."""
    obs_t = torch.from_numpy(E["obs"]).float().to(DEVICE)
    act_t = torch.from_numpy(E["act"]).float().to(DEVICE)
    _, state = warm(model, obs_t, act_t)
    preds = []
    for k in range(K):
        h = model.flat_state(state)
        p = model.decode_action(h, act_t[:, EF + k])
        preds.append(p)
        _, state = model.gru_step(p, state, prev_action=act_t[:, EF + k])
    roll = torch.stack(preds, 1)
    gt = obs_t[:, EF : EF + K, :]

    def tv(x):
        return float(x.diff(dim=-1).abs().mean())

    return {
        "freerun_rmse": float(((roll - gt) ** 2).mean().sqrt()),
        "sharpness_tv_ratio": tv(roll) / max(tv(gt), 1e-9),
        "nextstep_rmse": float(((roll[:, 0] - gt[:, 0]) ** 2).mean().sqrt()),
    }


def obs_baselines(E, sim, refl):
    """Dataset-level RMSE reference lines (repo convention, see pim/eval/baselines.py):
    copy-previous-frame ("identity") and the observation noise floor."""
    from pim.eval.baselines import compute_obs_baselines

    obs = E["obs"]
    clean = np.zeros_like(obs)
    for b in range(obs.shape[0]):
        for t in range(obs.shape[1]):
            clean[b, t] = render_clean(E["pos"][b, t], refl, sim)[0]
    bl = compute_obs_baselines(obs, clean, sim.obs_noise_std)
    return {
        "identity_rmse": bl.identity_rmse,  # copy the previous frame
        "noise_floor_rmse": bl.noise_floor_rmse,  # noisy vs clean obs
        "random_rmse": bl.random_rmse,
    }


# ── metrics ───────────────────────────────────────────────────────────────────
def rms_mask(a, b, mask):
    d = (a - b) ** 2
    return float(np.sqrt(d[mask].mean())) if mask.sum() else float("nan")


def scorecard(ROLL, ed, E):
    e0, u0, s0 = ROLL[ed][:, 0], ROLL["Unsteered"][:, 0], ROLL["True-swap"][:, 0]
    allm = np.ones_like(E["kmask"])
    ref = rms_mask(s0, u0, E["kmask"])
    reach = 100 * rms_mask(e0, u0, E["kmask"]) / max(ref, 1e-9)
    collat = 100 * rms_mask(e0, u0, E["othermask"]) / max(ref, 1e-9)
    ghost = float(e0[E["ghostmask"]].mean() / max(u0[E["ghostmask"]].mean(), 1e-6))
    chg0 = rms_mask(e0, u0, allm)
    chg_late = np.mean(
        [rms_mask(ROLL[ed][:, s], ROLL["Unsteered"][:, s], allm) for s in range(10, K)]
    )
    # per-step RMSE against the static post-edit target render (see the notebook's note on
    # why the reference is static: each editor would otherwise induce a different future).
    step_rmse = [
        float(np.sqrt(((ROLL[ed][:, s] - E["tgt_obs"]) ** 2).mean())) for s in range(K)
    ]
    return dict(
        reach=reach,
        collat=collat,
        ghost=ghost,
        select=reach / max(reach + collat, 1e-9),
        persist=chg_late / max(chg0, 1e-9),
        target_rmse=step_rmse[0],
        step_rmse_to_target=step_rmse,
    )


# ── editors ───────────────────────────────────────────────────────────────────
def build_editors(model, E, hp, Hb, Pb, W, b):
    tgt_t = torch.from_numpy(E["tgt"].astype(np.float32)).to(DEVICE)
    Wk = torch.tensor(W[:, 0:2], dtype=torch.float32, device=DEVICE)
    Wk_pinv = torch.tensor(
        np.linalg.pinv(W[:, 0:2]), dtype=torch.float32, device=DEVICE
    )
    bk = torch.tensor(b[0:2], dtype=torch.float32, device=DEVICE)
    h_t = torch.tensor(hp, dtype=torch.float32, device=DEVICE)

    def linear_inject(h, t):
        return h + (t - (h @ Wk + bk)) @ Wk_pinv

    out = {"Readout injection": linear_inject(h_t, tgt_t)}

    bank = torch.tensor(Hb, dtype=torch.float32, device=DEVICE)
    sub = fit_state_subspace(bank, var_threshold=0.99)
    out["Global-PCA projection"] = manifold_steer(
        h_t, tgt_t, linear_inject, sub, n_iters=25
    )
    out["PCA geodesic"] = manifold_steer_local(
        h_t, tgt_t, linear_inject, bank, k_neighbors=256, n_iters=50, bank_size=50_000
    )

    # MLP-probe gradient: freeze an MLP probe h→pos(obj0), steer h until it reads the target
    probe = nn.Sequential(
        nn.Linear(Hb.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    ).to(DEVICE)
    optp = torch.optim.Adam(probe.parameters(), lr=1e-3)
    Hb_t = torch.tensor(Hb, device=DEVICE)
    Pb_t = torch.tensor(Pb[:, 0:2], device=DEVICE)
    for _ in range(400):
        optp.zero_grad()
        ((probe(Hb_t) - Pb_t) ** 2).mean().backward()
        optp.step()
    for p_ in probe.parameters():
        p_.requires_grad_(False)
    h = h_t.clone().requires_grad_(True)
    opt = torch.optim.Adam([h], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        ((probe(h) - tgt_t) ** 2).mean().backward()
        opt.step()
    out["MLP-probe gradient"] = h.detach()

    # Decoder gradient (oracle): match the true target obs through the decoder
    h = h_t.clone().requires_grad_(True)
    noop = model._noop(h.shape[0], DEVICE)
    tgt_obs_t = torch.from_numpy(E["tgt_obs"]).float().to(DEVICE)
    opt = torch.optim.Adam([h], lr=0.05)
    for _ in range(250):
        opt.zero_grad()
        ((model.decode_action(h, noop) - tgt_obs_t) ** 2).mean().backward()
        opt.step()
    out["Decoder gradient"] = h.detach()
    return out


def eval_role(model, E, mode):
    obs_t = torch.from_numpy(E["obs"]).float().to(DEVICE)
    h_pre, state_pre = warm(model, obs_t, torch.from_numpy(E["act"]).float().to(DEVICE))
    with torch.no_grad():
        hbank = model.get_hidden_states(obs_t).cpu().numpy()
    pbank = E["pos"][:, : hbank.shape[1]].reshape(E["pos"].shape[0], hbank.shape[1], 4)
    Hb = hbank.reshape(-1, hbank.shape[-1])
    Pb = pbank.reshape(-1, 4)
    A = np.concatenate([Hb, np.ones((len(Hb), 1))], 1)
    sol, *_ = np.linalg.lstsq(A, Pb, rcond=None)
    W, b = sol[:-1], sol[-1]
    hp = h_pre.detach().cpu().numpy()
    with torch.no_grad():
        h_swap = model.flat_state(
            model.gru_step(
                torch.from_numpy(E["tgt_obs"]).float().to(DEVICE), state_pre
            )[1]
        )
    H = {
        "Unsteered": h_pre,
        "True-swap": h_swap,
        **build_editors(model, E, hp, Hb, Pb, W, b),
    }
    ROLL = {k: rollout(model, v, mode).cpu().numpy() for k, v in H.items()}
    cards = {ed: scorecard(ROLL, ed, E) for ed in EDITORS if ed != "Unsteered"}
    return cards, ROLL


# ── waterfalls (FIXED: each column shows its OWN rollout from step 0) ──────────
def waterfall(E, ROLL, fname, title):
    cols = ["GT (sim target)"] + EDITORS
    samples = list(
        np.argsort(np.linalg.norm(E["tgt"] - E["pos"][:, EF, 0], axis=1))[::-1][:3]
    )
    DARK, TXT, EDGE = "#0a0a14", "#a3adc2", "#fa8850"
    fig, axes = plt.subplots(
        len(samples),
        len(cols),
        figsize=(2.5 * len(cols), 3.2 * len(samples)),
        squeeze=False,
        facecolor=DARK,
    )
    for r, smp in enumerate(samples):
        tc = np.where(E["kmask"][smp])[0]
        gc = np.where(E["ghostmask"][smp])[0]
        tcx = tc.mean() if len(tc) else np.nan
        gcx = gc.mean() if len(gc) else np.nan
        ctx = E["obs"][smp, EF - N_CTX : EF]  # real (noisy) observed context
        for c, name in enumerate(cols):
            ax = axes[r][c]
            ax.set_facecolor(DARK)
            if name == "GT (sim target)":
                body = np.tile(
                    E["tgt_obs"][smp], (K, 1)
                )  # static edit target, reference
            else:
                body = ROLL[name][smp]  # this column's OWN free-run, step 0 first
            panel = np.clip(np.concatenate([ctx, body], 0), 0, 1)
            ax.imshow(
                panel,
                aspect="auto",
                origin="upper",
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.axhline(N_CTX - 0.5, color=EDGE, lw=1.2, ls="--", alpha=0.9)
            if not np.isnan(tcx):
                ax.axvline(tcx, color="#00E676", lw=1.4)
            if not np.isnan(gcx):
                ax.axvline(gcx, color="#FF5252", ls="--", lw=1.4)
            if r == 0:
                ax.set_title(name, fontsize=8, color=("#00E676" if c == 0 else TXT))
            if c == 0:
                ax.set_ylabel(f"sample {smp}\nsim frame", fontsize=8, color=TXT)
                ax.set_yticks([0, N_CTX, N_CTX + 7, N_CTX + 14])
                ax.set_yticklabels(
                    [EF - N_CTX, EF, EF + 7, EF + 14], fontsize=7, color=TXT
                )
            else:
                ax.set_yticks([])
            ax.set_xticks([])
    handles = [
        Line2D([0], [0], color="#00E676", lw=2, label="object-0 target location"),
        Line2D(
            [0],
            [0],
            color="#FF5252",
            ls="--",
            lw=2,
            label="object-0 ghost (pre-edit) location",
        ),
        Line2D(
            [0],
            [0],
            color=EDGE,
            ls="--",
            lw=2,
            label="edit applied here — every row below is that column's OWN free-run (no teacher forcing)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        fontsize=8.5,
        frameon=False,
        labelcolor=TXT,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(title, y=1.0, fontsize=11, color=TXT)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = ROOT / "edit_figs_v2" / f"{fname}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print("saved", out)


@torch.no_grad()
def action_interface_test(model, E, sim, refl, steps=K):
    """Complement to latent surgery: can object 0 be driven toward the target through the
    trained ACTION channel?  Runs the SAME action sequence in (a) the real simulator — the
    ground-truth authority of the channel — and (b) the model's imagination, then scores the
    model's imagined rollout with the SAME reach/ghost metrics used for the latent editors.

    A "button, not a handle" world model shows: action channel moves the object (ghost drops),
    while latent surgery does not (ghost stays ~1).
    """
    n = E["obs"].shape[0]
    obs_t = torch.from_numpy(E["obs"]).float().to(DEVICE)
    _, state0 = warm(model, obs_t, torch.from_numpy(E["act"]).float().to(DEVICE))

    # (a) real simulator from the edit-frame state, driven by a bang-bang PD controller on
    #     object 0.  (A constant push overshoots and bounces off the walls, which would
    #     understate the channel's authority.)  Object 1 is held at no-op.
    pos0 = E["pos"][:, EF, 0]
    icfg = InteractiveConfig(
        dynamics="force", death_on_collision=False, death_on_wall=False, init_speed=0.28
    )
    d0 = np.linalg.norm(E["tgt"] - pos0, axis=1)
    acts = np.zeros((n, steps, 2, 2), np.float32)
    real_obs = np.zeros((n, steps, E["obs"].shape[-1]), np.float32)
    d_end = np.zeros(n, np.float32)
    for b in range(n):
        w = InteractiveWorld(sim, icfg, seed=b)
        w._pos = E["pos"][b, EF].astype(np.float64).copy()
        w._vel = np.zeros((2, 2))
        for k in range(steps):
            desired = (E["tgt"][b] - w.positions[0]) - 4.0 * w.velocities[0]  # P - D
            acts[b, k, 0] = np.sign(desired)
            o, _ = w.step(acts[b, k])
            real_obs[b, k] = o
        d_end[b] = np.linalg.norm(E["tgt"][b] - w.positions[0])
    closed = float(
        np.mean(1.0 - d_end / np.maximum(d0, 1e-6))
    )  # fraction of distance closed

    # (b) the model's imagination under the same actions
    acts_t = torch.from_numpy(acts).to(DEVICE)
    state = (
        tuple(s.clone() for s in state0)
        if isinstance(state0, tuple)
        else state0.clone()
    )
    preds = []
    for k in range(steps):
        h = model.flat_state(state)
        p = model.decode_action(h, acts_t[:, k])
        preds.append(p)
        _, state = model.gru_step(p, state, prev_action=acts_t[:, k])
    imagined = torch.stack(preds, 1).cpu().numpy()

    # score the imagined rollout with the same handle metrics (vs the no-op rollout)
    h_pre = model.flat_state(state0)
    uns = rollout(model, h_pre, "noop").cpu().numpy()
    swap_state = model.gru_step(
        torch.from_numpy(E["tgt_obs"]).float().to(DEVICE), state0
    )[1]
    swp = rollout(model, model.flat_state(swap_state), "noop").cpu().numpy()
    card = scorecard({"act": imagined, "Unsteered": uns, "True-swap": swp}, "act", E)
    # per-step model-vs-real RMSE: isolates COMPOUNDING free-run error from the (small)
    # one-step error, which is what the teacher-forced animations actually show.
    step_rmse = [
        float(np.sqrt(((imagined[:, k] - real_obs[:, k]) ** 2).mean()))
        for k in range(steps)
    ]
    return {
        "real_frac_distance_closed": closed,
        "model_vs_real_rmse": float(np.sqrt(((imagined - real_obs) ** 2).mean())),
        "model_vs_real_step_rmse": step_rmse,
        **{f"imagined_{k}": v for k, v in card.items()},
    }


def load(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    mcfg = EndogenousActorConfig(**ck["model_cfg"])
    a = EndogenousActorGRU(mcfg).to(DEVICE).eval()
    a.load_state_dict(ck["actor"])
    o = EndogenousActorGRU(mcfg).to(DEVICE).eval()
    o.load_state_dict(ck["observer"])
    return a, o, SimConfig(**ck["sim_cfg"]), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["L3", "L3s0", "L3s1"])
    ap.add_argument("--modes", nargs="+", default=["self", "noop"])
    ap.add_argument("--n-edits", type=int, default=64)
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    results = {}
    for tag in args.runs:
        ck_path = ROOT / tag / "ckpt_final.pt"
        if not ck_path.exists():
            print(f"skip {tag}: no checkpoint")
            continue
        actor, observer, sim, ckd = load(ck_path)
        refl = np.linspace(sim.refl_min, sim.refl_max, 2).astype(np.float32)
        E = build_edits(actor, sim, refl, args.n_edits, seed=ckd["args"]["seed"])
        results[tag] = {
            "baselines": obs_baselines(E, sim, refl),
            "level": ckd["level"],
            "iters": ckd["args"]["iters"],
            "hidden": ckd["model_cfg"]["hidden_size"],
            "enc_layers": ckd["model_cfg"].get("enc_layers", 1),
            "multistep": ckd["args"].get("multistep", 1),
        }
        for role, model in [("actor", actor), ("observer", observer)]:
            q = quality_gate(model, E)
            ai = action_interface_test(model, E, sim, refl)
            results[tag][role] = {"quality": q, "action_interface": ai}
            print(
                f"\n### {tag} {role} — ACTION-CHANNEL control: real sim closes "
                f"{100*ai['real_frac_distance_closed']:.0f}% of the distance | model-vs-real RMSE "
                f"{ai['model_vs_real_rmse']:.4f} | imagined reach {ai['imagined_reach']:.1f}% "
                f"ghost {ai['imagined_ghost']:.3f}"
            )
            print(
                f"\n### {tag} {role} — QUALITY GATE: free-run RMSE {q['freerun_rmse']:.4f} | "
                f"sharpness TV ratio {q['sharpness_tv_ratio']:.3f} (1.0 = as sharp as GT) | "
                f"next-step RMSE {q['nextstep_rmse']:.4f}"
            )
            for mode in args.modes:
                cards, ROLL = eval_role(model, E, mode)
                results[tag][role][mode] = cards
                print(f"  -- rollout mode: {mode} --")
                for ed in EDITORS[1:]:
                    c = cards[ed]
                    print(
                        f"     {ed:22s} reach {c['reach']:6.1f}%  collat {c['collat']:6.1f}%  "
                        f"ghost {c['ghost']:.3f}  select {c['select']:.2f}  persist {c['persist']:.2f}"
                    )
                if (
                    mode == "self"
                ):  # render for BOTH roles so the observer is visible too
                    waterfall(
                        E,
                        ROLL,
                        f"edit_{tag}_{role}_{mode}",
                        f"§4 editability — {LABEL.get(tag, tag)} · {role} (rollout: {mode})",
                    )
    (ROOT / "editability_metrics_v2.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {ROOT / 'editability_metrics_v2.json'}")


if __name__ == "__main__":
    main()
