#!/usr/bin/env python3
"""Evaluate the endogenous-action ACTOR vs its OBSERVER twin (per level).

For each level's checkpoint we roll the actor out (deterministic policy) to drive the
world, collect a clean eval trace (obs + GT positions/velocities, masking death/rebirth
frames), teacher-force BOTH models on the same obs to get their passive latents, and
compute — actor vs observer:

  §2 recoverability : linear + MLP probe R² of (pos, vel) read off the latent   (↑ = legible)
  §3 canonicality   : fiber residual  ‖h − g(pos,vel)‖/‖h‖,  g linear + MLP      (↓ = canonical)
  prediction        : next-step obs RMSE (action-conditioned decode)             (↓)

plus the final survival/reward from each training log.  Writes runs/endogenous/eval_metrics.json.
The actor-vs-observer delta (esp. at L3, where the policy gradient shapes the actor's shared
trunk) is the headline: does *acting* make the latent more identifiable/canonical than *observing*?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pim.simulator.config import SimConfig
from pim.simulator.interactive import InteractiveConfig, InteractiveWorld
from pim.world_models.actor_gru import EndogenousActorConfig, EndogenousActorGRU

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("runs/endogenous")


# ── probes (self-contained; train/test split → held-out numbers) ──────────────
def _split(n, frac=0.7, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    k = int(frac * n)
    return idx[:k], idx[k:]


def linear_r2(X, Y, tr, te):
    A = np.concatenate([X[tr], np.ones((len(tr), 1))], 1)
    W, *_ = np.linalg.lstsq(A, Y[tr], rcond=None)
    pred = np.concatenate([X[te], np.ones((len(te), 1))], 1) @ W
    ss_res = ((Y[te] - pred) ** 2).sum()
    ss_tot = ((Y[te] - Y[tr].mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def _mlp_fit(X, Y, tr, te, out_dim, epochs=400, hid=128):
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    Yt = torch.tensor(Y, dtype=torch.float32, device=DEVICE)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hid),
        torch.nn.ReLU(),
        torch.nn.Linear(hid, hid),
        torch.nn.ReLU(),
        torch.nn.Linear(hid, out_dim),
    ).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    tri = torch.tensor(tr, device=DEVICE)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((net(Xt[tri]) - Yt[tri]) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(Xt[torch.tensor(te, device=DEVICE)]).cpu().numpy()
    return pred


def mlp_r2(X, Y, tr, te):
    pred = _mlp_fit(X, Y, tr, te, Y.shape[1])
    ss_res = ((Y[te] - pred) ** 2).sum()
    ss_tot = ((Y[te] - Y[tr].mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def fiber_resid(H, PV, tr, te, mode):
    if mode == "linear":
        A = np.concatenate([PV[tr], np.ones((len(tr), 1))], 1)
        W, *_ = np.linalg.lstsq(A, H[tr], rcond=None)
        pred = np.concatenate([PV[te], np.ones((len(te), 1))], 1) @ W
    else:
        pred = _mlp_fit(PV, H, tr, te, H.shape[1])
    resid = np.linalg.norm(H[te] - pred, axis=1) / (
        np.linalg.norm(H[te], axis=1) + 1e-8
    )
    return float(resid.mean())


# ── eval-trace collection (actor drives; clean frames only) ───────────────────
@torch.no_grad()
def collect_eval(actor, sim, icfg, n_traj, T, seed):
    worlds = [
        InteractiveWorld(sim, icfg, seed=seed + 10_000 + b) for b in range(n_traj)
    ]
    cur = np.stack([w.reset(seed=seed + 10_000 + b) for b, w in enumerate(worlds)])
    R, n_obj = worlds[0].obs_res, worlds[0].n
    obs = np.zeros((T + 1, n_traj, R), np.float32)
    act = np.zeros((T, n_traj, n_obj, 2), np.float32)
    pos = np.zeros((T, n_traj, n_obj * 2), np.float32)
    vel = np.zeros((T, n_traj, n_obj * 2), np.float32)
    valid = np.zeros((T, n_traj), bool)
    obs[0] = cur
    state = None
    c = actor.cfg
    prev_a = torch.zeros(n_traj, c.n_obj, c.n_axes, device=DEVICE)
    for t in range(T):
        h, state = actor.gru_step(
            torch.from_numpy(cur).float().to(DEVICE), state, prev_action=prev_a
        )
        action, *_ = actor.act(h, deterministic=True)
        prev_a = action
        a_np = action.cpu().numpy()
        act[t] = a_np
        for b, w in enumerate(worlds):
            o, info = w.step(a_np[b])
            obs[t + 1, b] = o
            pos[t, b] = info["positions"].reshape(-1)
            vel[t, b] = info["velocities"].reshape(-1)
            valid[t, b] = (
                info["alive"]
                and not info["died"]
                and not info["dying"]
                and not info["rebirth"]
            )
        cur = obs[t + 1]
    return dict(obs=obs, act=act, pos=pos, vel=vel, valid=valid)


@torch.no_grad()
def latents_and_pred(model, obs, act):
    """Teacher-force → h_seq (B,T,H) and next-step pred (B,T,R)."""
    obs_t = torch.from_numpy(obs).permute(1, 0, 2).to(DEVICE)  # (B,T+1,R)
    act_t = torch.from_numpy(act).permute(1, 0, 2, 3).to(DEVICE)  # (B,T,n,2)
    pred, h_seq = model.predict_sequence(obs_t, act_t)
    return h_seq.cpu().numpy(), pred.cpu().numpy(), obs_t[:, 1:, :].cpu().numpy()


def eval_role(model, data):
    # h_seq[:,t] aligns with pos/vel[:,t]; valid[t] marks clean frames
    h, pred, target = latents_and_pred(model, data["obs"], data["act"])  # (B,T,·)
    T = h.shape[1]
    valid = data["valid"].T[:, :T]  # (B,T)
    pos = data["pos"].transpose(1, 0, 2)[:, :T]
    vel = data["vel"].transpose(1, 0, 2)[:, :T]
    m = valid.reshape(-1)
    H = h.reshape(-1, h.shape[-1])[m]
    POS = pos.reshape(-1, pos.shape[-1])[m]
    VEL = vel.reshape(-1, vel.shape[-1])[m]
    PV = np.concatenate([POS, VEL], 1)
    tr, te = _split(len(H))
    rmse = float(
        np.sqrt((((pred - target) ** 2).reshape(-1, pred.shape[-1])[m]).mean())
    )
    return {
        "n_samples": int(len(H)),
        "pos_r2_lin": linear_r2(H, POS, tr, te),
        "pos_r2_mlp": mlp_r2(H, POS, tr, te),
        "vel_r2_lin": linear_r2(H, VEL, tr, te),
        "vel_r2_mlp": mlp_r2(H, VEL, tr, te),
        "fiber_lin": fiber_resid(H, PV, tr, te, "linear"),
        "fiber_mlp": fiber_resid(H, PV, tr, te, "mlp"),
        "nextstep_rmse": rmse,
    }


def load_models(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    mcfg = EndogenousActorConfig(**ck["model_cfg"])
    actor = EndogenousActorGRU(mcfg).to(DEVICE).eval()
    observer = EndogenousActorGRU(mcfg).to(DEVICE).eval()
    actor.load_state_dict(ck["actor"])
    observer.load_state_dict(ck["observer"])
    sim = SimConfig(**ck["sim_cfg"])
    icfg = InteractiveConfig(**ck["interactive_cfg"])
    return actor, observer, sim, icfg, ck


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs", nargs="+", default=["L1", "L2", "L3", "L3b", "L2s0", "L3s0", "L3s1"]
    )
    ap.add_argument("--out", default=str(ROOT / "eval_metrics.json"))
    cli = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    runs = {r: ROOT / r for r in cli.runs}
    out = {}
    for name, d in runs.items():
        ck_path = d / "ckpt_final.pt"
        if not ck_path.exists():
            print(f"skip {name}: no checkpoint")
            continue
        actor, observer, sim, icfg, ck = load_models(ck_path)
        data = collect_eval(actor, sim, icfg, n_traj=96, T=48, seed=ck["args"]["seed"])
        log = ck["log"]
        out[name] = {
            "level": ck["level"],
            "actor": eval_role(actor, data),
            "observer": eval_role(observer, data),
            "survival_final": log["survival"][-1] if log["survival"] else None,
            "reward_final": log["mean_reward"][-1] if log["mean_reward"] else None,
            "survival_curve": list(zip(log["it"], log["survival"])),
            "deaths_curve": log.get(
                "deaths", []
            ),  # raw per-iteration death counts (for the death RATE)
            "reward_curve": list(zip(log["it"], log["mean_reward"])),
            "pred_rmse_actor_curve": list(zip(log["it"], log["pred_rmse_actor"])),
            "pred_rmse_obs_curve": list(zip(log["it"], log["pred_rmse_obs"])),
        }
        a, o = out[name]["actor"], out[name]["observer"]
        print(f"\n=== {name} (level {ck['level']}) — n={a['n_samples']} ===")
        print(
            f"  pos R² lin  actor {a['pos_r2_lin']:.3f}  observer {o['pos_r2_lin']:.3f}  Δ {a['pos_r2_lin']-o['pos_r2_lin']:+.3f}"
        )
        print(
            f"  vel R² lin  actor {a['vel_r2_lin']:.3f}  observer {o['vel_r2_lin']:.3f}  Δ {a['vel_r2_lin']-o['vel_r2_lin']:+.3f}"
        )
        print(
            f"  fiber MLP   actor {a['fiber_mlp']:.3f}  observer {o['fiber_mlp']:.3f}  Δ {a['fiber_mlp']-o['fiber_mlp']:+.3f}"
        )
        print(
            f"  nextstep    actor {a['nextstep_rmse']:.4f} observer {o['nextstep_rmse']:.4f}"
        )
        if out[name]["survival_final"]:
            print(
                f"  survival final {out[name]['survival_final']:.1f}  reward {out[name]['reward_final']:+.3f}"
            )
    Path(cli.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {cli.out}")


if __name__ == "__main__":
    main()
