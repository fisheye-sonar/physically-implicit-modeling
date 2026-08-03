#!/usr/bin/env python3
"""Train the endogenous-action ACTOR and its OBSERVER twin in the interactive world.

See research/directions/endogenous-action-interactive-world.md.

Levels
------
  1  "shift" dynamics, prediction-only  (efference-copy ablation; guarded, no death)
  2  "force" dynamics, prediction-only  (physical momentum; death possible but no goal)
  3  "force" dynamics + REINFORCE       (goal = survive: avoid object/wall collisions)

Both models are the SAME architecture (EndogenousActorGRU).  Each iteration we roll the
ACTOR out on-policy in B parallel worlds; then we teacher-force the collected (obs, action)
chunk and train:
  * the ACTOR's predictor  (next-step obs MSE)  + at L3 its policy/value via REINFORCE —
    the policy gradient flows into the SHARED GRU trunk (the mechanism we test: does having
    to act reshape the latent?);
  * the OBSERVER's predictor on the SAME (obs, action) trace (it is *fed* the actor's
    actions; it never acts).  Its trunk sees only the prediction signal.
The actor-vs-observer difference therefore isolates the effect of *generating + acting*.

Checkpoints + a JSON training log are written to --out.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pim.simulator.config import SimConfig
from pim.simulator.interactive import InteractiveConfig, InteractiveWorld
from pim.simulator.interactive_batched import BatchedInteractiveWorld
from pim.world_models.actor_gru import EndogenousActorConfig, EndogenousActorGRU


# ── world / reward config per level ───────────────────────────────────────────
def make_worlds(n_obj, obs_res, level, n_worlds, base_seed, obs_noise=0.2):
    sim = SimConfig(
        n_objects=n_obj,
        radius=0.5,
        obs_res=obs_res,
        obs_noise_std=obs_noise,  # repo standard is 0.2 (every dataset 0..8); see --obs-noise
        fixed_reflectivities=True,
        boundary="bounce",
    )
    dyn = "shift" if level == 1 else "force"
    icfg = InteractiveConfig(
        dynamics=dyn,
        death_on_collision=(level >= 2),  # L1 shift is guarded (no death); L2/L3 lethal
        death_on_wall=(level >= 2),
        reset_on_death=True,
        reset_noise_frames=4,  # SMiRL-style surprise on death
        init_speed=0.28,  # initial momentum (challenge)
    )
    worlds = [InteractiveWorld(sim, icfg, seed=base_seed + b) for b in range(n_worlds)]
    return sim, icfg, worlds


def multistep_loss(model, obs, act, h_seq, pmask, W, n_start, rng):
    """Free-run rollout loss: from teacher-forced states, imagine W steps feeding the
    model's OWN predictions back (with the recorded actions) and match the true obs.

    Pure next-step training leaves free-run rollouts blurry (the model is never asked to
    consume its own output); this is the direct fix, and free-run is exactly what the
    editability waterfalls/metrics use.
    """
    T = h_seq.shape[1]
    if T <= W:
        return torch.zeros((), device=obs.device)
    losses = []
    for s in rng.choice(T - W, size=min(n_start, T - W), replace=False):
        s = int(s)
        state = h_seq[:, s].unsqueeze(0).contiguous()  # (1,B,H) GRU state at step s
        for k in range(W):
            a_k = act[:, s + k]
            pred = model.decode_action(model.flat_state(state), a_k)
            tgt = obs[:, s + k + 1, :]
            m = pmask[:, s + k]
            losses.append(
                (((pred - tgt) ** 2).mean(-1) * m).sum() / m.sum().clamp(min=1.0)
            )
            # feed own prediction back; the action also drives the transition when enabled
            _, state = model.gru_step(pred, state, prev_action=a_k)
    return torch.stack(losses).mean()


def compute_returns(rewards, dones, gamma, bootstrap=None):
    """rewards, dones: (B, T) torch. Discounted return-to-go, reset at done.

    ``bootstrap`` (B,): value estimate of the state AFTER the last step of the chunk. Without it
    the chunk end is treated as terminal, which tells the agent that surviving is only worth the
    steps remaining in this 48-frame window. That is approximately right when the recurrent state
    is zeroed at every boundary (each chunk really is an episode), but it is a systematic
    under-valuation of survival once the state is CARRIED across boundaries — and it is what
    destabilised the --carry-state runs (policy entropy collapsed to 0).
    """
    B, T = rewards.shape
    ret = torch.zeros_like(rewards)
    running = (
        torch.zeros(B, device=rewards.device)
        if bootstrap is None
        else bootstrap.detach()
    )
    for t in reversed(range(T)):
        running = rewards[:, t] + gamma * running * (1.0 - dones[:, t])
        ret[:, t] = running
    return ret


# ── on-policy collection ──────────────────────────────────────────────────────
@torch.no_grad()
def collect(actor, worlds, cur_obs, T, device, deterministic=False, state=None):
    """Roll the actor out for T steps in the B worlds. Returns numpy arrays."""
    B = len(worlds)
    R = worlds[0].obs_res
    n_obj = worlds[0].n
    obs_seq = np.zeros((T + 1, B, R), np.float32)
    act_seq = np.zeros((T, B, n_obj, 2), np.float32)
    idx_seq = np.zeros((T, B, n_obj, 2), np.int64)
    rew = np.zeros((T, B), np.float32)
    died = np.zeros((T, B), np.float32)
    unpred = np.zeros((T, B), np.float32)  # obs[t+1] is noise/rebirth (mask predictor)
    ent_sum = 0.0

    obs_seq[0] = cur_obs
    prev_a = torch.zeros(B, n_obj, 2, device=device)  # a_{-1} = no-op
    for t in range(T):
        obs_t = torch.from_numpy(cur_obs).float().to(device)
        h, state = actor.gru_step(obs_t, state, prev_action=prev_a)
        action, logp, ent, idx = actor.act(h, deterministic=deterministic)
        ent_sum += float(ent.mean())
        a_np = action.cpu().numpy()
        act_seq[t] = a_np
        idx_seq[t] = idx.cpu().numpy()
        nxt = np.zeros((B, R), np.float32)
        for b, w in enumerate(worlds):
            o, info = w.step(a_np[b])
            nxt[b] = o
            if info["died"]:
                rew[t, b] = -1.0
                died[t, b] = 1.0
            elif info["dying"] or info["rebirth"]:
                rew[t, b] = 0.0
                unpred[t, b] = 1.0
            else:
                rew[t, b] = 0.1
        obs_seq[t + 1] = nxt
        cur_obs = nxt
        prev_a = action
    return (
        dict(
            state_out=None if state is None else state.detach(),
            obs=obs_seq,
            act=act_seq,
            idx=idx_seq,
            rew=rew,
            died=died,
            unpred=unpred,
            ent=ent_sum / T,
        ),
        cur_obs,
    )


@torch.no_grad()
def collect_batched(actor, world, cur_obs, T, device, state=None):
    """Same contract as :func:`collect`, but the environment is a single
    :class:`BatchedInteractiveWorld` stepped as tensors (no per-world Python loop).

    Rewards match the scalar path exactly: +0.1 for a surviving step, -1.0 on death, 0.0 on
    the death-noise / rebirth frames (which are also masked out of the predictor loss).
    """
    B, R, n_obj = world.B, world.obs_res, world.n
    obs_seq = torch.zeros(T + 1, B, R, device=device)
    act_seq = torch.zeros(T, B, n_obj, 2, device=device)
    idx_seq = torch.zeros(T, B, n_obj, 2, dtype=torch.long, device=device)
    rew = torch.zeros(T, B, device=device)
    died = torch.zeros(T, B, device=device)
    unpred = torch.zeros(T, B, device=device)
    ent_sum = 0.0

    obs_seq[0] = cur_obs
    prev_a = torch.zeros(B, n_obj, 2, device=device)
    for t in range(T):
        h, state = actor.gru_step(obs_seq[t], state, prev_action=prev_a)
        action, logp, ent, idx = actor.act(h)
        ent_sum += float(ent.mean())
        act_seq[t], idx_seq[t] = action, idx
        nxt, info = world.step(action)
        obs_seq[t + 1] = nxt
        d = info["died"].to(rew.dtype)
        u = (info["dying"] | info["rebirth"]).to(rew.dtype)
        died[t] = d
        unpred[t] = u * (1.0 - d)
        rew[t] = -1.0 * d + 0.1 * (1.0 - d) * (1.0 - u)
        prev_a = action
    return (
        dict(
            obs=obs_seq,
            act=act_seq,
            idx=idx_seq,
            rew=rew,
            died=died,
            unpred=unpred,
            ent=ent_sum / T,
            state_out=None if state is None else state.detach(),
        ),
        obs_seq[T],
    )


# ── training ──────────────────────────────────────────────────────────────────
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sim, icfg, worlds = make_worlds(
        args.n_obj,
        args.obs_res,
        args.level,
        args.batch,
        args.seed,
        obs_noise=args.obs_noise,
    )
    bworld = (
        BatchedInteractiveWorld(
            sim, icfg, batch=args.batch, seed=args.seed, device=device
        )
        if args.batched_sim
        else None
    )
    mcfg = EndogenousActorConfig(
        input_dim=args.obs_res,
        hidden_size=args.hidden,
        n_obj=args.n_obj,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        action_in_transition=args.action_in_transition,
    )
    actor = EndogenousActorGRU(mcfg).to(device)
    observer = EndogenousActorGRU(mcfg).to(device)
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_o = torch.optim.Adam(observer.parameters(), lr=args.lr)

    ms_rng = np.random.default_rng(args.seed + 999)
    if bworld is not None:
        cur_obs = bworld.reset(seed=args.seed)
    else:
        cur_obs = np.stack([w.reset(seed=args.seed + b) for b, w in enumerate(worlds)])
    log = {
        "it": [],
        "pred_rmse_actor": [],
        "pred_rmse_obs": [],
        "mean_reward": [],
        "survival": [],
        "deaths": [],
        "policy_entropy": [],
        "coll_rate": [],
    }
    t0 = time.time()

    # Recurrent state carried across iteration boundaries (--carry-state). The ACTOR's carry
    # comes from collection (the state that actually drove the world); the OBSERVER never acts,
    # so its carry comes from its own teacher-forced pass. Both detached => truncated BPTT.
    carry, obs_carry = None, None
    for it in range(args.iters):
        h0 = carry if args.carry_state else None
        h0_obs = obs_carry if args.carry_state else None
        if bworld is not None:
            data, cur_obs = collect_batched(
                actor, bworld, cur_obs, args.rollout, device
            )
            obs = data["obs"].permute(1, 0, 2)  # (B,T+1,R) — already on device
            act = data["act"].permute(1, 0, 2, 3)
            idx = data["idx"].permute(1, 0, 2, 3)
            rew = data["rew"].permute(1, 0)
            died = data["died"].permute(1, 0)
            unpred = data["unpred"].permute(1, 0)
        else:
            data, cur_obs = collect(
                actor, worlds, cur_obs, args.rollout, device, state=h0
            )
            obs = torch.from_numpy(data["obs"]).permute(1, 0, 2).to(device)  # (B,T+1,R)
            act = (
                torch.from_numpy(data["act"]).permute(1, 0, 2, 3).to(device)
            )  # (B,T,n,2)
            idx = (
                torch.from_numpy(data["idx"]).permute(1, 0, 2, 3).to(device)
            )  # (B,T,n,2)
            rew = torch.from_numpy(data["rew"]).permute(1, 0).to(device)  # (B,T)
            died = torch.from_numpy(data["died"]).permute(1, 0).to(device)
            unpred = torch.from_numpy(data["unpred"]).permute(1, 0).to(device)
        target = obs[:, 1:, :]
        pmask = 1.0 - unpred  # predictor mask (B,T)
        denom = pmask.sum().clamp(min=1.0)

        # ---- actor: predictor (+ policy at L3) ----
        pred_a, h_a = actor.predict_sequence(obs, act, h0=h0)
        pred_loss_a = (((pred_a - target) ** 2).mean(-1) * pmask).sum() / denom
        actor_loss = pred_loss_a
        if args.multistep > 1:
            actor_loss = actor_loss + args.ms_coef * multistep_loss(
                actor, obs, act, h_a, pmask, args.multistep, args.ms_starts, ms_rng
            )
        if args.level == 3:
            logp, ent = actor.logp_entropy(h_a, idx)  # (B,T)
            val = actor.value_of(h_a)  # (B,T)
            boot = None
            if args.bootstrap_value:
                # value of the state the chunk ends in (detached) — makes the truncated
                # return an estimate of the true infinite-horizon return
                s_end = data.get("state_out")
                if s_end is not None:
                    boot = actor.value_of(s_end.squeeze(0)).detach()
            returns = compute_returns(rew, died, args.gamma, bootstrap=boot)
            adv = returns - val.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            polmask = 1.0 - unpred
            pol_denom = polmask.sum().clamp(min=1.0)
            pi_loss = -((logp * adv) * polmask).sum() / pol_denom
            v_loss = (((val - returns) ** 2) * polmask).sum() / pol_denom
            ent_loss = -((ent * polmask).sum() / pol_denom)
            actor_loss = (
                pred_loss_a
                + args.pi_coef * pi_loss
                + args.v_coef * v_loss
                + args.ent_coef * ent_loss
            )
        opt_a.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        opt_a.step()

        # ---- observer: predictor only, SAME (obs, action) trace ----
        pred_o, h_o = observer.predict_sequence(obs, act, h0=h0_obs)
        pred_loss_o = (((pred_o - target) ** 2).mean(-1) * pmask).sum() / denom
        obs_loss = pred_loss_o
        if args.multistep > 1:
            obs_loss = obs_loss + args.ms_coef * multistep_loss(
                observer, obs, act, h_o, pmask, args.multistep, args.ms_starts, ms_rng
            )
        opt_o.zero_grad()
        obs_loss.backward()
        torch.nn.utils.clip_grad_norm_(observer.parameters(), 5.0)
        opt_o.step()

        if args.carry_state:
            carry = data.get("state_out")
            if carry is not None:
                carry = carry.detach()
                if args.reset_state_on_death:
                    # a world that died mid-chunk is now a FRESH world, but its carried state
                    # still encodes the dead episode -> clear those slices
                    dead_any = died.any(dim=1)  # (B,)
                    if bool(dead_any.any()):
                        carry = carry.clone()
                        carry[:, dead_any, :] = 0.0
            obs_carry = h_o[:, -1].detach().unsqueeze(0).contiguous()

        # ---- logging ----
        n_deaths = float(data["died"].sum())  # works for numpy and torch alike
        surv = args.batch * args.rollout / max(n_deaths, 1.0)  # mean frames per life
        coll_rate = n_deaths / (args.batch * args.rollout)
        if it % args.log_every == 0 or it == args.iters - 1:
            log["it"].append(it)
            log["pred_rmse_actor"].append(float(pred_loss_a.detach() ** 0.5))
            log["pred_rmse_obs"].append(float(pred_loss_o.detach() ** 0.5))
            log["mean_reward"].append(float(rew.mean()))
            log["survival"].append(surv)
            log["deaths"].append(n_deaths)
            log["policy_entropy"].append(float(data["ent"]))
            log["coll_rate"].append(coll_rate)
            print(
                f"[L{args.level}] it {it:4d} | pred RMSE a {log['pred_rmse_actor'][-1]:.4f} "
                f"o {log['pred_rmse_obs'][-1]:.4f} | reward {log['mean_reward'][-1]:+.3f} "
                f"| survival {surv:6.1f} | deaths {int(n_deaths):4d} | H {log['policy_entropy'][-1]:.2f} "
                f"| {time.time() - t0:5.0f}s",
                flush=True,
            )
        if args.ckpt_every and it > 0 and it % args.ckpt_every == 0:
            _save(out, actor, observer, mcfg, sim, icfg, args, log, tag=f"it{it}")

    _save(out, actor, observer, mcfg, sim, icfg, args, log, tag="final")
    print(f"[L{args.level}] DONE in {time.time() - t0:.0f}s → {out}", flush=True)


def _save(out, actor, observer, mcfg, sim, icfg, args, log, tag):
    import dataclasses

    torch.save(
        {
            "actor": actor.state_dict(),
            "observer": observer.state_dict(),
            "model_cfg": dataclasses.asdict(mcfg),
            "sim_cfg": dataclasses.asdict(sim),
            "interactive_cfg": dataclasses.asdict(icfg),
            "level": args.level,
            "args": vars(args),
            "log": log,
        },
        out / f"ckpt_{tag}.pt",
    )
    (out / "log.json").write_text(json.dumps(log, indent=1))


def parse_args():
    p = argparse.ArgumentParser(description="Train endogenous-action actor + observer")
    p.add_argument("--level", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--rollout", type=int, default=48)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument(
        "--enc-layers", type=int, default=1, help="encoder depth (>1 adds MLP layers)"
    )
    p.add_argument(
        "--dec-layers", type=int, default=1, help="decoder depth (>1 adds residual MLP)"
    )
    p.add_argument(
        "--reset-state-on-death",
        action="store_true",
        default=False,
        help="with --carry-state, zero the carried state of worlds that died during the chunk "
        "(their state describes an episode that no longer exists)",
    )
    p.add_argument(
        "--bootstrap-value",
        action="store_true",
        default=False,
        help="bootstrap the truncated return with V(state at the chunk end). Required for "
        "correctness when --carry-state is on (the chunk boundary is then not a real episode end)",
    )
    p.add_argument(
        "--carry-state",
        action="store_true",
        default=False,
        help="carry the recurrent state ACROSS iteration boundaries (detached; truncated BPTT). "
        "Without this the hidden state is zeroed every --rollout frames while the world "
        "continues, so the model is always trained from a cold start on a mid-stream world.",
    )
    p.add_argument(
        "--batched-sim",
        action="store_true",
        default=False,
        help="use the vectorised BatchedInteractiveWorld (GPU-resident, no per-world Python "
        "loop). Parity with the scalar world is enforced by tests/test_interactive_batched.py",
    )
    p.add_argument(
        "--action-in-transition",
        action="store_true",
        default=False,
        help="feed the previous action into the GRU input so actions affect the STATE, not just "
        "the decoded observation (standard action-conditioned world model)",
    )
    p.add_argument(
        "--multistep",
        type=int,
        default=1,
        help="W-step free-run rollout loss (1 = off)",
    )
    p.add_argument(
        "--ms-coef", type=float, default=1.0, help="weight on the multistep loss"
    )
    p.add_argument(
        "--ms-starts",
        type=int,
        default=4,
        help="rollout start points sampled per iteration",
    )
    p.add_argument("--n-obj", type=int, default=2)
    p.add_argument("--obs-res", type=int, default=128)
    p.add_argument(
        "--obs-noise",
        type=float,
        default=0.2,
        help="observation noise std. REPO STANDARD = 0.2 (all datasets 0..8). "
        "The 2026-07-28/29 endogenous runs used 0.05 by mistake — see ENDOGENOUS_RUNS.md.",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--pi-coef", type=float, default=1.0)
    p.add_argument("--v-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
