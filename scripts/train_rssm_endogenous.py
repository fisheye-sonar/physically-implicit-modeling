#!/usr/bin/env python3
"""Train an endogenous-action RSSM ACTOR (and its OBSERVER twin) in the interactive world.

See ``research/directions/endogenous-action-rssm.md``.

Why this exists: the GRU actor learned the survival task but its imagination decoupled from reality
within ~10-20 closed-loop steps, and that survived every plumbing fix plus 16x batch and 3x data
(closed-loop deaths/1000: 85.0 -> 72.2 -> 87.2). Nothing in the GRU objective tied the *imagined*
latent to the *observation-informed* one. This script uses standard RSSM/Dreamer practice, where
that link is the KL term and the policy is trained inside imagination.

Objective (standard, deliberately NOT the pure-predictive variant used previously)
---------------------------------------------------------------------------------
world model :  recon(obs)  +  beta * KL_balanced(post || prior) with FREE BITS
               KL balancing (DreamerV2): kl_post * KL(sg[post] || prior)
                                       + kl_prior * KL(post || sg[prior])
               plus reward and continue heads regressed on real collected data
actor       :  lambda-returns over an IMAGINED rollout (horizon H) from posterior states,
               REINFORCE with the value baseline (discrete actions) + entropy bonus
critic      :  regression onto the same lambda-returns

The OBSERVER is the same architecture trained on the **world-model loss only**, fed the actor's
actions, never acting — so actor-vs-observer isolates agency while RSSM-vs-GRU isolates architecture.

Levels: 2 = force dynamics, lethal, NO goal (control).  3 = force, lethal, + survival goal.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from pim.simulator.config import SimConfig
from pim.simulator.interactive import InteractiveConfig
from pim.simulator.interactive_batched import BatchedInteractiveWorld
from pim.world_models.rssm.model import RSSMState
from pim.world_models.rssm_actor import RSSMActor, RSSMActorConfig


def make_world(args, device):
    sim = SimConfig(
        n_objects=args.n_obj,
        radius=0.5,
        obs_res=args.obs_res,
        obs_noise_std=args.obs_noise,  # repo standard 0.2
        fixed_reflectivities=True,
        boundary="bounce",
    )
    icfg = InteractiveConfig(
        dynamics="force",
        death_on_collision=True,
        death_on_wall=True,
        reset_on_death=True,
        reset_noise_frames=4,
        init_speed=0.28,
    )
    world = BatchedInteractiveWorld(
        sim, icfg, batch=args.batch, seed=args.seed, device=device
    )
    return sim, icfg, world


@torch.no_grad()
def collect(actor, world, cur_obs, T, device, state=None):
    """Roll the actor out on-policy for T steps, filtering with the POSTERIOR.

    Returns tensors on-device plus the final state (detached) for carrying across boundaries.
    """
    B, R, n = world.B, world.obs_res, world.n
    obs = torch.zeros(T + 1, B, R, device=device)
    act = torch.zeros(T, B, n, 2, device=device)
    idx = torch.zeros(T, B, n, 2, dtype=torch.long, device=device)
    rew = torch.zeros(T, B, device=device)
    died = torch.zeros(T, B, device=device)
    unpred = torch.zeros(T, B, device=device)
    ent_sum = 0.0

    obs[0] = cur_obs
    st = actor._initial_state(B, device) if state is None else state
    prev_a = torch.zeros(B, n, 2, device=device)
    for t in range(T):
        st, _, _ = actor.observe_step(obs[t], st, prev_action=prev_a)
        a, _, ent, i = actor.act(st)
        ent_sum += float(ent.mean())
        act[t], idx[t] = a, i
        nxt, info = world.step(a)
        obs[t + 1] = nxt
        d = info["died"].to(rew.dtype)
        u = (info["dying"] | info["rebirth"]).to(rew.dtype)
        died[t] = d
        unpred[t] = u * (1.0 - d)
        rew[t] = -1.0 * d + 0.1 * (1.0 - d) * (1.0 - u)
        prev_a = a
    return (
        dict(
            obs=obs,
            act=act,
            idx=idx,
            rew=rew,
            died=died,
            unpred=unpred,
            ent=ent_sum / T,
            state_out=RSSMState(st.h.detach(), st.s.detach()),
        ),
        obs[T],
    )


def kl_balanced(post_mu, post_std, prior_mu, prior_std, free_nats, w_post, w_prior):
    """DreamerV2 KL balancing with free bits, applied per stochastic dimension."""

    def _kl(qm, qs, pm, ps):
        return kl_divergence(Normal(qm, qs), Normal(pm, ps))

    kl_to_prior = _kl(post_mu.detach(), post_std.detach(), prior_mu, prior_std)
    kl_to_post = _kl(post_mu, post_std, prior_mu.detach(), prior_std.detach())
    # free bits: no gradient once a dimension is already below the floor
    kl_to_prior = kl_to_prior.clamp(min=free_nats)
    kl_to_post = kl_to_post.clamp(min=free_nats)
    return w_post * kl_to_prior.mean() + w_prior * kl_to_post.mean(), float(
        _kl(post_mu, post_std, prior_mu, prior_std).mean()
    )


def lambda_returns(rew, cont, values, gamma, lam):
    """Dreamer lambda-returns over an imagined rollout. All (B, H)."""
    H = rew.shape[1]
    out = [None] * H
    nxt = values[:, -1]
    for t in reversed(range(H)):
        disc = gamma * cont[:, t]
        if t == H - 1:
            out[t] = rew[:, t] + disc * nxt
        else:
            out[t] = rew[:, t] + disc * (
                (1 - lam) * values[:, t + 1] + lam * out[t + 1]
            )
    return torch.stack(out, dim=1)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sim, icfg, world = make_world(args, device)
    mcfg = RSSMActorConfig(
        input_dim=args.obs_res,
        embed_dim=args.embed,
        det_size=args.det,
        stoch_size=args.stoch,
        hidden_dim=args.hidden,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        n_obj=args.n_obj,
    )
    actor = RSSMActor(mcfg).to(device)
    observer = RSSMActor(mcfg).to(device)
    opt_wm = torch.optim.Adam(
        [
            p
            for n, p in actor.named_parameters()
            if not n.startswith(("policy", "value"))
        ],
        lr=args.lr,
    )
    opt_ac = torch.optim.Adam(actor.policy.parameters(), lr=args.actor_lr)
    opt_cr = torch.optim.Adam(actor.value.parameters(), lr=args.actor_lr)
    opt_ob = torch.optim.Adam(observer.parameters(), lr=args.lr)

    cur_obs = world.reset(seed=args.seed)
    carry = None
    log = {
        k: []
        for k in [
            "it",
            "recon_rmse_actor",
            "recon_rmse_obs",
            "kl",
            "mean_reward",
            "deaths",
            "policy_entropy",
            "imag_return",
        ]
    }
    t0 = time.time()

    for it in range(args.iters):
        data, cur_obs = collect(
            actor, world, cur_obs, args.rollout, device, state=carry
        )
        obs = data["obs"].permute(1, 0, 2)  # (B,T+1,R)
        act = data["act"].permute(1, 0, 2, 3)  # (B,T,n,2)
        # NB: the collected action indices are deliberately unused — unlike the GRU thread's
        # REINFORCE-on-real-rollouts, the policy here is trained on IMAGINED actions only.
        rew = data["rew"].permute(1, 0)
        died = data["died"].permute(1, 0)
        unpred = data["unpred"].permute(1, 0)
        pmask = 1.0 - unpred
        obs_in = obs[:, : args.rollout, :]  # RSSM reconstructs the CURRENT obs

        # ---------- world model (actor) ----------
        rec, feats, pmu, pstd, qmu, qstd, st_end = actor.observe_sequence_with_dists(
            obs_in, act
        )
        recon = (((rec - obs_in) ** 2).mean(-1) * pmask).sum() / pmask.sum().clamp(
            min=1
        )
        kl_term, kl_raw = kl_balanced(
            qmu, qstd, pmu, pstd, args.free_nats, args.kl_post, args.kl_prior
        )
        rew_loss = (
            (actor.reward_of(feats) - rew) ** 2 * pmask
        ).sum() / pmask.sum().clamp(min=1)
        cont_loss = F.binary_cross_entropy_with_logits(
            actor.cont_logit(feats), 1.0 - died, reduction="none"
        ).mul(pmask).sum() / pmask.sum().clamp(min=1)
        wm_loss = recon + args.beta * kl_term + rew_loss + cont_loss
        opt_wm.zero_grad()
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
        opt_wm.step()

        # ---------- actor + critic, trained INSIDE imagination ----------
        imag_ret = float("nan")
        if args.level == 3 and it >= args.wm_warmup:
            flat = feats.detach().reshape(-1, feats.shape[-1])
            sel = torch.randperm(flat.shape[0], device=device)[: args.imag_starts]
            st0 = actor.state_from_flat(flat[sel])
            f_i, lp_i, en_i, rw_i, ct_i = actor.imagine_for_actor(st0, args.horizon)
            with torch.no_grad():
                v_i = actor.value_of(f_i)
            ret = lambda_returns(rw_i, ct_i, v_i, args.gamma, args.lam)
            adv = (ret - v_i).detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            pi_loss = -(lp_i * adv).mean() - args.ent_coef * en_i.mean()
            opt_ac.zero_grad()
            pi_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(actor.policy.parameters(), 100.0)
            opt_ac.step()
            v_pred = actor.value_of(f_i.detach())
            cr_loss = ((v_pred - ret.detach()) ** 2).mean()
            opt_cr.zero_grad()
            cr_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.value.parameters(), 100.0)
            opt_cr.step()
            imag_ret = float(ret.mean())

        # ---------- observer: world-model loss ONLY, same (obs, action) trace ----------
        rec_o, feats_o, pmu_o, pstd_o, qmu_o, qstd_o, _ = (
            observer.observe_sequence_with_dists(obs_in, act)
        )
        recon_o = (((rec_o - obs_in) ** 2).mean(-1) * pmask).sum() / pmask.sum().clamp(
            min=1
        )
        kl_o, _ = kl_balanced(
            qmu_o, qstd_o, pmu_o, pstd_o, args.free_nats, args.kl_post, args.kl_prior
        )
        rl_o = (
            (observer.reward_of(feats_o) - rew) ** 2 * pmask
        ).sum() / pmask.sum().clamp(min=1)
        cl_o = F.binary_cross_entropy_with_logits(
            observer.cont_logit(feats_o), 1.0 - died, reduction="none"
        ).mul(pmask).sum() / pmask.sum().clamp(min=1)
        opt_ob.zero_grad()
        (recon_o + args.beta * kl_o + rl_o + cl_o).backward()
        torch.nn.utils.clip_grad_norm_(observer.parameters(), 100.0)
        opt_ob.step()

        # carry the recurrent state; clear worlds that died (GRU-thread lesson: omitting this
        # collapsed the policy, because a reborn world kept a state describing the dead episode)
        s_end = data["state_out"]
        dead_any = died.any(dim=1)
        h, s = s_end.h.clone(), s_end.s.clone()
        if bool(dead_any.any()):
            h[dead_any] = 0.0
            s[dead_any] = 0.0
        carry = RSSMState(h, s)

        n_deaths = float(died.sum())
        if it % args.log_every == 0 or it == args.iters - 1:
            log["it"].append(it)
            log["recon_rmse_actor"].append(float(recon.detach() ** 0.5))
            log["recon_rmse_obs"].append(float(recon_o.detach() ** 0.5))
            log["kl"].append(kl_raw)
            log["mean_reward"].append(float(rew.mean()))
            log["deaths"].append(n_deaths)
            log["policy_entropy"].append(float(data["ent"]))
            log["imag_return"].append(imag_ret)
            print(
                f"[R{args.level}] it {it:5d} | recon RMSE a {log['recon_rmse_actor'][-1]:.4f} "
                f"o {log['recon_rmse_obs'][-1]:.4f} | KL {kl_raw:6.3f} | reward "
                f"{log['mean_reward'][-1]:+.3f} | deaths {int(n_deaths):5d} | H "
                f"{log['policy_entropy'][-1]:.2f} | imagR {imag_ret:+.2f} | "
                f"{time.time()-t0:5.0f}s",
                flush=True,
            )
        if args.ckpt_every and it > 0 and it % args.ckpt_every == 0:
            _save(out, actor, observer, mcfg, sim, icfg, args, log, f"it{it}")

    _save(out, actor, observer, mcfg, sim, icfg, args, log, "final")
    print(f"[R{args.level}] DONE in {time.time()-t0:.0f}s -> {out}", flush=True)


def _save(out, actor, observer, mcfg, sim, icfg, args, log, tag):
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
    p = argparse.ArgumentParser(description="Endogenous-action RSSM actor + observer")
    p.add_argument("--level", type=int, choices=[2, 3], required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--rollout", type=int, default=48)
    p.add_argument("--obs-res", type=int, default=128)
    p.add_argument("--obs-noise", type=float, default=0.2, help="repo standard = 0.2")
    p.add_argument("--n-obj", type=int, default=2)
    # model
    p.add_argument("--det", type=int, default=256)
    p.add_argument("--stoch", type=int, default=32)
    p.add_argument("--embed", type=int, default=200)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--enc-layers", type=int, default=2)
    p.add_argument("--dec-layers", type=int, default=2)
    # world-model objective
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta", type=float, default=1.0, help="KL weight")
    p.add_argument(
        "--free-nats", type=float, default=3.0 / 32, help="free bits PER stochastic dim"
    )
    p.add_argument(
        "--kl-post", type=float, default=0.8, help="weight on KL(sg[post]||prior)"
    )
    p.add_argument(
        "--kl-prior", type=float, default=0.2, help="weight on KL(post||sg[prior])"
    )
    # actor / critic in imagination
    p.add_argument("--actor-lr", type=float, default=8e-5)
    p.add_argument("--horizon", type=int, default=15, help="imagination horizon")
    p.add_argument(
        "--imag-starts",
        type=int,
        default=512,
        help="latents to imagine from per update",
    )
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.003)
    p.add_argument(
        "--wm-warmup",
        type=int,
        default=200,
        help="iterations of world-model-only training before the actor loss switches on",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
