"""Action-conditioned RSSM with a policy and value head — the endogenous-action **actor**.

NEW module.  Subclasses :class:`pim.world_models.rssm.RSSMModel` and does not modify it, so every
existing RSSM checkpoint and result is untouched.  See
``research/directions/endogenous-action-rssm.md``.

What this adds to the base RSSM
-------------------------------
1. **The action enters the transition.**  The base model's deterministic core is
   ``h_t = GRUCell(s_{t-1}, h_{t-1})`` — it has no action input at all.  Here it becomes

       h_t = GRUCell([ s_{t-1} , proj(a_{t-1}) ] , h_{t-1})

   The **previous** action, because ``a_t = pi(state_t)`` is produced *from* the state at t (using
   ``a_t`` would be circular); ``a_{t-1}`` is what drove the transition into t.  This matters more
   than it sounds: in the GRU thread the equivalent omission meant feeding opposite actions produced
   a *bit-identical* next state, so the model could not imagine an action's consequences at all.
2. **Policy + value heads** on the combined latent ``[h, s]``.  The policy is the same factored
   discrete space as the GRU actor — per object, per axis, over ``{-1, 0, +1}`` — so the keyboard
   overlay in ``scripts/play.py`` keeps working for visualisation.

Why RSSM at all: the GRU actor learned the survival task but its imagination decoupled from reality
within ~10-20 closed-loop steps, and that survived every plumbing fix plus 16x batch and 3x data.
The missing ingredient is a term tying the *imagined* latent to the *observation-informed* one —
i.e. KL(posterior || prior), which the base RSSM already provides and which the training script
uses with KL balancing and free bits.

Protocol
--------
Every action argument defaults to **no-op (zeros)**, so the inherited ``HiddenStateModel`` methods
(``observe_sequence`` / ``predict_step`` / ``decode`` / ``flat_state`` / ``state_from_flat``) behave
as a passive RSSM and the whole existing eval / probe / editor suite runs unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from pim.world_models.rssm.model import ModelConfig, RSSMModel, RSSMState


@dataclass
class RSSMActorConfig(ModelConfig):
    """Base RSSM config plus the action space and head sizes."""

    n_obj: int = 2
    n_axes: int = 2  # (x, y) per object
    n_bins: int = 3  # categorical bins per axis; 3 = {-1, 0, +1}
    action_proj_dim: int = 16
    head_hidden: int = 200  # width of the policy / value MLPs


class RSSMActor(RSSMModel):
    """RSSM whose transition is action-conditioned, with policy and value heads."""

    def __init__(self, cfg: RSSMActorConfig) -> None:
        super().__init__(cfg)
        self.cfg: RSSMActorConfig = cfg
        self._adim = cfg.n_obj * cfg.n_axes

        # action -> transition input (separate from anything in the base model)
        self.action_proj = nn.Linear(self._adim, cfg.action_proj_dim)
        # replace the deterministic core so it consumes [s_{t-1}, proj(a_{t-1})]
        self.gru_cell = nn.GRUCell(cfg.stoch_size + cfg.action_proj_dim, cfg.det_size)

        feat = cfg.det_size + cfg.stoch_size
        self.policy = nn.Sequential(
            nn.Linear(feat, cfg.head_hidden),
            nn.ELU(),
            nn.Linear(cfg.head_hidden, self._adim * cfg.n_bins),
        )
        self.value = nn.Sequential(
            nn.Linear(feat, cfg.head_hidden),
            nn.ELU(),
            nn.Linear(cfg.head_hidden, 1),
        )
        # Training the actor INSIDE imagination requires predicting the reward and whether the
        # episode continues, because no simulator is available there (Dreamer's reward + discount
        # heads). Both are trained on real collected data and then queried on imagined latents.
        self.reward = nn.Sequential(
            nn.Linear(feat, cfg.head_hidden),
            nn.ELU(),
            nn.Linear(cfg.head_hidden, 1),
        )
        self.cont = nn.Sequential(  # logit of P(episode continues, i.e. did NOT die)
            nn.Linear(feat, cfg.head_hidden),
            nn.ELU(),
            nn.Linear(cfg.head_hidden, 1),
        )
        self.register_buffer("bin_values", torch.linspace(-1.0, 1.0, cfg.n_bins))

    # ── action helpers ────────────────────────────────────────────────────────
    def noop(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, self.cfg.n_obj, self.cfg.n_axes, device=device)

    def _aproj(self, action: torch.Tensor | None, batch: int, device) -> torch.Tensor:
        a = self.noop(batch, device) if action is None else action
        return F.relu(self.action_proj(a.reshape(*a.shape[:-2], self._adim)))

    def _core(self, state: RSSMState, prev_action: torch.Tensor | None) -> torch.Tensor:
        """Action-conditioned deterministic transition: h_t = GRUCell([s,proj(a)], h)."""
        a = self._aproj(prev_action, state.s.shape[0], state.s.device)
        return self.gru_cell(torch.cat([state.s, a], dim=-1), state.h)

    # ── policy / value ────────────────────────────────────────────────────────
    def policy_logits(self, feat: torch.Tensor) -> torch.Tensor:
        """feat (..., det+stoch) -> logits (..., n_obj, n_axes, n_bins)."""
        return self.policy(feat).reshape(
            *feat.shape[:-1], self.cfg.n_obj, self.cfg.n_axes, self.cfg.n_bins
        )

    def value_of(self, feat: torch.Tensor) -> torch.Tensor:
        return self.value(feat).squeeze(-1)

    def reward_of(self, feat: torch.Tensor) -> torch.Tensor:
        """Predicted reward for a latent (needed to score imagined rollouts)."""
        return self.reward(feat).squeeze(-1)

    def cont_logit(self, feat: torch.Tensor) -> torch.Tensor:
        """Logit of P(episode continues). Used as the discount inside imagination."""
        return self.cont(feat).squeeze(-1)

    def act(self, state: RSSMState, deterministic: bool = False):
        """Sample an action from the policy at ``state``.

        Returns ``(action, logp, entropy, idx)`` with action in {-1,0,+1}; logp and entropy are
        summed over the (object, axis) factors.
        """
        dist = torch.distributions.Categorical(
            logits=self.policy_logits(self.flat_state(state))
        )
        idx = dist.logits.argmax(-1) if deterministic else dist.sample()
        return (
            self.bin_values[idx],
            dist.log_prob(idx).flatten(1).sum(1),
            dist.entropy().flatten(1).sum(1),
            idx,
        )

    def logp_entropy(self, feat: torch.Tensor, idx: torch.Tensor):
        """Recompute (logp, entropy) of stored bin indices; works on (B,·) or (B,T,·)."""
        dist = torch.distributions.Categorical(logits=self.policy_logits(feat))
        n = idx.dim()
        lp, en = dist.log_prob(idx), dist.entropy()
        if n == 4:  # (B, T, n_obj, n_axes)
            return lp.flatten(2).sum(2), en.flatten(2).sum(2)
        return lp.flatten(1).sum(1), en.flatten(1).sum(1)

    # ── action-conditioned core operations (override the base model) ───────────
    def observe_step(
        self,
        obs_t: torch.Tensor,
        state: RSSMState,
        prev_action: torch.Tensor | None = None,
    ) -> tuple[RSSMState, Normal, Normal]:
        """Filter one real observation. ``prev_action`` drove the transition into this step."""
        h = self._core(state, prev_action)
        e = self.encoder(obs_t)
        prior = self._prior(h)
        posterior = self._posterior(h, e)
        s = posterior.rsample() if self.sample else posterior.mean
        return RSSMState(h, s), prior, posterior

    def imagine_step(
        self, state: RSSMState, action: torch.Tensor | None = None
    ) -> tuple[RSSMState, Normal]:
        """Evolve the state with the prior only (no observation), under ``action`` taken AT ``state``."""
        h = self._core(state, action)
        prior = self._prior(h)
        s = prior.rsample() if self.sample else prior.mean
        return RSSMState(h, s), prior

    def observe_sequence_with_dists(
        self, obs: torch.Tensor, actions: torch.Tensor | None = None
    ):
        """Teacher-forced pass returning states + raw prior/posterior parameters.

        ``obs`` (B,T,R); ``actions`` (B,T,n_obj,n_axes) is the action taken AT each step, so the
        transition into step t consumes ``actions[:, t-1]`` (no-op at t=0).

        Returns ``(recons, feats, prior_mu, prior_std, post_mu, post_std, final_state)``.
        """
        B, T, _ = obs.shape
        state = self._initial_state(B, obs.device)
        recons, feats, pmu, pstd, qmu, qstd = [], [], [], [], [], []
        for t in range(T):
            prev = None if (actions is None or t == 0) else actions[:, t - 1]
            state, prior, post = self.observe_step(obs[:, t], state, prev_action=prev)
            recons.append(self.decode(state))
            feats.append(self.flat_state(state))
            pmu.append(prior.loc)
            pstd.append(prior.scale)
            qmu.append(post.loc)
            qstd.append(post.scale)
        st = lambda x: torch.stack(x, dim=1)  # noqa: E731
        return st(recons), st(feats), st(pmu), st(pstd), st(qmu), st(qstd), state

    def imagine_for_actor(self, state: RSSMState, horizon: int):
        """Differentiable imagination used to train the actor.

        Starting from a (detached) posterior state, roll ``horizon`` steps under the model's own
        policy using the PRIOR only — no observations. Returns the per-step features, log-probs and
        entropies, plus predicted rewards and continue-probabilities, all with gradient so that
        lambda-returns can be backpropagated into the policy.
        """
        feats, logps, ents, rews, conts = [], [], [], [], []
        st = RSSMState(state.h.detach(), state.s.detach())
        for _ in range(horizon):
            f = self.flat_state(st)
            dist = torch.distributions.Categorical(logits=self.policy_logits(f))
            idx = dist.sample()
            a = self.bin_values[idx]
            logps.append(dist.log_prob(idx).flatten(1).sum(1))
            ents.append(dist.entropy().flatten(1).sum(1))
            st, _ = self.imagine_step(st, a)
            f_next = self.flat_state(st)
            feats.append(f_next)
            rews.append(self.reward_of(f_next))
            conts.append(torch.sigmoid(self.cont_logit(f_next)))
        stk = lambda x: torch.stack(x, dim=1)  # noqa: E731
        return stk(feats), stk(logps), stk(ents), stk(rews), stk(conts)

    @torch.no_grad()
    def imagine_rollout_actions(self, state: RSSMState, steps: int):
        """Free-run imagination driven by the model's OWN policy (no observations).

        Returns ``(obs_pred (B,steps,R), actions (B,steps,n_obj,n_axes))``.
        """
        obs_out, act_out = [], []
        for _ in range(steps):
            a, _, _, _ = self.act(state, deterministic=True)
            act_out.append(a)
            state, _ = self.imagine_step(state, a)
            obs_out.append(self.decode(state))
        return torch.stack(obs_out, 1), torch.stack(act_out, 1)
