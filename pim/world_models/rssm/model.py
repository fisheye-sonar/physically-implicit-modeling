"""Recurrent State Space Model (RSSM) — uncontrolled, observation-only.

Classic RSSM from Dreamer with no actions.  Separates the world state into a
deterministic recurrent component h_t and a stochastic latent s_t.  Training
uses a standard ELBO: reconstruction loss + KL regularisation.

Notation
--------
o   = observation (1D intensity scan, shape R)
e   = observation embedding (encoder output)
h   = deterministic recurrent hidden state  (GRUCell, shape det_size)
s   = stochastic latent state               (diagonal Gaussian, shape stoch_size)

Observe pass (training + filtering)
------------------------------------
For each timestep t:
    h_t  = GRUCell(s_{t-1}, h_{t-1})          deterministic transition
    e_t  = encoder(o_t)                        encode current observation
    prior:     p(s_t | h_t)     via prior_net(h_t)
    posterior: q(s_t | h_t, e_t) via posterior_net(cat([h_t, e_t]))
    s_t ~ posterior                            (rsample during training)
    o_hat_t = decoder(h_t, s_t)               reconstruct current observation

Training loss (ELBO)
--------------------
    recon_loss = MSE(decoder(h_t, s_t), o_t)    averaged over T, B, R
    kl_loss    = KL(q(s_t|h_t,e_t) || p(s_t|h_t))  averaged over T, B, stoch_size
    loss       = recon_loss + kl_weight * kl_loss

Evaluation
----------
    # Next-step prediction (WorldModel protocol):
    pred_{t+1}, state_t = model.step(o_t, state_{t-1})
        — observe o_t via posterior → state_t = (h_t, s_t)
        — imagine one step forward via prior → (h_{t+1}, s_{t+1})
        — decode o_hat_{t+1} = decoder(h_{t+1}, s_{t+1})

    # Long-horizon imagination (pure prior):
    state_{t+1} = model.imagine_step(state_t)
    o_hat_{t+1} = model.decode(state_{t+1})

    # Hidden states for probing (HiddenStateModel protocol):
    h_flat = model.get_hidden_states(obs)  # (B, T-1, det_size + stoch_size)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


# ── State type ────────────────────────────────────────────────────────────────


class RSSMState(NamedTuple):
    """Compact representation of the RSSM latent state at one timestep.

    h : (B, det_size)   — deterministic recurrent hidden state
    s : (B, stoch_size) — stochastic latent (posterior sample during training,
                          prior sample during imagination)
    """
    h: torch.Tensor
    s: torch.Tensor


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    input_dim: int = 128    # obs_res — overridden from dataset at train time
    embed_dim: int = 128    # observation encoder output dimension
    det_size: int = 200     # GRUCell hidden size (deterministic state h)
    stoch_size: int = 30    # stochastic latent dimension (state s)
    hidden_dim: int = 200   # MLP width for prior, posterior, and decoder nets


# ── Model ─────────────────────────────────────────────────────────────────────


class RSSMModel(nn.Module):
    """Uncontrolled RSSM world model.

    Implements both WorldModel and HiddenStateModel protocols.

    The combined latent state for probing is cat([h_t, s_t]), accessible via
    get_hidden_states().  Linear probes and MLP probes operate on this
    (det_size + stoch_size)-dimensional vector.

    Parameters
    ----------
    cfg:
        Model configuration.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Observation encoder: o_t → e_t
        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.embed_dim),
            nn.ReLU(),
        )

        # Deterministic recurrent core: GRUCell(s_{t-1}, h_{t-1}) → h_t
        # Input = stoch_size (s_{t-1}), hidden = det_size
        self.gru_cell = nn.GRUCell(cfg.stoch_size, cfg.det_size)

        # Prior: p(s_t | h_t)  →  (mu, log_std), shape (2 * stoch_size)
        self.prior_net = nn.Sequential(
            nn.Linear(cfg.det_size, cfg.hidden_dim),
            nn.ELU(),
            nn.Linear(cfg.hidden_dim, 2 * cfg.stoch_size),
        )

        # Posterior: q(s_t | h_t, e_t)  →  (mu, log_std)
        self.posterior_net = nn.Sequential(
            nn.Linear(cfg.det_size + cfg.embed_dim, cfg.hidden_dim),
            nn.ELU(),
            nn.Linear(cfg.hidden_dim, 2 * cfg.stoch_size),
        )

        # Observation decoder: (h_t, s_t) → o_hat_t
        self.decoder = nn.Sequential(
            nn.Linear(cfg.det_size + cfg.stoch_size, cfg.hidden_dim),
            nn.ELU(),
            nn.Linear(cfg.hidden_dim, cfg.input_dim),
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def hidden_size(self) -> int:
        """Combined latent dimension for probing: det_size + stoch_size."""
        return self.cfg.det_size + self.cfg.stoch_size

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        h = torch.zeros(batch_size, self.cfg.det_size, device=device)
        s = torch.zeros(batch_size, self.cfg.stoch_size, device=device)
        return RSSMState(h, s)

    def _prior(self, h: torch.Tensor) -> Normal:
        """Diagonal Gaussian prior p(s | h)."""
        stats = self.prior_net(h)
        mu, log_std = stats.chunk(2, dim=-1)
        return Normal(mu, F.softplus(log_std) + 1e-4)

    def _posterior(self, h: torch.Tensor, e: torch.Tensor) -> Normal:
        """Diagonal Gaussian posterior q(s | h, e)."""
        stats = self.posterior_net(torch.cat([h, e], dim=-1))
        mu, log_std = stats.chunk(2, dim=-1)
        return Normal(mu, F.softplus(log_std) + 1e-4)

    def _flat_state(self, state: RSSMState) -> torch.Tensor:
        """Flatten (h, s) → (B, det_size + stoch_size) for probing/steering."""
        return torch.cat([state.h, state.s], dim=-1)

    def _state_from_flat(self, flat: torch.Tensor) -> RSSMState:
        """Reconstruct RSSMState from flat hidden vector."""
        h = flat[..., : self.cfg.det_size]
        s = flat[..., self.cfg.det_size :]
        return RSSMState(h, s)

    # ── Core RSSM operations ──────────────────────────────────────────────────

    def observe_step(
        self,
        obs_t: torch.Tensor,
        state: RSSMState,
    ) -> tuple[RSSMState, Normal, Normal]:
        """Single observe step: incorporate one real observation.

        Parameters
        ----------
        obs_t : (B, R) — current observation
        state : previous RSSMState (h_{t-1}, s_{t-1})

        Returns
        -------
        new_state : RSSMState — filtered posterior state (h_t, s_t)
        prior     : p(s_t | h_t)
        posterior : q(s_t | h_t, e_t)
        """
        h = self.gru_cell(state.s, state.h)    # (B, det_size)
        e = self.encoder(obs_t)                 # (B, embed_dim)
        prior = self._prior(h)
        posterior = self._posterior(h, e)
        s = posterior.rsample()                 # reparameterized
        return RSSMState(h, s), prior, posterior

    def imagine_step(
        self,
        state: RSSMState,
    ) -> tuple[RSSMState, Normal]:
        """Single imagination step: evolve state using the prior only.

        Parameters
        ----------
        state : current RSSMState (h_t, s_t)

        Returns
        -------
        next_state : RSSMState — prior state at t+1 = (h_{t+1}, s_{t+1})
        prior      : p(s_{t+1} | h_{t+1})
        """
        h = self.gru_cell(state.s, state.h)    # (B, det_size)
        prior = self._prior(h)
        s = prior.rsample()                     # prior sample
        return RSSMState(h, s), prior

    def decode(self, state: RSSMState) -> torch.Tensor:
        """Decode observation from a latent state.

        Parameters
        ----------
        state : RSSMState — (h, s), any leading batch dims

        Returns
        -------
        obs_hat : (..., R) predicted observation
        """
        return self.decoder(self._flat_state(state))

    # ── Protocol interface ────────────────────────────────────────────────────

    def forward(
        self,
        obs: torch.Tensor,
        h0: RSSMState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full training observe pass over a sequence.

        Runs the posterior at every timestep and collects reconstruction
        targets and KL terms for the ELBO.

        Parameters
        ----------
        obs : (B, T, R) — observation sequence
        h0  : optional initial state (zeros if None)

        Returns
        -------
        recons   : (B, T, R) — reconstructed observations decoder(h_t, s_t)
        kl_terms : (B, T) — per-timestep KL(q(s_t|h_t,e_t) || p(s_t|h_t)),
                   summed over stoch_size dimension
        """
        B, T, _ = obs.shape
        state = h0 if h0 is not None else self._initial_state(B, obs.device)

        recons, kls = [], []
        for t in range(T):
            state, prior, posterior = self.observe_step(obs[:, t], state)
            recons.append(self.decode(state))
            kls.append(kl_divergence(posterior, prior).sum(-1))  # (B,)

        return torch.stack(recons, dim=1), torch.stack(kls, dim=1)

    def step(
        self,
        obs_t: torch.Tensor,
        state: RSSMState | None = None,
    ) -> tuple[torch.Tensor, RSSMState]:
        """WorldModel protocol: observe obs_t, predict next observation.

        Filters on obs_t via posterior to get the current state (h_t, s_t),
        then imagines one step forward with the prior and decodes the predicted
        next observation.

        Parameters
        ----------
        obs_t : (B, R) — current observation
        state : previous RSSMState, or None (initialised to zeros)

        Returns
        -------
        pred_next : (B, R) — predicted next observation  o_hat_{t+1}
        state_t   : RSSMState — filtered state at t, for probing/steering
        """
        if state is None:
            state = self._initial_state(obs_t.shape[0], obs_t.device)
        state_t, _, _ = self.observe_step(obs_t, state)
        next_state, _ = self.imagine_step(state_t)
        pred_next = self.decode(next_state)
        return pred_next, state_t

    @torch.no_grad()
    def get_hidden_states(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract combined posterior states via teacher-forcing.

        Runs observe_step on obs[0..T-2] and returns the filtered latent
        states.  Uses the posterior mean (not a stochastic sample) for
        deterministic, reproducible probing.

        Parameters
        ----------
        obs : (B, T, R) — observation sequence

        Returns
        -------
        h_flat : (B, T-1, det_size + stoch_size)
            h_flat[:, t, :] = cat([h_t, s_t]) after seeing obs[:, t, :].
            Aligns with positions[:, t, :] and is_visible[:, t, :].
        """
        B, T, _ = obs.shape
        state = self._initial_state(B, obs.device)
        all_h = []
        for t in range(T - 1):
            h = self.gru_cell(state.s, state.h)
            e = self.encoder(obs[:, t])
            # Use posterior mean for deterministic, stable probing
            stats = self.posterior_net(torch.cat([h, e], dim=-1))
            mu, _ = stats.chunk(2, dim=-1)
            state = RSSMState(h, mu)
            all_h.append(self._flat_state(state))
        return torch.stack(all_h, dim=1)   # (B, T-1, hidden_size)
