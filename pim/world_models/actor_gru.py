"""Endogenous-action GRU world model — the **actor** (and its **observer** twin).

NEW module (does not modify the plain GRU or the exogenous ``action_gru*``).  See
``research/directions/endogenous-action-interactive-world.md``.

The actor is a plain next-step GRU predictor **plus** two extra heads read off the
same hidden state ``h_t``:

  * a **policy head** — a per-object, per-axis categorical over ``{-1, 0, +1}`` (matches
    the keyboard); it *generates* the action ``a_t`` that is applied to the world;
  * a **value head** — ``V(h_t)`` for the REINFORCE/A2C baseline at L3.

Crucially the encoder is **obs-only** (no action-input channel): the action is carried
forward implicitly through the recurrence (``h_t → h_{t+1}``) and re-observed in
``o_{t+1}``.  The action only conditions the **decoder** for the immediate prediction —
``pred o_{t+1} = decoder([h_t, proj(a_t)])`` — because a *sampled* (stochastic) action is
not determined by ``h_t`` and ``o_{t+1}`` is a delayed/lossy view of it (see the design
doc, decision 1).

The **observer** is the *same class* (identical architecture) used in a different causal
role: it never acts and never touches the world; it is fed the actor's actions as the
decoder conditioning and trained only on prediction.  Keeping the architectures identical
makes the actor-vs-observer contrast isolate *generation/agency*, not capacity.

Protocol conformance
--------------------
Conforms to ``HiddenStateModel`` with the action defaulting to **no-op (zeros)**, so the
whole passive-latent eval suite (extractors / editors / eval) runs UNCHANGED on the
passive state — ``decode`` / ``predict_step`` / ``observe_sequence`` use the no-op action.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EndogenousActorConfig:
    input_dim: int = 128  # obs_res
    hidden_size: int = 256
    num_layers: int = 1
    n_obj: int = 2
    n_axes: int = 2  # (x, y) per object
    n_bins: int = 3  # categorical bins per axis; 3 = {-1, 0, +1}
    action_proj_dim: int = 16
    # ---- capacity (defaults reproduce the original 1-Linear enc/dec exactly, so
    #      older checkpoints load unchanged; >1 adds extra MLP layers) ----
    enc_layers: int = 1  # encoder depth (1 = single Linear, as originally)
    dec_layers: int = 1  # decoder depth (1 = single Linear, as originally)
    # If True the PREVIOUS action is concatenated to the GRU input, so the action enters the
    # *transition* (h_{t+1} = GRU([enc(o_t), proj(a_{t-1})], h_t)) and not only the decoder.
    # Without this the model cannot imagine an action's effect on its own state: the effect has
    # to pass through decoder -> predicted obs -> encoder, a lossy bottleneck.  Default False
    # reproduces the original architecture so existing checkpoints still load.
    action_in_transition: bool = False


class EndogenousActorGRU(nn.Module):
    """GRU predictor + policy head + value head + action-conditioned decoder."""

    def __init__(self, cfg: EndogenousActorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._adim = cfg.n_obj * cfg.n_axes
        self.encoder = nn.Linear(cfg.input_dim, cfg.hidden_size)  # obs-only
        self.act_trans_proj = (
            nn.Linear(self._adim, cfg.action_proj_dim)
            if cfg.action_in_transition
            else None
        )
        self.gru = nn.GRU(
            input_size=cfg.hidden_size
            + (cfg.action_proj_dim if cfg.action_in_transition else 0),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
        )
        self.action_proj = nn.Linear(self._adim, cfg.action_proj_dim)
        self.decoder = nn.Linear(cfg.hidden_size + cfg.action_proj_dim, cfg.input_dim)
        # Optional extra capacity. Kept as SEPARATE modules (rather than rebuilding
        # `encoder`/`decoder` as Sequentials) so that state-dict keys of the original
        # architecture are preserved and old checkpoints load unchanged.
        H, D = cfg.hidden_size, cfg.hidden_size + cfg.action_proj_dim
        self.enc_extra = (
            nn.Sequential(
                *sum(
                    ([nn.Linear(H, H), nn.ReLU()] for _ in range(cfg.enc_layers - 1)),
                    [],
                )
            )
            if cfg.enc_layers > 1
            else None
        )
        self.dec_extra = (  # residual (starts near-identity) so it only adds capacity
            nn.Sequential(
                *sum(
                    ([nn.Linear(D, D), nn.ReLU()] for _ in range(cfg.dec_layers - 1)),
                    [],
                )
            )
            if cfg.dec_layers > 1
            else None
        )
        self.policy = nn.Linear(cfg.hidden_size, self._adim * cfg.n_bins)
        self.value = nn.Linear(cfg.hidden_size, 1)
        # categorical bin centres in [-1, 1]  (n_bins=3 → [-1, 0, 1])
        self.register_buffer("bin_values", torch.linspace(-1.0, 1.0, cfg.n_bins))

    @property
    def hidden_size(self) -> int:
        return self.cfg.hidden_size

    # ── action helpers ────────────────────────────────────────────────────────
    def _noop(self, *lead, device=None):
        """Zeros action with the given leading dims: (*lead, n_obj, n_axes)."""
        if device is None:
            lead, device = lead[:-1], lead[-1]
        return torch.zeros(*lead, self.cfg.n_obj, self.cfg.n_axes, device=device)

    def _proj(self, action: torch.Tensor) -> torch.Tensor:
        """action (..., n_obj, n_axes) → projected embedding (..., proj_dim)."""
        flat = action.reshape(*action.shape[:-2], self._adim)
        return F.relu(self.action_proj(flat))

    def policy_logits(self, h: torch.Tensor) -> torch.Tensor:
        """h (..., H) → logits (..., n_obj, n_axes, n_bins)."""
        return self.policy(h).reshape(
            *h.shape[:-1], self.cfg.n_obj, self.cfg.n_axes, self.cfg.n_bins
        )

    def value_of(self, h: torch.Tensor) -> torch.Tensor:
        return self.value(h).squeeze(-1)

    def act(self, h: torch.Tensor, deterministic: bool = False):
        """Sample an action from the policy at state ``h``.

        Returns ``(action, logp, entropy, idx)``:
          action  : (B, n_obj, n_axes) float in {-1, 0, +1}
          logp    : (B,) summed log-prob over all (obj, axis)
          entropy : (B,) summed categorical entropy
          idx     : (B, n_obj, n_axes) long — the sampled bin indices
        """
        dist = torch.distributions.Categorical(logits=self.policy_logits(h))
        idx = dist.logits.argmax(-1) if deterministic else dist.sample()
        logp = dist.log_prob(idx).flatten(1).sum(1)
        ent = dist.entropy().flatten(1).sum(1)
        action = self.bin_values[idx]
        return action, logp, ent, idx

    def logp_entropy(self, h: torch.Tensor, idx: torch.Tensor):
        """Recompute (logp, entropy) of stored bin indices under the current policy."""
        dist = torch.distributions.Categorical(logits=self.policy_logits(h))
        logp = (
            dist.log_prob(idx).flatten(2).sum(2)
            if idx.dim() == 4
            else dist.log_prob(idx).flatten(1).sum(1)
        )
        ent = (
            dist.entropy().flatten(2).sum(2)
            if idx.dim() == 4
            else dist.entropy().flatten(1).sum(1)
        )
        return logp, ent

    def _enc(
        self, obs: torch.Tensor, prev_action: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Single encode choke-point: obs (..., R) → GRU input.

        When ``action_in_transition`` is set, the PREVIOUS action is appended so the action
        participates in the state transition (standard action-conditioned world model).
        """
        x = F.relu(self.encoder(obs))
        if self.enc_extra is not None:
            x = self.enc_extra(x)
        if self.act_trans_proj is not None:
            a = (
                self._noop(*obs.shape[:-1], obs.device)
                if prev_action is None
                else prev_action
            )
            flat = a.reshape(*a.shape[:-2], self._adim)
            x = torch.cat([x, F.relu(self.act_trans_proj(flat))], dim=-1)
        return x

    def decode_action(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict the next obs from state ``h`` conditioned on the chosen ``action``."""
        z = torch.cat([h, self._proj(action)], dim=-1)
        if self.dec_extra is not None:
            z = z + self.dec_extra(z)  # residual
        return self.decoder(z)

    # ── online single step (for rollout in the world) ─────────────────────────
    def gru_step(self, obs_t, state=None, prev_action=None):
        """obs_t (B, R), state (L, B, H) → (h_t (B, H), state_next).

        ``prev_action`` (B, n_obj, n_axes) is used only when ``action_in_transition`` is set.
        """
        x = self._enc(obs_t, prev_action).unsqueeze(1)
        h_out, state_next = self.gru(x, state)
        return h_out.squeeze(1), state_next

    # ── teacher-forced sequence prediction (for the predictor loss) ───────────
    def predict_sequence(self, obs: torch.Tensor, actions: torch.Tensor, h0=None):
        """obs (B, T, R), actions (B, T-1, n_obj, n_axes) driving o_t→o_{t+1}.

        ``h0`` is the recurrent state the chunk starts from.  Pass the state that collection
        started from so the teacher-forced states match the ones the policy actually acted on
        (needed when the hidden state is carried across iteration boundaries).

        Returns ``(pred (B, T-1, R), h_seq (B, T-1, H))``; ``h_seq[:, t]`` is the state
        after seeing ``obs[:, t]`` and predicts ``obs[:, t+1]`` given ``actions[:, t]``.
        """
        prev = None
        if self.act_trans_proj is not None:
            # h_t consumes a_{t-1}; a_{-1} is a no-op
            prev = torch.cat([torch.zeros_like(actions[:, :1]), actions[:, :-1]], dim=1)
        x = self._enc(obs[:, :-1, :], prev)
        h_seq, _ = self.gru(x, h0)
        pred = self.decode_action(h_seq, actions)
        return pred, h_seq

    # ── HiddenStateModel protocol (passive / no-op semantics for eval) ────────
    def forward(self, obs: torch.Tensor, h0=None, actions: torch.Tensor | None = None):
        a = (
            self._noop(obs.shape[0], obs.device)
            .unsqueeze(1)
            .expand(-1, obs.shape[1] - 1, -1, -1)
            if actions is None
            else actions
        )
        x = self._enc(obs[:, :-1, :])
        h_seq, h_n = self.gru(x, h0)
        pred = self.decode_action(h_seq, a)
        return pred, h_n

    def step(self, obs_t: torch.Tensor, state=None, action: torch.Tensor | None = None):
        h, state_next = self.gru_step(obs_t, state)
        a = self._noop(obs_t.shape[0], obs_t.device) if action is None else action
        return self.decode_action(h, a), state_next

    @torch.no_grad()
    def get_hidden_states(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._enc(obs[:, :-1, :])
        h, _ = self.gru(x)
        return h

    def flat_state(self, state: torch.Tensor) -> torch.Tensor:
        return state[-1]

    def state_from_flat(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.unsqueeze(0)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        h = state[-1]
        return self.decode_action(h, self._noop(h.shape[0], h.device))

    @torch.no_grad()
    def observe_sequence(self, obs: torch.Tensor):
        x = self._enc(obs[:, :-1, :])
        h_seq, _ = self.gru(x)
        a = (
            self._noop(obs.shape[0], obs.device)
            .unsqueeze(1)
            .expand(-1, h_seq.shape[1], -1, -1)
        )
        pred = self.decode_action(h_seq, a)
        return pred, h_seq

    def predict_step(self, state: torch.Tensor):
        obs_hat = self.decode(state)
        return self.step(obs_hat, state)
