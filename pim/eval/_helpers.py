"""Inference helpers — the only place in pim/eval/ that calls models.

These functions bridge world models and the pure-array eval functions.
They run inference and return numpy arrays that eval functions consume.

All functions are decorated with @torch.no_grad() and accept numpy or tensor
inputs for convenience.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from pim.world_models.protocol import HiddenStateModel, WorldModel


@torch.no_grad()
def run_autoregressive(
    model: WorldModel,
    obs: np.ndarray,      # (T, R) single sample
    n_context: int,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Warm up on obs[:n_context], then roll out autoregressively.

    Parameters
    ----------
    model     : WorldModel
    obs       : (T, R) float32 single observation sequence
    n_context : frames used to build hidden state before rollout begins

    Returns
    -------
    obs_rollout     : (T - n_context, R) float32 — predicted observations
    internal_states : (T - 1, H) float32 if model is HiddenStateModel, else None
                      Hidden states for all T-1 steps (context + rollout).
    """
    T = obs.shape[0]
    obs_t = torch.from_numpy(obs).float().to(device)

    h = None
    all_h = []   # collect hidden states if available
    is_hsm = isinstance(model, HiddenStateModel)

    # Context warm-up
    last_pred = None
    for t in range(n_context):
        pred_t, h = model.step(obs_t[t].unsqueeze(0), h)
        if is_hsm:
            # h shape: (num_layers, 1, H) — take last layer, squeeze batch
            all_h.append(h[-1, 0].cpu().numpy())
        last_pred = pred_t

    # Autoregressive rollout
    preds = []
    x = last_pred
    for _ in range(T - n_context):
        x, h = model.step(x, h)
        preds.append(x.squeeze(0).cpu().numpy())
        if is_hsm:
            all_h.append(h[-1, 0].cpu().numpy())

    obs_rollout = np.stack(preds, axis=0)           # (T - n_context, R)
    internal_states = np.stack(all_h) if all_h else None  # (T-1, H) or None
    return obs_rollout, internal_states


@torch.no_grad()
def run_teacher_forcing(
    model: HiddenStateModel,
    loader: DataLoader,
    device: str = "cpu",
    obs_key: str = "obs_intensity",
) -> tuple[np.ndarray, np.ndarray]:
    """Run teacher-forcing over a full loader; collect predictions + hidden states.

    Parameters
    ----------
    model    : HiddenStateModel (must implement get_hidden_states)
    loader   : DataLoader yielding batches with obs_key
    device   : torch device string
    obs_key  : key in batch dict for the observation sequence

    Returns
    -------
    obs_pred        : (N, T-1, R) float32 — teacher-forcing predictions
    internal_states : (N, T-1, H) float32 — hidden states from get_hidden_states
    """
    all_pred, all_h = [], []

    for batch in loader:
        obs = batch[obs_key].float().to(device)     # (B, T, R)
        pred, _ = model(obs)                         # (B, T-1, R)
        h = model.get_hidden_states(obs)             # (B, T-1, H)
        all_pred.append(pred.cpu().numpy())
        all_h.append(h.cpu().numpy())

    obs_pred = np.concatenate(all_pred, axis=0)         # (N, T-1, R)
    internal_states = np.concatenate(all_h, axis=0)     # (N, T-1, H)
    return obs_pred, internal_states


@torch.no_grad()
def collect_rollout(
    model: HiddenStateModel,
    obs: np.ndarray,      # (T, R)
    n_context: int,
    n_rollout: int,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect context and rollout hidden states + rollout observations.

    Parameters
    ----------
    model     : HiddenStateModel
    obs       : (T, R) observation sequence
    n_context : frames for context warm-up
    n_rollout : frames to roll out autoregressively

    Returns
    -------
    h_context   : (n_context, H) hidden states during context
    h_rollout   : (n_rollout, H) hidden states during rollout
    obs_rollout : (n_rollout, R) predicted observations during rollout
    """
    obs_t = torch.from_numpy(obs).float().to(device)
    h = None
    h_ctx, h_roll, obs_roll = [], [], []

    # Context warm-up (teacher forcing)
    last_pred = None
    for t in range(n_context):
        pred_t, h = model.step(obs_t[t].unsqueeze(0), h)
        h_ctx.append(h[-1, 0].cpu().numpy())
        last_pred = pred_t

    # Autoregressive rollout
    x = last_pred
    for _ in range(n_rollout):
        x, h = model.step(x, h)
        h_roll.append(h[-1, 0].cpu().numpy())
        obs_roll.append(x.squeeze(0).cpu().numpy())

    return (
        np.stack(h_ctx),    # (n_context, H)
        np.stack(h_roll),   # (n_rollout, H)
        np.stack(obs_roll), # (n_rollout, R)
    )
