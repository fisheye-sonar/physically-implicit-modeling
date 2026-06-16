"""Inference helpers — the only place in pim/eval/ that calls models.

Model-agnostic via the HiddenStateModel SSM protocol (step, predict_step,
flat_state, observe_sequence). Functions here run inference and return numpy
arrays that the rest of pim/eval/ consumes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pim.world_models.protocol import HiddenStateModel


def _as_obs_tensor(obs: np.ndarray | torch.Tensor, device: str) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs.float().to(device)
    return torch.from_numpy(obs).float().to(device)


@torch.no_grad()
def teacher_force(
    model: HiddenStateModel,
    loader: DataLoader,
    device: str = "cpu",
    obs_key: str = "obs_intensity",
) -> tuple[np.ndarray, np.ndarray]:
    """Single-pass teacher forcing over a full loader.

    Returns
    -------
    obs_pred : (N, T-1, R) — next-step predictions; obs_pred[i, t] ≈ obs[i, t+1]
    states   : (N, T-1, H) — flat hidden state aligned to obs[:, :-1]
    """
    all_pred, all_h = [], []
    for batch in loader:
        obs = batch[obs_key].float().to(device)
        pred, h = model.observe_sequence(obs)
        all_pred.append(pred.cpu().numpy())
        all_h.append(h.cpu().numpy())
    return np.concatenate(all_pred, axis=0), np.concatenate(all_h, axis=0)


def _ar_single(
    model: HiddenStateModel,
    obs: np.ndarray,
    n_context: int,
    n_rollout: int,
    device: str,
    *,
    collect_hidden: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Core single-sample AR loop: context via model.step, rollout via model.predict_step."""
    obs_t = _as_obs_tensor(obs, device)

    state = None
    last_pred = None
    h_ctx, h_roll, obs_roll = [], [], []

    for t in range(n_context):
        last_pred, state = model.step(obs_t[t].unsqueeze(0), state)
        if collect_hidden:
            h_ctx.append(model.flat_state(state).squeeze(0).cpu().numpy())

    if last_pred is None:
        raise ValueError("n_context must be at least 1")

    for _ in range(n_rollout):
        obs_roll.append(last_pred.squeeze(0).cpu().numpy())
        last_pred, state = model.predict_step(state)
        if collect_hidden:
            h_roll.append(model.flat_state(state).squeeze(0).cpu().numpy())

    h_context = np.stack(h_ctx) if collect_hidden else None
    h_rollout = np.stack(h_roll) if collect_hidden else None
    return h_context, h_rollout, np.stack(obs_roll)


@torch.no_grad()
def autoregressive_rollout(
    model: HiddenStateModel,
    obs: np.ndarray,
    n_context: int,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Single-sample AR rollout: warm up on obs[:n_context], then roll out to end.

    Returns
    -------
    obs_rollout : (T - n_context, R) — obs_rollout[0] predicts obs[n_context]
    states      : (T, H) — flat hidden states for context + rollout (concatenated)
    """
    T = obs.shape[0]
    h_ctx, h_roll, obs_roll = _ar_single(
        model, obs, n_context, T - n_context, device, collect_hidden=True
    )
    assert h_ctx is not None and h_roll is not None
    return obs_roll, np.concatenate([h_ctx, h_roll], axis=0)


@torch.no_grad()
def autoregressive_rollouts(
    model: HiddenStateModel,
    obs_array: np.ndarray,
    n_context: int,
    device: str = "cpu",
    *,
    desc: str = "AR rollout",
) -> np.ndarray:
    """Batched AR rollout over many samples.

    Parameters
    ----------
    obs_array : (N, T, R)

    Returns
    -------
    obs_rollout : (N, T - n_context, R)
    """
    T = obs_array.shape[1]
    rollouts = []
    for i in tqdm(range(len(obs_array)), desc=desc, leave=False):
        _, _, obs_roll = _ar_single(
            model, obs_array[i], n_context, T - n_context, device, collect_hidden=False
        )
        rollouts.append(obs_roll)
    return np.stack(rollouts)


@torch.no_grad()
def collect_rollouts(
    model: HiddenStateModel,
    obs_array: np.ndarray,
    n_context: int,
    n_rollout: int,
    device: str = "cpu",
    *,
    desc: str = "rollouts",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-window rollout for many samples: warm up n_context, roll out n_rollout.

    Parameters
    ----------
    obs_array : (N, T, R)

    Returns
    -------
    h_context   : (N, n_context, H)
    h_rollout   : (N, n_rollout, H)
    obs_rollout : (N, n_rollout, R) — obs_rollout[i, 0] predicts obs_array[i, n_context]
    """
    h_ctxs, h_rolls, obs_rolls = [], [], []
    for i in tqdm(range(len(obs_array)), desc=desc, leave=False):
        h_ctx, h_roll, obs_roll = _ar_single(
            model, obs_array[i], n_context, n_rollout, device, collect_hidden=True
        )
        h_ctxs.append(h_ctx)
        h_rolls.append(h_roll)
        obs_rolls.append(obs_roll)
    return np.stack(h_ctxs), np.stack(h_rolls), np.stack(obs_rolls)


@torch.no_grad()
def decode_states_multi(
    probes,  # list[ProbeSpec]
    states: np.ndarray,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Apply each probe to the same hidden-state array.

    Parameters
    ----------
    probes : list of ProbeSpec
    states : (..., H) hidden states (any leading shape)

    Returns
    -------
    decoded : dict[probe.name -> (..., *state_shape)] decoded states per probe
    """
    h = torch.from_numpy(states).float().to(device)
    out: dict[str, np.ndarray] = {}
    for p in probes:
        p.probe.to(device).eval()
        out[p.name] = p.probe(h).cpu().numpy()
    return out
