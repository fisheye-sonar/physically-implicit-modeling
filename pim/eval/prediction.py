"""Predictive quality evaluation.

Criterion 1: how well does the model predict the next observation?

All functions except eval_mse_by_context take pre-computed arrays.
eval_mse_by_context requires the model because each context length
needs a fresh AR sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from pim.world_models.protocol import WorldModel


@dataclass
class PredictionMetrics:
    """Results from a single-step prediction evaluation."""

    mean_mse: float
    std_mse: float
    per_sample_mse: np.ndarray  # (N,)


def eval_single_step(
    obs_actual: np.ndarray,  # (N, T, R)
    obs_predicted: np.ndarray,  # (N, T-1, R)  teacher-forcing predictions
) -> PredictionMetrics:
    """Next-step MSE under teacher forcing.

    Parameters
    ----------
    obs_actual    : (N, T, R) ground-truth observations
    obs_predicted : (N, T-1, R) model's teacher-forcing predictions
                    (obs_predicted[:, t] is the model's prediction for obs_actual[:, t+1])

    Returns
    -------
    PredictionMetrics with mean, std, and per-sample MSE
    """
    targets = obs_actual[:, 1:, :]  # (N, T-1, R)
    sq_err = (obs_predicted - targets) ** 2
    per_sample_mse = sq_err.mean(axis=(1, 2))  # (N,)
    return PredictionMetrics(
        mean_mse=float(per_sample_mse.mean()),
        std_mse=float(per_sample_mse.std()),
        per_sample_mse=per_sample_mse,
    )


def eval_horizon_mse(
    obs_actual: np.ndarray,  # (N, T, R)
    obs_rollout: np.ndarray,  # (N, T_rollout, R)  AR predictions
    n_context: int,
) -> np.ndarray:
    """MSE of autoregressive predictions at each horizon step.

    Parameters
    ----------
    obs_actual   : (N, T, R) ground-truth
    obs_rollout  : (N, T_rollout, R) AR predictions starting at frame n_context
    n_context    : number of context frames before rollout starts

    Returns
    -------
    mse_by_horizon : (T_rollout,) — MSE at each step ahead
    """
    n_rollout = obs_rollout.shape[1]
    targets = obs_actual[:, n_context : n_context + n_rollout, :]  # (N, T_rollout, R)
    sq_err = (obs_rollout - targets) ** 2
    return sq_err.mean(axis=(0, 2))  # (T_rollout,)


@torch.no_grad()
def eval_mse_by_context(
    model: WorldModel,
    loader: DataLoader,
    n_steps_ahead: int = 1,
    device: str = "cpu",
    obs_key: str = "obs_intensity",
) -> tuple[np.ndarray, np.ndarray]:
    """MSE of an n_steps_ahead prediction as a function of context length.

    For each timestep t in [1, T - n_steps_ahead], warms up for t frames,
    then predicts n_steps_ahead autoregressively and measures MSE against
    the ground truth.

    Parameters
    ----------
    model         : WorldModel
    loader        : DataLoader (obs_key must be present)
    n_steps_ahead : prediction horizon
    device        : torch device string
    obs_key       : batch key for observations

    Returns
    -------
    context_lengths : (T - n_steps_ahead,) int array
    mse_by_context  : (T - n_steps_ahead,) float array
    """
    # Peek at T and R from the first batch
    first_batch = next(iter(loader))
    T = first_batch[obs_key].shape[1]
    max_t = T - n_steps_ahead

    sum_sq_err = np.zeros(max_t)
    counts = np.zeros(max_t)

    for batch in loader:
        obs = batch[obs_key].float().to(device)  # (B, T, R)
        B = obs.shape[0]

        for t in range(1, max_t + 1):
            # Warm up for t steps
            h = None
            for step in range(t):
                pred, h = model.step(obs[:, step, :], h)

            # Roll out n_steps_ahead - 1 more steps (we have 1 pred already)
            x = pred
            for _ in range(n_steps_ahead - 1):
                x, h = model.predict_step(h)

            target = obs[:, t + n_steps_ahead - 1, :]  # (B, R)
            sq_err = ((x - target) ** 2).mean(dim=1).sum().item()
            sum_sq_err[t - 1] += sq_err
            counts[t - 1] += B

    mse_by_context = sum_sq_err / counts
    context_lengths = np.arange(1, max_t + 1)
    return context_lengths, mse_by_context
