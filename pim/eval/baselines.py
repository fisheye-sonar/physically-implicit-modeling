"""Reference RMSE baselines for annotating prediction/recovery plots.

These are properties of the dataset, not the model. They give the reader
something to compare model RMSE against (e.g., a model RMSE below the noise
floor is impossibly good and indicates a leak).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ObsBaselines:
    """Observation RMSE baselines."""
    random_rmse: float        # sqrt(2 * Var(obs)) — predicting a random independent frame
    identity_rmse: float      # RMSE of copying the previous frame (obs[t] ≈ obs[t-1])
    noise_std: float          # additive obs noise σ (from dataset config)
    noise_floor_rmse: float   # empirical RMSE of noisy vs clean obs (clipping-aware)


@dataclass
class PosBaselines:
    """Position RMSE baselines."""
    random_rmse: float        # sqrt(2 * Var(positions))
    identity_rmse: float      # RMSE of copying the previous position (pos[t] ≈ pos[t-1])
    noise_std: float          # per-step position diffusion noise σ (from dataset config)


def compute_obs_baselines(
    obs: np.ndarray,            # (N, T, R) noisy observations
    clean_obs: np.ndarray,      # (N, T, R) reconstructed noiseless observations
    noise_std: float,           # additive obs noise σ from dataset config
) -> ObsBaselines:
    """Compute observation RMSE baselines from arrays + config."""
    return ObsBaselines(
        random_rmse=float(np.sqrt(2.0 * obs.var())),
        identity_rmse=float(np.sqrt(((obs[:, 1:] - obs[:, :-1]) ** 2).mean())),
        noise_std=float(noise_std),
        noise_floor_rmse=float(np.sqrt(((obs - clean_obs) ** 2).mean())),
    )


def compute_pos_baselines(
    positions: np.ndarray,      # (N, T, n_obj, 2)
    noise_std: float,           # per-step position diffusion σ
) -> PosBaselines:
    """Compute position RMSE baselines from positions + config."""
    return PosBaselines(
        random_rmse=float(np.sqrt(2.0 * positions.var())),
        identity_rmse=float(np.sqrt(((positions[:, 1:] - positions[:, :-1]) ** 2).mean())),
        noise_std=float(noise_std),
    )
