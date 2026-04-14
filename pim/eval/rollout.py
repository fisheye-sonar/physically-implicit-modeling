"""Rollout consistency evaluation.

Criterion 3: does the model produce plausible observations and decoded
env states during extended autoregressive rollout?

Two sub-criteria:
  - Observation drift: MSE of AR-predicted observations vs ground truth at
    each rollout horizon.
  - Trajectory coherence: smoothness of decoded env states during rollout,
    measured via relative acceleration (dimensionless, lower = smoother).

All functions take pre-computed arrays — no model calls here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoherenceMetrics:
    """Results from a trajectory coherence evaluation."""
    mean_score: float                     # mean relative acceleration (population)
    std_score: float
    per_component_scores: np.ndarray      # (output_dim,) per state component
    mean_jump_ratio: float                # max_step / median_step (teleportation proxy)


def rollout_coherence(
    decoded_states: np.ndarray,   # (T, *state_shape) single sample
) -> tuple[float, np.ndarray, float]:
    """Compute trajectory smoothness for one rollout.

    Measures the relative acceleration: mean |Δ²s| / (mean |Δs| + ε),
    where Δ²s is the second difference (acceleration) and Δs is the first
    difference (velocity). Lower = smoother trajectory.

    Also computes the jump ratio: max |Δs| / (median |Δs| + ε), which
    flags sudden teleportation-like discontinuities.

    Parameters
    ----------
    decoded_states : (T, *state_shape) trajectory of decoded env states

    Returns
    -------
    score        : float — mean relative acceleration across all components
    per_component: (output_dim,) — per state-component scores
    jump_ratio   : float — max step / median step (teleportation indicator)
    """
    T = decoded_states.shape[0]
    flat = decoded_states.reshape(T, -1)   # (T, D)

    if T < 3:
        D = flat.shape[1]
        return 0.0, np.zeros(D), 1.0

    vel = np.diff(flat, axis=0)     # (T-1, D) first differences
    acc = np.diff(vel, axis=0)      # (T-2, D) second differences

    mean_vel = np.abs(vel).mean(axis=0) + 1e-8   # (D,)
    mean_acc = np.abs(acc).mean(axis=0)           # (D,)
    per_component = mean_acc / mean_vel            # (D,)
    score = float(per_component.mean())

    # Jump ratio
    step_norms = np.linalg.norm(vel, axis=1)      # (T-1,)
    jump_ratio = float(step_norms.max() / (np.median(step_norms) + 1e-8))

    return score, per_component, jump_ratio


def eval_observation_drift(
    obs_actual: np.ndarray,     # (N, T, R)
    obs_rollout: np.ndarray,    # (N, T_rollout, R)
    n_context: int,
) -> np.ndarray:
    """Per-step observation MSE during AR rollout.

    Parameters
    ----------
    obs_actual   : (N, T, R) ground-truth observations
    obs_rollout  : (N, T_rollout, R) AR predicted observations
    n_context    : context frames before rollout (alignment offset)

    Returns
    -------
    drift_mse : (T_rollout,) — MSE at each rollout step
    """
    n_rollout = obs_rollout.shape[1]
    targets = obs_actual[:, n_context : n_context + n_rollout, :]   # (N, T_rollout, R)
    sq_err = (obs_rollout - targets) ** 2
    return sq_err.mean(axis=(0, 2))   # (T_rollout,)


def eval_trajectory_coherence(
    decoded_states_rollout: np.ndarray,   # (N, T_rollout, *state_shape)
) -> CoherenceMetrics:
    """Population coherence scoring over N rollout trajectories.

    Parameters
    ----------
    decoded_states_rollout : (N, T_rollout, *state_shape) decoded trajectories

    Returns
    -------
    CoherenceMetrics
    """
    N = decoded_states_rollout.shape[0]
    scores, per_comp_list, jumps = [], [], []

    for i in range(N):
        score, per_comp, jump = rollout_coherence(decoded_states_rollout[i])
        scores.append(score)
        per_comp_list.append(per_comp)
        jumps.append(jump)

    return CoherenceMetrics(
        mean_score=float(np.mean(scores)),
        std_score=float(np.std(scores)),
        per_component_scores=np.stack(per_comp_list).mean(axis=0),
        mean_jump_ratio=float(np.mean(jumps)),
    )
