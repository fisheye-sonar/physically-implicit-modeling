"""Rollout consistency evaluation.

Two sub-criteria:
  * Observation drift — MSE of AR-predicted observations vs ground truth at
    each rollout horizon.
  * Trajectory coherence — smoothness of decoded env states during rollout,
    via relative acceleration (dimensionless, lower = smoother).

All functions take pre-computed numpy arrays — no model calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoherenceMetrics:
    """Results from a trajectory coherence evaluation."""
    mean_score: float
    std_score: float
    per_component_scores: np.ndarray
    mean_jump_ratio: float


def rollout_coherence(
    decoded_states: np.ndarray,   # (T, *state_shape) single sample
) -> tuple[float, np.ndarray, float]:
    """Smoothness score for one rollout: mean |Δ²s| / (mean |Δs| + ε).

    Returns
    -------
    score        : float — mean relative acceleration
    per_component: (output_dim,) — per state-component scores
    jump_ratio   : max step / median step (teleportation indicator)
    """
    T = decoded_states.shape[0]
    flat = decoded_states.reshape(T, -1)
    if T < 3:
        return 0.0, np.zeros(flat.shape[1]), 1.0
    vel = np.diff(flat, axis=0)
    acc = np.diff(vel, axis=0)
    mean_vel = np.abs(vel).mean(axis=0) + 1e-8
    mean_acc = np.abs(acc).mean(axis=0)
    per_component = mean_acc / mean_vel
    step_norms = np.linalg.norm(vel, axis=1)
    jump_ratio = float(step_norms.max() / (np.median(step_norms) + 1e-8))
    return float(per_component.mean()), per_component, jump_ratio


def per_sample_coherence(
    decoded_states_rollout: np.ndarray,   # (N, T_rollout, *state_shape)
) -> np.ndarray:
    """Per-sample smoothness scores. Returns (N,) array."""
    return np.array(
        [rollout_coherence(decoded_states_rollout[i])[0]
         for i in range(decoded_states_rollout.shape[0])]
    )


def eval_observation_drift(
    obs_actual: np.ndarray,     # (N, T, R)
    obs_rollout: np.ndarray,    # (N, T_rollout, R)
    n_context: int,
) -> np.ndarray:
    """Per-step MSE of AR-predicted observations vs ground truth (aligned)."""
    n_rollout = obs_rollout.shape[1]
    targets = obs_actual[:, n_context : n_context + n_rollout, :]
    sq_err = (obs_rollout - targets) ** 2
    return sq_err.mean(axis=(0, 2))


def eval_position_drift(
    decoded_positions: np.ndarray,    # (N, T_rollout, *state_shape)
    positions_gt: np.ndarray,         # (N, T_full, *state_shape) — full sequence
    n_context: int,
) -> np.ndarray:
    """Per-step MSE of decoded positions vs GT positions across the rollout."""
    n_rollout = decoded_positions.shape[1]
    n_obj = decoded_positions.shape[2]
    gt = positions_gt[:, n_context : n_context + n_rollout, :n_obj, :]
    return ((decoded_positions - gt) ** 2).mean(axis=(0, 2, 3))


def eval_trajectory_coherence(
    decoded_states_rollout: np.ndarray,   # (N, T_rollout, *state_shape)
) -> CoherenceMetrics:
    """Population coherence over N rollout trajectories."""
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
