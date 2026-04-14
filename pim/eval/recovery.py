"""Recovery / inverse problem evaluation.

Criterion 2: how well can a trained extractor recover env state from
model internal states?

All functions take pre-computed arrays — no model calls here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class RecoveryMetrics:
    """Results from a recovery evaluation."""
    overall_mse: float
    per_component_mse: np.ndarray   # (output_dim,) — MSE per state component
    mse_by_context: np.ndarray      # (T,) — MSE as function of context length


def eval_recovery(
    env_states_gt: np.ndarray,     # (N, T, *state_shape)
    internal_states: np.ndarray,   # (N, T, H)
    extractor: nn.Module,
    *,
    mask: np.ndarray | None = None,   # (N, T) bool — True = include
    use_hungarian: bool = False,
    device: str = "cpu",
    batch_size: int = 512,
) -> RecoveryMetrics:
    """Evaluate how well extractor(internal_states) matches env_states_gt.

    Parameters
    ----------
    env_states_gt   : (N, T, *state_shape) ground-truth environment state
    internal_states : (N, T, H) hidden states from run_teacher_forcing()
    extractor       : trained extractor module
    mask            : (N, T) bool — only masked timesteps contribute to metrics
    use_hungarian   : if True, apply Hungarian matching (for object-centric states
                      with ambiguous ordering)
    device          : torch device
    batch_size      : samples per forward pass

    Returns
    -------
    RecoveryMetrics
    """
    extractor = extractor.to(device).eval()
    N, T = internal_states.shape[:2]

    all_pred, all_gt = [], []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            h_b = torch.from_numpy(internal_states[i:i+batch_size]).float().to(device)
            pred_b = extractor(h_b).cpu().numpy()   # (B, T, *state_shape)
            all_pred.append(pred_b)
            all_gt.append(env_states_gt[i:i+batch_size])

    pred_all = np.concatenate(all_pred, axis=0)   # (N, T, *state_shape)
    gt_all = np.concatenate(all_gt, axis=0)       # (N, T, *state_shape)

    if use_hungarian:
        pred_all, gt_all = _apply_hungarian(pred_all, gt_all)

    sq_err = (pred_all - gt_all) ** 2   # (N, T, *state_shape)

    if mask is not None:
        # Expand mask to (N, T, 1, 1, ...) to broadcast against sq_err's state_shape dims
        m = mask
        for _ in range(sq_err.ndim - mask.ndim):
            m = m[..., None]
        sq_err = sq_err * m

    # Per-component MSE (flatten state_shape)
    flat_sq = sq_err.reshape(N, T, -1)   # (N, T, output_dim)
    if mask is not None:
        denom = mask.sum()
    else:
        denom = N * T
    per_component_mse = flat_sq.sum(axis=(0, 1)) / denom   # (output_dim,)
    overall_mse = float(per_component_mse.mean())

    # MSE by context length (mean over samples and state dims at each t)
    mse_by_context = flat_sq.mean(axis=(0, 2))   # (T,)
    if mask is not None:
        mask_per_t = mask.mean(axis=0)  # (T,) fraction of valid samples
        mse_by_context = np.where(mask_per_t > 0, mse_by_context / mask_per_t, 0.0)

    return RecoveryMetrics(
        overall_mse=overall_mse,
        per_component_mse=per_component_mse,
        mse_by_context=mse_by_context,
    )


def _apply_hungarian(
    pred: np.ndarray,   # (N, T, n_obj, D)
    gt: np.ndarray,     # (N, T, n_obj, D)
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder pred objects to best match gt per sample-timestep (2-object case)."""
    if pred.ndim < 4 or pred.shape[2] != 2:
        return pred, gt   # only supported for (N, T, 2, D)

    loss_01 = ((pred - gt) ** 2).mean(axis=(-2, -1))              # (N, T)
    pred_swap = np.stack([pred[:, :, 1], pred[:, :, 0]], axis=2)  # (N, T, 2, D)
    loss_10 = ((pred_swap - gt) ** 2).mean(axis=(-2, -1))         # (N, T)

    swap_mask = loss_10 < loss_01   # (N, T)
    pred_out = pred.copy()
    pred_out[swap_mask] = pred_swap[swap_mask]
    return pred_out, gt
