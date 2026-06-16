"""Recovery / inverse problem evaluation.

How well can a trained extractor recover env state from model internal
states? All functions take pre-computed numpy arrays plus probes — no
model calls here.

fit_probes / eval_recovery_multi accept a list of ProbeSpec so that any
number of probe types (linear, MLP, future) can be evaluated uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from pim.extractors.matching import identity_mse
from pim.extractors.spec import ProbeSpec


@dataclass
class RecoveryMetrics:
    """Results from a recovery evaluation."""
    overall_mse: float
    per_component_mse: np.ndarray   # (output_dim,) — per state component
    mse_by_context: np.ndarray      # (T,) — MSE as a function of context length


def fit_probes(
    probes: list[ProbeSpec],
    internal_states: np.ndarray,    # (N, T, H)
    env_states_gt: np.ndarray,      # (N, T, *state_shape)
    *,
    mask: np.ndarray | None = None,
    loss_fn: Callable = identity_mse,
    device: str = "cpu",
) -> dict[str, float]:
    """Train each probe in place; return final training loss keyed by probe name."""
    return {
        p.name: p.probe.fit(
            internal_states, env_states_gt,
            mask=mask, loss_fn=loss_fn, device=device,
        )
        for p in probes
    }


def eval_recovery_multi(
    probes: list[ProbeSpec],
    internal_states: np.ndarray,    # (N, T, H)
    env_states_gt: np.ndarray,      # (N, T, *state_shape)
    *,
    mask: np.ndarray | None = None,
    use_hungarian: bool = False,
    device: str = "cpu",
) -> dict[str, RecoveryMetrics]:
    """Evaluate each fitted probe; return metrics keyed by probe name."""
    return {
        p.name: eval_recovery(
            env_states_gt, internal_states, p.probe,
            mask=mask, use_hungarian=use_hungarian, device=device,
        )
        for p in probes
    }


def eval_recovery(
    env_states_gt: np.ndarray,     # (N, T, *state_shape)
    internal_states: np.ndarray,   # (N, T, H)
    extractor: nn.Module,
    *,
    mask: np.ndarray | None = None,
    use_hungarian: bool = False,
    device: str = "cpu",
    batch_size: int = 512,
) -> RecoveryMetrics:
    """Evaluate how well extractor(internal_states) matches env_states_gt."""
    extractor = extractor.to(device).eval()
    N, T = internal_states.shape[:2]

    all_pred, all_gt = [], []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            h_b = torch.from_numpy(internal_states[i:i+batch_size]).float().to(device)
            pred_b = extractor(h_b).cpu().numpy()
            all_pred.append(pred_b)
            all_gt.append(env_states_gt[i:i+batch_size])

    pred_all = np.concatenate(all_pred, axis=0)
    gt_all = np.concatenate(all_gt, axis=0)

    if use_hungarian:
        pred_all, gt_all = _apply_hungarian(pred_all, gt_all)

    sq_err = (pred_all - gt_all) ** 2

    if mask is not None:
        m = mask
        for _ in range(sq_err.ndim - mask.ndim):
            m = m[..., None]
        sq_err = sq_err * m

    flat_sq = sq_err.reshape(N, T, -1)
    denom = mask.sum() if mask is not None else N * T
    per_component_mse = flat_sq.sum(axis=(0, 1)) / denom
    overall_mse = float(per_component_mse.mean())

    mse_by_context = flat_sq.mean(axis=(0, 2))
    if mask is not None:
        mask_per_t = mask.mean(axis=0)
        mse_by_context = np.where(mask_per_t > 0, mse_by_context / mask_per_t, 0.0)

    return RecoveryMetrics(
        overall_mse=overall_mse,
        per_component_mse=per_component_mse,
        mse_by_context=mse_by_context,
    )


def _apply_hungarian(
    pred: np.ndarray,
    gt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder pred objects to best match gt (2-object case)."""
    if pred.ndim < 4 or pred.shape[2] != 2:
        return pred, gt
    loss_01 = ((pred - gt) ** 2).mean(axis=(-2, -1))
    pred_swap = np.stack([pred[:, :, 1], pred[:, :, 0]], axis=2)
    loss_10 = ((pred_swap - gt) ** 2).mean(axis=(-2, -1))
    swap_mask = loss_10 < loss_01
    pred_out = pred.copy()
    pred_out[swap_mask] = pred_swap[swap_mask]
    return pred_out, gt
