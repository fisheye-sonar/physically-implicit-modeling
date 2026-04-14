"""Assignment / matching losses for extractor training.

When ground-truth object ordering is ambiguous (e.g. random reflectivities),
Hungarian matching finds the permutation that minimises MSE before computing
the loss.  When ordering is fixed (e.g. fixed_reflectivities=True), use
identity_mse which is cheaper and exact.
"""

from __future__ import annotations

import torch


def identity_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE with no permutation matching.

    Use when object ordering is fixed (e.g. fixed_reflectivities=True datasets).

    Parameters
    ----------
    pred   : (..., *state_shape) predictions
    target : (..., *state_shape) ground truth

    Returns
    -------
    scalar MSE loss
    """
    return ((pred - target) ** 2).mean()


def hungarian_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE with Hungarian matching over the leading object dimension.

    Handles permutation ambiguity by finding the optimal assignment for each
    sample independently. Currently implemented as an explicit enumeration
    of both permutations for the 2-object case; falls back to identity for
    1 object.

    Parameters
    ----------
    pred   : (B, n_objects, per_object_dim) or (B, T, n_objects, per_object_dim)
    target : same shape as pred

    Returns
    -------
    scalar MSE loss (mean over batch and matched pairs)

    Notes
    -----
    For n_objects > 2, upgrade to scipy.optimize.linear_sum_assignment.
    The current fast path only handles n_objects ≤ 2.
    """
    # Flatten to (..., n_objects, per_object_dim) for generality
    # Object dimension is second-to-last, feature dim is last
    n_obj = pred.shape[-2]

    if n_obj == 1:
        return identity_mse(pred, target)

    if n_obj == 2:
        # Enumerate both permutations and take the one with lower MSE per sample
        loss_01 = ((pred - target) ** 2).mean(dim=(-2, -1))          # original order
        pred_swap = torch.stack([pred[..., 1, :], pred[..., 0, :]], dim=-2)
        loss_10 = ((pred_swap - target) ** 2).mean(dim=(-2, -1))     # swapped order
        # Per-sample minimum
        loss = torch.minimum(loss_01, loss_10).mean()
        return loss

    # Fallback for n_obj > 2: identity (upgrade to scipy if needed)
    return identity_mse(pred, target)
