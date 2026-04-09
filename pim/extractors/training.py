"""Training routines for extractors.

Both functions take pre-computed arrays rather than a model + dataloader.
The caller runs inference first (e.g. via pim.eval._helpers.run_teacher_forcing)
and passes the resulting arrays here. This keeps training logic decoupled
from model architecture.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .linear import LinearExtractor
from .matching import identity_mse


def train_extractor(
    extractor: nn.Module,
    internal_states: np.ndarray,  # (N, T, H)
    env_states_gt: np.ndarray,  # (N, T, *state_shape)
    *,
    n_epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 512,
    loss_fn: Callable = identity_mse,
    mask: np.ndarray | None = None,  # (N, T) bool — True = include
    device: str = "cpu",
) -> list[float]:
    """Train an extractor via gradient descent on pre-computed hidden states.

    Parameters
    ----------
    extractor       : nn.Module to train (LinearExtractor or MLPExtractor)
    internal_states : (N, T, H) hidden states from run_teacher_forcing()
    env_states_gt   : (N, T, *state_shape) ground-truth env state
    n_epochs        : training epochs
    lr              : learning rate for Adam
    batch_size      : samples per gradient step
    loss_fn         : loss function, e.g. identity_mse or hungarian_mse
    mask            : (N, T) bool mask; if provided, only masked timesteps
                      contribute to the loss
    device          : torch device string

    Returns
    -------
    per_epoch_losses : list of float, length n_epochs
    """
    extractor = extractor.to(device).train()
    opt = Adam(extractor.parameters(), lr=lr)

    h_t = torch.from_numpy(internal_states.astype(np.float32))  # (N, T, H)
    gt_t = torch.from_numpy(env_states_gt.astype(np.float32))  # (N, T, *S)
    if mask is not None:
        m_t = torch.from_numpy(mask.astype(bool))  # (N, T)
    else:
        m_t = None

    N = h_t.shape[0]
    losses = []

    for _ in range(n_epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            h_b = h_t[idx].to(device)  # (B, T, H)
            gt_b = gt_t[idx].to(device)  # (B, T, *S)

            pred_b = extractor(h_b)  # (B, T, *S)

            if m_t is not None:
                mb = m_t[idx].to(device)  # (B, T)
                # Expand mask to match state dims
                extra_dims = pred_b.dim() - mb.dim()
                for _ in range(extra_dims):
                    mb = mb.unsqueeze(-1)
                mb = mb.expand_as(pred_b)
                pred_b = pred_b[mb]
                gt_b = gt_b[mb]

            loss = loss_fn(pred_b, gt_b)
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        losses.append(epoch_loss / max(n_batches, 1))

    extractor.eval()
    return losses


def fit_lstsq(
    extractor: LinearExtractor,
    internal_states: np.ndarray,  # (N, T, H)
    env_states_gt: np.ndarray,  # (N, T, *state_shape)
    *,
    mask: np.ndarray | None = None,  # (N, T) bool
) -> float:
    """Fit a LinearExtractor via exact least-squares (closed-form).

    Collects all (hidden_state, env_state) pairs, solves the normal equations,
    and writes the solution back into extractor.linear.weight and .bias.

    Parameters
    ----------
    extractor       : LinearExtractor to fit (modified in-place)
    internal_states : (N, T, H) hidden states
    env_states_gt   : (N, T, *state_shape) ground-truth env state
    mask            : (N, T) bool mask; if provided, only masked timesteps used

    Returns
    -------
    train_mse : float — MSE of the fitted solution on the training data
    """
    N, T, H = internal_states.shape
    output_dim = extractor.state_def.output_dim

    h_flat = internal_states.reshape(N * T, H)  # (N*T, H)
    gt_flat = env_states_gt.reshape(N * T, output_dim)  # (N*T, D)

    if mask is not None:
        keep = mask.reshape(N * T).astype(bool)
        h_flat = h_flat[keep]
        gt_flat = gt_flat[keep]

    # Augment with bias column: [H | 1]
    ones = np.ones((h_flat.shape[0], 1), dtype=np.float32)
    A = np.concatenate([h_flat, ones], axis=1)  # (M, H+1)

    # Solve: A @ [W; b]^T ≈ gt_flat   →   least-squares solution
    sol, _, _, _ = np.linalg.lstsq(A, gt_flat, rcond=None)  # (H+1, D)
    W = sol[:-1].T.astype(np.float32)  # (D, H)
    b = sol[-1].astype(np.float32)  # (D,)

    with torch.no_grad():
        extractor.linear.weight.copy_(torch.from_numpy(W))
        extractor.linear.bias.copy_(torch.from_numpy(b))

    # Compute train MSE
    pred = A @ sol  # (M, D)
    mse = float(np.mean((pred - gt_flat) ** 2))
    return mse
