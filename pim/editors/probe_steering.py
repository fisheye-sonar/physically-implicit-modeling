"""Probe-based hidden state steering.

Uses the pseudoinverse of a trained LinearExtractor to decompose hidden states
into row-space (probe-controlled) and null-space (probe-invariant) components,
then injects a target env state while preserving the null-space component.

This is the method used in the Counterfactual Controllability evaluation.
It requires a LinearExtractor (not MLP) because pseudoinverse decomposition
is only defined for linear maps.

All functions are pure: they take tensors and return tensors without modifying
any module or model state.
"""

from __future__ import annotations

import torch

from pim.extractors.linear import LinearExtractor


def probe_decomposition(
    extractor: LinearExtractor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract (A, b, A_pinv) from a trained LinearExtractor.

    Returns
    -------
    A      : (output_dim, hidden_size) — weight matrix
    b      : (output_dim,) — bias vector
    A_pinv : (hidden_size, output_dim) — Moore-Penrose pseudoinverse of A

    The pseudoinverse satisfies A @ A_pinv ≈ I_{output_dim}.
    """
    A = extractor.linear.weight.detach()      # (D, H)
    b = extractor.linear.bias.detach()        # (D,)
    A_pinv = torch.linalg.pinv(A)            # (H, D)
    return A, b, A_pinv


def decompose_hidden(
    h: torch.Tensor,
    A: torch.Tensor,
    A_pinv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose h into row-space and null-space components of A.

    Parameters
    ----------
    h      : (..., hidden_size) hidden state(s) to decompose
    A      : (output_dim, hidden_size) probe weight matrix
    A_pinv : (hidden_size, output_dim) pseudoinverse

    Returns
    -------
    h_parallel : (..., hidden_size) — projection onto row(A), probe-controlled
    h_perp     : (..., hidden_size) — null-space component, probe-invariant

    By construction: h_parallel + h_perp == h.
    """
    # h_parallel = A_pinv @ (A @ h^T) for each h vector
    Ah = h @ A.T          # (..., output_dim)
    h_parallel = Ah @ A_pinv.T  # (..., hidden_size)
    h_perp = h - h_parallel
    return h_parallel, h_perp


def inject_state(
    h: torch.Tensor,
    target_state: torch.Tensor,
    A: torch.Tensor,
    A_pinv: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Return h' such that the probe reads target_state: A @ h' + b = target_flat.

    The null-space component of h is preserved, so only the probe-controlled
    part of the hidden state changes.

    Parameters
    ----------
    h            : (..., hidden_size) original hidden state
    target_state : (..., output_dim) desired probe output (flattened state)
    A            : (output_dim, hidden_size) probe weight matrix
    A_pinv       : (hidden_size, output_dim) pseudoinverse
    b            : (output_dim,) probe bias

    Returns
    -------
    h_edited : (..., hidden_size)
        Satisfies: A @ h_edited + b ≈ target_state  (exactly if A has full row rank)
        Preserves: h_edited - h_parallel == h_perp (null-space unchanged)
    """
    _, h_perp = decompose_hidden(h, A, A_pinv)
    # h_new_parallel = A_pinv @ (target_state - b)
    target_flat = target_state - b   # (..., output_dim)
    h_new_parallel = target_flat @ A_pinv.T  # (..., hidden_size)
    return h_new_parallel + h_perp
