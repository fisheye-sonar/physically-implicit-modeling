"""Manifold-constrained hidden state steering.

The pseudoinverse (`probe_steering.inject_state`) and gradient
(`gradient_steering.gradient_steer`) editors both find a *minimum-effort* edit
that makes the probe read a target — but neither cares whether the resulting
hidden state is one the model ever actually visits. A min-norm edit almost
always walks off the thin manifold of on-trajectory states, and the recurrent
map, never trained on off-manifold inputs, simply projects it away. That makes
a failed edit ambiguous: it could be a real structural fact (the decoded
direction is not a generative one) or a mundane off-manifold artifact.

This module supplies the missing piece for disambiguating the two: a cheap,
linear approximation of the state manifold (the PCA subspace of visited states)
and a projection onto it. The *edit* itself is supplied by the caller as an
`edit_fn` — so the same machinery wraps the linear pseudoinverse edit or the
MLP gradient edit interchangeably.

`manifold_steer` alternates edit ↦ project. For the linear editor (where
`inject_state` is exactly the orthogonal projection onto the affine readout
constraint {h : A·h + b = target}), repeating this is projection-onto-convex-
sets: it converges to the on-manifold state that best matches the target
readout — i.e. the most on-manifold edit possible.

All functions are pure: no model or probe weights are modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass
class StateSubspace:
    """A linear (affine) approximation of the hidden-state manifold.

    A PCA subspace fit to a bank of visited hidden states: states are assumed
    to live near `mean + span(basis)`.

    Attributes
    ----------
    mean  : (H,) mean of the fitted states.
    basis : (H, k) orthonormal columns — the top-k principal directions.
    explained_variance_ratio : (k,) fraction of variance per retained component.
    total_explained : scalar — fraction of total variance captured by the k components.
    """

    mean: torch.Tensor
    basis: torch.Tensor
    explained_variance_ratio: torch.Tensor
    total_explained: float

    @property
    def n_components(self) -> int:
        return self.basis.shape[1]

    @property
    def hidden_size(self) -> int:
        return self.basis.shape[0]


def _pca_subspace(
    X: torch.Tensor,
    *,
    n_components: int | None,
    var_threshold: float,
) -> StateSubspace:
    """Core PCA: eigendecomposition of the covariance of an (M, H) matrix."""
    mean = X.mean(dim=0)
    Xc = X - mean
    n = Xc.shape[0]
    cov = (Xc.T @ Xc) / max(n - 1, 1)  # (H, H)

    evals, evecs = torch.linalg.eigh(cov)  # ascending
    evals = evals.flip(0).clamp_min(0.0)  # descending
    evecs = evecs.flip(1)
    ratio = evals / evals.sum().clamp_min(1e-12)

    if n_components is None:
        cum = torch.cumsum(ratio, dim=0)
        thr = torch.tensor(var_threshold, device=cum.device)
        k = int(torch.searchsorted(cum, thr).item()) + 1
        k = min(k, evecs.shape[1])
    else:
        k = min(n_components, evecs.shape[1])

    basis = evecs[:, :k].contiguous()  # (H, k)
    return StateSubspace(
        mean=mean,
        basis=basis,
        explained_variance_ratio=ratio[:k],
        total_explained=float(ratio[:k].sum().item()),
    )


def fit_state_subspace(
    states: np.ndarray | torch.Tensor,
    *,
    n_components: int | None = None,
    var_threshold: float = 0.99,
    max_samples: int = 100_000,
    seed: int = 0,
) -> StateSubspace:
    """Fit a PCA subspace to a bank of visited hidden states.

    Parameters
    ----------
    states       : (..., H) any-shape array of hidden states; flattened to (M, H).
    n_components : if given, keep exactly this many components.
    var_threshold: if n_components is None, keep the smallest k explaining at
                   least this fraction of total variance.
    max_samples  : subsample at most this many state vectors before fitting
                   (bounds the M×H matmul; the covariance is H×H regardless).
    seed         : RNG seed for the subsample.

    Returns
    -------
    StateSubspace
    """
    X = torch.as_tensor(states, dtype=torch.float32).reshape(-1, np.shape(states)[-1])
    if X.shape[0] > max_samples:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(X.shape[0], generator=g)[:max_samples]
        X = X[idx]
    return _pca_subspace(X, n_components=n_components, var_threshold=var_threshold)


def project_to_subspace(h: torch.Tensor, subspace: StateSubspace) -> torch.Tensor:
    """Orthogonally project hidden state(s) onto the affine PCA subspace.

    Parameters
    ----------
    h        : (..., H) hidden state(s).
    subspace : fitted StateSubspace.

    Returns
    -------
    (..., H) the nearest point in `mean + span(basis)`.
    """
    centered = h - subspace.mean
    coeffs = centered @ subspace.basis  # (..., k)
    return subspace.mean + coeffs @ subspace.basis.T


def offmanifold_residual(h: torch.Tensor, subspace: StateSubspace) -> torch.Tensor:
    """Euclidean distance from h to its projection — how far off-manifold it is.

    Returns
    -------
    (...,) per-state residual norm ‖h - project(h)‖.
    """
    return torch.linalg.norm(h - project_to_subspace(h, subspace), dim=-1)


def manifold_steer(
    h: torch.Tensor,
    target: torch.Tensor,
    edit_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    subspace: StateSubspace,
    *,
    n_iters: int = 1,
) -> torch.Tensor:
    """Edit h toward `target`, constrained to stay on the state manifold.

    Alternates the caller-supplied edit with a projection onto the PCA
    subspace. With the linear pseudoinverse editor (where `edit_fn` is the
    orthogonal projection onto the readout constraint), `n_iters > 1` is
    projection-onto-convex-sets and converges to the on-manifold state that
    best matches the target readout.

    Parameters
    ----------
    h        : (..., H) starting hidden state(s).
    target   : (..., D) desired probe readout, passed through to `edit_fn`.
    edit_fn  : (h, target) -> h_edited. E.g.
               linear:   lambda h, t: inject_state(h, t, A, A_pinv, b)
               gradient: lambda h, t: gradient_steer(h, t, probe, ...)[0]
    subspace : fitted StateSubspace defining the manifold.
    n_iters  : number of edit↦project alternations (>=1).

    Returns
    -------
    (..., H) the manifold-constrained edited state.
    """
    out = h
    for _ in range(max(1, n_iters)):
        out = edit_fn(out, target)
        out = project_to_subspace(out, subspace)
    return out


def _subsample_bank(
    bank: np.ndarray | torch.Tensor,
    *,
    bank_size: int,
    seed: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Flatten a state bank to (M, H) and subsample to at most bank_size rows."""
    B = torch.as_tensor(bank, dtype=torch.float32).reshape(-1, np.shape(bank)[-1])
    if B.shape[0] > bank_size:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(B.shape[0], generator=g)[:bank_size]
        B = B[idx]
    return B.to(device)


def fit_local_subspace(
    bank: np.ndarray | torch.Tensor,
    query: torch.Tensor,
    *,
    k_neighbors: int = 512,
    n_components: int | None = None,
    var_threshold: float = 0.99,
    bank_size: int = 50_000,
    seed: int = 0,
) -> StateSubspace:
    """Fit a *local* tangent-PCA subspace around a single query state.

    Global PCA captures only a global linear hull; the reachable-state manifold
    is curved. Fitting PCA on the k nearest visited states to `query`
    approximates the manifold's tangent plane there, so projecting onto it
    respects curvature. The subspace `mean` is the local neighborhood mean.

    Parameters
    ----------
    bank        : (..., H) bank of visited hidden states; flattened to (M, H).
    query       : (H,) or (1, H) state to localize around.
    k_neighbors : number of nearest neighbors defining the local patch.
    n_components / var_threshold : passed to the PCA core (applied within the patch).
    bank_size   : subsample the bank to at most this many states before the kNN.
    seed        : RNG seed for the subsample.

    Returns
    -------
    StateSubspace on the same device as `query`.
    """
    q = torch.as_tensor(query, dtype=torch.float32).reshape(-1)
    B = _subsample_bank(bank, bank_size=bank_size, seed=seed, device=q.device)
    k = min(k_neighbors, B.shape[0])
    d = torch.cdist(q[None], B)[0]  # (M,)
    nn_idx = torch.topk(d, k, largest=False).indices
    return _pca_subspace(
        B[nn_idx], n_components=n_components, var_threshold=var_threshold
    )


def manifold_steer_local(
    h: torch.Tensor,
    target: torch.Tensor,
    edit_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    bank: np.ndarray | torch.Tensor,
    *,
    k_neighbors: int = 512,
    n_iters: int = 50,
    var_threshold: float = 0.99,
    bank_size: int = 50_000,
    seed: int = 0,
) -> torch.Tensor:
    """Per-state edit constrained to the *local* tangent manifold.

    For each row of `h`, fit a local tangent-PCA subspace around it
    (`fit_local_subspace`) and run the alternating edit↦project of
    `manifold_steer` against that local subspace. This is the curvature-aware
    counterpart of `manifold_steer` (which uses one global subspace).

    Parameters
    ----------
    h        : (..., H) starting hidden state(s).
    target   : (..., D) desired probe readout, broadcast to match `h` rows.
    edit_fn  : (h, target) -> h_edited, e.g. the linear pseudoinverse projection.
    bank     : (..., H) bank of visited states used to define local manifolds.
    k_neighbors / var_threshold / bank_size / seed : local-subspace fit params.
    n_iters  : edit↦project alternations per state.

    Returns
    -------
    (..., H) the locally-manifold-constrained edited states.
    """
    h2 = h.reshape(-1, h.shape[-1])
    t2 = target.reshape(h2.shape[0], -1)
    outs = []
    for i in range(h2.shape[0]):
        sub = fit_local_subspace(
            bank,
            h2[i],
            k_neighbors=k_neighbors,
            var_threshold=var_threshold,
            bank_size=bank_size,
            seed=seed,
        )
        outs.append(
            manifold_steer(h2[i : i + 1], t2[i : i + 1], edit_fn, sub, n_iters=n_iters)
        )
    return torch.cat(outs, dim=0).reshape(h.shape)
