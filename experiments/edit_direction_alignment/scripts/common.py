"""Alignment between a probe's writable subspace and the TRUE edit direction Δ = h_cf − h,
where h_cf is the residual the model produces on an oracle counterfactual history.

Everything is measured in the probe's z-space (residual standardised by the probe's own
x_mean / x_std — the space PI and ND write in). Subspaces compared against Δ:
  rows     span of the probe weight rows W[rows] (the min-norm read-out directions)
  haufe    span of the Haufe et al. (2014) activation patterns A = Σ_z Wᵀ (Σ_ŷ)⁻¹ for the
           same rows — the FORWARD directions along which the read-outs actually covary
           in the data, which a min-norm backward probe systematically misses
  pca-k    the rows projected onto the top-k principal components of the residual
A "fraction" is ‖P Δ‖² / ‖Δ‖² for the orthogonal projector P onto the subspace; the
generic baseline is the same fraction for an unrelated displacement of the same kind.
"""
from __future__ import annotations

import numpy as np
import torch


def zspace(probe, h: torch.Tensor) -> torch.Tensor:
    return (h - probe.x_mean) / probe.x_std


def orth(M: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis (d, r) of the span of the rows of M (r, d)."""
    q, _ = torch.linalg.qr(M.T.double())
    return q.float()


def frac_in(delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """(B,) fraction of each row of `delta` (B, d) lying in span(basis) (d, r)."""
    proj = delta @ basis @ basis.T
    return (proj.norm(dim=1) ** 2) / (delta.norm(dim=1) ** 2).clamp_min(1e-12)


def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)


def haufe_patterns(W: torch.Tensor, cov_z: torch.Tensor) -> torch.Tensor:
    """(d_out, d) forward patterns for backward weights W (d_out, d): A = Σ_z Wᵀ Σ_ŷ⁻¹."""
    Wd = W.double(); C = cov_z.double()
    cov_y = Wd @ C @ Wd.T
    A = C @ Wd.T @ torch.linalg.pinv(cov_y, hermitian=True)
    return A.T.float()                                # (d_out, d)


def pca_basis(cov_z: torch.Tensor, k: int) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(cov_z.double())
    return evecs[:, -k:].float()                      # (d, k)


def subspace_fracs(delta_z, W_rows, cov_z, ks=(16, 64)) -> dict:
    """All fractions for one layer: rows / haufe / pca-k(rows) ; delta_z (B, d), W_rows (r, d)."""
    out = {}
    B_rows = orth(W_rows)
    out["rows"] = frac_in(delta_z, B_rows)
    A = haufe_patterns(W_rows, cov_z)
    out["haufe"] = frac_in(delta_z, orth(A))
    for k in ks:
        P = pca_basis(cov_z, k)                       # (d, k)
        Wp = (W_rows @ P) @ P.T                       # rows projected onto the PC subspace
        out[f"pca{k}"] = frac_in(delta_z, orth(Wp))
        out[f"pcaspace{k}"] = frac_in(delta_z, P)     # how much of Δ the top-k PCs hold at all
    return out


def summarise(d: dict) -> dict:
    return {k: (float(v.mean()), float(v.std()) if hasattr(v, "std") else 0.0) for k, v in d.items()}
