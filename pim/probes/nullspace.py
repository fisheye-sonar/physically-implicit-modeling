"""The iterative nullspace-projection probe — a CASCADE of orthogonal linear probes.

Ported 2026-08-31 from ``iterative_probing/iterative_probing.ipynb``. Non-default: the
workhorse decodability numbers come from ``pim.probes.linear`` / ``pim.probes.mlp``;
this probe exists to measure how LARGE the linearly-readable code is, and to power the
multi-probe editor (``pim.editors.nullspace``) that writes to all of it at once —
the direct answer to "discworld fails editability because one probe's row space is only
d_out of d_model dimensions".

The construction: fit a min-norm least-squares probe, remove its row space from every
state (deflation), refit on what remains, repeat until held-out R² falls below a stop
threshold. ``np.linalg.lstsq`` returns the MINIMUM-NORM solution, which is what makes
the accumulated row spaces orthogonal: each probe's rows land inside the remaining
(already-deflated) subspace. Orthogonality is asserted at every step, not assumed.

⛔ float64 throughout: 40 successive projections in float32 accumulate enough error to
break the orthogonality this whole construction rests on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def fit_lstsq_probe(X: np.ndarray, Y: np.ndarray, tr, te) -> dict:
    """Min-norm least-squares probe on X[tr] → Y[tr], scored on X[te].

    Returns ``A`` (d_out, D), ``b``, and the held-out fit — R² against the TRAIN mean,
    the same convention as every probe in the repo.
    """
    Aug = np.concatenate([X[tr], np.ones((X[tr].shape[0], 1))], 1)
    sol, *_ = np.linalg.lstsq(Aug, Y[tr], rcond=None)
    W, b = sol[:-1], sol[-1]  # W: (D, d_out)
    pred = X[te] @ W + b
    ss_res = ((pred - Y[te]) ** 2).sum()
    ss_tot = ((Y[te] - Y[tr].mean(0)) ** 2).sum()
    return dict(A=W.T, b=b,
                r2=float(1 - ss_res / ss_tot),
                rmse=float(np.sqrt(((pred - Y[te]) ** 2).sum(1).mean())))


def rowspace_basis(A: np.ndarray, tol: float = 1e-8) -> tuple[np.ndarray, int]:
    """Orthonormal basis of row(A) as columns, plus its numerical rank."""
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    rank = int((s > s[0] * tol).sum()) if s.size and s[0] > 0 else 0
    return vt[:rank].T, rank  # (D, rank)


def deflate(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Remove span(B) from every row of X."""
    return X - (X @ B) @ B.T


def random_basis(remaining_B: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k random orthonormal directions from INSIDE the current remaining subspace.

    The matched control: removes the same number of dimensions from the same space, so
    the only difference is whether those dimensions were chosen to carry the target.
    """
    G = rng.standard_normal((remaining_B.shape[1], k))
    Q, _ = np.linalg.qr(G)
    return remaining_B @ Q


@dataclass
class NullspaceCascade:
    """The fitted cascade: K orthogonal probes exhausting the linear code for Y.

    ``probes[k]`` holds ``A`` (d_out, D), ``b``, ``r2`` (held-out, at fit time — i.e.
    after k earlier row spaces were removed), and ``B`` (D, rank_k), the orthonormal
    basis removed after fitting probe k.
    """

    probes: list[dict] = field(default_factory=list)

    @property
    def n_probes(self) -> int:
        return len(self.probes)

    @property
    def total_rank(self) -> int:
        return sum(p["B"].shape[1] for p in self.probes)

    def subspace(self) -> np.ndarray:
        """(D, total_rank) orthonormal basis of the full accumulated code subspace."""
        return np.concatenate([p["B"] for p in self.probes], 1)

    def read(self, k: int, h: np.ndarray) -> np.ndarray:
        """Probe k's readout applied to the ORIGINAL state.

        Valid without deflating ``h`` first because ``A_k`` is orthogonal to every
        earlier removed basis by construction.
        """
        p = self.probes[k]
        return h @ p["A"].T + p["b"]


def fit_nullspace_cascade(H: np.ndarray, Y: np.ndarray, tr, te, *,
                          max_iter: int = 50, r2_stop: float = 0.02,
                          log=print) -> NullspaceCascade:
    """Run the deflation until the target is linearly exhausted.

    H : (N, D) states — cast to float64 internally (see module ⛔).
    Y : (N, d_out) targets.
    tr, te : row slices/index arrays, split BY SEQUENCE by the caller (the same leak
             rule as every probe fit in the repo).
    """
    X = H.astype(np.float64).copy()
    Y = Y.astype(np.float64)
    casc = NullspaceCascade()
    removed: list[np.ndarray] = []
    for k in range(1, max_iter + 1):
        p = fit_lstsq_probe(X, Y, tr, te)
        B, rank = rowspace_basis(p["A"])
        if rank == 0:
            if log:
                log(f"iteration {k}: probe is rank 0 — nothing left to remove; stopping.")
            break
        # ASSERT the two properties the whole construction depends on.
        if removed:
            prev = np.concatenate(removed, 1)
            leak = float(np.abs(prev.T @ B).max())
            assert leak < 1e-6, (
                f"iteration {k}: new basis not orthogonal to removed subspace "
                f"(max |inner| {leak:.2e})")
        orth = float(np.abs(B.T @ B - np.eye(rank)).max())
        assert orth < 1e-9, f"iteration {k}: basis not orthonormal ({orth:.2e})"

        casc.probes.append(dict(A=p["A"], b=p["b"], r2=p["r2"], rmse=p["rmse"], B=B))
        removed.append(B)
        X = deflate(X, B)
        if p["r2"] < r2_stop:
            if log:
                log(f"iteration {k}: held-out R² {p['r2']:.4f} < {r2_stop} — "
                    f"target exhausted; stopping.")
            break
    return casc
