"""INVERSE MAP — g: environment state → residual at one point, the state-conditional mean of the
residual E[h | state] (2026-09-14, Sevan's instrument; canonical 2026-09-15).

The mirror of the canonical MLP-128 probe: the same ``fit_probe`` body, one hidden layer of
width ``CANONICAL_HIDDEN``, the probes' 200-epoch recipe, the same seeded 80/20 split BY
SEQUENCE — with the state as input and the residual as output. One g per residual point.
Cached like every probe (``pim.probes.cache``; kind ``inverse_map``).

Beside it, the probe-free RETRIEVAL form: the mean residual of the k training frames whose
state is nearest the query (standardised Euclidean on a continuous state, one-hot agreement —
Hamming — on a categorical one). It is not cached: it needs the training residuals, which
the fitter has in hand.

Measured 2026-09-14 (`findings/inverse-probe.md`): the overwrite h′ = g(s_post) edits every
discworld regression state (+0.61 … +0.87) and standard Othello (+0.81), not the non-flip
Othello variants; retrieval edits where states repeat (discworld), never on Othello.
"""

from __future__ import annotations

import numpy as np
import torch

from pim.probes.base import FIT_BATCH, FIT_EPOCHS, FIT_LR, WorldStateProbe, fit_probe
from pim.probes.mlp import CANONICAL_HIDDEN

INVERSE_HIDDEN = CANONICAL_HIDDEN     # the mirror: the canonical MLP probe's width
INVERSE_EPOCHS = FIT_EPOCHS
INVERSE_SEED = 0
RETRIEVAL_K = 10
R2_ROWS = 20_000                       # held-out rows used for RetrievalBank.r2 (see its docstring)


def fit_inverse_map(s_tr, h_tr, s_te, h_te, *, hidden: int = INVERSE_HIDDEN,
                    epochs: int = INVERSE_EPOCHS, seed: int = INVERSE_SEED,
                    device: str = "cuda") -> tuple[WorldStateProbe, dict]:
    """g: state (N, m) → residual (N, d), fitted with the canonical probe body (regression
    fit, standardised target space; ``forward`` returns residuals in RAW units). Returns the
    frozen map and its held-out stats (``r2`` = how much of the residual the state explains)."""
    with torch.enable_grad():            # callers write under no_grad; the fit must not inherit it
        g, st = fit_probe(s_tr, h_tr, s_te, h_te, hidden=hidden, epochs=epochs, lr=FIT_LR,
                          batch=FIT_BATCH, device=device, seed=seed, n_classes=None)
    g.eval()
    for p in g.parameters():
        p.requires_grad_(False)
    return g, st


class RetrievalBank:
    """The k-nearest-state mean residual over one point's TRAINING rows.

    ``metric="euclidean"``: distance in standardised state units (continuous states —
    discworld position + velocity). ``metric="onehot"``: agreement count between one-hot
    rows, i.e. Hamming distance over categorical tiles (Othello boards). The distances are
    computed by one matmul per chunk — never a (chunk, rows, m) broadcast.

    ⛔ The residual bank is FLOAT32 (2026-09-16). It was half — 0.9 M × 512 rows in ~1 GB —
    until an Othello run showed why that is unsafe: this project's token models carry outlier
    residual features up to ~1.1e5 at the deep points, above half's 65504, so ~1 % of rows
    held ±inf and every retrieval mean touching them was inf. Discworld peaks at ~3.5e3 and was
    never affected. The QUERY side stays half (one-hot boards / standardised states, both O(1)).
    """

    def __init__(self, states: torch.Tensor, resid: torch.Tensor, *,
                 metric: str = "euclidean", k: int = RETRIEVAL_K) -> None:
        if metric not in ("euclidean", "onehot"):
            raise ValueError(f"metric must be 'euclidean' or 'onehot', got {metric!r}")
        self.k, self.metric = int(k), metric
        if not torch.isfinite(resid).all():
            raise ValueError("retrieval bank got non-finite residuals")
        self.H = resid.float()
        if metric == "euclidean":
            self.mu = states.float().mean(0)
            self.sd = states.float().std(0).clamp_min(1e-6)
            self.A = (states.float() - self.mu) / self.sd
            self.a2 = (self.A * self.A).sum(1)
        else:
            self.A = states.half()

    @torch.no_grad()
    def mean(self, s_query: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """(n, d) the mean residual of the k nearest training states to each query."""
        out = torch.zeros(len(s_query), self.H.shape[1], device=self.H.device)
        for i in range(0, len(s_query), chunk):
            q = s_query[i:i + chunk]
            if self.metric == "euclidean":
                Q = (q.float() - self.mu) / self.sd
                d2 = self.a2[None, :] - 2 * Q @ self.A.T + (Q * Q).sum(1)[:, None]
                idx = d2.topk(self.k, dim=1, largest=False).indices
            else:
                idx = (q.half() @ self.A.T).topk(self.k, dim=1, largest=True).indices
            out[i:i + chunk] = self.H[idx].float().mean(1)
        return out

    @torch.no_grad()
    def r2(self, s_te: torch.Tensor, h_te: torch.Tensor, max_rows: int = R2_ROWS) -> float:
        """Held-out R² of the retrieval mean as a predictor of the residual — the same statistic
        as the inverse map's ``r2`` (``pim.metrics.decodability.r2`` against the TRAINING mean),
        so the two instruments are compared on one axis (2026-09-16).

        Every query scores the WHOLE bank, so the held-out set is subsampled to ``max_rows``
        (deterministically, generator seed 0) — Othello's is ~450k rows against a 1.8M-row bank,
        which is minutes per point for a number that is converged in the thousands.
        """
        from pim.metrics.decodability import r2

        if len(s_te) > max_rows:
            idx = torch.from_numpy(
                np.random.default_rng(0).choice(len(s_te), max_rows, replace=False)
            ).to(s_te.device)
            s_te, h_te = s_te[idx], h_te[idx]
        pred = self.mean(s_te).cpu().numpy()
        return float(r2(pred, h_te.float().cpu().numpy(), self.H.float().mean(0).cpu().numpy()))
