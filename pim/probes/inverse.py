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


# ── the CATEGORICAL inverse map (2026-09-20, Sevan) ────────────────────────────────────────
#
# On a categorical discworld block the state g inverts is the block's OWN categorical state —
# the labels its probes read, one-hot per tile — followed by the discs' continuous Cartesian
# velocity. The velocity is deliberate and asymmetric: the forward probes on a categorical
# target read labels only, but the residual also encodes how the discs move, and a map from
# labels alone returns the mean residual OVER velocities — writing it would erase the model's
# velocity estimate along with moving the disc. "Set where the discs are, leave how they move."
# Until this date every categorical block's IM arm was the CONTINUOUS full-state map of its
# basis scored on the categorical bench; those arms were removed from every scores.json.
#
# Recipe = the mirror of the forward probe for the SAME target: the categorical probes'
# large corpus and epochs (``arms.GRID_PROBE_RECIPE``), streamed — the residual stack stays on
# disk exactly as it does for ``fit_probe_stream``, only here it is the TARGET, not the input.

CATEGORICAL_STATE = "onehot-labels+cartesian-velocity"      # goes into the cache key and scores.json


def encode_categorical_state(labels: torch.Tensor, extra: torch.Tensor | None, n_classes: int) -> torch.Tensor:
    """(R, n_tiles) long labels [+ (R, m) floats] → (R, n_tiles · n_classes + m) float: one-hot per
    tile on the shared class axis, the continuous values appended RAW (the map standardises them)."""
    oh = torch.nn.functional.one_hot(labels.long(), n_classes).reshape(len(labels), -1).float()
    return oh if extra is None else torch.cat([oh, extra.float()], dim=1)


class CategoricalState:
    """Rows source for the categorical inverse map's INPUT — the same ``build(seq, frame)`` surface as
    ``baselines.MemmapRows``. Holds only the compact tensors on the GPU — ``labels`` (N, T, n_tiles)
    and ``extra`` (N, T, m) — and builds the one-hot per minibatch (1,028 wide on dw-128ray: 16 MB a
    batch, against 26 GB if the whole 200k-sequence design matrix were materialised)."""

    kind = "categorical_state"

    def __init__(self, labels, n_classes: int, extra=None, device="cuda") -> None:
        self.device = torch.device(device)
        self.labels = torch.as_tensor(labels).long().to(self.device)
        self.extra = None if extra is None else torch.as_tensor(extra).float().to(self.device)
        self.n, self.T, self.n_tiles = self.labels.shape
        self.n_classes = int(n_classes)
        self.m = 0 if self.extra is None else int(self.extra.shape[-1])
        self.dim = self.n_tiles * self.n_classes + self.m

    def build(self, seq: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        return encode_categorical_state(self.labels[seq, frame],
                                        None if self.extra is None else self.extra[seq, frame], self.n_classes)

    def moments(self, tr_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Input standardisation: the one-hot columns are left as 0/1 (mean 0, sd 1 — a rare class's
        sd is ~0.03, and dividing by it would hand that column a 30x gain); the continuous tail is
        standardised on the TRAIN sequences."""
        xm, xs = torch.zeros(self.dim, device=self.device), torch.ones(self.dim, device=self.device)
        if self.m:
            v = self.extra[tr_seq].reshape(-1, self.m)
            xm[-self.m:], xs[-self.m:] = v.mean(0), v.std(0).clamp_min(1e-6)
        return xm, xs


def fit_inverse_map_stream(state, H, tr_seq, te_seq, *, hidden: int = INVERSE_HIDDEN,
                           epochs: int = INVERSE_EPOCHS, batch: int = FIT_BATCH,
                           seed: int = INVERSE_SEED, log=None) -> tuple[WorldStateProbe, dict]:
    """The STREAMED mirror of ``fit_inverse_map``: ``state`` serves the input rows
    (``CategoricalState``), ``H`` the residual TARGET rows from disk (``baselines.MemmapRows``), both by
    ``build(seq, frame)``. Same probe body, optimiser, learning rate, batch size and standardised-target
    loss as the dense fit; as in ``fit_probe_stream`` the minibatches are drawn by SEQUENCE block so the
    disk reads are contiguous. Stats: ``r2`` / ``r2_insample`` (against the TRAIN mean, pooled over every
    residual dimension — ``pim.metrics.decodability.r2``'s definition, accumulated) and ``rmse``."""
    from pim.probes.baselines import _moments, _row_index

    torch.manual_seed(seed)
    dev = H.device
    tr_seq = torch.as_tensor(tr_seq, device=dev).sort().values
    te_seq = torch.as_tensor(te_seq, device=dev).sort().values
    s_tr, f_tr = _row_index(tr_seq, H.T, dev)
    s_te, f_te = _row_index(te_seq, H.T, dev)
    ym, ys = _moments(H, s_tr, f_tr)
    xm, xs = state.moments(tr_seq)
    g = WorldStateProbe(state.dim, H.dim, hidden, x_mean=xm, x_std=xs, y_mean=ym, y_std=ys,
                        n_classes=None).to(dev)
    with torch.enable_grad():            # callers write under no_grad; the fit must not inherit it
        opt = torch.optim.Adam(g.parameters(), lr=FIT_LR)
        ys_t = g.y_std.detach()
        bseq = max(1, batch // H.T)
        for ep in range(epochs):
            perm = tr_seq[torch.randperm(len(tr_seq), device=dev)]
            for i in range(0, len(perm), bseq):
                s_b, f_b = _row_index(perm[i:i + bseq], H.T, dev)
                loss = (((g(state.build(s_b, f_b)) - H.build(s_b, f_b)) / ys_t) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if log and (ep + 1) % 10 == 0:
                log(f"      epoch {ep + 1}/{epochs} loss {float(loss.detach()):.5f}")
    g.eval()
    for p in g.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def _sse(s, f, chunk=8192):
        sse = sst = 0.0
        n = 0
        for i in range(0, len(s), chunk):
            h = H.build(s[i:i + chunk], f[i:i + chunk]).double()
            pr = g(state.build(s[i:i + chunk], f[i:i + chunk])).double()
            sse += float(((pr - h) ** 2).sum())
            sst += float(((h - ym.double()) ** 2).sum())
            n += h.numel()
        return sse, sst, n

    sse_te, sst_te, n_te = _sse(s_te, f_te)
    sse_tr, sst_tr, _ = _sse(s_tr, f_tr)
    if sst_te <= 0:
        raise ValueError("trivial predictor has zero error — the residual is constant on this split")
    return g, {"r2": 1.0 - sse_te / sst_te, "r2_insample": 1.0 - sse_tr / sst_tr,
               "rmse": float(np.sqrt(sse_te / n_te)), "kind": "inverse_map_stream",
               "state": CATEGORICAL_STATE, "d_in": int(state.dim), "rows_train": int(len(s_tr))}


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
