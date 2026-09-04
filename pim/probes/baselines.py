"""Decodability baselines — the two floors every Probe Skill number is read against.

A probe skill of 0.98 means nothing on its own. Two controls say what it is measuring,
and they rule out different things:

**Observation baseline** — the SAME probe families fitted to the raw observation history
instead of the model's residual stream. Answers *how much of the state does a shallow
read of the input already give you?* Without it, a high skill might only mean the state
is sitting in the input in probe-readable form.

**Random-init baseline** — the same architecture, seeded, never trained, probed
identically. Answers *how much comes from training rather than from random features of
the right shape?* (This is the control that was once accidentally served the trained
model's probes — see ``pim.probes.cache``.)

Everything is matched to the model probes except the one thing under test: same probe
families, same targets, same seeded 80/20 split BY SEQUENCE, same Probe Skill, same
stats dict — so a baseline row and a model row are the same measurement on different
features.

⚠ **Why this file has its own fit loop.** The causal-history feature is
``T × R`` wide (discworld 40 × 128 = 5,120; Othello 60 × 61 = 3,660), so the dense train
matrix ``pim.probes.base.fit_probe`` expects would be 960k × 5,120 × 4 B ≈ **19.7 GB**.
The features are therefore built per MINIBATCH on the device and never materialised.
The probe object, the standardisation, the R²/stats and the optimiser settings are all
still the shared ones, and ``tests/test_baselines.py`` gates this fit against
``fit_probe`` on a small dense problem so the two can never drift.
"""

from __future__ import annotations

import numpy as np
import torch

from pim.probes.base import WorldStateProbe, _r2

# The canonical fit hyper-parameters, imported in spirit from pim.probes.base.fit_probe.
# They are repeated (not re-derived) because the loop below must own its batching.
EPOCHS, LR, BATCH = 200, 1e-3, 4096


class CausalHistory:
    """The causal, zero-left-padded observation history, built on demand.

    Row ``(s, t)`` is the whole history ``obs[s, 0..t]`` laid end to end and zero-filled
    for frames after ``t`` — the same information the model has consumed when its
    residual stream is probed at frame ``t``. Nothing beyond ``t`` is ever visible.

    ``dense``   discworld: ``src`` is (N, T, R) float observations.
    ``one_hot`` Othello:   ``src`` is (N, T) int token ids, expanded to R = vocab.
    """

    def __init__(self, src: torch.Tensor, kind: str = "dense", vocab: int | None = None):
        self.src, self.kind = src, kind
        self.device = src.device
        self.n, self.T = src.shape[0], src.shape[1]
        self.R = int(src.shape[2]) if kind == "dense" else int(vocab)
        self.dim = self.T * self.R
        self._ar = torch.arange(self.T, device=src.device)

    def build(self, seq: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        """(B, T*R) features for the given (sequence, frame) rows."""
        if self.kind == "dense":
            x = self.src[seq]                                   # (B, T, R)
        else:
            x = torch.zeros(len(seq), self.T, self.R, device=self.src.device)
            x.scatter_(2, self.src[seq].unsqueeze(-1).long(), 1.0)
        # ⛔ the causal mask: everything strictly after the probed frame is zeroed
        x = x * (self._ar[None, :] <= frame[:, None]).unsqueeze(-1).to(x.dtype)
        return x.reshape(len(seq), self.dim)


class MemmapRows:
    """Rows of a residual stack on disk — ``(N, T, d)`` float32, as written by
    ``collect_residuals(memmap=…, points=[ℓ])`` — served per minibatch.

    The probe-capacity sweep fits probes on ~10M rows, where the dense ``fit_probe``
    path (``X[tr]`` copies, an augmented lstsq matrix, a GPU-resident tensor) cannot fit
    beside a 20 GB stack under the memory cap. This source keeps the stack on the nvme
    and the fit streams it; the fit loop batches by SEQUENCE so every read is a
    contiguous block of frames rather than a random row.
    """

    kind = "memmap"

    def __init__(self, arr, device="cuda"):
        self.mm = arr
        self.n, self.T, self.dim = arr.shape
        self.device = torch.device(device)

    def build(self, seq: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        s, f = seq.cpu().numpy(), frame.cpu().numpy()
        return torch.from_numpy(np.ascontiguousarray(self.mm[s, f])).to(self.device)


def _row_index(seq: torch.Tensor, T: int, device, mask=None):
    """The (sequence, frame) rows to fit on, as two flat index tensors.

    ``mask`` (N, T) drops invalid rows — Othello games run 9-60 moves, and fitting a
    probe on the zero-padding past a game's end would report an easy constant as skill.
    Discworld passes None: every frame of every episode is real.
    """
    s = seq.repeat_interleave(T)
    f = torch.arange(T, device=device).repeat(len(seq))
    if mask is not None:
        keep = mask[s, f]
        s, f = s[keep], f[keep]
    return s, f


def _moments(hist, s, f, chunk: int = 4096):
    """Streamed train-set mean/std of the features (never materialises the matrix)."""
    tot = torch.zeros(hist.dim, dtype=torch.float64, device=hist.device)
    sq = torch.zeros_like(tot)
    for i in range(0, len(s), chunk):
        x = hist.build(s[i : i + chunk], f[i : i + chunk]).double()
        tot += x.sum(0)
        sq += (x * x).sum(0)
    n = float(len(s))
    mean = tot / n
    var = (sq / n - mean * mean).clamp_min(0)
    return mean.float(), var.sqrt().float().clamp_min(1e-6)


@torch.no_grad()
def _predict(probe, hist, s, f, chunk: int = 8192, classify: bool = False):
    out = []
    for i in range(0, len(s), chunk):
        p = probe(hist.build(s[i : i + chunk], f[i : i + chunk]))
        out.append((p.argmax(-1) if classify else p).cpu().numpy())
    return np.concatenate(out, 0)


def fit_probe_stream(hist, y: torch.Tensor, tr_seq, te_seq, *,
                     hidden: int | None, n_classes: int | None = None,
                     row_mask: torch.Tensor | None = None, seed: int = 0,
                     epochs: int = EPOCHS, batch: int = BATCH, log=None):
    """Fit one probe on a STREAMED row source — ``CausalHistory`` (the observation
    floor) or ``MemmapRows`` (a residual stack too large to hold densely). Mirrors
    ``base.fit_probe``: same probe object, standardisation, loss, optimiser and stats;
    only the delivery of rows differs, and the SGD minibatches are drawn by SEQUENCE
    block (all frames of ~batch/T sequences) so a disk-backed source reads contiguously.

    ``y`` is (N, T, d_out) — targets aligned frame-for-frame with the rows.
    Returns ``(probe, stats)`` with the same keys the model probes report, including
    the in-sample counterpart of every score (the overfit check).
    """
    torch.manual_seed(seed)
    dev = hist.device
    tr_seq = torch.as_tensor(tr_seq, device=dev).sort().values   # contiguous reads
    te_seq = torch.as_tensor(te_seq, device=dev).sort().values
    d_out = y.shape[-1]

    s_tr, f_tr = _row_index(tr_seq, hist.T, dev, row_mask)
    s_te, f_te = _row_index(te_seq, hist.T, dev, row_mask)
    n = len(s_tr)

    xm, xs = _moments(hist, s_tr, f_tr)
    if n_classes is None:
        y_tr = y[s_tr, f_tr].float()
        ym, ys = y_tr.mean(0), y_tr.std(0).clamp_min(1e-6)
    else:                                    # classification: the y affine is meaningless
        ym, ys = torch.zeros(d_out, device=dev), torch.ones(d_out, device=dev)

    probe = WorldStateProbe(hist.dim, d_out, hidden, x_mean=xm, x_std=xs,
                            y_mean=ym, y_std=ys, n_classes=n_classes).to(dev)

    if hidden is None and n_classes is None:
        # Closed-form least squares, streamed as NORMAL EQUATIONS. Standardising with the
        # train mean makes z zero-mean, so the intercept solves to exactly 0 and the
        # system reduces to (ZᵀZ) W = Ztw — the same minimum-norm solution np.linalg.lstsq
        # returns densely (gated in tests/test_baselines.py).
        ZtZ = torch.zeros(hist.dim, hist.dim, dtype=torch.float64, device=dev)
        Ztw = torch.zeros(hist.dim, d_out, dtype=torch.float64, device=dev)
        for i in range(0, n, 8192):
            sl = slice(i, i + 8192)
            z = (hist.build(s_tr[sl], f_tr[sl]) - xm) / xs
            w = (y[s_tr[sl], f_tr[sl]].float() - ym) / ys
            ZtZ += (z.T @ z).double()        # matmul in fp32, accumulate in fp64
            Ztw += (z.T @ w).double()
        W = torch.linalg.lstsq(ZtZ, Ztw).solution.float()
        with torch.no_grad():
            probe.net.weight.copy_(W.T)
            probe.net.bias.zero_()
    else:
        opt = torch.optim.Adam(probe.parameters(), lr=LR)
        ys_t = probe.y_std.detach()
        bseq = max(1, batch // hist.T)                 # sequences per minibatch
        for ep in range(epochs):
            perm = tr_seq[torch.randperm(len(tr_seq), device=dev)]
            for i in range(0, len(perm), bseq):
                s_b, f_b = _row_index(perm[i : i + bseq], hist.T, dev, row_mask)
                if len(s_b) == 0:
                    continue
                x = hist.build(s_b, f_b)
                tgt = y[s_b, f_b]
                if n_classes is None:
                    # loss in STANDARDISED target space, so every output dim counts
                    # equally (base.fit_probe documents why at length)
                    loss = (((probe(x) - tgt.float()) / ys_t) ** 2).mean()
                else:
                    loss = torch.nn.functional.cross_entropy(
                        probe(x).reshape(-1, n_classes), tgt.reshape(-1).long())
                opt.zero_grad()
                loss.backward()
                opt.step()
            if log and (ep + 1) % 50 == 0:
                log(f"      epoch {ep + 1}/{epochs} loss {float(loss.detach()):.5f}")
    probe.eval()

    classify = n_classes is not None
    hat_te = _predict(probe, hist, s_te, f_te, classify=classify)
    hat_tr = _predict(probe, hist, s_tr, f_tr, classify=classify)
    g_te = y[s_te, f_te].cpu().numpy()
    g_tr = y[s_tr, f_tr].cpu().numpy()
    if classify:
        g_te, g_tr = g_te.astype(int), g_tr.astype(int)
        stats = {
            "error_rate": float((hat_te != g_te).mean() * 100.0),
            "error_rate_insample": float((hat_tr != g_tr).mean() * 100.0),
            "accuracy": float((hat_te == g_te).mean() * 100.0),
            "per_tile_error_rate": ((hat_te != g_te).mean(0) * 100.0).tolist(),
            "majority_class_error_rate": float(
                (1.0 - np.bincount(g_tr.reshape(-1), minlength=n_classes).max()
                 / g_tr.size) * 100.0),
        }
    else:
        ymn = ym.cpu().numpy()
        stats = {
            "r2": _r2(hat_te, g_te, ymn),
            "r2_insample": _r2(hat_tr, g_tr, ymn),
            "rmse": float(np.sqrt(((hat_te - g_te) ** 2).mean())),
            "per_dim_r2": [_r2(hat_te[:, [j]], g_te[:, [j]], ymn[[j]])
                           for j in range(d_out)],
        }
    stats.update({"kind": probe.kind, "n_train_rows": int(n),
                  "n_test_rows": int(len(s_te)), "d_in": hist.dim})
    return probe, stats


def random_init_model(arch: str, model_config: dict, seed: int = 0, device: str = "cpu"):
    """An UNTRAINED model of the canonical architecture — the random-init floor.

    Seeded, so its parameter fingerprint (and therefore every probe cache key derived
    from it) is reproducible: the same call always yields the same control. It is then
    probed through the ordinary ``fit_probes`` path — a random-init baseline needs a
    different MODEL, never a different measurement.
    """
    from pim.models.registry import build

    torch.manual_seed(seed)
    return build(arch, model_config).to(device).eval()


fit_baseline_probe = fit_probe_stream   # the observation floor's name for the same fit
