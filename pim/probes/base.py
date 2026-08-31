"""The probe body shared by the two canonical probes, plus the residual harvest.

Moved verbatim 2026-08-31 from ``othello_gpt/othello_probe.py`` (the probing half; the
intervention half moved to ``pim.editors.grad_steer``). The public faces are
``pim.probes.linear`` and ``pim.probes.mlp`` — import those; this module holds the one
implementation both delegate to, because Li et al.'s §3.1 linear probe and §3.2 MLP
probe deliberately differ ONLY in the middle map (``net``), and duplicating the
standardisation/scoring scaffolding per probe is how numbers drift apart.

What is copied exactly from Li et al. (arXiv:2210.13382), and what necessarily differs
for a continuous world (regression instead of 3-way tile classification; held-out split
by SEQUENCE, never by frame; standardisation inside the probe), is documented at the
original length in the class and function docstrings below — none of that changed in
the move.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# ── Probes ────────────────────────────────────────────────────────────────────


class WorldStateProbe(nn.Module):
    """Paper §3 probe: linear, or a 2-layer MLP with ONE hidden layer.

    Two output modes, same body:

    * ``n_classes=None`` (default) — **regression**. Maps a **raw** residual-stream
      activation to world-state values in **raw sim units**; ``d_out`` is the number of
      scalars. This is what runs on this repo's own world model.
    * ``n_classes=C`` — **C-way classification over ``d_out`` tiles**, added 2026-08-20
      for `othello_transfer/`, where the state is 64 ternary tiles rather than continuous
      scalars. ``forward`` then returns ``(B, d_out, C)`` logits and the ``y`` affine is
      skipped (it is meaningless on logits). The body is otherwise **identical**, and at
      ``C=3, d_out=64`` it is exactly the shape of Li et al.'s own
      ``BatteryProbeClassificationTwoLayer``.

    Standardisation moments are non-trainable buffers either way, so the module is a
    pure function of the raw activation and ``∂L/∂x`` is taken in the space the
    paper's update rule assumes.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden: int | None = 512,
        *,
        x_mean=None,
        x_std=None,
        y_mean=None,
        y_std=None,
        n_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.d_in, self.d_out, self.hidden = d_in, d_out, hidden
        self.n_classes = n_classes
        n_final = d_out if n_classes is None else d_out * n_classes
        if hidden is None:  # paper §3.1 — linear probe
            self.net = nn.Linear(d_in, n_final)
        else:  # paper §3.2 — softmax(W1 ReLU(W2 x)), one hidden layer
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, n_final)
            )
        z = torch.zeros(d_in)
        o = torch.ones(d_in)
        self.register_buffer("x_mean", z.clone() if x_mean is None else x_mean)
        self.register_buffer("x_std", o.clone() if x_std is None else x_std)
        zo, oo = torch.zeros(d_out), torch.ones(d_out)
        self.register_buffer("y_mean", zo.clone() if y_mean is None else y_mean)
        self.register_buffer("y_std", oo.clone() if y_std is None else y_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, d_in) raw activation → (B, d_out) values, or (B, d_out, n_classes) logits."""
        z = (x - self.x_mean) / self.x_std
        out = self.net(z)
        if self.n_classes is not None:
            return out.reshape(*out.shape[:-1], self.d_out, self.n_classes)
        return out * self.y_std + self.y_mean

    @property
    def kind(self) -> str:
        base = "linear" if self.hidden is None else f"mlp-{self.hidden}"
        return base if self.n_classes is None else f"{base}-{self.n_classes}way"

    @property
    def act_scale(self) -> float:
        """Typical per-dim spread of the activations this probe was fit on.

        The residual points differ in scale by ~17x (median per-dim std 0.16 at point 0
        to 2.76 at point 4), so a single absolute step size cannot converge at every
        depth. Intervention step sizes are quoted **relative** to this, which makes one
        `alpha` mean the same thing at every residual point.
        """
        return float(self.x_std.median())


# ── Residual collection ───────────────────────────────────────────────────────


@torch.no_grad()
def collect_residuals(model, obs: np.ndarray, batch: int = 128) -> np.ndarray:
    """(n_points, N, T, d_model) residual stream at every residual point/position.

    Uses the model's one-pass banded forward, which *is* the sliding window (the band
    mask is the window), so this is exactly the activation a step-by-step rollout
    would produce at each position — verified by
    ``tests/test_transformer.py::test_buffer_rollout_matches_full_sequence``.
    """
    dev = next(model.parameters()).device
    out: list[np.ndarray] = []
    for i in range(0, len(obs), batch):
        o = torch.from_numpy(obs[i : i + batch]).float().to(dev)
        tokens = model.embed(o)
        _, resids = model._run(
            tokens, model._seq_mask(o.shape[1], dev), want_resid=True
        )
        out.append(torch.stack(resids, 0).float().cpu().numpy())
    return np.concatenate(out, axis=1)


# ── Fitting ───────────────────────────────────────────────────────────────────


def _r2(pred: np.ndarray, gt: np.ndarray, train_mean: np.ndarray) -> float:
    """R² against the TRAIN mean (repo convention), pooled over output dims."""
    ss_res = ((pred - gt) ** 2).sum()
    ss_tot = ((gt - train_mean) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def fit_probe(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    hidden: int | None = 512,
    epochs: int = 200,
    lr: float = 1e-3,
    batch: int = 4096,
    device: str = "cuda",
    seed: int = 0,
    n_classes: int | None = None,
) -> tuple[WorldStateProbe, dict]:
    """Fit one probe. `hidden=None` gives the paper's linear probe.

    ``n_classes=None`` (default) is the regression fit used on this repo's own world
    model: squared error in standardised target space, and the linear probe solved in
    closed form.

    ``n_classes=C`` (added 2026-08-20) is the **classification** fit for a state made of
    ``d_out`` C-way tiles, which is what Li et al. actually train. ``y`` then holds
    integer class labels ``(N, d_out)``, the loss is per-tile cross-entropy, and quality
    is reported as an **error rate (%)** so it sits beside their Tables 1 and 2. The
    linear probe is trained by the same SGD loop rather than ``lstsq``, which has no
    multinomial analogue — so unlike the regression path, the linear-vs-MLP gap here
    carries an optimiser difference of zero rather than an optimiser *advantage* to the
    linear side.
    """
    torch.manual_seed(seed)
    xm, xs = x_tr.mean(0), x_tr.std(0)
    # Floor the per-dim scale. Residual point 0 spans a 600x range of per-dim std
    # (2.6e-4 against a 0.16 median): dividing by the smallest makes the standardised
    # objective so anisotropic that activation-space gradient descent diverges at any
    # step size large enough to move the read-out. The floor costs nothing in fit
    # quality (the probe compensates with larger weights) and makes the descent
    # well-conditioned, which is what the paper's update rule assumes.
    xs = np.maximum(xs, 1e-2 * np.median(xs)) + 1e-8
    if n_classes is None:
        ym, ys = y_tr.mean(0), y_tr.std(0) + 1e-6
    else:  # logits carry no target affine
        ym = np.zeros(y_tr.shape[1], np.float32)
        ys = np.ones(y_tr.shape[1], np.float32)

    probe = WorldStateProbe(
        x_tr.shape[1],
        y_tr.shape[1],
        hidden,
        x_mean=torch.tensor(xm, dtype=torch.float32),
        x_std=torch.tensor(xs, dtype=torch.float32),
        y_mean=torch.tensor(ym, dtype=torch.float32),
        y_std=torch.tensor(ys, dtype=torch.float32),
        n_classes=n_classes,
    ).to(device)

    if n_classes is not None:
        # Classification path: the paper's own objective, per-tile cross-entropy.
        xt = torch.tensor(x_tr, dtype=torch.float32, device=device)
        yt = torch.tensor(y_tr, dtype=torch.long, device=device)
        opt = torch.optim.Adam(probe.parameters(), lr=lr)
        n = len(xt)
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch):
                idx = perm[i : i + batch]
                logits = probe(xt[idx])  # (b, d_out, C)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, n_classes), yt[idx].reshape(-1)
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
        probe.eval()

        def _pred(x):
            out = []
            with torch.no_grad():
                for i in range(0, len(x), 8192):
                    xb = torch.tensor(x[i : i + 8192], dtype=torch.float32, device=device)
                    out.append(probe(xb).argmax(-1).cpu().numpy())
            return np.concatenate(out, 0)

        hat_te, hat_tr = _pred(x_te), _pred(x_tr)
        per_tile_err = (hat_te != y_te).mean(0)
        stats = {
            "error_rate": float((hat_te != y_te).mean() * 100.0),
            "error_rate_insample": float((hat_tr != y_tr).mean() * 100.0),
            "accuracy": float((hat_te == y_te).mean() * 100.0),
            "per_tile_error_rate": (per_tile_err * 100.0).tolist(),
            "majority_class_error_rate": float(
                (1.0 - np.bincount(y_tr.reshape(-1), minlength=n_classes).max()
                 / y_tr.size) * 100.0
            ),
            "n_train_rows": int(len(x_tr)),
            "n_test_rows": int(len(x_te)),
            "kind": probe.kind,
        }
        return probe, stats

    if hidden is None:
        # Closed-form least squares: strictly better than SGD for a linear map, so the
        # linear-vs-MLP gap is a statement about the probe family, not the optimiser.
        zx = (x_tr - xm) / xs
        zy = (y_tr - ym) / ys
        A = np.concatenate([zx, np.ones((len(zx), 1), dtype=np.float32)], 1)
        W, *_ = np.linalg.lstsq(A, zy, rcond=None)
        with torch.no_grad():
            probe.net.weight.copy_(torch.tensor(W[:-1].T, dtype=torch.float32))
            probe.net.bias.copy_(torch.tensor(W[-1], dtype=torch.float32))
    else:
        xt = torch.tensor(x_tr, dtype=torch.float32, device=device)
        yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
        opt = torch.optim.Adam(probe.parameters(), lr=lr)
        n = len(xt)
        # The loss is taken in STANDARDISED target space, so every output dimension
        # contributes equally.
        #
        # `forward` un-standardises (`net(z) * y_std + y_mean`), so a raw-units loss is
        # implicitly weighted by `y_std**2` per dimension — and on a combined
        # position+velocity target that is a ~1000x weighting toward position (variance
        # 3.0-3.6 vs 0.0033 in sim units). The velocity dimensions were then barely
        # trained: measured 2026-08-19, mean velocity R2 0.158 with the raw-units loss
        # vs 0.276 here, with the x-components roughly doubling (0.211 -> 0.518,
        # 0.450 -> 0.730) and position paying almost nothing (0.938 -> 0.927). The
        # balanced fit also matches a dedicated velocity-only probe (0.272), so the
        # imbalance was the whole gap.
        #
        # The tell was that the MLP scored BELOW the linear probe on velocity
        # (0.158 vs 0.200). A strictly more expressive probe losing to a linear one is a
        # training failure, never a fact about the representation - the same diagnostic
        # the repo standard records for its own 2026-08-11 undertraining fix.
        ys_t = probe.y_std.detach()
        for ep in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch):
                idx = perm[i : i + batch]
                loss = (((probe(xt[idx]) - yt[idx]) / ys_t) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    probe.eval()
    with torch.no_grad():
        pr_te = (
            probe(torch.tensor(x_te, dtype=torch.float32, device=device)).cpu().numpy()
        )
        pr_tr = (
            probe(torch.tensor(x_tr, dtype=torch.float32, device=device)).cpu().numpy()
        )
    stats = {
        "r2": _r2(pr_te, y_te, ym),
        "r2_insample": _r2(pr_tr, y_tr, ym),
        "rmse": float(np.sqrt(((pr_te - y_te) ** 2).mean())),
        "per_dim_r2": [
            _r2(pr_te[:, [j]], y_te[:, [j]], ym[[j]]) for j in range(y_te.shape[1])
        ],
        "kind": probe.kind,
    }
    return probe, stats


# ── Paper §4.1 intervention ───────────────────────────────────────────────────

