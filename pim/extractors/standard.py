"""The standard readability probes — one definition, used everywhere.

Why this module exists
----------------------
Across the editability notebooks the "MLP probe R²" was being produced two
different ways, and the numbers are not comparable:

* `MLPExtractor` with its defaults (one 128-unit hidden layer, 30 epochs) scored
  **in-sample**, in `00_master_editability` and the `controls/` notebooks;
* a hand-rolled two-hidden-layer 256-unit probe scored on a **held-out** split, in
  `iterative_probing` and `nonlinear_gru`.

Two axes were conflated: probe *capacity*, and whether the score is in-sample.
The second is simply an error — an in-sample R² is not a readability claim — and
the first makes a shallow probe under-report how much is nonlinearly decodable.

`fit_readability_probes` fixes both by fitting the linear and MLP probes on the
**same by-sequence split** and scoring both on the **same held-out sequences**.
Use it for every reported position/velocity R².

Not to be confused with the *steering* probe
--------------------------------------------
The **MLP Grad Steering** editor writes through a frozen `MLPExtractor` with the
original 1×128 defaults. That is deliberate and must not be changed: the editor's
published results are tied to that exact frozen probe. Report readability with
this module; steer with `MLPExtractor(...)` directly, and never quote one as the
other.
"""

from __future__ import annotations

import numpy as np
import torch

import warnings

from .base import StateDefinition
from .mlp import MLPExtractor

# The standard, so notebooks state one number rather than re-deciding it.
#
# STD_EPOCHS raised 30 → 300 on 2026-08-11: `train_extractor` batches over SEQUENCES
# (batch_size 512), so at a typical N of 500–1500 sequences, 30 "epochs" is only
# ~30–90 Adam steps — badly undertrained, which made the MLP under-read the linear
# probe everywhere (flagged by Sevan on the DiT probe grid, where the MLP went
# *negative* while linear read 0.70). Validated on GRU H256 h-states (N=500):
# MLP R² 0.17 @ 30 epochs → 0.89 @ 300 epochs, vs linear 0.81 — the MLP ≥ linear
# ordering a strictly-more-expressive probe must satisfy. Input z-scoring was also
# tested and is NOT needed once training is adequate (0.890 vs 0.891).
# ⚠ MLP R² values reported between 2026-08-06 and 2026-08-11 under-read and are
# not comparable to post-fix numbers; linear R² is unaffected (exact lstsq).
STD_HOLDOUT = 0.2
STD_MLP_HIDDEN = 256
STD_MLP_LAYERS = 2
STD_EPOCHS = 300
STD_LR = 1e-3


def _r2(pred: np.ndarray, y: np.ndarray, train_mean: np.ndarray) -> float:
    """R² against the TRAIN mean — the held-out baseline a probe must beat."""
    return float(1 - ((pred - y) ** 2).sum() / ((y - train_mean) ** 2).sum())


def fit_readability_probes(
    states: np.ndarray,  # (N, T, H)
    targets: np.ndarray,  # (N, T, D)
    *,
    mask: np.ndarray | None = None,  # (N, T) bool — True = include
    holdout: float = STD_HOLDOUT,
    mlp_hidden: int = STD_MLP_HIDDEN,
    n_hidden_layers: int = STD_MLP_LAYERS,
    n_epochs: int = STD_EPOCHS,
    lr: float = STD_LR,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Fit a linear and an MLP probe on one split; score both on the held-out part.

    The split is **by sequence**, never by row: consecutive frames of one sequence are
    near-duplicates, so a random row split leaks them across the boundary and inflates
    every R² reported from it.

    Returns
    -------
    dict with `linear_r2`, `mlp_r2`, `gap` (mlp − linear), `linear_rmse`, `mlp_rmse`,
    `linear_r2_insample` (an overfit check, never the headline), `n_train_seq`,
    `n_heldout_seq`, and the fitted `linear` (A, b) and `mlp` probe.
    """
    if states.ndim != 3 or targets.ndim != 3:
        raise ValueError(
            f"expected (N, T, ·) arrays, got {states.shape} and {targets.shape}"
        )
    # Mixed-scale targets are a trap here: `train_extractor` takes an unweighted MSE in
    # RAW target units, so a dimension's gradient share scales with its variance. Fitting
    # position and velocity in ONE probe would weight position ~1000x (variance 3.0-3.6 vs
    # 0.0033 in sim units) and leave velocity barely trained. The repo has always fit them
    # SEPARATELY, so no published number is affected — this guard exists so that stays true.
    _v = targets.reshape(-1, targets.shape[-1]).var(0)
    if _v.size > 1 and _v.max() > 100.0 * max(_v.min(), 1e-12):
        warnings.warn(
            f"fit_readability_probes: target dimensions span {_v.max() / max(_v.min(), 1e-12):.0f}x "
            "in variance. The MSE is in raw units, so low-variance dimensions will be "
            "under-trained. Fit them as separate probes (as eval_controls.py does), or "
            "standardise the targets first.",
            stacklevel=2,
        )

    n_seq = states.shape[0]
    n_tr = int(round((1 - holdout) * n_seq))
    if not 0 < n_tr < n_seq:
        raise ValueError(
            f"holdout={holdout} leaves {n_tr} of {n_seq} sequences for training"
        )

    def _flat(a, sl):
        out = a[sl].reshape(-1, a.shape[-1])
        if mask is None:
            return out
        return out[mask[sl].reshape(-1)]

    tr, te = slice(0, n_tr), slice(n_tr, None)
    Xtr, Ytr = _flat(states, tr).astype(np.float64), _flat(targets, tr).astype(
        np.float64
    )
    Xte, Yte = _flat(states, te).astype(np.float64), _flat(targets, te).astype(
        np.float64
    )
    mu = Ytr.mean(0)

    # ── linear probe: least squares on train, scored on held-out ──────────────
    aug = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    sol, *_ = np.linalg.lstsq(aug, Ytr, rcond=None)
    A, b = sol[:-1].T, sol[-1]
    pred_te = Xte @ sol[:-1] + sol[-1]

    # ── MLP probe: same split, same held-out scoring ──────────────────────────
    torch.manual_seed(seed)
    sdef = StateDefinition(
        name="readability", state_shape=(targets.shape[-1],), extract_fn=lambda x: x
    )
    mlp = MLPExtractor(
        states.shape[-1],
        sdef,
        mlp_hidden=mlp_hidden,
        n_hidden_layers=n_hidden_layers,
        n_epochs=n_epochs,
        lr=lr,
    )
    mlp.fit(
        states[tr], targets[tr], mask=None if mask is None else mask[tr], device=device
    )
    mlp = mlp.to(device).eval()
    with torch.no_grad():
        pm = mlp(torch.tensor(Xte, dtype=torch.float32, device=device)).cpu().numpy()

    lin_r2 = _r2(pred_te, Yte, mu)
    mlp_r2 = _r2(pm, Yte, mu)
    return dict(
        linear_r2=lin_r2,
        mlp_r2=mlp_r2,
        gap=mlp_r2 - lin_r2,
        linear_rmse=float(np.sqrt(((pred_te - Yte) ** 2).mean())),
        mlp_rmse=float(np.sqrt(((pm - Yte) ** 2).mean())),
        linear_r2_insample=_r2(Xtr @ sol[:-1] + sol[-1], Ytr, mu),
        n_train_seq=n_tr,
        n_heldout_seq=n_seq - n_tr,
        A=A.astype(np.float32),
        b=b.astype(np.float32),
        mlp=mlp,
        spec=f"linear=lstsq · MLP={n_hidden_layers}x{mlp_hidden} ReLU, {n_epochs} epochs, lr={lr} · "
        f"held-out {holdout:.0%} of sequences",
    )
