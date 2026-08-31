"""The canonical LINEAR probe — Li et al. §3.1, one affine map.

Definition
----------
Regression (discworld):    ``y = (A z + b) * y_std + y_mean``, ``z = (h − x_mean)/x_std``
Classification (Othello):  ``softmax(W z)`` per tile — their exact probe shape.

The probe standardises both ends internally (the moments are non-trainable buffers), so
it is a pure function of the RAW activation — which is what lets editors take
``∂/∂h`` and pseudoinverses in the space the intervention actually writes.

Fit
---
Regression is solved in **closed form** (least squares on the standardised problem) —
strictly better than SGD for a linear map, so the linear-vs-MLP gap is a statement
about the probe *family*, never about the optimiser. Classification has no multinomial
lstsq analogue and trains by the same SGD loop as the MLP, so there the gap carries an
optimiser difference of zero rather than an advantage to the linear side.

Both this and ``pim.probes.mlp`` delegate to the single verified implementation in
``pim.probes.base`` — the two probes differ only in the middle map, and that is
enforced by construction rather than by discipline.
"""

from __future__ import annotations

import numpy as np

from pim.probes.base import WorldStateProbe, fit_probe


def fit_linear(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    n_classes: int | None = None,
    epochs: int = 200,
    lr: float = 1e-3,
    batch: int = 4096,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[WorldStateProbe, dict]:
    """Fit the canonical linear probe. See the module docstring for what runs.

    Returns ``(probe, stats)`` — stats carry ``r2`` / ``per_dim_r2`` (regression) or
    ``error_rate`` / ``per_tile_error_rate`` (classification), each with its in-sample
    counterpart for the overfit check.
    """
    return fit_probe(x_tr, y_tr, x_te, y_te, hidden=None, n_classes=n_classes,
                     epochs=epochs, lr=lr, batch=batch, device=device, seed=seed)
