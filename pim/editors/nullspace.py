"""Multi-probe nullspace editor — write to the WHOLE linear code, not one probe's slice.

Non-default; pairs with ``pim.probes.nullspace``. Kept (2026-08-31, ported from
``iterative_probing/iterative_probing.ipynb`` cell [9]) to answer a specific objection
to the discworld negative: *"one probe's row space is only d_out of d_model dimensions
— maybe the edit fails because the code is written redundantly and you only moved one
copy."* This editor moves every copy the cascade found:

    Δh = Σ_k A_k⁺ (target_k − p_k(h))

which is exact for every probe simultaneously because the cascade's row spaces are
orthogonal by construction. With oracle per-probe targets it reduces to the projection
of the true state change onto the accumulated code subspace, ``P_S(Δh_true)`` — an
identity the original notebook verified numerically rather than asserted.

⚠ The uniform arm is *expected* to blow up at large K: a low-R² probe has a small
``A_k`` hence a large ``A_k⁺``, and its readout sits near the population mean, so its
δ is large too — and orthogonal blocks add in quadrature. The R²-shrunk arm
(``shrink=True``: target_k = μ + R²_k·(t − μ)) is the principled damping.
"""

from __future__ import annotations

import numpy as np

from pim.probes.nullspace import NullspaceCascade


def multiprobe_delta(
    cascade: NullspaceCascade,
    h: np.ndarray,
    target: np.ndarray,
    *,
    K: int | None = None,
    shrink: bool = False,
    mu: np.ndarray | None = None,
    per_probe_targets: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Δh writing the first K probes of the cascade to their targets (float64).

    h      : (B, D) states — cast to float64; apply the returned Δh to the float32
             state yourself (the notebook's convention, kept).
    target : (B, d_out) the shared raw-units target every probe is asked to read.
    K      : how many probes to write (default: all of them).
    shrink : R²-shrink each probe's target toward ``mu`` (required when shrink=True):
             ``target_k = mu + r2_k * (target - mu)``.
    per_probe_targets : overrides target/shrink — one (B, d_out) per probe. Passing
             each probe's readout of an ORACLE state makes Δh = P_S(Δh_true) exactly.
    """
    h = h.astype(np.float64)
    target = np.asarray(target, np.float64)
    K = cascade.n_probes if K is None else K
    if shrink and mu is None:
        raise ValueError("shrink=True needs mu (the population mean target)")
    dh = np.zeros_like(h)
    for k in range(K):
        p = cascade.probes[k]
        if per_probe_targets is not None:
            t = np.asarray(per_probe_targets[k], np.float64)
        elif shrink:
            t = mu + p["r2"] * (target - mu)
        else:
            t = target
        dh += (t - cascade.read(k, h)) @ np.linalg.pinv(p["A"]).T
    return dh
