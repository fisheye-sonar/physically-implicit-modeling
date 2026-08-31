"""The canonical MLP probe — Li et al. §3.2: ONE hidden layer, width 128 (their shape).

Definition
----------
Regression:     ``y = (W₁ ReLU(W₂ z) …) * y_std + y_mean`` — one hidden layer of 128.
Classification: ``softmax(W₁ ReLU(W₂ z))`` per tile — at width 128 this is exactly the
shape of Li et al.'s own ``BatteryProbeClassificationTwoLayer`` (their ``mid_dim``).

Width 128 became canonical 2026-08-31: the earlier 512 was an inherited default with no
justification, and MLP-128 at ≥20k probe sequences reproduces every 512 conclusion
(verified on the Othello 20M model: gradient steering EI −0.0014 vs −0.0007, same
guards) while carrying 4× fewer parameters — which is what keeps 30k-sequence probe
corpora out of the memorisation regime that produced the 2026-08-22 bad numbers.

The regression fit takes its loss in **standardised target space** so every output
dimension contributes equally — a raw-units loss is implicitly ``y_std²``-weighted and
on a position+velocity target that is a ~1000× tilt toward position (the 2026-08-19
velocity-undertraining bug). Full derivation in ``pim.probes.base.fit_probe``.

The tripwire
------------
``check_probe_sanity`` — an MLP can represent the linear map exactly, so on held-out
data it must never score *below* the linear probe on the same features. When it does,
the MLP is fitting the probe training set, not the representation, and every
decodability number from that fit is meaningless. Run it after every paired fit.
"""

from __future__ import annotations

import numpy as np

from pim.probes.base import WorldStateProbe, fit_probe

CANONICAL_HIDDEN = 128


def fit_mlp(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    hidden: int = CANONICAL_HIDDEN,
    n_classes: int | None = None,
    epochs: int = 200,
    lr: float = 1e-3,
    batch: int = 4096,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[WorldStateProbe, dict]:
    """Fit the canonical MLP probe (see module docstring). Returns ``(probe, stats)``."""
    return fit_probe(x_tr, y_tr, x_te, y_te, hidden=hidden, n_classes=n_classes,
                     epochs=epochs, lr=lr, batch=batch, device=device, seed=seed)


class ProbeSanityError(AssertionError):
    """An MLP probe scored WORSE than a linear one on held-out data."""


def check_probe_sanity(lin: dict, mlp: dict, *, tol: float = 0.01, strict: bool = True,
                       label: str = "", log=print) -> dict:
    """Tripwire: a held-out MLP probe must not be beaten by a linear probe.

    ``lin`` / ``mlp`` map residual point → ``(probe, stats)`` as returned by the fit
    functions, fitted on the same features and targets.

    This is not hypothetical. On 2026-08-22 MLP probes on 1,500 sequences reported
    in-sample velocity R² of 0.954–0.959 against held-out −0.073 and −0.090 — 262k
    probe parameters on 48k rows. The numbers looked like "velocity is barely
    decodable" and were quoted as such; refitting on 10k sequences moved held-out vy R²
    to 0.83. This check catches it at the point of fitting instead of two findings
    later.

    Returns a report dict (always); raises ``ProbeSanityError`` when ``strict``.
    """
    rows, bad = [], []
    for ell in sorted(set(lin) & set(mlp)):
        sl, sm = lin[ell][1], mlp[ell][1]
        r_lin, r_mlp = float(sl["r2"]), float(sm["r2"])
        gap_mlp = float(sm.get("r2_insample", np.nan)) - r_mlp
        gap_lin = float(sl.get("r2_insample", np.nan)) - r_lin
        row = {"point": ell, "r2_linear": r_lin, "r2_mlp": r_mlp,
               "mlp_minus_linear": r_mlp - r_lin,
               "insample_gap_mlp": gap_mlp, "insample_gap_linear": gap_lin}
        rows.append(row)
        if r_mlp < r_lin - tol:
            bad.append(row)
    report = {"label": label, "tol": tol, "rows": rows, "n_violations": len(bad)}
    if log:
        worst = max(rows, key=lambda r: r["insample_gap_mlp"]) if rows else None
        if worst is not None:
            log(f"    probe sanity{' [' + label + ']' if label else ''}: "
                f"{len(bad)}/{len(rows)} points where MLP < linear; "
                f"worst MLP in-sample gap {worst['insample_gap_mlp']:+.4f} "
                f"@ point {worst['point']}")
    if bad and strict:
        det = "\n".join(f"      point {r['point']}: linear {r['r2_linear']:+.4f} > "
                        f"MLP {r['r2_mlp']:+.4f} (by {-r['mlp_minus_linear']:.4f}), "
                        f"MLP in-sample gap {r['insample_gap_mlp']:+.4f}" for r in bad)
        raise ProbeSanityError(
            f"MLP probe beaten by linear probe at {len(bad)} residual point(s)"
            f"{' for ' + label if label else ''} — the MLP is fitting the probe training "
            f"set, not the representation. Refit with more probe sequences (>=10k) before "
            f"trusting any decodability number from it.\n{det}")
    return report
