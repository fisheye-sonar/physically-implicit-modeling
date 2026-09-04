"""PI — pseudoinverse injection: the minimum-norm write that lands the linear read-out.

One of the three workhorse editors. The primitive (``inject_state``) is the repo's
original ``pim.editors.probe_steering`` implementation, kept verbatim; what this module
adds (2026-08-31) is the **decomposition layer** — how a ``WorldStateProbe``'s three-stage
sandwich ``h → z → net(z) → y`` is collapsed into the single affine map the primitive
inverts. Getting that collapse wrong is not hypothetical:

⛔ **The y-affine bug (found 2026-08-31).** The probe's forward pass is
``(A_z z + b_z) * y_std + y_mean`` with ``z = (h − x_mean)/x_std``. The original
discworld decomposition dropped the ``* y_std + y_mean`` stage, so the editor solved
``standardised-read-out = raw-units target`` — on the cartesian depth row a ~4σ unit
mismatch. Every published discworld PI number before this date is the ``"legacy"``
space below; fixing it roughly tripled the best Edit Index (+0.05 → +0.16 cartesian,
+0.09 → +0.24 frustum) while leaving the conclusion (not editable; the α=1 write does
nothing) intact. Othello was structurally immune: classification probes carry
``y_mean=0, y_std=1``.

The three named spaces (``pinv_maps``):

    "zspace"  CANONICAL — solve in the probe's standardised input space, y-affine
              included; the write is minimum-norm in units of activation s.d., which
              is the on-manifold choice (a raw-space min-norm write will happily push
              50σ along a direction the model has never seen vary). This is what the
              Othello-side editor always did.
    "raw"     y-affine included, pinv taken in raw activation units. Correct target,
              variance-blind norm. Kept because the z-vs-raw gap is itself a finding.
    "legacy"  the pre-fix decomposition, kept ONLY so old numbers stay reproducible.
              Never quote it as PI.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pim.probes.base import WorldStateProbe

# ── the primitive (verbatim from the original pim.editors.probe_steering) ────


def decompose_hidden(
    h: torch.Tensor, A: torch.Tensor, A_pinv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split h into row-space (probe-controlled) and null-space (probe-invariant) parts.

    By construction ``h_parallel + h_perp == h``.
    """
    Ah = h @ A.T  # (..., d_out)
    h_parallel = Ah @ A_pinv.T  # (..., d_in)
    h_perp = h - h_parallel
    return h_parallel, h_perp


def inject_state(
    h: torch.Tensor,
    target: torch.Tensor,
    A: torch.Tensor,
    A_pinv: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Return h' with ``A h' + b = target`` (exactly, when A has full row rank),
    preserving the null-space component of h — the minimum-norm write in the space
    the operators live in."""
    _, h_perp = decompose_hidden(h, A, A_pinv)
    h_new_parallel = (target - b) @ A_pinv.T
    return h_new_parallel + h_perp


# ── the decomposition layer ──────────────────────────────────────────────────


@dataclass
class PinvMap:
    """One collapsed affine read-out plus the space its pseudoinverse lives in."""

    space: str  # "zspace" | "raw" | "legacy"
    A: torch.Tensor  # (d_out, d_in) — read-out weight in this space
    A_pinv: torch.Tensor  # (d_in, d_out)
    b: torch.Tensor  # (d_out,)


def pinv_maps(probe: WorldStateProbe) -> dict[str, PinvMap]:
    """The three named decompositions of a LINEAR ``WorldStateProbe``.

    All three agree on what the probe *reads*; they differ in which vector the editor
    solves for and in which coordinates "minimum norm" is measured. See module doc.
    """
    lin = probe.net
    Az, bz = lin.weight.detach(), lin.bias.detach()  # (D, H), (D)
    xs, xm = probe.x_std, probe.x_mean
    ys, ym = probe.y_std, probe.y_mean

    # legacy: standardised-y read-out expressed over raw h — the y-affine is DROPPED
    A_leg = Az / xs
    b_leg = bz - (Az / xs) @ xm
    # raw: same map with the y-affine restored, so the target is in raw sim units
    A_raw = ys[:, None] * A_leg
    b_raw = ym + ys * b_leg
    return {
        "zspace": PinvMap("zspace", Az, torch.linalg.pinv(Az), bz),
        "raw": PinvMap("raw", A_raw, torch.linalg.pinv(A_raw), b_raw),
        "legacy": PinvMap("legacy", A_leg, torch.linalg.pinv(A_leg), b_leg),
    }


@torch.no_grad()
def pinv_step(h0: torch.Tensor, target: torch.Tensor, probe: WorldStateProbe,
              space: str = "zspace", dims=None) -> torch.Tensor:
    """The α=1 write: Δh such that the probe reads ``target`` at ``h0 + Δh``.

    ``target`` is ALWAYS in the probe's output units (raw sim units for regression) —
    the space choice changes the solve, never the meaning of the target. Scale the
    returned step by α for a sweep; α=1 is the exact jump and the honest headline.

    ``dims`` restricts the solve to a SUBSET of the probe's output dimensions: the
    write then lands only those read-outs and leaves the rest free, so a probe fitted
    on the full state can drive position alone. The pseudoinverse is taken on the
    sub-matrix, so this is a genuine rank-len(dims) solve with a correspondingly larger
    null space (a smaller write), not a masked version of the full one.
    """
    maps = pinv_maps(probe)
    if space not in maps:
        raise KeyError(f"space must be one of {sorted(maps)}, got {space!r}")
    m = maps[space]
    idx = None if dims is None else list(dims)
    if m.space == "zspace":
        z0 = (h0 - probe.x_mean) / probe.x_std
        tgt_net = (target - probe.y_mean) / probe.y_std
        A, b = m.A, m.b
        if idx is not None:
            A, b, tgt_net = A[idx], b[idx], tgt_net[..., idx]
        z1 = inject_state(z0, tgt_net, A, torch.linalg.pinv(A), b)
        return (z1 - z0) * probe.x_std
    # raw and legacy solve directly over h (legacy against the WRONG-units target;
    # that is exactly what makes it legacy)
    A, b, tgt = m.A, m.b, target
    if idx is not None:
        A, b, tgt = A[idx], b[idx], target[..., idx]
        return inject_state(h0, tgt, A, torch.linalg.pinv(A), b) - h0
    return inject_state(h0, target, m.A, m.A_pinv, m.b) - h0


@torch.no_grad()
def readout_error(h: torch.Tensor, target: torch.Tensor, probe: WorldStateProbe,
                  dims=None) -> float:
    """Mean ‖probe(h) − target‖ in the probe's OUTPUT units — the landing check.

    Uses ``probe.forward`` itself, so a decomposition can never disagree with it
    silently (the failure mode the y-affine bug lived in for nine days).

    ``dims`` restricts the norm to the read-outs the write actually DROVE, so a
    dims-restricted solve is still scored on whether it landed. Measuring a rank-k
    write against all d outputs reports the undriven dims as error and makes an exact
    landing look like a miss.
    """
    err = probe(h) - target
    if dims is not None:
        err = err[..., list(dims)]
    return float(err.norm(dim=-1).mean())
