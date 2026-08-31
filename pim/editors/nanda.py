"""ND — Nanda et al. (2309.00941) §4.1 direction addition. One of the three workhorses.

Their method: ``x' ← x + α·‖x‖·d̂``, where ``d`` is the LINEAR probe's weight row (or
row combination) for the target read-out, added to the residual stream. One vector
addition, no gradients, no solve. Their Table 2 anchors: null 2.723, non-linear (Li)
0.12, linear addition 0.10.

Consolidates (2026-08-31) the three copies that used to exist —
``othello_arch/editability.nanda_addition``, ``othello_transfer/linear_intervention``'s
``"add"``/``"add_raw"`` modes, and the inline hook in ``nanda_on_discworld.py`` — into
two primitives:

* ``probe_direction`` — which raw-space direction the probe implies. The probe
  standardises its input, so ``w / x_std`` is the true raw-space gradient of that
  read-out; ``standardised=False`` gives the weight row as-is, which is what Nanda's
  own un-standardised probe would use (kept as the ``add_raw`` comparison arm).
* ``addition_delta`` — the write itself, scaled by the CURRENT activation's norm so
  one α means the same size of write at every residual point (the residual scale
  varies ~3× across points; their α is unstated).

Row selection is the caller's, because it is task-shaped: discworld sums the edited
object's read-out rows (``rows=out_dims``); Othello picks the (tile, class) row per
case, optionally subtracting the current class's row (their target−current variant).
"""

from __future__ import annotations

import torch

from pim.probes.base import WorldStateProbe


def probe_direction(probe: WorldStateProbe, rows, *, subtract_rows=None,
                    standardised: bool = True) -> torch.Tensor:
    """(d_in,) unit direction implied by the LINEAR probe for the selected read-out rows.

    rows          : indices into the probe's output dims — summed (the edited object's
                    read-out rows on discworld; a single (tile,class) row on Othello).
    subtract_rows : optional rows to subtract first (the target−current variant).
    standardised  : divide by ``x_std`` (the raw-space gradient — canonical);
                    False = the bare weight row (``add_raw`` comparison arm).
    """
    W = probe.net.weight.detach()
    scale = probe.x_std if standardised else torch.ones_like(probe.x_std)
    # divide-then-sum, matching the original implementation's float op order exactly
    d = (W[list(rows)] / scale).sum(0)
    if subtract_rows is not None:
        d = d - (W[list(subtract_rows)] / scale).sum(0)
    return d / d.norm()


def addition_delta(x_last: torch.Tensor, d_hat: torch.Tensor, alpha: float) -> torch.Tensor:
    """The write: ``α · ‖x‖ · d̂`` per row of ``x_last`` (B, d_in)."""
    return alpha * x_last.norm(dim=-1, keepdim=True) * d_hat


def addition_hook(ell: int, d_hat: torch.Tensor, alpha: float):
    """A `_run`-style edit hook applying the addition at residual point ``ell``,
    last position — the form every model's ``decode(edit=...)`` accepts."""

    def hook(layer, x):
        if layer != ell:
            return x
        out = x.clone()
        out[:, -1] = x[:, -1] + addition_delta(x[:, -1], d_hat, alpha)
        return out

    return hook
