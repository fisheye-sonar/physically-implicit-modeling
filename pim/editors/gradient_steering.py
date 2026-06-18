"""Gradient-based hidden state steering.

Finds a latent edit h* that minimises ||mlp(h*) - target||² by treating
h as a free variable and back-propagating through a frozen MLP probe.
Works with any differentiable probe (MLPExtractor, LinearExtractor).

Unlike pseudoinverse injection, this does not assume linearity — it finds
the edit numerically.  An optional L2 regulariser anchors h* to the
original h so the edit stays in-distribution:

    loss = ||probe(h*) - target||² + reg_weight * ||h* - h_orig||²

All functions are pure: no model or probe weights are modified.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def gradient_steer(
    h: torch.Tensor,
    target: torch.Tensor,
    probe: nn.Module,
    *,
    n_steps: int = 200,
    lr: float = 0.01,
    reg_weight: float = 0.0,
) -> tuple[torch.Tensor, float]:
    """Optimise h to minimise ||probe(h) - target||² via gradient descent.

    Parameters
    ----------
    h           : (1, H) initial hidden state — used as the starting point
                  and, when reg_weight > 0, as the anchor for regularisation.
    target      : (1, D) desired probe output (flat env state).
    probe       : frozen differentiable probe (MLPExtractor or LinearExtractor).
    n_steps     : number of gradient steps.
    lr          : Adam learning rate.
    reg_weight  : L2 penalty weight on ||h* - h_orig||.  0 = disabled.

    Returns
    -------
    h_edited    : (1, H) optimised hidden state (detached, no grad).
    final_loss  : scalar MSE at the final step (for injection_error bookkeeping).
    """
    h_orig = h.detach().clone()

    # h_opt is what we optimise; init from the warmed-up state
    h_opt = h_orig.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([h_opt], lr=lr)

    probe_frozen = probe.eval()
    for param in probe_frozen.parameters():
        param.requires_grad_(False)

    final_loss = 0.0
    for _ in range(n_steps):
        optimizer.zero_grad()
        pred = probe_frozen(h_opt)                       # (1, *state_shape)
        pred_flat = pred.reshape(1, -1)
        mse = ((pred_flat - target) ** 2).mean()
        loss = mse
        if reg_weight > 0.0:
            loss = loss + reg_weight * ((h_opt - h_orig) ** 2).mean()
        loss.backward()
        optimizer.step()
        final_loss = mse.item()

    for param in probe_frozen.parameters():
        param.requires_grad_(True)

    return h_opt.detach(), final_loss
