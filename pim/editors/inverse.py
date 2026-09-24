"""IM — the INVERSE-MAP write: replace the residual by the state-conditional mean of the
target state (2026-09-14, Sevan; canonical 2026-09-15). Pairs with ``pim.probes.inverse``.

    overwrite   h′ = g(s_post)                          THE canonical form — parameter-free
    delta       h′ = h + α (g(s_post) − g(s_pre))       kept, non-default (keeps what h carries
                                                        beyond the state; ≤ +0.15 over the
                                                        overwrite at its best α, never a
                                                        different verdict — inverse-probe.md)
    retrieval   h′ = bank.mean(s_post)                  IM-NN: the probe-free lookup, scored
                                                        beside IM and kept out of the tables

Editors write; they never score. The caller supplies the target state in the same
coordinates g was fitted in (the bench's PRE-dynamics full state on discworld; the mine /
theirs board on Othello) and scores the rollout with the canonical scorecards.
"""

from __future__ import annotations

import torch

from pim.probes.inverse import RetrievalBank


@torch.no_grad()
def inverse_overwrite(g, s_post: torch.Tensor) -> torch.Tensor:
    """(n, d) the residual written for target state ``s_post``: E[h | s_post]."""
    return g(s_post)


@torch.no_grad()
def inverse_delta(g, h: torch.Tensor, s_pre: torch.Tensor, s_post: torch.Tensor,
                  alpha: float = 1.0) -> torch.Tensor:
    """(n, d) h moved by α times the conditional-mean displacement from s_pre to s_post."""
    return h + alpha * (g(s_post) - g(s_pre))


@torch.no_grad()
def retrieval_overwrite(bank: RetrievalBank, s_post: torch.Tensor) -> torch.Tensor:
    """(n, d) the mean residual of the k training states nearest ``s_post``."""
    return bank.mean(s_post)
