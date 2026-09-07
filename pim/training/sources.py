"""DataSources: how each environment feeds the one canonical loop.

Two, matching the two environments' physically different scales:

* ``discworld_source`` — the 410 GB flat memmap, streamed via ``BlockStream``
  (from ``discworld_scale/train.py``'s recipe: last 10% of the pool is val, read
  in order; train blocks shuffled).
* ``othello_source`` — 1.2 GB of int8 tokens resident ON the GPU, sampled with a
  seeded generator (from ``ours_on_othello/train.py``; no loader, no host→device
  copy in the step loop).

Both honour ``limit`` as a strict PREFIX of the pool — sequences are index/seed-
generated, so a smaller rung is a subset of a larger one by construction, and a
data-scale sweep varies diversity, never the sampling law.
"""

from __future__ import annotations

import functools

import numpy as np
import torch
import torch.nn.functional as F

from pim.environments.othello.data import T_MODEL
from pim.training.stream import BlockStream
from pim.training.train import DataSource, ce_next_move, mse_next_move_onehot, mse_next_obs, xy_tokens, IGNORE


def discworld_source(obs, *, n_total: int, batch_size: int, seed: int,
                     device: str = "cuda", block: int = 2_048,
                     val_fraction: float = 0.1, val_batches: int = 64,
                     limit: int | None = None, meta: dict | None = None) -> DataSource:
    """Stream a flat (N, T, R) float32 memmap. Val = the LAST val_fraction of the pool.

    With ``limit`` the pool is the first ``limit`` sequences and val is the last 10% OF
    THAT PREFIX — so the training-time val loss is measured on a different set at every
    data-scale rung and is not comparable across rungs. Every canonical score (probe
    fits, the editability bench, the gates) uses the instance's fixed eval/probe/edits
    splits, which do not depend on ``limit`` and are comparable across rungs.
    """
    n_total = min(limit, n_total) if limit else n_total
    n_val = max(block * 2, int(val_fraction * n_total))
    n_train = n_total - n_val
    tr = BlockStream(obs, 0, n_train, batch_size, block, seed).batches()
    va_src = BlockStream(obs, n_train, n_total, batch_size, block, seed + 1, shuffle=False)

    def batches():
        while True:
            yield next(tr).to(device, non_blocking=True)

    @torch.no_grad()
    def validate(model) -> float:
        g, tot = va_src.batches(), 0.0
        for _ in range(val_batches):
            x = next(g).to(device, non_blocking=True)
            tot += mse_next_obs(model, x).item()   # same alignment dispatch as training
        return tot / val_batches

    return DataSource(batches=batches(), loss_fn=mse_next_obs, validate=validate,
                      steps_per_epoch=n_train / batch_size,
                      meta={"env": "discworld", "n_total": n_total, "n_train": n_train,
                            "n_val": n_val, "objective": "mse", **(meta or {})})


def token_source(tok_np: np.ndarray, ln_np: np.ndarray, *, block: int, env: str,
                 batch_size: int, seed: int, device: str = "cuda", val_fraction: float = 0.1,
                 limit: int | None = None, objective: str = "ce",
                 meta: dict | None = None) -> DataSource:
    """Whole token corpus on the GPU; train/val split by a seeded permutation.

    Shared by every tokenised environment: Othello moves (block 59) and discworld
    frames-as-tokens (block T-1, 2026-09-05). ``block`` is the model's INPUT length.
    ``objective``: "ce" (canonical cross-entropy) or "mse_onehot" (MSE against the
    one-hot next token, 2026-09-04) — selects the loss AND the matching val loss.
    """
    losses = {"ce": ce_next_move, "mse_onehot": mse_next_move_onehot}
    if objective not in losses:
        raise ValueError(f"unknown objective {objective!r}; one of {sorted(losses)}")
    loss_fn = functools.partial(losses[objective], block=block)
    if limit:
        tok_np, ln_np = tok_np[:limit], ln_np[:limit]
    if not tok_np.flags.writeable:          # a read-only memmap → one resident copy
        tok_np = np.array(tok_np)
    n = len(tok_np)
    cut = int((1 - val_fraction) * n)
    perm = np.random.default_rng(seed).permutation(n)
    tok = torch.from_numpy(np.ascontiguousarray(tok_np)).to(device)
    ln = torch.from_numpy(np.ascontiguousarray(ln_np)).to(device)
    tr_i = torch.from_numpy(perm[:cut]).to(device)
    va_i = torch.from_numpy(perm[cut:]).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)

    def batches():
        while True:
            idx = tr_i[torch.randint(len(tr_i), (batch_size,), device=device, generator=gen)]
            yield (tok[idx], ln[idx])

    @torch.no_grad()
    def validate(model) -> float:
        tot, cnt = 0.0, 0
        for i in range(0, len(va_i), 1024):
            idx = va_i[i: i + 1024]
            x, y = xy_tokens(tok[idx], ln[idx], block)
            lg = model.logits(x)
            m = y != IGNORE
            if objective == "mse_onehot":      # the mean-over-elements MSE, summed here
                oh = F.one_hot(y[m].clamp_min(0), lg.shape[-1]).to(lg.dtype)
                tot += F.mse_loss(lg[m], oh, reduction="sum").item() / lg.shape[-1]
            else:
                tot += F.cross_entropy(lg[m], y[m], reduction="sum").item()
            cnt += int(m.sum())
        return tot / max(cnt, 1)

    return DataSource(batches=batches(), loss_fn=loss_fn, validate=validate,
                      steps_per_epoch=len(tr_i) / batch_size,
                      meta={"env": env, "n_total": n, "n_train": int(cut),
                            "n_val": int(n - cut), "objective": objective, "block": block,
                            **(meta or {})})


def othello_source(tok_np: np.ndarray, ln_np: np.ndarray, *, batch_size: int, seed: int,
                   device: str = "cuda", val_fraction: float = 0.1,
                   limit: int | None = None, objective: str = "ce",
                   meta: dict | None = None) -> DataSource:
    """Othello's corpus: 1.2 GB of int8 move tokens on the GPU, block T_MODEL=59 (``token_source``)."""
    return token_source(tok_np, ln_np, block=T_MODEL, env="othello", batch_size=batch_size,
                        seed=seed, device=device, val_fraction=val_fraction, limit=limit,
                        objective=objective, meta=meta)
