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

import numpy as np
import torch
import torch.nn.functional as F

from pim.training.stream import BlockStream
from pim.training.train import DataSource, ce_next_move, mse_next_obs, xy_tokens, IGNORE


def discworld_source(obs, *, n_total: int, batch_size: int, seed: int,
                     device: str = "cuda", block: int = 2_048,
                     val_fraction: float = 0.1, val_batches: int = 64,
                     limit: int | None = None, meta: dict | None = None) -> DataSource:
    """Stream a flat (N, T, R) float32 memmap. Val = the LAST val_fraction of the pool."""
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


def othello_source(tok_np: np.ndarray, ln_np: np.ndarray, *, batch_size: int, seed: int,
                   device: str = "cuda", val_fraction: float = 0.1,
                   limit: int | None = None, meta: dict | None = None) -> DataSource:
    """Whole token corpus on the GPU; train/val split by a seeded permutation."""
    if limit:
        tok_np, ln_np = tok_np[:limit], ln_np[:limit]
    n = len(tok_np)
    cut = int((1 - val_fraction) * n)
    perm = np.random.default_rng(seed).permutation(n)
    tok = torch.from_numpy(tok_np).to(device)
    ln = torch.from_numpy(ln_np).to(device)
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
            x, y = xy_tokens(tok[idx], ln[idx], 59)
            lg = model.logits(x)
            m = y != IGNORE
            tot += F.cross_entropy(lg[m], y[m], reduction="sum").item()
            cnt += int(m.sum())
        return tot / max(cnt, 1)

    return DataSource(batches=batches(), loss_fn=ce_next_move, validate=validate,
                      steps_per_epoch=len(tr_i) / batch_size,
                      meta={"env": "othello", "n_total": n, "n_train": int(cut),
                            "n_val": int(n - cut), "objective": "ce", **(meta or {})})
