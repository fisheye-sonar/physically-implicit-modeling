"""BlockStream — shuffled-block reader over a flat memmap, with one-block prefetch.

Moved verbatim 2026-08-31 from ``discworld_scale/train.py``. The trick that makes
streaming trivial here: the corpus is i.i.d. BY CONSTRUCTION (every sequence from its
own seed), so a contiguous block IS a random sample and there is no global shuffle to
solve. Blocks are read in a shuffled order, shuffled within, and the next block is
prefetched on a thread so I/O hides behind compute.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import torch


class BlockStream:
    """Yields (batch, ...) torch tensors from ``obs[lo:hi]`` forever."""

    def __init__(self, obs, lo: int, hi: int, batch: int, block: int, seed: int,
                 shuffle: bool = True):
        self.obs, self.lo, self.hi = obs, lo, hi
        self.batch, self.block, self.shuffle = batch, block, shuffle
        self.rng = np.random.default_rng(seed)
        self.q: queue.Queue = queue.Queue(maxsize=2)
        self.starts = np.arange(lo, hi - block + 1, block)
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            order = self.rng.permutation(self.starts) if self.shuffle else self.starts
            for s in order:
                blk = np.asarray(self.obs[s: s + self.block])
                if self.shuffle:
                    blk = blk[self.rng.permutation(len(blk))]
                self.q.put(blk)

    def batches(self):
        while True:
            blk = self.q.get()
            for i in range(0, len(blk) - self.batch + 1, self.batch):
                yield torch.from_numpy(blk[i: i + self.batch])
