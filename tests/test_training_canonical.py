"""The canonical training loop: config guards, schedules, the CE padding mask.

CPU-only; the end-to-end pipeline (train → run dir artifacts → registry load) is
exercised by the Phase-4 smoke runs, not here.
"""

from __future__ import annotations

import pytest
import torch

from pim.training import TrainConfig, xy_tokens
from pim.training.train import IGNORE, _ckpt_schedule


def test_ckpt_base_zero_refused():
    """`s *= 2` never advances from 0 — the 2026-08-24 stall, now unrepresentable."""
    with pytest.raises(ValueError, match="ckpt_base"):
        TrainConfig(steps=100, ckpt_base=0)


def test_ckpt_schedule_is_log_union_epochs():
    cfg = TrainConfig(steps=1000, ckpt_base=100)
    ck = _ckpt_schedule(cfg, steps_per_epoch=300.0)
    assert {100, 200, 400, 800} <= ck          # log-spaced
    assert {300, 600, 900} <= ck               # per-epoch
    assert 1000 in ck and all(0 < c <= 1000 for c in ck)


def test_canonical_recipe_defaults():
    """The matched BIG20M recipe — changing a default is changing every future run."""
    cfg = TrainConfig(steps=10)
    assert (cfg.batch_size, cfg.lr, cfg.weight_decay, cfg.grad_clip,
            cfg.lr_schedule, cfg.warmup_steps, cfg.seed) == (
        256, 1e-3, 1e-4, 1.0, "constant", 2000, 0)


def test_xy_tokens_masks_padding():
    tok = torch.tensor([[5, 6, 7, 0, 0, 0]])
    ln = torch.tensor([3])
    x, y = xy_tokens(tok, ln, block=5)
    assert x.tolist() == [[5, 6, 7, 0, 0]]
    # targets past position len-1 are IGNORE: the model is never rewarded for pad
    assert y.tolist() == [[6, 7, IGNORE, IGNORE, IGNORE]]
