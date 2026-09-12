"""Resume from ckpt/latest.pt (2026-09-11): a run stopped and resumed reproduces an
uninterrupted run batch for batch (token source), can be EXTENDED past its original step
count, and a run dir with a resumable state is refused without --resume."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from pim.models.transformer_l import TransformerLTokens
from pim.training import TrainConfig
from pim.training.sources import token_source
from pim.training.train import train

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _source(seed=0):
    rng = np.random.default_rng(1)
    tok = rng.integers(1, 61, (300, 60)).astype(np.int8)
    ln = np.full(300, 60, np.int8)
    return token_source(tok, ln, block=59, env="othello", batch_size=16, seed=seed, device=DEV)


def _model():
    torch.manual_seed(0)
    return TransformerLTokens(vocab=61, block_size=59, n_layer=1, n_head=2, n_embd=32, dropout=0.0)


def _params(m):
    return torch.cat([p.detach().flatten().cpu() for p in m.parameters()])


def test_resume_reproduces_uninterrupted_run_and_extends(tmp_path):
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg10 = TrainConfig(steps=10, batch_size=16, warmup_steps=2, val_every=5, ckpt_base=4)
    straight = _model()
    train(straight, _source(), cfg10, tmp_path / "straight", arch="transformer_l_tokens", model_config={}, device=DEV, log=lambda s: None)

    cfg6 = TrainConfig(steps=6, batch_size=16, warmup_steps=2, val_every=3, ckpt_base=4)
    stopped = _model()
    train(stopped, _source(), cfg6, tmp_path / "resumed", arch="transformer_l_tokens", model_config={}, device=DEV, log=lambda s: None)
    assert (tmp_path / "resumed" / "ckpt" / "latest.pt").exists()
    # without --resume the run dir is refused rather than clobbered
    with pytest.raises(SystemExit, match="resum"):
        train(_model(), _source(), cfg10, tmp_path / "resumed", arch="transformer_l_tokens", model_config={}, device=DEV, log=lambda s: None)
    # resume and EXTEND to 10 steps: same parameters as the uninterrupted run
    resumed = _model()
    out = train(resumed, _source(), cfg10, tmp_path / "resumed", arch="transformer_l_tokens", model_config={}, device=DEV, log=lambda s: None, resume=True)
    assert torch.allclose(_params(straight), _params(resumed), atol=1e-5, rtol=1e-4)
    cfg = json.loads((tmp_path / "resumed" / "config.json").read_text())
    assert cfg["train"]["steps"] == 10 and cfg["resumed"][0]["from_step"] == 6 and cfg["resumed"][0]["batch_order_exact"]
    rows = [json.loads(line) for line in (tmp_path / "resumed" / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [3, 6, 10]          # the earlier passes, then the extension's
    assert out["best_step"] in {3, 6, 10}
    # a second resume with nothing left to do is a no-op
    again = train(resumed, _source(), cfg10, tmp_path / "resumed", arch="transformer_l_tokens", model_config={}, device=DEV, log=lambda s: None, resume=True)
    assert again["best_step"] == out["best_step"]
