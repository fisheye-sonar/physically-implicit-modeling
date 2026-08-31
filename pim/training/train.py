"""The canonical training setup — ONE loop, two objectives, both environments.

Merges (2026-08-31) the three trainers that produced every canonical run —
``scripts/train_transformer.py`` (the S recipe), ``discworld_scale/train.py`` (the
matched 20M recipe: streaming, log-spaced ∪ per-epoch checkpoints), and
``ours_on_othello/train.py`` (padded-CE, on-GPU tokens) — so that "the training setup
is matched across environments" is enforced by a shared code path instead of by three
files promising to mirror each other.

The canonical hyperparameters are the BIG20M recipe, matched exactly across both 20M
runs and used for every new canonical run unless a config overrides them explicitly:

    AdamW lr 1e-3 · weight_decay 1e-4 (all params) · betas torch-default
    grad clip 1.0 · batch 256 · 2,000-step warmup then CONSTANT lr · seed 0

Constant LR is deliberate: under a decaying schedule a checkpoint at step k is "a
model k steps into a schedule", not "a model trained for k steps" — the 14-epoch
Othello run gained 2.096 → 2.084 during annealing alone. Constant LR removes the
schedule-position confound from every training-length axis and lets a run be extended.

What differs per environment is packaged as a ``DataSource`` (batches + a validate
function + the loss): MSE on next observation (discworld) or padded cross-entropy on
the next move (Othello). The loop itself neither knows nor cares which world it is in.
"""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

IGNORE = -100  # CE ignore index for padded positions


@dataclass
class TrainConfig:
    """The canonical setup. Defaults ARE the matched BIG20M recipe — change with care."""

    steps: int
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    lr_schedule: str = "constant"  # "constant" | "cosine"
    warmup_steps: int = 2_000
    ckpt_base: int = 1_000  # log-spaced ckpts at base, 2·base, …; MUST be >= 1 (see below)
    val_every: int = 5_000
    seed: int = 0

    def __post_init__(self):
        # ⛔ `s *= 2` never advances from 0, so ckpt_base=0 would spin forever in the
        # schedule builder — with no output and the GPU idle, it looks exactly like a
        # stalled data loader (cost three smoke tests on 2026-08-24).
        if self.ckpt_base < 1:
            raise ValueError("ckpt_base must be >= 1")


@dataclass
class DataSource:
    """What an environment supplies to the loop.

    batches       : infinite iterator of training batches (already collated).
    loss_fn       : (model, batch) -> scalar loss. The ONLY objective-specific code.
    validate      : (model) -> float, the val loss on a fixed protocol.
    steps_per_epoch : for the per-epoch checkpoint schedule.
    meta          : provenance recorded into config.json (env instance, split sizes…).
    """

    batches: Iterator
    loss_fn: Callable
    validate: Callable
    steps_per_epoch: float
    meta: dict = field(default_factory=dict)


# ── the two objectives ───────────────────────────────────────────────────────


def mse_next_obs(model, x: torch.Tensor) -> torch.Tensor:
    """(B, T, R) observations → MSE on the next frame at every position.

    The two architectures kept their historical forward conventions (both are gated
    bit-identical against the checkpoints that trained under them), so the alignment
    is dispatched here, in one visible place, rather than papered over in the models:
    Transformer-S takes the FULL sequence and slices internally, returning
    ``(pred, state)`` with pred aligned to ``x[:, 1:]``; Transformer-L predicts at
    every position of whatever it is given, so it gets ``x[:, :-1]``.
    """
    from pim.models.transformer_s import TransformerS

    if isinstance(model, TransformerS):
        pred, _ = model(x)
    else:
        pred = model(x[:, :-1])
    return F.mse_loss(pred, x[:, 1:])


def xy_tokens(tok: torch.Tensor, ln: torch.Tensor, block: int):
    """Next-move pairs over a right-padded batch; padded targets are IGNORE so the CE
    is comparable across corpora with different length distributions."""
    x = tok[:, :block].long()
    y = tok[:, 1: block + 1].long()
    pos = torch.arange(block, device=tok.device)[None, :]
    y = y.masked_fill(pos >= (ln[:, None].long() - 1), IGNORE)
    return x, y


def ce_next_move(model, batch, block: int = 59) -> torch.Tensor:
    """batch = (tok, ln) int tensors → padded CE on the next move.

    ``block`` is the model's INPUT length (59 = the first 59 moves of a 60-move game,
    the same for both architectures) — not ``state_span``, which for Transformer-S is
    the receptive field (61 at window 16), a different quantity.
    """
    tok, ln = batch
    x, y = xy_tokens(tok, ln, block)
    lg = model.logits(x)
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), ignore_index=IGNORE)


# ── the loop ─────────────────────────────────────────────────────────────────


def _ckpt_schedule(cfg: TrainConfig, steps_per_epoch: float) -> set[int]:
    """Log-spaced (the early curve) UNION every epoch (the long tail)."""
    ck = set()
    s = cfg.ckpt_base
    while s < cfg.steps:
        ck.add(s)
        s *= 2
    for e in range(1, int(cfg.steps / steps_per_epoch) + 1):
        ck.add(int(round(e * steps_per_epoch)))
    ck.add(cfg.steps)
    return {c for c in ck if 0 < c <= cfg.steps}


def _commit_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def train(model, source: DataSource, cfg: TrainConfig, run_dir: str | Path, *,
          arch: str, model_config: dict, device: str = "cuda", log=print) -> dict:
    """Run the canonical loop. Writes into ``run_dir``:

    config.json   arch, model, train, data meta, n_params, commit_sha
    commit_sha    the code identity, one line (runs/ is gitignored; without this
                  nothing ties an artifact to the code that made it)
    metrics.jsonl one row per val pass
    best_model.pt / ckpt/step_*.pt   all stamped with ``arch`` for the registry
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = model.to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sha = _commit_sha()
    (run_dir / "commit_sha").write_text(sha + "\n")
    (run_dir / "config.json").write_text(json.dumps({
        "arch": arch, "model": model_config, "train": dataclasses.asdict(cfg),
        "data": source.meta, "n_params": n_par, "commit_sha": sha,
        "steps_per_epoch": source.steps_per_epoch,
        "epochs": cfg.steps / source.steps_per_epoch,
    }, indent=2))

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps
        if cfg.lr_schedule == "constant":
            return 1.0
        prog = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    ck_steps = _ckpt_schedule(cfg, source.steps_per_epoch)
    log(f"{run_dir.name}: {n_par:,} params · arch {arch} · {cfg.steps:,} steps "
        f"({cfg.steps / source.steps_per_epoch:.2f} epochs) · {len(ck_steps)} checkpoints")

    def save(path: Path, step: int, va: float | None):
        torch.save({"arch": arch, "step": step, "model_state": model.state_dict(),
                    "model_config": model_config,
                    "train_config": dataclasses.asdict(cfg),
                    "val_loss": va, "epoch": step / source.steps_per_epoch}, path)

    best, hist, t0 = float("inf"), [], time.perf_counter()
    model.train()
    for step in range(1, cfg.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = cfg.lr * lr_at(step)
        batch = next(source.batches)
        loss = source.loss_fn(model, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.val_every == 0 or step == cfg.steps:
            model.eval()
            va = source.validate(model)
            model.train()
            rec = {"step": step, "train_loss": float(loss.item()), "val_loss": float(va),
                   "lr": opt.param_groups[0]["lr"],
                   "elapsed_s": round(time.perf_counter() - t0, 1)}
            hist.append(rec)
            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            mark = ""
            if va < best:
                best, mark = va, "  *"
                save(run_dir / "best_model.pt", step, va)
            log(f"  step {step:>9,}/{cfg.steps:,}  train {loss.item():.6f}  "
                f"val {va:.6f}{mark}  [{(time.perf_counter() - t0) / 60:.1f} min]")

        # checkpoints run on their OWN cadence, outside the val branch
        if step in ck_steps:
            (run_dir / "ckpt").mkdir(exist_ok=True)
            save(run_dir / "ckpt" / f"step_{step:09d}.pt", step,
                 hist[-1]["val_loss"] if hist else None)
            log(f"  [ckpt] step {step:,} (epoch {step / source.steps_per_epoch:.2f})")

    best_step = min(hist, key=lambda r: r["val_loss"])["step"] if hist else -1
    out = {"best_val": best, "best_step": best_step,
           "minutes": (time.perf_counter() - t0) / 60}
    log(f"done {run_dir.name}: best val {best:.6f} at step {best_step:,}/{cfg.steps:,} "
        f"· {out['minutes']:.1f} min")
    return out
