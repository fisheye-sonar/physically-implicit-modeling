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
    skip          : optional (n) -> None advancing the batch stream by n batches WITHOUT
                    materialising them — what makes a resumed run reproduce an
                    uninterrupted one batch for batch (2026-09-11). A source without it
                    can still resume; its batch order after the resume then differs, and
                    the loop records that in config.json.
    """

    batches: Iterator
    loss_fn: Callable
    validate: Callable
    steps_per_epoch: float
    meta: dict = field(default_factory=dict)
    skip: Callable[[int], None] | None = None


# ── the two objectives ───────────────────────────────────────────────────────


def mse_next_obs(model, x: torch.Tensor) -> torch.Tensor:
    """(B, T, R) observations → MSE on the next frame at every position.

    ONE alignment rule for every architecture: ``model(frames)`` predicts the next frame
    at every given position (the protocol's forward convention since 2026-09-07), so the
    model sees ``x[:, :-1]`` and is scored against ``x[:, 1:]``.
    """
    return F.mse_loss(model(x[:, :-1]), x[:, 1:])


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


def mse_next_move_onehot(model, batch, block: int = 59) -> torch.Tensor:
    """batch = (tok, ln) → MSE between the head's RAW outputs and the ONE-HOT next move.

    The regression counterpart of ``ce_next_move`` (2026-09-04): identical inputs,
    targets and masking, the identical 61-way head — but its outputs are read as
    probability estimates and pulled to the one-hot target by squared error (the Brier
    score) instead of by −log softmax. Same population minimiser (the conditional
    next-move distribution), different gradient geometry, and the outputs are NOT
    constrained to the simplex — the model carries ``output_kind="raw"`` so scoring reads
    them as they are (`pim.environments.othello.data.move_probs`). Padded positions are
    dropped exactly as CE's ``ignore_index`` drops them; the loss is the mean over the
    kept positions x all 61 outputs (the pad output's target is always 0).
    """
    tok, ln = batch
    x, y = xy_tokens(tok, ln, block)
    out = model.logits(x)                                    # (B, T, 61) raw head outputs
    keep = y != IGNORE
    target = F.one_hot(y.clamp_min(0), out.shape[-1]).to(out.dtype)
    return F.mse_loss(out[keep], target[keep])


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
          arch: str, model_config: dict, device: str = "cuda", log=print,
          resume: bool = False) -> dict:
    """Run the canonical loop. Writes into ``run_dir``:

    config.json   arch, model, train, data meta, n_params, commit_sha
    commit_sha    the code identity, one line (runs/ is gitignored; without this
                  nothing ties an artifact to the code that made it)
    metrics.jsonl one row per val pass
    best_model.pt / ckpt/step_*.pt   all stamped with ``arch`` for the registry
    ckpt/latest.pt   the RESUMABLE state (model + optimizer + RNG + history), rewritten
                  atomically at every val pass and at the end (2026-09-11)

    ``resume=True`` continues a run from ``ckpt/latest.pt`` up to ``cfg.steps`` — which
    may exceed the original, so a finished run can be EXTENDED. The step counter, the
    warm-up / schedule, the checkpoint cadence, the best-val tracking and the elapsed
    clock all continue; the batch stream is fast-forwarded through ``source.skip`` when
    the source provides it, so the resumed run reproduces an uninterrupted one batch for
    batch (token sources do; the discworld memmap stream does not — recorded in
    config.json as ``batch_order_exact``). Without ``resume``, a run dir that already
    holds ``latest.pt`` is refused rather than clobbered.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = model.to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    latest = run_dir / "ckpt" / "latest.pt"
    start_step, best, hist, t_off, resumed, exact = 1, float("inf"), [], 0.0, [], True
    if latest.exists():
        if not resume:
            raise SystemExit(f"{run_dir} already holds a resumable training state ({latest.name}); "
                             f"pass --resume to continue it, or choose a new --run-name")
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        start_step, best, hist, t_off = int(ck["step"]) + 1, float(ck["best"]), list(ck["hist"]), float(ck["elapsed_s"])
        torch.set_rng_state(ck["rng"]["torch"].cpu())
        if ck["rng"].get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() for r in ck["rng"]["cuda"]])
        np.random.set_state(ck["rng"]["numpy"])
        if source.skip is not None:
            source.skip(int(ck["step"]))
        else:
            exact = False
            log("  ⚠ source has no skip(): batch order after the resume differs from an uninterrupted run")
        resumed = list(ck.get("resumed", [])) + [{"from_step": int(ck["step"]), "to_steps": cfg.steps,
                                                  "at": time.strftime("%Y-%m-%d %H:%M"), "batch_order_exact": exact}]
        log(f"{run_dir.name}: RESUMED at step {start_step:,} (best val so far {best:.6f}), training to {cfg.steps:,}")
    elif resume:
        log(f"{run_dir.name}: --resume given but no {latest.name} yet — starting fresh")
    sha = _commit_sha()
    (run_dir / "commit_sha").write_text(sha + "\n")
    (run_dir / "config.json").write_text(json.dumps({
        "arch": arch, "model": model_config, "train": dataclasses.asdict(cfg),
        "data": source.meta, "n_params": n_par, "commit_sha": sha,
        "steps_per_epoch": source.steps_per_epoch,
        "epochs": cfg.steps / source.steps_per_epoch,
        **({"resumed": resumed} if resumed else {}),
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

    def save_latest(step: int):
        """The resumable state — written to a temp file and renamed, so a crash mid-write
        leaves the previous latest.pt intact."""
        (run_dir / "ckpt").mkdir(exist_ok=True)
        tmp = latest.with_suffix(".tmp")
        torch.save({"arch": arch, "step": step, "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(), "model_config": model_config,
                    "train_config": dataclasses.asdict(cfg), "best": best, "hist": hist,
                    "elapsed_s": t_off + time.perf_counter() - t0, "resumed": resumed,
                    "rng": {"torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                            "numpy": np.random.get_state()}}, tmp)
        tmp.replace(latest)

    t0 = time.perf_counter()
    if start_step > cfg.steps:
        log(f"{run_dir.name}: already at step {start_step - 1:,} >= {cfg.steps:,}; nothing to do")
        best_step = min(hist, key=lambda r: r["val_loss"])["step"] if hist else -1
        return {"best_val": best, "best_step": best_step, "minutes": t_off / 60}
    model.train()
    for step in range(start_step, cfg.steps + 1):
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
                   "elapsed_s": round(t_off + time.perf_counter() - t0, 1)}
            hist.append(rec)
            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            mark = ""
            if va < best:
                best, mark = va, "  *"
                save(run_dir / "best_model.pt", step, va)
            save_latest(step)
            log(f"  step {step:>9,}/{cfg.steps:,}  train {loss.item():.6f}  "
                f"val {va:.6f}{mark}  [{(t_off + time.perf_counter() - t0) / 60:.1f} min]")

        # checkpoints run on their OWN cadence, outside the val branch
        if step in ck_steps:
            (run_dir / "ckpt").mkdir(exist_ok=True)
            save(run_dir / "ckpt" / f"step_{step:09d}.pt", step,
                 hist[-1]["val_loss"] if hist else None)
            log(f"  [ckpt] step {step:,} (epoch {step / source.steps_per_epoch:.2f})")

    save_latest(cfg.steps)
    best_step = min(hist, key=lambda r: r["val_loss"])["step"] if hist else -1
    out = {"best_val": best, "best_step": best_step,
           "minutes": (t_off + time.perf_counter() - t0) / 60}
    log(f"done {run_dir.name}: best val {best:.6f} at step {best_step:,}/{cfg.steps:,} "
        f"· {out['minutes']:.1f} min")
    return out
