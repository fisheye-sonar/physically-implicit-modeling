"""Train OUR transformer on Othello, with OUR discworld recipe held fixed.

`scripts/train_transformer.py` line for line, with two substitutions: the batch is move
tokens instead of observations, and the loss is cross-entropy instead of MSE. Optimiser,
schedule, warmup fraction, gradient clipping, batch size, `val_fraction` and the
best-val-checkpoint rule are all the ones that produced `runs/transformers/W16`.

The step budget is the point
----------------------------
`W16` ran 300 epochs over 90,000 episodes at batch 256 — **95,100 optimiser steps**. Every
rung of the scale ladder runs that same number of steps with the same schedule, and varies
only how many unique games the sampler draws from (`corpus.LADDER`). Anything that differs
across the ladder is therefore data diversity and not compute. Arm `F` is the one exception
and is flagged as such: it gets 8 passes over the full 20M pool and exists to answer "can
this architecture do Othello at all", never the scale question.

    M   90,000 games x 300 epochs  |
    L1   1,000,000 x  27           |  95,100 steps each, identical schedule
    L2   5,000,000 x   5.4         |
    D   20,000,000 x   1.35        |
    F   20,000,000 x   8 passes    <- 8x the compute, not a scale datapoint
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
for _p in (str(_HERE), str(_HERE.parent / "othello_transfer"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import corpus as cp  # noqa: E402
from model import BLOCK, build  # noqa: E402

IGNORE = -100
W16_STEPS = 300 * math.ceil(90_000 * 0.9 / 256)  # the discworld schedule, in steps


def _parse():
    p = argparse.ArgumentParser(description="Train our transformer on Othello")
    p.add_argument("--rung", required=True, choices=list(cp.LADDER) + ["F"])
    p.add_argument("--window", type=int, default=16)
    p.add_argument(
        "--arch", choices=("ours", "theirs"), default="ours",
        help="'ours' = this repo's transformer (d256/L4/RoPE/banded). 'theirs' = Li et al.'s "
        "minGPT verbatim (d512/L8/full-causal/learned-abs). The 'theirs' arm is the "
        "environment control from 2026-08-22: their architecture at OUR pilot's data volume "
        "and epoch count, so ONLY the environment differs from `othello_arch/A_pilot_900k`.",
    )
    p.add_argument("--epochs", type=int, default=None,
                   help="override the step budget with a plain epoch count over the pool")
    p.add_argument(
        "--limit", type=int, default=None,
        help="use only the first N games of the pool. Games are index-seeded, so this is a "
        "strict PREFIX and a smaller rung stays a subset of a larger one. Added 2026-08-22 to "
        "match the Othello arm's training-set size EXACTLY to the discworld arm's (810,000 "
        "after the 10%% val split, 3,164 steps/epoch on both sides) — previously the Othello "
        "side ran 11%% more sequences and steps.",
    )
    p.add_argument("--run-name", default=None)
    # every default below is `scripts/train_transformer.py`'s, unchanged
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=W16_STEPS)
    p.add_argument("--val-every", type=int, default=2000, help="steps between val passes")
    p.add_argument(
        "--lr-schedule", choices=("cosine", "constant"), default="cosine",
        help="'constant' = warmup then a flat LR. Use it whenever intermediate CHECKPOINTS are "
        "the deliverable: under a decaying schedule a checkpoint at step k is 'the model k steps "
        "into a schedule', not 'a model trained for k steps', and the two differ a lot — the "
        "14-epoch Othello run gained 2.0961 -> 2.0841 during annealing alone. Constant LR removes "
        "that schedule-position confound and lets the run be extended indefinitely.",
    )
    p.add_argument("--warmup-steps", type=int, default=None,
                   help="absolute warmup length; overrides --warmup-frac. For a long constant-LR "
                        "run a fraction is the wrong unit (5%% of a 250-epoch schedule at 20M "
                        "games is 879k steps — longer than a whole overnight run).")
    p.add_argument("--ckpt-base", type=int, default=0,
                   help="if >0, save a checkpoint at base, 2*base, 4*base, ... steps, giving "
                        "log-spaced points for the training-length axis. 0 disables.")
    p.add_argument("--run-dir", default=str(_REPO / "runs" / "ours_on_othello"))
    return p.parse_args()


def xy(tok: torch.Tensor, ln: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-move prediction over a right-padded batch.

    `x = t[0 : L-1]`, `y = t[1 : L]`, padded to `BLOCK`. Positions at or beyond `L-1` are
    `IGNORE` in `y`, so padding contributes nothing to the loss — without this the model is
    rewarded for predicting the pad token and the CE is not comparable across corpora with
    different length distributions.
    """
    x = tok[:, :BLOCK].long()
    y = tok[:, 1 : BLOCK + 1].long()
    pos = torch.arange(BLOCK, device=tok.device)[None, :]
    y = y.masked_fill(pos >= (ln[:, None].long() - 1), IGNORE)
    return x, y


def main() -> None:
    a = _parse()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = a.run_name or (f"{a.rung}_w{a.window}" if a.arch == "ours" else f"{a.rung}_theirs")

    pool = "D" if a.rung == "F" else a.rung
    # Generate only the pool this rung needs. Pools are prefixes of index range [0, n), so
    # `M ⊂ L1 ⊂ L2 ⊂ D` holds regardless of which files happen to exist.
    paths = cp.build(cp.LADDER[pool], log=print, only=("train",))
    tok_np, ln_np = cp.rung(paths["train"], pool)
    if a.limit is not None:
        tok_np, ln_np = tok_np[: a.limit], ln_np[: a.limit]
    n = len(tok_np)
    cut = int((1 - a.val_fraction) * n)
    g = np.random.default_rng(a.seed).permutation(n)
    tr_i, va_i = g[:cut], g[cut:]

    # The whole corpus lives on the GPU: 20M x 60 int8 is 1.2 GB, so there is no loader and
    # no host->device copy in the step loop. This is what makes 95,100 steps take ~17 min.
    tok = torch.from_numpy(tok_np).to(dev)
    ln = torch.from_numpy(ln_np).to(dev)
    tr_i = torch.from_numpy(tr_i).to(dev)
    va_i = torch.from_numpy(va_i).to(dev)

    if a.epochs is not None:
        total = a.epochs * math.ceil(len(tr_i) / a.batch_size)
    else:
        total = a.steps if a.rung != "F" else 8 * math.ceil(len(tr_i) / a.batch_size)
    warmup = a.warmup_steps if a.warmup_steps is not None else max(1, int(a.warmup_frac * total))
    if a.arch == "theirs":
        sys.path.insert(0, str(_HERE.parent / "othello_arch"))
        from model_othello import build as build_theirs

        model = build_theirs().to(dev)
    else:
        model = build(a.d_model, a.n_layers, a.n_heads, a.window, a.mlp_ratio).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        if a.lr_schedule == "constant":
            return 1.0
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    # Log-spaced checkpoint steps: base, 2*base, 4*base, ... plus the final step.
    ckpt_steps = set()
    if a.ckpt_base > 0:
        s = a.ckpt_base
        while s < total:
            ckpt_steps.add(s)
            s *= 2
        ckpt_steps.add(total)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    run_dir = Path(a.run_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    mcfg = (dataclasses.asdict(model.cfg) if dataclasses.is_dataclass(model.cfg)
            else {k: v for k, v in vars(model.cfg).items() if isinstance(v, (int, float, str))})
    (run_dir / "config.json").write_text(json.dumps({
        "train": vars(a), "model": mcfg, "arch": a.arch, "rung": a.rung,
        "unique_games": n, "train_games": len(tr_i), "val_games": len(va_i),
        "total_steps": total, "warmup_steps": warmup, "n_params": n_par,
        "state_span": model.state_span, "w16_reference_steps": W16_STEPS,
        "epochs_over_pool": total * a.batch_size / len(tr_i),
    }, indent=2))
    print(f"{name}: {n_par:,} params · arch {a.arch} · span {model.state_span} · "
          f"{n:,} unique games ({len(tr_i):,} train / {len(va_i):,} val)")
    print(f"  {total:,} steps ({total * a.batch_size / len(tr_i):.2f} epochs over the pool), "
          f"{warmup:,} warmup, lr {a.lr_schedule} · W16 ran {W16_STEPS:,}")
    if ckpt_steps:
        print(f"  {len(ckpt_steps)} checkpoints at: "
              + ", ".join(f"{s:,}" for s in sorted(ckpt_steps)), flush=True)

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        tot, cnt = 0.0, 0
        for i in range(0, len(va_i), 1024):
            idx = va_i[i : i + 1024]
            x, y = xy(tok[idx], ln[idx])
            lg = model.logits(x)
            m = y != IGNORE
            tot += F.cross_entropy(lg[m], y[m], reduction="sum").item()
            cnt += int(m.sum())
        model.train()
        return tot / max(cnt, 1)

    gen = torch.Generator(device=dev).manual_seed(a.seed)
    best, hist, t0 = float("inf"), [], time.perf_counter()
    model.train()
    for step in range(1, total + 1):
        idx = tr_i[torch.randint(len(tr_i), (a.batch_size,), device=dev, generator=gen)]
        x, y = xy(tok[idx], ln[idx])
        lg = model.logits(x)
        loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        sched.step()
        if step % a.val_every == 0 or step == total:
            va = validate()
            rec = {"step": step, "train_loss": loss.item(), "val_loss": va,
                   "lr": opt.param_groups[0]["lr"],
                   "elapsed_s": round(time.perf_counter() - t0, 1)}
            hist.append(rec)
            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            ck = {"step": step, "model_state": model.state_dict(),
                  "model_config": mcfg, "train_config": vars(a),
                  "val_loss": va, "rung": a.rung, "vocab": model.vocab}
            torch.save(ck, run_dir / "latest.pt")
            if va < best:
                best, ck["best"] = va, True
                torch.save(ck, run_dir / "best_model.pt")
            print(f"  step {step:>7,}/{total:,}  train {loss.item():.4f}  val {va:.4f}"
                  f"{'  *' if va == best else ''}  "
                  f"[{(time.perf_counter() - t0) / 60:.1f} min]", flush=True)
        # Checkpoints run on their OWN cadence, independent of --val-every, so the
        # training-length axis is evenly spaced on a log x-axis whatever the val cadence is.
        # Must sit outside the validation branch — `va` does not exist on a non-val step.
        if step in ckpt_steps:
            cdir = run_dir / "ckpt"
            cdir.mkdir(exist_ok=True)
            torch.save({"step": step, "model_state": model.state_dict(),
                        "model_config": mcfg, "train_config": vars(a),
                        "val_loss": hist[-1]["val_loss"] if hist else None,
                        "rung": a.rung, "vocab": model.vocab, "arch": a.arch},
                       cdir / f"step_{step:09d}.pt")
            print(f"  [ckpt] step {step:,} -> ckpt/step_{step:09d}.pt", flush=True)

    el = time.perf_counter() - t0
    best_step = min(hist, key=lambda r: r["val_loss"])["step"]
    print(f"done {name}: best val {best:.5f} at step {best_step:,}/{total:,} "
          f"({100 * best_step / total:.0f}% through) · {el / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
