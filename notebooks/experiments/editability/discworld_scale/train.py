"""Transformer L on discworld at 20M sequences — the matched counterpart to BIG20M_othello_L.

MATCHED TO THE OTHELLO RUN, exactly:
    8 layers / 8 heads / n_embd 512 · dropout 0.1 on embd+resid+attn
    AdamW lr 1e-3, weight_decay 1e-4 (ALL params), betas (0.9, 0.999) — torch defaults
    grad clip 1.0 · batch 256 · 2,000-step warmup then CONSTANT lr
    780,000 steps · seed 0 · 18M train / 2M val (val_fraction 0.1 of the 20M pool)

NOT matched, and cannot be — these ARE the environment:
    block_size 39 vs 59      (40 frames vs 60 moves)
    Linear(128,512) encoder / Linear(512,128) decoder  vs  Embedding(61,512) / Linear(512,61)
    MSE  vs  cross-entropy
    25,371,776 params vs 25,312,768 (+0.2%, the continuous encoder/decoder)

STREAMING: the corpus is 410 GB and RAM is 59 GB, so obs is a flat memmap. The sequences are
i.i.d. by construction (each from its own seed), so a contiguous block IS a random sample — there
is no global shuffle to solve. We read whole blocks in a shuffled block order, shuffle within the
block, and prefetch the next block on a thread so I/O hides behind compute.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

# Cap the CPU thread pool. This loop is GPU-bound — the CPU side is one memmap read plus a
# shuffle — so one thread per core (32 here) buys nothing. Measured 2026-08-24: capped and
# uncapped are INDISTINGUISHABLE (300 steps in 0.2 min both ways, even with 32 generator
# processes running). Kept only as cheap insurance against pool thrashing on a busier box.
#
# ⚠ If you keep them, these lines MUST precede `import torch`: OpenMP reads its thread count when
# the library LOADS, so setting them afterwards is a silent no-op.
#
# ⛔ HISTORICAL NOTE, so nobody re-derives a wrong conclusion: an apparent ">230x speedup from
# capping threads" was recorded here on 2026-08-24 and was WRONG. The runs it compared were
# hanging in the `ckpt_base` loop below (`--ckpt-base 0` spins forever), not thrashing. Threads
# were never the problem.
_THREADS = "4"
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _THREADS)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(int(_THREADS))

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
for _p in (str(_HERE), str(_HERE.parent / "othello_arch"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import corpus as cp  # noqa: E402

from model import build as build_dw  # noqa: E402  (othello_arch/model.py)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class BlockStream:
    """Shuffled-block reader over the flat memmap, with a one-block prefetch thread."""

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=780_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--ckpt-base", type=int, default=1_000)
    p.add_argument("--val-every", type=int, default=5_000)
    p.add_argument("--val-batches", type=int, default=64)
    p.add_argument("--block", type=int, default=2_048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="BIG20M_discworld_L")
    p.add_argument("--run-dir", default=str(_REPO / "runs" / "discworld_scale"))
    p.add_argument("--limit", type=int, default=None, help="cap the pool (smoke tests only)")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    run_dir = Path(a.run_dir) / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    obs = cp.open_obs("r")
    n_total = a.limit or cp.N_TOTAL
    n_val = min(cp.VAL_N, max(a.block * 2, n_total // 10))
    n_train = n_total - n_val
    model = build_dw(obs_res=cp.OBS_RES, block_size=cp.FRAMES - 1, n_layer=a.n_layer,
                     n_head=a.n_head, n_embd=a.n_embd, dropout=a.dropout).to(DEV)
    n_par = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    spe = n_train / a.batch_size
    print(f"{a.run_name}: {n_par:,} params · block {cp.FRAMES - 1} · "
          f"{n_total:,} sequences ({n_train:,} train / {n_val:,} val)", flush=True)
    print(f"  {a.steps:,} steps ({a.steps / spe:.2f} epochs), {a.warmup_steps:,} warmup, "
          f"lr {a.lr_schedule} {a.lr}", flush=True)

    def lr_at(step: int) -> float:
        if step < a.warmup_steps:
            return step / a.warmup_steps
        if a.lr_schedule == "constant":
            return 1.0
        prog = (step - a.warmup_steps) / max(1, a.steps - a.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    # checkpoints: log-spaced (the early curve) UNION every epoch (Sevan's request)
    ck_steps = set()
    # ⛔ ckpt_base must be >= 1: `s *= 2` never advances from 0, so `--ckpt-base 0` spins forever
    # here with no output and the GPU idle — it looks exactly like a stalled data loader.
    # (Cost three smoke tests on 2026-08-24 before the cause was found.) 0 = "no log spacing".
    s = a.ckpt_base
    while s >= 1 and s < a.steps:
        ck_steps.add(s)
        s *= 2
    for e in range(1, int(a.steps / spe) + 1):
        ck_steps.add(int(round(e * spe)))
    ck_steps.add(a.steps)
    ck_steps = {c for c in ck_steps if 0 < c <= a.steps}
    print(f"  {len(ck_steps)} checkpoints: {sorted(ck_steps)}", flush=True)

    tr = BlockStream(obs, 0, n_train, a.batch_size, a.block, a.seed).batches()
    va_src = BlockStream(obs, n_train, n_total, a.batch_size, a.block, a.seed + 1,
                         shuffle=False)

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        g, tot, k = va_src.batches(), 0.0, 0
        for _ in range(a.val_batches):
            x = next(g).to(DEV, non_blocking=True)
            tot += F.mse_loss(model(x[:, :-1]), x[:, 1:]).item()
            k += 1
        model.train()
        return tot / k

    best, t0, hist = float("inf"), time.perf_counter(), []
    model.train()
    for step in range(1, a.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = a.lr * lr_at(step)
        x = next(tr).to(DEV, non_blocking=True)
        loss = F.mse_loss(model(x[:, :-1]), x[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()

        if step % a.val_every == 0 or step == a.steps:
            va = validate()
            rec = {"step": step, "train_loss": float(loss.item()), "val_loss": float(va),
                   "lr": opt.param_groups[0]["lr"],
                   "elapsed_s": time.perf_counter() - t0}
            hist.append(rec)
            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            mark = ""
            if va < best:
                best, mark = va, "  *"
                torch.save({"step": step, "model_state": model.state_dict(),
                            "model_config": {"obs_res": cp.OBS_RES, "block_size": cp.FRAMES - 1,
                                             "n_layer": a.n_layer, "n_head": a.n_head,
                                             "n_embd": a.n_embd, "dropout": a.dropout},
                            "train_config": vars(a), "val_loss": va, "best": True},
                           run_dir / "best_model.pt")
            print(f"  step {step:>7,}/{a.steps:,}  train {loss.item():.6f}  val {va:.6f}"
                  f"{mark}  [{(time.perf_counter() - t0) / 60:.1f} min]", flush=True)

        if step in ck_steps:
            cd = run_dir / "ckpt"
            cd.mkdir(exist_ok=True)
            torch.save({"step": step, "model_state": model.state_dict(),
                        "model_config": {"obs_res": cp.OBS_RES, "block_size": cp.FRAMES - 1,
                                         "n_layer": a.n_layer, "n_head": a.n_head,
                                         "n_embd": a.n_embd, "dropout": a.dropout},
                        "train_config": vars(a),
                        "val_loss": hist[-1]["val_loss"] if hist else None,
                        "epoch": step / spe},
                       cd / f"step_{step:09d}.pt")
            print(f"  [ckpt] step {step:,} (epoch {step / spe:.2f})", flush=True)

    bs = min(hist, key=lambda r: r["val_loss"])["step"] if hist else -1
    print(f"done {a.run_name}: best val {best:.6f} at step {bs:,}/{a.steps:,} · "
          f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
