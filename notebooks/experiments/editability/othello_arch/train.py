"""Train Li et al.'s architecture on discworld — the run-A architecture pilot.

⚠ **Not the full run-A brief.** `directions/othello-architecture-on-discworld.md` specifies a
25M-episode corpus; this trains on an existing one so the *architecture* variable can be tested
without first building a streaming dataloader. What is faithful: the architecture (their minGPT,
8 blocks / 8 heads / `n_embd` 512, learned absolute positions, dropout 0.1, their init) and the
task substitution (continuous in, continuous out, MSE). What is not: the data scale.

Everything else mirrors `scripts/train_transformer.py`, so the result sits on the same axes as
every other discworld transformer: AdamW, 5% warmup + cosine, grad-clip 1.0, `val_fraction` 0.1,
best-val checkpointing, and the identical `build_inmemory_dataloaders` split RNG.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[4]
for _p in (str(_REPO), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pim.world_models.dataloader import build_inmemory_dataloaders  # noqa: E402

from model import build  # noqa: E402


def _parse():
    p = argparse.ArgumentParser(description="Li et al.'s architecture on discworld")
    p.add_argument("--dataset-path", default="datasets/17_scale_900k/train.h5")
    p.add_argument("--run-name", required=True)
    p.add_argument("--run-dir", default="runs/othello_arch")
    p.add_argument("--n-epochs", type=int, default=4,
                   help="Sevan's pilot call, 2026-08-21: ~4 to start, then read the val curve")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    a = _parse()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(a.dataset_path, "r") as f:
        obs_res, n_frames = f["obs_intensity"].shape[2], f["obs_intensity"].shape[1]

    train_loader, val_loader = build_inmemory_dataloaders(
        a.dataset_path, val_fraction=a.val_fraction, batch_size=a.batch_size,
        seed=a.seed, device=dev, limit=a.limit)

    model = build(obs_res=obs_res, block_size=n_frames - 1, n_layer=a.n_layer,
                  n_head=a.n_head, n_embd=a.n_embd, dropout=a.dropout).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps_per_epoch = len(train_loader)
    total = steps_per_epoch * a.n_epochs
    warmup = max(1, int(a.warmup_frac * total))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    run_dir = Path(a.run_dir) / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "train": vars(a), "n_params": n_par, "obs_res": obs_res, "block_size": n_frames - 1,
        "steps_per_epoch": steps_per_epoch, "total_steps": total,
        "note": "run-A ARCHITECTURE pilot; NOT the 25M-episode corpus the brief specifies",
    }, indent=2))
    print(f"{a.run_name}: {n_par:,} params (theirs ~25.3M on Othello) · {a.n_layer} blocks · "
          f"n_embd {a.n_embd} · block_size {n_frames - 1}")
    print(f"  {total:,} steps ({a.n_epochs} epochs x {steps_per_epoch:,}), {warmup:,} warmup",
          flush=True)

    def run_epoch(loader, train: bool) -> float:
        model.train(train)
        tot, n = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                obs = batch["obs_intensity"]
                pred = model(obs[:, :-1, :])
                loss = F.mse_loss(pred, obs[:, 1:, :])
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
                    opt.step()
                    sched.step()
                tot += loss.item()
                n += 1
        return tot / max(n, 1)

    best, t0 = float("inf"), time.perf_counter()
    for epoch in range(1, a.n_epochs + 1):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        rec = {"epoch": epoch, "train_loss": tr, "val_loss": va,
               "lr": opt.param_groups[0]["lr"],
               "elapsed_s": round(time.perf_counter() - t0, 1)}
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        ck = {"epoch": epoch, "model_state": model.state_dict(),
              "train_config": vars(a), "val_loss": va, "obs_res": obs_res,
              "block_size": n_frames - 1}
        torch.save(ck, run_dir / "latest.pt")
        if va < best:
            best = va
            torch.save(ck, run_dir / "best_model.pt")
        print(f"  epoch {epoch}/{a.n_epochs}  train {tr:.5f}  val {va:.5f}"
              f"{'  *' if va == best else ''}  [{(time.perf_counter() - t0) / 60:.1f} min]",
              flush=True)

    print(f"Best val loss {best:.5f} | {(time.perf_counter() - t0) / 60:.1f} min")
    print("W16 reference on dataset 4 (90k episodes): best val 0.02359")
    print("⚠ val still falling at the last epoch => the pilot is undertrained, extend it.")


if __name__ == "__main__":
    main()
