#!/usr/bin/env python3
"""Train the causal transformer world model — same task/objective as the GRU.

Deliberately mirrors `scripts/train_gru.py` (same dataset, same MSE next-step
objective, same in-memory loader, same checkpoint format) so that "recurrence vs
attention" is the only variable.  The two deviations are the ones transformers
actually need: **LR warmup** and **gradient clipping**, without which a pre-norm
transformer at this depth trains far less stably than a GRU.

Checkpoints are written in the `train_gru.py` format, so `load_checkpoint`,
`scripts/eval_controls.py` and the whole editability suite work unchanged.

Usage
-----
    python scripts/train_transformer.py --window 16 --run-name W16
    python scripts/train_transformer.py --window 4  --run-name W4 --lr 3e-4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pim.world_models.dataloader import build_inmemory_dataloaders
from pim.world_models.transformer import ModelConfig, TransformerModel


def _parse():
    p = argparse.ArgumentParser(description="Train a causal transformer world model")
    p.add_argument("--dataset-path", default="datasets/4_fixed_refl_inview/train.h5")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--n-epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-frac", type=float, default=0.05, help="fraction of steps")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir", default="runs/transformers")
    p.add_argument("--run-name", required=True)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="train on only the first N samples. Samples are written in seed order, so this "
        "selects a PREFIX of the seed range — a smaller rung is a strict subset of a larger "
        "one, and the split RNG depends only on the count and the seed, so the partition is "
        "reproducible. Used by the data-scaling ladder to vary volume and nothing else.",
    )
    return p.parse_args()


def main() -> None:
    a = _parse()
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with h5py.File(a.dataset_path, "r") as f:
        obs_res = f["obs_intensity"].shape[2]
    train_loader, val_loader = build_inmemory_dataloaders(
        a.dataset_path,
        val_fraction=a.val_fraction,
        batch_size=a.batch_size,
        seed=a.seed,
        device=device,
        limit=a.limit,
    )

    mcfg = ModelConfig(
        input_dim=obs_res,
        d_model=a.d_model,
        n_layers=a.n_layers,
        n_heads=a.n_heads,
        mlp_ratio=a.mlp_ratio,
        window=a.window,
    )
    model = TransformerModel(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * a.n_epochs
    warmup = max(1, int(a.warmup_frac * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))  # cosine to zero

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    run_dir = Path(a.run_dir) / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train": vars(a),
                "model": dataclasses.asdict(mcfg),
                "state_span": model.state_span,
                "n_params": n_params,
                "device": str(device),
            },
            indent=2,
        )
    )
    print(f"Run dir    : {run_dir}")
    print(f"Model      : {n_params:,} params | d_model={a.d_model} layers={a.n_layers}")
    print(
        f"Window     : {a.window}  ->  carried state_span = {model.state_span} frames"
    )
    print(f"Schedule   : {total_steps} steps, {warmup} warmup, cosine decay\n")

    def run_epoch(loader, train: bool) -> float:
        model.train(train)
        tot, n = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                obs = batch["obs_intensity"]
                pred, _ = model(obs)
                loss = F.mse_loss(pred, obs[:, 1:, :])
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
                    opt.step()
                    sched.step()
                tot += loss.item()
                n += 1
        return tot / n

    best = float("inf")
    t0 = time.perf_counter()
    with tqdm(range(1, a.n_epochs + 1), desc="epochs", unit="epoch") as bar:
        for epoch in bar:
            tr = run_epoch(train_loader, True)
            va = run_epoch(val_loader, False)
            bar.set_postfix(train=f"{tr:.4f}", val=f"{va:.4f}")
            with open(metrics_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "train_loss": tr,
                            "val_loss": va,
                            "lr": opt.param_groups[0]["lr"],
                        }
                    )
                    + "\n"
                )
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "model_config": dataclasses.asdict(mcfg),
                "train_config": vars(a),
                "val_loss": va,
            }
            torch.save(ckpt, run_dir / "latest.pt")
            if va < best:
                best = va
                torch.save(ckpt, run_dir / "best_model.pt")

    el = time.perf_counter() - t0
    print(
        f"\nDone. Best val loss {best:.5f} | {el/60:.1f} min ({el/a.n_epochs:.1f}s/epoch)"
    )
    print(f"Checkpoints: {run_dir}")


if __name__ == "__main__":
    main()
