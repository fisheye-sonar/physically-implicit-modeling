"""Train a GRU implicit world model on 1D observation sequences.

Self-supervised with teacher forcing: the model learns to predict
obs[t+1] from obs[t] via MSE loss.  No position or velocity supervision.

Usage
-----
    # Smoke test (2 epochs, single worker)
    python scripts/train_gru.py --n-epochs 2 --num-workers 0 --run-name smoke_test

    # Full run with defaults
    python scripts/train_gru.py --run-name gru_baseline

    # Override hyperparameters
    python scripts/train_gru.py --hidden-size 512 --lr 3e-4 --batch-size 512
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pim.models.dataloader import build_dataloaders
from pim.models.gru import GRUModel, ModelConfig


# ── Configs ───────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    # Data
    dataset_path: str = "datasets/initial_train_100k/dataset.h5"
    val_fraction: float = 0.1
    # Optimization
    n_epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # System
    num_workers: int = 4
    device: str = "auto"  # "auto" → cuda > mps > cpu
    seed: int = 0
    # Output
    run_dir: str = "runs"
    run_name: str = ""  # auto-generated from timestamp if empty
    # Model hyperparameters (forwarded to ModelConfig)
    hidden_size: int = 256
    num_layers: int = 1
    dropout: float = 0.0


# ── CLI parsing ───────────────────────────────────────────────────────────────


def _parse_args() -> TrainConfig:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description="Train GRU implicit world model")

    # Data
    p.add_argument("--dataset-path", default=defaults.dataset_path)
    p.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    # Optimization
    p.add_argument("--n-epochs", type=int, default=defaults.n_epochs)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    # System
    p.add_argument("--num-workers", type=int, default=defaults.num_workers)
    p.add_argument("--device", default=defaults.device)
    p.add_argument("--seed", type=int, default=defaults.seed)
    # Output
    p.add_argument("--run-dir", default=defaults.run_dir)
    p.add_argument("--run-name", default=defaults.run_name)
    # Model
    p.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    p.add_argument("--num-layers", type=int, default=defaults.num_layers)
    p.add_argument("--dropout", type=float, default=defaults.dropout)

    a = p.parse_args()
    return TrainConfig(
        dataset_path=a.dataset_path,
        val_fraction=a.val_fraction,
        n_epochs=a.n_epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        weight_decay=a.weight_decay,
        num_workers=a.num_workers,
        device=a.device,
        seed=a.seed,
        run_dir=a.run_dir,
        run_name=a.run_name,
        hidden_size=a.hidden_size,
        num_layers=a.num_layers,
        dropout=a.dropout,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def _read_obs_res(h5_path: str) -> int:
    with h5py.File(h5_path, "r") as f:
        return f["obs_intensity"].shape[2]  # (N, T, R)


def _make_run_dir(tcfg: TrainConfig) -> Path:
    name = tcfg.run_name or f"gru_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(tcfg.run_dir) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# ── Train / val loops ─────────────────────────────────────────────────────────


def _run_epoch(
    model: GRUModel,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    batch_bar: tqdm | None = None,
) -> float:
    """Run one epoch.  Pass optimizer=None for validation (no-grad)."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            obs = batch["obs_intensity"].to(device)  # (B, T, R)
            pred, _ = model(obs)  # (B, T-1, R)
            loss = F.mse_loss(pred, obs[:, 1:, :])

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if batch_bar is not None:
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")
                batch_bar.update(1)

    return total_loss / n_batches


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    tcfg = _parse_args()
    device = _resolve_device(tcfg.device)

    torch.manual_seed(tcfg.seed)

    # ── Dataset ───────────────────────────────────────────────────────────
    obs_res = _read_obs_res(tcfg.dataset_path)
    train_loader, val_loader = build_dataloaders(
        tcfg.dataset_path,
        val_fraction=tcfg.val_fraction,
        batch_size=tcfg.batch_size,
        seed=tcfg.seed,
        num_workers=tcfg.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    mcfg = ModelConfig(
        input_dim=obs_res,
        hidden_size=tcfg.hidden_size,
        num_layers=tcfg.num_layers,
        dropout=tcfg.dropout,
    )
    model = GRUModel(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay
    )

    # ── Run directory ─────────────────────────────────────────────────────
    run_dir = _make_run_dir(tcfg)
    metrics_path = run_dir / "metrics.jsonl"
    best_path = run_dir / "best_model.pt"
    latest_path = run_dir / "latest.pt"

    config_snapshot = {
        "train": dataclasses.asdict(tcfg),
        "model": dataclasses.asdict(mcfg),
        "device": str(device),
        "n_params": n_params,
    }
    (run_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2))

    print(f"Run dir  : {run_dir}")
    print(f"Device   : {device}")
    print(f"Model    : {n_params:,} parameters")
    print(f"Train    : {len(train_loader.dataset):,} samples")
    print(f"Val      : {len(val_loader.dataset):,} samples")
    print()

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    with tqdm(range(1, tcfg.n_epochs + 1), desc="epochs", unit="epoch") as epoch_bar:
        for epoch in epoch_bar:
            with tqdm(
                total=n_train_batches + n_val_batches,
                desc=f"epoch {epoch}",
                unit="batch",
                leave=False,
            ) as batch_bar:
                train_loss = _run_epoch(
                    model, train_loader, device, optimizer, batch_bar
                )
                batch_bar.set_description(f"epoch {epoch} [val]")
                val_loss = _run_epoch(
                    model, val_loader, device, optimizer=None, batch_bar=batch_bar
                )

            epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

            # ── Logging ───────────────────────────────────────────────────
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            # ── Checkpoints ───────────────────────────────────────────────
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_config": dataclasses.asdict(tcfg),
                "model_config": dataclasses.asdict(mcfg),
                "val_loss": val_loss,
            }
            torch.save(ckpt, latest_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt, best_path)

    print(f"\nDone.  Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {run_dir}")


if __name__ == "__main__":
    main()
