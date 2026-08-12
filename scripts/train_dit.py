"""Train a DiT (diffusion transformer) implicit world model on 1D observations.

Self-supervised next-step prediction, same task as the GRU/RSSM: the model
learns to predict obs[t+1] from obs[<=t] (windowed causal attention), trained
with a flow-matching loss at every sequence position (diffusion forcing).

Three validation metrics with different roles:
  * val_loss     — flow-matching loss on the val split (same units as train);
  * val_mse_mean — MSE of the deterministic conditional-mean predictions
    (predict_mode="mean") vs noisy targets, every --val-mse-every epochs.
    The GRU/RSSM-comparable number; selects the best checkpoint (the
    diffusion loss is NOT a reliable model-selection signal — cf. the RSSM
    best-by-ELBO bug).
  * val_mse_sample — same but for the K-step ODE samples
    (predict_mode="sample").  Rises as the model learns to generate
    distribution-typical observations (including a noise realisation);
    tracked as a generative-quality signal, not for selection.

Usage
-----
    # Smoke test (2 epochs, single worker)
    python scripts/train_dit.py --n-epochs 2 --num-workers 0 --run-name smoke_test

    # Full run with defaults
    python scripts/train_dit.py --run-name dit_baseline

    # Override hyperparameters
    python scripts/train_dit.py --d-model 192 --n-layers 6 --window 32
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import torch
from tqdm import tqdm

from pim.world_models.dataloader import build_dataloaders
from pim.world_models.dit import (
    DiTModel,
    ModelConfig,
    SingleFrameConfig,
    SingleFrameDiTModel,
)

# ── Configs ───────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    # Data
    dataset_path: str = "datasets/4_fixed_refl_inview/train.h5"
    val_fraction: float = 0.1
    # Optimization
    n_epochs: int = 150
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 500
    lr_min_factor: float = 0.1  # cosine decays to lr * this
    grad_clip: float = 1.0
    p_clean: float = 0.3  # per-position probability of a clean (τ=0) token
    p_one: float = 0.1  # per-position probability of a τ=1 token (mean readout)
    # Validation
    val_mse_every: int = 5  # epochs between sampled next-step MSE evals
    val_mse_samples: int = 2048  # val subset size for the sampled MSE
    # System
    num_workers: int = 4  # ignored when in_memory (forced to 0)
    in_memory: bool = True  # cache the dataset in RAM (~2 GB; 30× faster)
    device: str = "auto"  # "auto" → cuda > mps > cpu
    seed: int = 0
    # Output
    run_dir: str = "runs/dit"
    run_name: str = ""  # auto-generated from timestamp if empty
    # Model variant: "concat" = paired-frame tokens (DiTModel),
    # "single_frame" = vanilla diffusion forcing (SingleFrameDiTModel)
    variant: str = "concat"
    # Model hyperparameters (forwarded to ModelConfig)
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: float = 4.0
    window: int = 16
    n_sample_steps: int = 8
    noise_seed: int = 0


# ── CLI parsing ───────────────────────────────────────────────────────────────


def _parse_args() -> TrainConfig:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description="Train DiT implicit world model")
    for f in dataclasses.fields(TrainConfig):
        arg = "--" + f.name.replace("_", "-")
        default = getattr(defaults, f.name)
        if isinstance(default, bool):
            p.add_argument(arg, action=argparse.BooleanOptionalAction, default=default)
        else:
            p.add_argument(arg, type=type(default), default=default)
    a = p.parse_args()
    return TrainConfig(**vars(a))


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
    name = tcfg.run_name or f"dit_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(tcfg.run_dir) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _lr_lambda(step: int, total_steps: int, tcfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to lr_min_factor."""
    if step < tcfg.warmup_steps:
        return (step + 1) / tcfg.warmup_steps
    progress = (step - tcfg.warmup_steps) / max(1, total_steps - tcfg.warmup_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return tcfg.lr_min_factor + (1.0 - tcfg.lr_min_factor) * cos


# ── Train / val loops ─────────────────────────────────────────────────────────


def _run_epoch(
    model: DiTModel,
    loader,
    device: torch.device,
    tcfg: TrainConfig,
    optimizer: torch.optim.Optimizer | None,
    scheduler=None,
    batch_bar: tqdm | None = None,
) -> float:
    """One epoch of the flow-matching loss.  optimizer=None → validation."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            obs = batch["obs_intensity"].to(device)  # (B, T, R)
            loss = model.diffusion_loss(obs, p_clean=tcfg.p_clean, p_one=tcfg.p_one)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if batch_bar is not None:
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")
                batch_bar.update(1)

    return total_loss / n_batches


@torch.no_grad()
def _next_step_mse(
    model: DiTModel,
    loader,
    device: torch.device,
    max_samples: int,
    mode: str,
) -> float:
    """MSE of deterministic next-step predictions vs noisy targets.

    Measures what the chosen predict_mode actually produces, not the
    denoising objective.  mode="mean" is the GRU/RSSM-comparable metric.
    """
    model.eval()
    prev_mode = model.predict_mode
    model.predict_mode = mode
    total_se = 0.0
    total_n = 0
    try:
        for batch in loader:
            obs = batch["obs_intensity"].to(device)
            pred, _ = model(obs)  # (B, T-1, R)
            total_se += ((pred - obs[:, 1:, :]) ** 2).sum().item()
            total_n += pred.numel()
            if total_n >= max_samples * pred.shape[1] * pred.shape[2]:
                break
    finally:
        model.predict_mode = prev_mode
    return total_se / total_n


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
        in_memory=tcfg.in_memory,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    if tcfg.variant not in ("concat", "single_frame"):
        raise ValueError(f"unknown variant: {tcfg.variant!r}")
    cfg_cls = ModelConfig if tcfg.variant == "concat" else SingleFrameConfig
    model_cls = DiTModel if tcfg.variant == "concat" else SingleFrameDiTModel
    mcfg = cfg_cls(
        input_dim=obs_res,
        d_model=tcfg.d_model,
        n_layers=tcfg.n_layers,
        n_heads=tcfg.n_heads,
        mlp_ratio=tcfg.mlp_ratio,
        window=tcfg.window,
        n_sample_steps=tcfg.n_sample_steps,
        noise_seed=tcfg.noise_seed,
    )
    model = model_cls(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay
    )
    total_steps = len(train_loader) * tcfg.n_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, total_steps, tcfg)
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
    best_val_mse = float("inf")

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
                    model, train_loader, device, tcfg, optimizer, scheduler, batch_bar
                )
                batch_bar.set_description(f"epoch {epoch} [val]")
                val_loss = _run_epoch(
                    model, val_loader, device, tcfg, optimizer=None, batch_bar=batch_bar
                )

            # Deterministic next-step MSE — selection (mean) + tracking (sample)
            val_mse_mean = val_mse_sample = None
            if epoch % tcfg.val_mse_every == 0 or epoch == 1 or epoch == tcfg.n_epochs:
                val_mse_mean = _next_step_mse(
                    model, val_loader, device, tcfg.val_mse_samples, mode="mean"
                )
                val_mse_sample = _next_step_mse(
                    model, val_loader, device, tcfg.val_mse_samples, mode="sample"
                )

            epoch_bar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                mse=f"{val_mse_mean:.4f}" if val_mse_mean is not None else "—",
            )

            # ── Logging ───────────────────────────────────────────────────
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mse_mean": val_mse_mean,
                "val_mse_sample": val_mse_sample,
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
                # val_loss key = the comparable metric (mean-mode MSE when
                # available), matching what load_checkpoint reports.
                "val_loss": val_mse_mean if val_mse_mean is not None else val_loss,
            }
            torch.save(ckpt, latest_path)

            if val_mse_mean is not None and val_mse_mean < best_val_mse:
                best_val_mse = val_mse_mean
                torch.save(ckpt, best_path)

    print(f"\nDone.  Best val next-step MSE: {best_val_mse:.4f}")
    print(f"Checkpoints: {run_dir}")


if __name__ == "__main__":
    main()
