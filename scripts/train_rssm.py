"""Train an RSSM implicit world model on 1D observation sequences.

Self-supervised with a standard ELBO:
    loss = MSE(decoder(h_t, s_t), o_t) + kl_weight * KL(q(s_t|h_t,e_t) || p(s_t|h_t))

No position or velocity supervision — all structure in the latent must emerge
from the predictive and reconstruction pressures alone.

Dataset paths
-------------
Pass either:
  --dataset-dir  <dir>    look for <dir>/train.h5 and <dir>/val.h5
  --train-h5 <path> --val-h5 <path>   explicit paths (overrides --dataset-dir)

Usage
-----
    # Smoke test (2 epochs, single worker)
    python scripts/train_rssm.py \\
        --dataset-dir datasets/my_dataset \\
        --n-epochs 2 --num-workers 0 --run-name smoke_rssm

    # Full run
    python scripts/train_rssm.py \\
        --dataset-dir datasets/my_dataset \\
        --run-name rssm_baseline

    # Explicit H5 paths
    python scripts/train_rssm.py \\
        --train-h5 datasets/my_dataset/train.h5 \\
        --val-h5   datasets/my_dataset/val.h5 \\
        --run-name rssm_baseline

    # Override hyperparameters
    python scripts/train_rssm.py \\
        --dataset-dir datasets/my_dataset \\
        --det-size 256 --stoch-size 64 --kl-weight 0.5
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader
from tqdm import tqdm

from pim.world_models.rssm import ModelConfig, RSSMModel

# ── Configs ───────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    # Data — set via CLI; exactly one of (dataset_dir) or (train_h5 + val_h5) must be given
    dataset_dir: str = ""
    train_h5: str = ""
    val_h5: str = ""
    # Optimization
    n_epochs: int = 100
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    kl_weight: float = 1.0  # β in β-VAE / ELBO
    kl_warmup_epochs: int = 10  # linearly ramp kl_weight from 0 over this many epochs
    free_nats: float = 3.0  # minimum KL per timestep (PlaNet/Dreamer default); 0 = disabled
    kl_balance_alpha: float = 0.0  # DreamerV2 KL balancing (0 = off, 0.8 = Dreamer default)
    # System
    num_workers: int = 4
    device: str = "auto"
    seed: int = 0
    # Output
    run_dir: str = "runs"
    run_name: str = ""
    # Model hyperparameters (forwarded to ModelConfig)
    embed_dim: int = 128
    det_size: int = 200
    stoch_size: int = 30
    hidden_dim: int = 200


# ── CLI parsing ───────────────────────────────────────────────────────────────


def _parse_args() -> TrainConfig:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description="Train RSSM implicit world model")

    # Data
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--dataset-dir",
        default="",
        dest="dataset_dir",
        help="Directory containing train.h5 and val.h5",
    )
    p.add_argument(
        "--train-h5",
        default="",
        dest="train_h5",
        help="Explicit path to training HDF5 (used together with --val-h5)",
    )
    p.add_argument(
        "--val-h5",
        default="",
        dest="val_h5",
        help="Explicit path to validation HDF5 (used together with --train-h5)",
    )

    # Optimization
    p.add_argument("--n-epochs", type=int, default=defaults.n_epochs)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    p.add_argument(
        "--kl-weight",
        type=float,
        default=defaults.kl_weight,
        help="KL divergence weight β (default 1.0 = standard ELBO)",
    )
    p.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=defaults.kl_warmup_epochs,
        help="Linearly ramp kl_weight from 0 over this many epochs (0 = disabled)",
    )
    p.add_argument(
        "--free-nats",
        type=float,
        default=defaults.free_nats,
        help="Minimum KL per timestep before penalty kicks in; 0 = disabled (PlaNet default: 3.0)",
    )
    p.add_argument(
        "--kl-balance-alpha",
        type=float,
        default=defaults.kl_balance_alpha,
        help="DreamerV2 KL balancing: alpha*KL(sg(q)||p) + (1-alpha)*KL(q||sg(p)); 0 = off (Dreamer default: 0.8)",
    )

    # System
    p.add_argument(
        "--num-workers", type=int, default=defaults.num_workers, help="(Unused)"
    )
    p.add_argument("--device", default=defaults.device)
    p.add_argument("--seed", type=int, default=defaults.seed)

    # Output
    p.add_argument("--run-dir", default=defaults.run_dir)
    p.add_argument("--run-name", default=defaults.run_name)

    # Model
    p.add_argument("--embed-dim", type=int, default=defaults.embed_dim)
    p.add_argument("--det-size", type=int, default=defaults.det_size)
    p.add_argument("--stoch-size", type=int, default=defaults.stoch_size)
    p.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)

    a = p.parse_args()

    # Validate dataset args
    if a.dataset_dir:
        train_h5 = str(Path(a.dataset_dir) / "train.h5")
        val_h5 = str(Path(a.dataset_dir) / "val.h5")
        for p_h5 in [train_h5, val_h5]:
            if not Path(p_h5).exists():
                raise FileNotFoundError(
                    f"Expected {p_h5} inside --dataset-dir {a.dataset_dir}. "
                    "Pass --train-h5 / --val-h5 explicitly if the files have different names."
                )
    elif a.train_h5 and a.val_h5:
        train_h5 = a.train_h5
        val_h5 = a.val_h5
    else:
        p.error(
            "Provide either --dataset-dir (containing train.h5 and val.h5) "
            "or both --train-h5 and --val-h5."
        )

    return TrainConfig(
        dataset_dir=a.dataset_dir,
        train_h5=train_h5,
        val_h5=val_h5,
        n_epochs=a.n_epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        weight_decay=a.weight_decay,
        kl_weight=a.kl_weight,
        kl_warmup_epochs=a.kl_warmup_epochs,
        free_nats=a.free_nats,
        kl_balance_alpha=a.kl_balance_alpha,
        num_workers=a.num_workers,
        device=a.device,
        seed=a.seed,
        run_dir=a.run_dir,
        run_name=a.run_name,
        embed_dim=a.embed_dim,
        det_size=a.det_size,
        stoch_size=a.stoch_size,
        hidden_dim=a.hidden_dim,
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


def _read_n_samples(h5_path: str) -> int:
    with h5py.File(h5_path, "r") as f:
        return f["obs_intensity"].shape[0]


def _make_run_dir(tcfg: TrainConfig) -> Path:
    # name = tcfg.run_name or f"rssm_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(tcfg.run_dir)  # / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _effective_kl_weight(tcfg: TrainConfig, epoch: int) -> float:
    """KL weight with optional linear warm-up."""
    if tcfg.kl_warmup_epochs <= 0:
        return tcfg.kl_weight
    frac = min(1.0, epoch / tcfg.kl_warmup_epochs)
    return frac * tcfg.kl_weight


class _RamDataset(torch.utils.data.Dataset):
    """Entire dataset pre-loaded into RAM as a float32 tensor.

    Eliminates per-sample HDF5 reads, which are the dominant bottleneck when
    the dataset fits in memory.  Workers are unnecessary with in-RAM data, so
    the loader is always created with num_workers=0.
    """

    def __init__(self, h5_path: str) -> None:
        with h5py.File(h5_path, "r") as f:
            self._data = torch.from_numpy(f["obs_intensity"][:].astype(np.float32))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"obs_intensity": self._data[i]}


def _make_loader(h5_path: str, tcfg: TrainConfig, shuffle: bool) -> DataLoader:
    ds = _RamDataset(h5_path)
    return DataLoader(
        ds,
        batch_size=tcfg.batch_size,
        shuffle=shuffle,
        num_workers=0,  # data is already in RAM — workers only add fork overhead
        pin_memory=True,
    )


# ── Train / val loops ─────────────────────────────────────────────────────────


def _run_epoch(
    model: RSSMModel,
    loader: DataLoader,
    device: torch.device,
    kl_w: float,
    optimizer: torch.optim.Optimizer | None,
    batch_bar: tqdm | None = None,
    free_nats: float = 0.0,
    kl_balance_alpha: float = 0.0,
) -> tuple[float, float, float]:
    """Run one epoch.  Pass optimizer=None for validation (no-grad).

    Returns
    -------
    (total_loss, recon_loss, kl_loss) — averaged over batches
    """
    training = optimizer is not None
    model.train(training)
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            obs = batch["obs_intensity"].to(device)  # (B, T, R)

            if kl_balance_alpha > 0.0:
                # DreamerV2 KL balancing: train prior harder, regularise posterior less.
                # alpha fraction of gradient flows only to the prior (posterior stopped);
                # (1-alpha) fraction flows only to the posterior (prior stopped).
                recons, p_mu, p_std, q_mu, q_std = model._forward_with_dists(obs)
                kl_prior = kl_divergence(
                    Normal(q_mu.detach(), q_std.detach()), Normal(p_mu, p_std)
                ).sum(-1)  # (B, T) — trains prior only
                kl_post = kl_divergence(
                    Normal(q_mu, q_std), Normal(p_mu.detach(), p_std.detach())
                ).sum(-1)  # (B, T) — trains posterior only
                kl_terms = kl_balance_alpha * kl_prior + (1.0 - kl_balance_alpha) * kl_post
            else:
                recons, kl_terms = model(obs)  # (B, T, R), (B, T)

            recon_loss = F.mse_loss(recons, obs)
            # Free nats: clamp per-(batch, timestep) KL before averaging so the
            # gradient vanishes once the stochastic component is informative enough.
            if free_nats > 0.0:
                kl_terms = torch.clamp(kl_terms, min=free_nats)
            kl_loss = kl_terms.mean()
            loss = recon_loss + kl_w * kl_loss

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss_sum += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            n_batches += 1

            if batch_bar is not None:
                batch_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    recon=f"{recon_loss.item():.4f}",
                    kl=f"{kl_loss.item():.4f}",
                )
                batch_bar.update(1)

    return (
        total_loss_sum / n_batches,
        recon_loss_sum / n_batches,
        kl_loss_sum / n_batches,
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    tcfg = _parse_args()
    device = _resolve_device(tcfg.device)

    torch.manual_seed(tcfg.seed)

    # ── Dataset ───────────────────────────────────────────────────────────
    obs_res = _read_obs_res(tcfg.train_h5)
    train_loader = _make_loader(tcfg.train_h5, tcfg, shuffle=True)
    val_loader = _make_loader(tcfg.val_h5, tcfg, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    mcfg = ModelConfig(
        input_dim=obs_res,
        embed_dim=tcfg.embed_dim,
        det_size=tcfg.det_size,
        stoch_size=tcfg.stoch_size,
        hidden_dim=tcfg.hidden_dim,
    )
    model = RSSMModel(mcfg).to(device)
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

    n_train = _read_n_samples(tcfg.train_h5)
    n_val = _read_n_samples(tcfg.val_h5)

    print(f"Run dir  : {run_dir}")
    print(f"Device   : {device}")
    print(
        f"Model    : {n_params:,} parameters  "
        f"(det={mcfg.det_size} stoch={mcfg.stoch_size} hidden={mcfg.hidden_dim})"
    )
    print(f"Train    : {n_train:,} samples  ({tcfg.train_h5})")
    print(f"Val      : {n_val:,} samples  ({tcfg.val_h5})")
    print(f"KL weight: {tcfg.kl_weight}  warm-up={tcfg.kl_warmup_epochs} epochs  free_nats={tcfg.free_nats}  kl_balance_alpha={tcfg.kl_balance_alpha}")
    print()

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    with tqdm(range(1, tcfg.n_epochs + 1), desc="epochs", unit="epoch") as epoch_bar:
        for epoch in epoch_bar:
            kl_w = _effective_kl_weight(tcfg, epoch)

            with tqdm(
                total=n_train_batches + n_val_batches,
                desc=f"epoch {epoch}",
                unit="batch",
                leave=False,
            ) as batch_bar:
                train_loss, train_recon, train_kl = _run_epoch(
                    model, train_loader, device, kl_w, optimizer, batch_bar,
                    free_nats=tcfg.free_nats, kl_balance_alpha=tcfg.kl_balance_alpha,
                )
                batch_bar.set_description(f"epoch {epoch} [val]")
                val_loss, val_recon, val_kl = _run_epoch(
                    model, val_loader, device, kl_w, optimizer=None, batch_bar=batch_bar,
                    free_nats=tcfg.free_nats, kl_balance_alpha=tcfg.kl_balance_alpha,
                )

            epoch_bar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                kl_w=f"{kl_w:.3f}",
            )

            # ── Logging ───────────────────────────────────────────────────
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "recon_loss": train_recon,
                "val_recon_loss": val_recon,
                "kl_loss": train_kl,
                "val_kl_loss": val_kl,
                "kl_weight": kl_w,
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
                "val_recon_loss": val_recon,
                "val_kl_loss": val_kl,
            }
            torch.save(ckpt, latest_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt, best_path)

    print(f"\nDone.  Best val ELBO loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {run_dir}")


if __name__ == "__main__":
    main()
