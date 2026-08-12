"""Train a latent DiT world model on top of a frozen, pre-trained observation VAE.

The VAE (`scripts/train_vae.py`) is loaded, frozen, and embedded in the
checkpoint, so a latent-DiT checkpoint is self-contained and
`load_checkpoint` returns a working observation-space world model.

Validation metrics
------------------
  * `val_loss`         — flow-matching loss in latent space (train-comparable);
  * `val_mse_mean`     — MSE of **decoded** mean-mode predictions vs the noisy
    observations. The number comparable to every other architecture in the repo
    (GRU 0.02362, pixel DiT concat W4 d256 0.02445), and what selects the best
    checkpoint. NB it is floored by the VAE's own reconstruction error, so a
    latent model cannot beat a pixel model on it by more than the VAE allows —
    report the VAE's recon RMSE alongside.
  * `val_mse_clean`    — same predictions vs the CLEAN render. A latent model
    partially denoises, so this is the fairer structure metric across the
    architecture boundary.
  * `val_mse_sample`   — decoded fresh-noise Euler samples ("sample_fresh") vs
    noisy targets: a generative-quality signal, never used for selection.

Usage
-----
    python scripts/train_latent_dit.py --vae-checkpoint runs/vae/vae_z16/best_model.pt \
        --window 4 --run-name latent_dit_z16_w4
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
import numpy as np
import torch
from tqdm import tqdm

from pim.simulator.dataset import reconstruct_clean_obs
from pim.world_models.dataloader import build_dataloaders
from pim.world_models.latent_dit import LatentDiTConfig, LatentDiTModel


@dataclass
class TrainConfig:
    dataset_path: str = "datasets/4_fixed_refl_inview/train.h5"
    vae_checkpoint: str = "runs/vae/vae_z16/best_model.pt"
    val_fraction: float = 0.1
    n_epochs: int = 150
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 500
    lr_min_factor: float = 0.1
    grad_clip: float = 1.0
    p_clean: float = 0.3
    p_one: float = 0.1
    val_mse_every: int = 5
    val_mse_batches: int = 8
    num_workers: int = 0
    in_memory: bool = True
    device: str = "auto"
    seed: int = 0
    run_dir: str = "runs/latent_dit"
    run_name: str = ""
    # DiT core hyperparameters
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: float = 4.0
    window: int = 4
    n_sample_steps: int = 8
    noise_seed: int = 0


def _parse_args() -> TrainConfig:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description="Train latent DiT world model")
    for f in dataclasses.fields(TrainConfig):
        default = getattr(defaults, f.name)
        arg = "--" + f.name.replace("_", "-")
        if isinstance(default, bool):
            p.add_argument(arg, action=argparse.BooleanOptionalAction, default=default)
        else:
            p.add_argument(arg, type=type(default), default=default)
    return TrainConfig(**vars(p.parse_args()))


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _lr_lambda(step: int, total: int, t: TrainConfig) -> float:
    if step < t.warmup_steps:
        return (step + 1) / t.warmup_steps
    prog = (step - t.warmup_steps) / max(1, total - t.warmup_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return t.lr_min_factor + (1.0 - t.lr_min_factor) * cos


def main() -> None:
    tcfg = _parse_args()
    device = _resolve_device(tcfg.device)
    torch.manual_seed(tcfg.seed)

    train_loader, val_loader = build_dataloaders(
        tcfg.dataset_path,
        val_fraction=tcfg.val_fraction,
        batch_size=tcfg.batch_size,
        seed=tcfg.seed,
        num_workers=tcfg.num_workers,
        in_memory=tcfg.in_memory,
    )

    # ── frozen VAE + latent DiT core ──────────────────────────────────────
    vae_ck = torch.load(tcfg.vae_checkpoint, map_location="cpu")
    vae_cfg = vae_ck["model_config"]
    mcfg = LatentDiTConfig(
        vae=vae_cfg,
        core=dict(
            d_model=tcfg.d_model,
            n_layers=tcfg.n_layers,
            n_heads=tcfg.n_heads,
            mlp_ratio=tcfg.mlp_ratio,
            window=tcfg.window,
            n_sample_steps=tcfg.n_sample_steps,
            noise_seed=tcfg.noise_seed,
        ),
        vae_checkpoint=str(tcfg.vae_checkpoint),
    )
    model = LatentDiTModel(mcfg).to(device)
    model.vae.load_state_dict(
        {k: v.to(device) for k, v in vae_ck["model_state"].items()}
    )
    model.vae.requires_grad_(False)
    model.vae.eval()
    n_core = sum(p.numel() for p in model.core.parameters())
    n_vae = sum(p.numel() for p in model.vae.parameters())

    opt = torch.optim.AdamW(
        model.core.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay
    )
    total_steps = len(train_loader) * tcfg.n_epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_lambda(s, total_steps, tcfg)
    )

    run_dir = Path(tcfg.run_dir) / (
        tcfg.run_name or f"latent_dit_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train": dataclasses.asdict(tcfg),
                "model": dataclasses.asdict(mcfg),
                "n_params_core": n_core,
                "n_params_vae_frozen": n_vae,
                "vae_val_recon_rmse_noisy": vae_ck.get("val_recon_rmse_noisy"),
                "vae_val_recon_rmse_clean": vae_ck.get("val_recon_rmse_clean"),
                "latent_scale": vae_cfg.get("latent_scale"),
                "device": str(device),
            },
            indent=2,
        )
    )
    metrics_path = run_dir / "metrics.jsonl"
    print(f"Run dir : {run_dir}\nDevice  : {device}")
    print(
        f"Model   : core {n_core:,} trainable + VAE {n_vae:,} frozen "
        f"(z={model.latent_dim}, latent_scale={vae_cfg.get('latent_scale'):.3f})"
    )
    print(
        f"VAE recon RMSE — noisy {vae_ck.get('val_recon_rmse_noisy'):.4f} / "
        f"clean {vae_ck.get('val_recon_rmse_clean'):.4f}"
    )
    print(
        f"Data    : {len(train_loader.dataset):,} train / {len(val_loader.dataset):,} val\n"
    )

    # clean renders for the val split, to score decoded predictions against
    with h5py.File(tcfg.dataset_path, "r") as f:
        val_idx = np.asarray(val_loader.dataset.indices)
        order = np.argsort(val_idx)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        if "obs_clean" in f:
            clean_val = f["obs_clean"][val_idx[order]].astype(np.float32)[inv]
        else:
            clean_val = reconstruct_clean_obs(
                f["obs_id"][val_idx[order]].astype(np.int8)[inv],
                f["reflectivities"][val_idx[order]].astype(np.float32)[inv],
            )
    clean_val_t = torch.from_numpy(np.asarray(clean_val))

    @torch.no_grad()
    def val_mse(mode: str) -> tuple[float, float]:
        """(MSE vs noisy targets, MSE vs clean targets) for decoded predictions."""
        model.eval()
        prev = model.predict_mode
        model.predict_mode = mode
        if mode == "sample_fresh":
            model.noise_gen = torch.Generator().manual_seed(1234)
        se_n = se_c = n = 0.0
        try:
            for bi, batch in enumerate(val_loader):
                if bi >= tcfg.val_mse_batches:
                    break
                obs = batch["obs_intensity"].to(device)
                cl = clean_val_t[
                    bi * tcfg.batch_size : bi * tcfg.batch_size + obs.shape[0]
                ].to(device)
                pred, _ = model(obs)
                se_n += float(((pred - obs[:, 1:]) ** 2).sum())
                se_c += float(((pred - cl[:, 1:]) ** 2).sum())
                n += pred.numel()
        finally:
            model.predict_mode = prev
            model.noise_gen = None
        return se_n / n, se_c / n

    def run_epoch(loader, train: bool) -> float:
        model.train(train)
        tot, nb = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                obs = batch["obs_intensity"].to(device)
                loss = model.diffusion_loss(obs, p_clean=tcfg.p_clean, p_one=tcfg.p_one)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.core.parameters(), tcfg.grad_clip
                    )
                    opt.step()
                    sched.step()
                tot += float(loss.detach())
                nb += 1
        return tot / max(nb, 1)

    best = float("inf")
    with tqdm(range(1, tcfg.n_epochs + 1), desc="epochs", unit="epoch") as bar:
        for epoch in bar:
            train_loss = run_epoch(train_loader, True)
            val_loss = run_epoch(val_loader, False)
            mse_mean = mse_clean = mse_sample = None
            if epoch % tcfg.val_mse_every == 0 or epoch in (1, tcfg.n_epochs):
                mse_mean, mse_clean = val_mse("mean")
                mse_sample, _ = val_mse("sample_fresh")
            bar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                mse=f"{mse_mean:.4f}" if mse_mean is not None else "—",
            )
            rec = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mse_mean": mse_mean,
                "val_mse_clean": mse_clean,
                "val_mse_sample": mse_sample,
                "lr": opt.param_groups[0]["lr"],
            }
            with open(metrics_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "train_config": dataclasses.asdict(tcfg),
                "model_config": dataclasses.asdict(mcfg),
                "val_loss": mse_mean if mse_mean is not None else val_loss,
            }
            torch.save(ckpt, run_dir / "latest.pt")
            if mse_mean is not None and mse_mean < best:
                best = mse_mean
                torch.save(ckpt, run_dir / "best_model.pt")

    print(f"\nDone.  Best decoded next-step MSE (mean mode, vs noisy): {best:.5f}")
    print(f"Checkpoints: {run_dir}")


if __name__ == "__main__":
    main()
