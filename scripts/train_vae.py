"""Train the per-frame observation VAE that the latent DiT runs inside.

Frames are treated as i.i.d. samples (the VAE carries no temporal information —
see `pim/world_models/vae.py`), so the whole train split is flattened to
(N·T, R) and shuffled.

Reported metrics — quote BOTH reconstruction numbers:
  * `val_recon_rmse_noisy` — vs the noisy input it was trained on;
  * `val_recon_rmse_clean` — vs the simulator's clean render. A tight latent
    partially denoises, so this is typically the *better* number and is the one
    that says whether the code kept the world state.
  * `val_kl` — tracked, not optimised hard (LDM-style tiny weight).

After training, `fit_latent_scale` measures the posterior-mean std and stores it
in the checkpoint config as `latent_scale`, so downstream models get ≈unit-scale
latents without re-deriving it.

Usage
-----
    python scripts/train_vae.py --run-name vae_z16
    python scripts/train_vae.py --latent-dim 8 --run-name vae_z8
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from pim.simulator.dataset import reconstruct_clean_obs
from pim.world_models.vae import ObsVAE, VAEConfig, fit_latent_scale


@dataclass
class TrainConfig:
    dataset_path: str = "datasets/4_fixed_refl_inview/train.h5"
    val_fraction: float = 0.05
    n_epochs: int = 12
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 0
    run_dir: str = "runs/vae"
    run_name: str = ""
    # model
    latent_dim: int = 16
    hidden: int = 256
    n_layers: int = 2
    kl_weight: float = 1e-6


def _parse_args() -> TrainConfig:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description="Train the per-frame observation VAE")
    for f in dataclasses.fields(TrainConfig):
        p.add_argument(
            "--" + f.name.replace("_", "-"),
            type=type(getattr(defaults, f.name)),
            default=getattr(defaults, f.name),
        )
    return TrainConfig(**vars(p.parse_args()))


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def main() -> None:
    tcfg = _parse_args()
    device = _resolve_device(tcfg.device)
    torch.manual_seed(tcfg.seed)
    rng = np.random.default_rng(tcfg.seed)

    # ── data: flatten sequences into frames; keep clean renders for scoring ──
    with h5py.File(tcfg.dataset_path, "r") as f:
        obs = f["obs_intensity"][:].astype(np.float32)  # (N, T, R)
        obs_id = f["obs_id"][:].astype(np.int8)
        refl = f["reflectivities"][:].astype(np.float32)
        clean = (
            f["obs_clean"][:].astype(np.float32)
            if "obs_clean" in f
            else reconstruct_clean_obs(obs_id, refl)
        )
    N, T, R = obs.shape
    frames = torch.from_numpy(obs.reshape(N * T, R))
    frames_clean = torch.from_numpy(np.asarray(clean).reshape(N * T, R))
    perm = torch.from_numpy(rng.permutation(N * T))
    frames, frames_clean = frames[perm], frames_clean[perm]
    n_val = int(tcfg.val_fraction * len(frames))
    tr, va = frames[n_val:].to(device), frames[:n_val].to(device)
    va_clean = frames_clean[:n_val].to(device)

    mcfg = VAEConfig(
        input_dim=R,
        latent_dim=tcfg.latent_dim,
        hidden=tcfg.hidden,
        n_layers=tcfg.n_layers,
        kl_weight=tcfg.kl_weight,
    )
    model = ObsVAE(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay
    )
    steps_per_epoch = max(1, len(tr) // tcfg.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=tcfg.n_epochs * steps_per_epoch
    )

    run_dir = Path(tcfg.run_dir) / (
        tcfg.run_name or f"vae_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train": dataclasses.asdict(tcfg),
                "model": dataclasses.asdict(mcfg),
                "n_params": n_params,
                "device": str(device),
            },
            indent=2,
        )
    )
    metrics_path = run_dir / "metrics.jsonl"
    print(
        f"Run dir : {run_dir}\nDevice  : {device}\nModel   : {n_params:,} params "
        f"(z={tcfg.latent_dim})\nFrames  : {len(tr):,} train / {len(va):,} val\n"
    )

    @torch.no_grad()
    def evaluate() -> dict:
        model.eval()
        out = model.forward(va, sample=False)  # deterministic (posterior mean)
        kl = -0.5 * (1 + out.logvar - out.mu.pow(2) - out.logvar.exp()).sum(-1).mean()
        return {
            "val_recon_rmse_noisy": float(((out.recon - va) ** 2).mean().sqrt()),
            "val_recon_rmse_clean": float(((out.recon - va_clean) ** 2).mean().sqrt()),
            "val_kl": float(kl),
        }

    best = float("inf")
    with tqdm(range(1, tcfg.n_epochs + 1), desc="epochs", unit="epoch") as bar:
        for epoch in bar:
            model.train()
            idx = torch.randperm(len(tr), device=device)
            tot, nb = 0.0, 0
            for i in range(0, len(tr) - tcfg.batch_size + 1, tcfg.batch_size):
                batch = tr[idx[i : i + tcfg.batch_size]]
                loss, _ = model.loss(batch)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                opt.step()
                sched.step()
                tot += float(loss.detach())
                nb += 1
            ev = evaluate()
            rec = {
                "epoch": epoch,
                "train_loss": tot / max(nb, 1),
                **ev,
                "lr": opt.param_groups[0]["lr"],
            }
            with open(metrics_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            bar.set_postfix(
                train=f"{rec['train_loss']:.5f}",
                rmse_noisy=f"{ev['val_recon_rmse_noisy']:.4f}",
                rmse_clean=f"{ev['val_recon_rmse_clean']:.4f}",
            )
            if ev["val_recon_rmse_noisy"] < best:
                best = ev["val_recon_rmse_noisy"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "model_config": dataclasses.asdict(mcfg),
                        "train_config": dataclasses.asdict(tcfg),
                        "val_loss": ev["val_recon_rmse_noisy"],
                        **ev,
                    },
                    run_dir / "best_model.pt",
                )

    # ── measure and bake in the latent scale (LDM scale factor) ──
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    scale = fit_latent_scale(model, tr[:200_000])
    model.cfg.latent_scale = scale
    ckpt["model_config"] = dataclasses.asdict(model.cfg)
    ckpt["latent_scale"] = scale
    torch.save(ckpt, run_dir / "best_model.pt")
    print(f"\nBest val recon RMSE (vs noisy): {best:.4f}")
    print(f"Latent scale (posterior-mean std): {scale:.4f}  → stored in checkpoint")
    print(f"Checkpoints: {run_dir}")


if __name__ == "__main__":
    main()
