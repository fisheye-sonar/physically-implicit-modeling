"""Train a continuous-action GRU (or a passive control) on a continuous-action dataset.

Standalone / decoupled so each model trains in one foreground call (WORKER.md).
Mirrors the Exp-2 in-notebook trainer: in-memory, teacher-forced next-step MSE,
AdamW, val split = seed-0 permutation, 0.1 fraction (byte-identical protocol to the
dataset-4 baseline run 7 and Exp-2 runs 8/9).

Usage
-----
    # action-conditioned model
    python scripts/train_action_gru_continuous.py \
        --dataset datasets/6_cont_dxdy/train.h5 --run-name M_dxdy --use-actions

    # perturbed-passive control (same trajectories, action channel withheld)
    python scripts/train_action_gru_continuous.py \
        --dataset datasets/7_cont_teleport/train.h5 --run-name M_teleport_ctrl
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pim.world_models.action_gru_continuous import (  # noqa: E402
    ActionContinuousModelConfig,
    ActionGRUContinuousModel,
)
from pim.world_models.gru import GRUModel, ModelConfig  # noqa: E402


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--run-dir", default="runs/gru")
    p.add_argument("--use-actions", action="store_true")
    p.add_argument("--n-epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--n-obj", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=-1, help="limit samples (-1 = all)")
    return p.parse_args()


def _epoch(model, idx, OBS, ACT, opt, use_actions, device, bs=256):
    training = opt is not None
    model.train(training)
    if training:
        idx = idx[torch.randperm(len(idx))]
    tot, nb = 0.0, 0
    with (torch.enable_grad() if training else torch.no_grad()):
        for i in range(0, len(idx), bs):
            bi = idx[i:i + bs]
            obs = OBS[bi].to(device)
            if use_actions:
                pred = model(obs, actions=ACT[bi].to(device))[0]
            else:
                pred = model(obs)[0]
            loss = F.mse_loss(pred, obs[:, 1:, :])
            if training:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += loss.item()
            nb += 1
    return tot / nb


def main():
    a = _parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    run_dir = Path(a.run_dir) / a.run_name
    ckpt = run_dir / "best_model.pt"
    if ckpt.exists():
        print(f"{a.run_name}: checkpoint already exists at {ckpt} — skipping.")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(a.dataset, "r") as f:
        n_all = f["obs_intensity"].shape[0]
        n = n_all if a.n_samples < 0 else min(a.n_samples, n_all)
        OBS = f["obs_intensity"][:n].astype(np.float32)
        obs_res = OBS.shape[2]
        ACT = f["actions"][:n, :, :a.n_obj, :].astype(np.float32) if a.use_actions else None
    OBS_t = torch.from_numpy(OBS)
    ACT_t = torch.from_numpy(ACT) if ACT is not None else None

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(n)
    nval = int(a.val_fraction * n)
    val_idx = torch.as_tensor(perm[:nval])
    tr_idx = torch.as_tensor(perm[nval:])

    if a.use_actions:
        mcfg = ActionContinuousModelConfig(
            input_dim=obs_res, hidden_size=a.hidden_size, n_obj=a.n_obj
        )
        model = ActionGRUContinuousModel(mcfg).to(device)
    else:
        mcfg = ModelConfig(input_dim=obs_res, hidden_size=a.hidden_size)
        model = GRUModel(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    best = float("inf")
    t0 = time.perf_counter()
    print(f"{a.run_name}: N={n} obs_res={obs_res} use_actions={a.use_actions} "
          f"params={n_params:,} device={device}")
    for ep in tqdm(range(1, a.n_epochs + 1), desc=a.run_name):
        _epoch(model, tr_idx, OBS_t, ACT_t, opt, a.use_actions, device, bs=a.batch_size)
        va = _epoch(model, val_idx, OBS_t, ACT_t, None, a.use_actions, device, bs=a.batch_size)
        if va < best:
            best = va
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "model_config": asdict(mcfg), "val_loss": va,
                        "use_actions": a.use_actions}, ckpt)
    json.dump({"run_name": a.run_name, "dataset": a.dataset, "use_actions": a.use_actions,
               "model_config": asdict(mcfg), "n_epochs": a.n_epochs, "n_samples": n,
               "best_val": best, "n_params": n_params},
              open(run_dir / "config.json", "w"), indent=2)
    print(f"{a.run_name}: {a.n_epochs} ep in {(time.perf_counter() - t0) / 60:.1f} min | "
          f"best val {best:.5f} | saved {ckpt}")


if __name__ == "__main__":
    main()
