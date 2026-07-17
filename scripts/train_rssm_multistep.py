"""Train an RSSM with a multi-step latent-overshooting objective (companion to train_rssm.py).

Standard single-step ELBO (recon + 1-step KL) is kept for every setting.  For an
overshoot horizon ``W >= 2`` we ADD PlaNet-style latent overshooting: from each of
``n_start`` randomly-chosen posterior states we imagine ``W`` steps forward through the
PRIOR (no observations), and for each imagined step d = 1..W add

    obs-overshoot : MSE(decode(prior_d), obs[t+d])
    latent-overshoot KL : KL( sg(posterior[t+d]) || prior_d )   (free-nats clamped)

so the multi-step prior is trained to predict future observations and to match the
posterior it would have inferred.  W=1 = the standard ELBO (no overshoot), matching the
refined RSSM objective; the sweep W in {1,2,5} varies ONLY the training horizon.

Starts are subsampled (``n_start``) to keep the per-epoch cost bounded.  Everything
else (KL warm-up, free-nats, recon-based best-checkpoint selection, in-RAM loader)
mirrors train_rssm.py so the comparison is clean.

Usage
-----
    python scripts/train_rssm_multistep.py --dataset-dir datasets/4_fixed_refl_inview \\
        --run-dir runs/rssm_multistep/w2 --run-name w2 --overshoot-w 2 --n-epochs 150
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
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader

from pim.world_models.rssm import ModelConfig, RSSMModel
from pim.world_models.rssm.model import RSSMState


@dataclass
class Cfg:
    dataset_dir: str = "datasets/4_fixed_refl_inview"
    n_epochs: int = 150
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    kl_weight: float = 1.0
    kl_warmup_epochs: int = 10
    free_nats: float = 3.0
    overshoot_w: int = 1          # 1 = pure ELBO; >=2 adds latent overshooting to horizon W
    n_start: int = 8              # subsampled overshoot start indices per sequence
    seed: int = 0
    run_dir: str = "runs/rssm_multistep/run"
    run_name: str = "run"
    embed_dim: int = 128
    det_size: int = 256
    stoch_size: int = 64
    hidden_dim: int = 256


def parse() -> Cfg:
    d = Cfg(); p = argparse.ArgumentParser()
    for f in dataclasses.fields(Cfg):
        p.add_argument("--" + f.name.replace("_", "-"), default=getattr(d, f.name), type=type(getattr(d, f.name)))
    a = p.parse_args()
    return Cfg(**{f.name: getattr(a, f.name) for f in dataclasses.fields(Cfg)})


class RamDS(torch.utils.data.Dataset):
    def __init__(self, h5_path: str):
        with h5py.File(h5_path, "r") as f:
            self.d = torch.from_numpy(f["obs_intensity"][:].astype(np.float32))
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return self.d[i]


def kl_weight_at(cfg: Cfg, epoch: int) -> float:
    if cfg.kl_warmup_epochs <= 0: return cfg.kl_weight
    return min(1.0, epoch / cfg.kl_warmup_epochs) * cfg.kl_weight


def run_epoch(model, loader, device, kl_w, opt, cfg, rng):
    training = opt is not None
    model.train(training)
    tot = dict(loss=0.0, recon=0.0, kl=0.0, os_recon=0.0, os_kl=0.0); nb = 0
    W = cfg.overshoot_w
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for obs in loader:
            obs = obs.to(device)                                   # (B,T,R)
            B, T, R = obs.shape
            state = model._initial_state(B, device)
            recons, kls, post_mu, post_std, states = [], [], [], [], []
            for t in range(T):
                state, prior, posterior = model.observe_step(obs[:, t], state)
                recons.append(model.decode(state))
                kls.append(kl_divergence(posterior, prior).sum(-1))
                post_mu.append(posterior.loc); post_std.append(posterior.scale); states.append(state)
            recon = F.mse_loss(torch.stack(recons, 1), obs)
            kl1 = torch.clamp(torch.stack(kls, 1), min=cfg.free_nats).mean()
            loss = recon + kl_w * kl1
            os_recon_v = torch.zeros((), device=device); os_kl_v = torch.zeros((), device=device)

            if W >= 2:
                n_valid = T - W
                if n_valid > 0:
                    k = min(cfg.n_start, n_valid)
                    starts = rng.choice(n_valid, size=k, replace=False)
                    acc_r = torch.zeros((), device=device); acc_k = torch.zeros((), device=device); cnt = 0
                    for t in starts.tolist():
                        img = RSSMState(states[t].h.detach(), states[t].s.detach())
                        for d in range(1, W + 1):
                            img, prior_d = model.imagine_step(img)
                            acc_r = acc_r + F.mse_loss(model.decode(img), obs[:, t + d])
                            post_d = Normal(post_mu[t + d].detach(), post_std[t + d].detach())
                            kl_d = kl_divergence(post_d, prior_d).sum(-1)
                            acc_k = acc_k + torch.clamp(kl_d, min=cfg.free_nats).mean()
                            cnt += 1
                    os_recon_v = acc_r / cnt; os_kl_v = acc_k / cnt
                    loss = loss + os_recon_v + kl_w * os_kl_v

            if training:
                opt.zero_grad(); loss.backward(); opt.step()
            _f = lambda x: float(x.detach()) if torch.is_tensor(x) else float(x)
            tot["loss"] += _f(loss); tot["recon"] += _f(recon); tot["kl"] += _f(kl1)
            tot["os_recon"] += _f(os_recon_v); tot["os_kl"] += _f(os_kl_v); nb += 1
    return {k: v / nb for k, v in tot.items()}


def main():
    cfg = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    tr = DataLoader(RamDS(f"{cfg.dataset_dir}/train.h5"), batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    va = DataLoader(RamDS(f"{cfg.dataset_dir}/val.h5"), batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    with h5py.File(f"{cfg.dataset_dir}/train.h5", "r") as f:
        obs_res = f["obs_intensity"].shape[2]
    mcfg = ModelConfig(input_dim=obs_res, embed_dim=cfg.embed_dim, det_size=cfg.det_size,
                       stoch_size=cfg.stoch_size, hidden_dim=cfg.hidden_dim)
    model = RSSMModel(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    run_dir = Path(cfg.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"train": dataclasses.asdict(cfg),
        "model": dataclasses.asdict(mcfg), "n_params": sum(p.numel() for p in model.parameters())}, indent=2))
    print(f"RSSM multistep | run={cfg.run_name} overshoot_w={cfg.overshoot_w} n_start={cfg.n_start} "
          f"epochs={cfg.n_epochs} params={sum(p.numel() for p in model.parameters()):,} device={device}", flush=True)
    best = float("inf"); t0 = time.perf_counter()
    for epoch in range(1, cfg.n_epochs + 1):
        kl_w = kl_weight_at(cfg, epoch)
        trm = run_epoch(model, tr, device, kl_w, opt, cfg, rng)
        vam = run_epoch(model, va, device, kl_w, None, cfg, rng)
        rec = {"epoch": epoch, "kl_weight": kl_w, **{f"train_{k}": v for k, v in trm.items()},
               **{f"val_{k}": v for k, v in vam.items()}}
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        ck = {"epoch": epoch, "model_state": model.state_dict(), "model_config": dataclasses.asdict(mcfg),
              "val_recon_loss": vam["recon"], "overshoot_w": cfg.overshoot_w}
        torch.save(ck, run_dir / "latest.pt")
        if vam["recon"] < best:
            best = vam["recon"]; torch.save(ck, run_dir / "best_model.pt")
        if epoch % 10 == 0 or epoch == 1:
            el = time.perf_counter() - t0
            print(f"  ep{epoch:3d}/{cfg.n_epochs} val_recon={vam['recon']:.4f} os_recon={vam['os_recon']:.4f} "
                  f"kl={vam['kl']:.3f} | {el/epoch:.1f}s/ep elapsed={el/60:.1f}m", flush=True)
    (run_dir / "DONE").write_text("done")
    print(f"DONE {cfg.run_name}: best val_recon={best:.4f} total={(time.perf_counter()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
