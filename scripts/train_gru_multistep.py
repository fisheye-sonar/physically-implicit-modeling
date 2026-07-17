"""Train a GRU implicit world model with a MULTI-STEP (free-running rollout) objective.

This is a NEW, standalone training helper for the `multistep-prediction-objective`
research direction. It does NOT modify `scripts/train_gru.py` or any `pim/` module.

Objective (free-running / "overshooting")
-----------------------------------------
Standard `train_gru.py` uses a pure single-step teacher-forcing loss: from the hidden
state after seeing obs[t], predict obs[t+1], MSE, done. Here instead, for a rollout
window `w`, we slide a start index `n` across the whole sequence; at each start the
model is teacher-forced up to obs[n] (giving hidden state h_n) and then FREE-RUNS `w`
steps, feeding its OWN decoded predictions back in (exactly the eval-time
`predict_step` semantics: decode current h -> feed as next obs -> step). All `w`
predicted frames are penalised against the true observations, and BPTT flows through
the entire w-step imagination. The loss is averaged over start indices and offsets.

Concretely, with a single teacher-forced forward pass giving h_seq[:, n] (the state
after seeing obs[n], n = 0..T-2):

    pred_1 = decode(h_seq[:, n])                 # predicts obs[n+1]   (== single-step)
    for j = 2..w:
        h  <- GRU(encode(pred_{j-1}), h)         # feed model's OWN prediction back
        pred_j = decode(h)                        # predicts obs[n+j]
    loss = mean_j mean_n MSE(pred_j, obs[n+j])

`w = 1` reduces EXACTLY to the single-step teacher-forcing objective of
`scripts/train_gru.py`, which is the baseline. Only the objective changes: identical
architecture, data, hidden size, optimizer, and epoch budget.

Usage
-----
    python scripts/train_gru_multistep.py --w 2 --run-name mstep_w2 \
        --dataset-path datasets/4_fixed_refl_inview/train.h5 \
        --n-epochs 400 --hidden-size 256 --run-dir runs/gru_multistep
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

from pim.world_models.gru import GRUModel, ModelConfig


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    dataset_path: str = "datasets/4_fixed_refl_inview/train.h5"
    val_fraction: float = 0.1
    w: int = 2                      # rollout window (free-run steps); w=1 == single-step
    n_epochs: int = 400
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "auto"
    seed: int = 0
    run_dir: str = "runs/gru_multistep"
    run_name: str = ""
    hidden_size: int = 256
    num_layers: int = 1
    dropout: float = 0.0


def _parse_args() -> TrainConfig:
    d = TrainConfig()
    p = argparse.ArgumentParser(description="Train GRU with a multi-step rollout objective")
    p.add_argument("--dataset-path", default=d.dataset_path)
    p.add_argument("--val-fraction", type=float, default=d.val_fraction)
    p.add_argument("--w", type=int, default=d.w)
    p.add_argument("--n-epochs", type=int, default=d.n_epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--device", default=d.device)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--run-dir", default=d.run_dir)
    p.add_argument("--run-name", default=d.run_name)
    p.add_argument("--hidden-size", type=int, default=d.hidden_size)
    p.add_argument("--num-layers", type=int, default=d.num_layers)
    p.add_argument("--dropout", type=float, default=d.dropout)
    a = p.parse_args()
    return TrainConfig(
        dataset_path=a.dataset_path, val_fraction=a.val_fraction, w=a.w,
        n_epochs=a.n_epochs, batch_size=a.batch_size, lr=a.lr,
        weight_decay=a.weight_decay, device=a.device, seed=a.seed,
        run_dir=a.run_dir, run_name=a.run_name, hidden_size=a.hidden_size,
        num_layers=a.num_layers, dropout=a.dropout,
    )


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# ── Multi-step free-running loss ──────────────────────────────────────────────


def multistep_loss(model: GRUModel, obs: torch.Tensor, w: int) -> torch.Tensor:
    """Free-running w-step overshoot MSE, averaged over start indices and offsets.

    obs : (B, T, R). Returns a scalar loss.

    Slides start index n over 0..T-1-w (so obs[n+w] exists). For every start the
    model free-runs w steps from h_n feeding its own predictions back; each of the
    w predictions is compared to the true obs at that offset.
    """
    B, T, R = obs.shape
    S = T - w                              # number of valid start indices (n = 0..T-1-w)

    # Teacher-forced context for every start at once: h_seq[:, n] = state after obs[n].
    x = F.relu(model.encoder(obs[:, :-1, :]))       # (B, T-1, H) encodes obs[0..T-2]
    h_seq, _ = model.gru(x)                          # (B, T-1, H); h_seq[:, n] after obs[n]

    cur = h_seq[:, :S, :].reshape(1, B * S, -1).contiguous()   # (1, B*S, H) hidden states
    # offset j = 1: decode current state -> predicts obs[n+1]  (pure teacher-forced next-step)
    pred = model.decoder(cur.squeeze(0))             # (B*S, R)

    total = obs.new_zeros(())
    for j in range(1, w + 1):
        target = obs[:, j:S + j, :].reshape(B * S, R)          # obs[n+j] for n=0..S-1
        total = total + F.mse_loss(pred, target)
        if j < w:
            # free-run one step: feed the model's OWN prediction back in
            inp = F.relu(model.encoder(pred)).unsqueeze(1)     # (B*S, 1, H)
            out, cur = model.gru(inp, cur)                     # (B*S, 1, H)
            pred = model.decoder(out.squeeze(1))               # (B*S, R)
    return total / w


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    tcfg = _parse_args()
    device = _resolve_device(tcfg.device)
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)

    # ── Data: load fully into memory (fast, matches baseline's in-memory training) ──
    with h5py.File(tcfg.dataset_path, "r") as f:
        obs_all = f["obs_intensity"][:].astype(np.float32)     # (N, T, R)
    N, T, R = obs_all.shape
    assert tcfg.w >= 1 and tcfg.w < T, f"w must be in [1, {T-1}], got {tcfg.w}"

    rng = np.random.default_rng(tcfg.seed)
    perm = rng.permutation(N)
    n_val = max(1, int(N * tcfg.val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    obs_t = torch.from_numpy(obs_all).to(device)               # keep full tensor on GPU
    train_idx_t = torch.from_numpy(train_idx).to(device)
    val_idx_t = torch.from_numpy(val_idx).to(device)

    # ── Model ──
    mcfg = ModelConfig(input_dim=R, hidden_size=tcfg.hidden_size,
                       num_layers=tcfg.num_layers, dropout=tcfg.dropout)
    model = GRUModel(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    run_dir = Path(tcfg.run_dir) / (tcfg.run_name or f"mstep_w{tcfg.w}_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = run_dir / "metrics.jsonl"
    best_path = run_dir / "best_model.pt"
    latest_path = run_dir / "latest.pt"

    (run_dir / "config.json").write_text(json.dumps({
        "train": dataclasses.asdict(tcfg), "model": dataclasses.asdict(mcfg),
        "device": str(device), "n_params": n_params, "objective": "multistep_freerun",
    }, indent=2))

    print(f"Run dir  : {run_dir}")
    print(f"Device   : {device}  |  w={tcfg.w} (free-run steps)  |  T={T}")
    print(f"Model    : {n_params:,} parameters")
    print(f"Train    : {len(train_idx):,} samples  |  Val: {len(val_idx):,} samples")

    @torch.no_grad()
    def val_multistep() -> float:
        model.eval()
        tot, nb = 0.0, 0
        for i in range(0, len(val_idx_t), tcfg.batch_size):
            idx = val_idx_t[i:i + tcfg.batch_size]
            tot += float(multistep_loss(model, obs_t[idx], tcfg.w)); nb += 1
        return tot / nb

    @torch.no_grad()
    def val_singlestep() -> float:
        """1-step teacher-forced MSE (comparable across all w; the baseline's metric)."""
        model.eval()
        tot, nb = 0.0, 0
        for i in range(0, len(val_idx_t), tcfg.batch_size):
            idx = val_idx_t[i:i + tcfg.batch_size]
            o = obs_t[idx]
            pred, _ = model(o)
            tot += float(F.mse_loss(pred, o[:, 1:, :])); nb += 1
        return tot / nb

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, tcfg.n_epochs + 1):
        model.train()
        ep_perm = train_idx_t[torch.randperm(len(train_idx_t), device=device)]
        tr, nb = 0.0, 0
        for i in range(0, len(ep_perm), tcfg.batch_size):
            idx = ep_perm[i:i + tcfg.batch_size]
            loss = multistep_loss(model, obs_t[idx], tcfg.w)
            opt.zero_grad(); loss.backward(); opt.step()
            tr += float(loss); nb += 1
        tr /= nb
        vloss = val_multistep()
        v1 = val_singlestep()

        with open(metrics_path, "a") as fh:
            fh.write(json.dumps({"epoch": epoch, "train_loss": tr, "val_loss": vloss,
                                 "val_1step_mse": v1, "lr": opt.param_groups[0]["lr"]}) + "\n")

        ckpt = {"epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(), "train_config": dataclasses.asdict(tcfg),
                "model_config": dataclasses.asdict(mcfg), "val_loss": vloss,
                "val_1step_mse": v1}
        torch.save(ckpt, latest_path)
        if vloss < best_val:
            best_val = vloss
            torch.save(ckpt, best_path)

        if epoch % 20 == 0 or epoch == 1:
            el = time.time() - t0
            print(f"  epoch {epoch:4d}/{tcfg.n_epochs}  train={tr:.5f}  "
                  f"val(w={tcfg.w})={vloss:.5f}  val_1step={v1:.5f}  "
                  f"[{el:.0f}s, {el/epoch:.2f}s/ep]", flush=True)

    print(f"\nDone. Best val(w={tcfg.w})={best_val:.5f}  total {time.time()-t0:.0f}s")
    print(f"Checkpoints: {run_dir}")
    (run_dir / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
