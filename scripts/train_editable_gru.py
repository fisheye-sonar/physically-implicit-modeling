#!/usr/bin/env python3
"""Can editability be INDUCED BY TRAINING?  Two mechanisms, one evaluation.

Every §4 result so far uses **inference-time** editors on a **frozen** world model, and every
probe-directed one fails: the model's own decoder-gradient oracle reaches the edited world
(Edit Index +0.94) while readout injection, PCA geodesic and friends sit at the unedited end
(-0.6 or worse).  `learn_to_edit` asked whether *training* fixes this and got two negatives —
but both under a deliberately **light** budget, and the heavier fine-tune was left OWED.
This script pays that debt and turns it into a controlled comparison.

Two arms:

**`finetune`** — the world model is fine-tuned so that a **FIXED, untrained** editor works.
    The editor is the linear-pseudoinverse *readout injection* whose probe `(A, b)` is fit once
    on the BASE model and then **frozen**, so the model must learn to honour writes along
    `A_pinv` as "put the object here".  Nothing about the editor is learned; all the adaptation
    is in the world model.  Loss:
        edit      = MSE(rollout(h_edited, K), clean_obs[ef : ef+K])
        retention = ordinary teacher-forced next-step MSE on train sequences
        total     = edit + retention_weight * retention
    The retention term is what separates "the model became editable" from "the model was
    destroyed and now outputs whatever the editor asks for".

**`amortized`** — the world model is **frozen** and an editor network `E_theta(h, target) -> dh`
    is trained instead (`learn_to_edit` Variant A at a larger budget).  Same edit loss.

Both are trained on a **disjoint slice** of the edits split from the one the notebooks report on,
so every reported number is held out.  `--train-object` restricts training to edits of one object
so the notebook can test **content generalisation** (train on object 0, evaluate on object 1).

Checkpoints are written in the same format as `scripts/train_gru.py`, so `load_checkpoint` and
`scripts/eval_controls.py` work on them unchanged.

Usage
-----
    python scripts/train_editable_gru.py --mode finetune  --steps 3000 --run-name FT_heavy
    python scripts/train_editable_gru.py --mode finetune  --steps 3000 --retention 0 --run-name FT_heavy_noret
    python scripts/train_editable_gru.py --mode amortized --steps 3000 --run-name AMORT
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pim.world_models import load_checkpoint, load_dataset  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_OBJ = 2
K = 15  # post-edit rollout steps optimised
HELD_OUT = 2000  # edits[:HELD_OUT] are NEVER trained on — the notebooks report on them


# ── the frozen write mechanism ────────────────────────────────────────────────


def fit_readout_probe(model, obs: np.ndarray, pos: np.ndarray):
    """Least-squares linear position probe on the BASE model's states; frozen thereafter.

    Returns (W, b, W_pinv) such that `h @ W + b` reads the flat (x0,y0,x1,y1) position and
    `h + (target - (h @ W + b)) @ W_pinv` is the readout-injection edit.
    """
    with torch.no_grad():
        H = (
            model.get_hidden_states(torch.from_numpy(obs).float().to(DEVICE))
            .cpu()
            .numpy()
        )
    T = H.shape[1]
    X = H.reshape(-1, H.shape[-1])
    Y = pos[:, :T].reshape(-1, N_OBJ * 2)
    A = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    sol, *_ = np.linalg.lstsq(A, Y, rcond=None)
    W = torch.tensor(sol[:-1], dtype=torch.float32, device=DEVICE)
    b = torch.tensor(sol[-1], dtype=torch.float32, device=DEVICE)
    W_pinv = torch.tensor(np.linalg.pinv(sol[:-1]), dtype=torch.float32, device=DEVICE)
    rmse = float(np.sqrt(((X @ sol[:-1] + sol[-1] - Y) ** 2).mean()))
    return W, b, W_pinv, rmse


def readout_inject(h, target, W, b, W_pinv):
    """The fixed, untrained editor.  Nothing here is learned."""
    return h + (target - (h @ W + b)) @ W_pinv


class AmortizedEditor(nn.Module):
    """E_theta(h, target) -> dh.  The learned alternative to a fixed write mechanism."""

    def __init__(self, hidden: int, target_dim: int = N_OBJ * 2, width: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden + target_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, hidden),
        )

    def forward(self, h, target):
        return h + self.net(torch.cat([h, target], -1))


# ── differentiable model ops ──────────────────────────────────────────────────


def warm_to_edit(model, obs_t, ef: int):
    """Teacher-force obs[0..ef-1]; gradients flow (the encoder/recurrence are being trained)."""
    state = None
    for t in range(ef):
        _, state = model.step(obs_t[:, t], state)
    return model.flat_state(state)


def rollout(model, h_flat, steps: int = K):
    """Free-run from an edited state; step 0 decodes the edit frame."""
    state = model.state_from_flat(h_flat)
    out = [model.decode(state)]
    for _ in range(steps - 1):
        p, state = model.predict_step(state)
        out.append(p)
    return torch.stack(out, 1)


# ── training ──────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["finetune", "amortized"], default="finetune")
    ap.add_argument("--base", default="runs/controls/H256/best_model.pt")
    ap.add_argument("--data", default="datasets/4_fixed_refl_inview")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--retention",
        type=float,
        default=1.0,
        help="weight on the prediction-retention loss",
    )
    ap.add_argument(
        "--train-object",
        type=int,
        default=-1,
        help="restrict training to edits of this object (-1 = both) — the content-generalisation control",
    )
    ap.add_argument("--run-dir", default="runs/trained_editability")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.run_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────────
    base_model, info = load_checkpoint(args.base, device=DEVICE)
    bundle = load_dataset(args.data, n_obj_keep=N_OBJ)
    edits, test = bundle.edits, bundle.test
    ef = edits.edit_frame

    obs_all = edits.obs.astype(np.float32)
    clean_all = edits.clean_obs.astype(np.float32)
    tgt_all = (
        edits.positions[:, ef, :N_OBJ, :]
        .reshape(len(obs_all), N_OBJ * 2)
        .astype(np.float32)
    )
    eobj = edits.edit_object.astype(int)

    train_idx = np.arange(HELD_OUT, len(obs_all))
    if args.train_object >= 0:
        train_idx = train_idx[eobj[train_idx] == args.train_object]
    print(
        f"edits: {len(obs_all)} total | held out for reporting: [0,{HELD_OUT}) | training pool: {len(train_idx)}"
    )

    # retention data: ordinary sequences from the test split (never the edits split)
    ret_obs = test.obs.astype(np.float32)

    # ── the frozen editor ─────────────────────────────────────────────────────
    W, b_, W_pinv, probe_rmse = fit_readout_probe(
        base_model,
        obs_all[HELD_OUT : HELD_OUT + 2000],
        edits.positions[HELD_OUT : HELD_OUT + 2000, :, :N_OBJ, :],
    )
    print(
        f"frozen linear position probe fit on the BASE model: train RMSE {probe_rmse:.3f} sim-units"
    )

    model = copy.deepcopy(base_model).to(DEVICE)
    editor = None
    if args.mode == "finetune":
        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
        params = list(model.parameters())
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        # NOTE: stay in train() mode even though the world model is frozen — cuDNN cannot
        # backprop through an RNN in eval mode, and this GRU has no dropout, so train()/eval()
        # are behaviourally identical here.  The freeze is enforced by requires_grad_(False).
        model.train()
        editor = AmortizedEditor(model.hidden_size).to(DEVICE)
        params = list(editor.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    log = []
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        idx = rng.choice(train_idx, size=args.batch, replace=False)
        o = torch.from_numpy(obs_all[idx]).to(DEVICE)
        tgt = torch.from_numpy(tgt_all[idx]).to(DEVICE)
        gt = torch.from_numpy(clean_all[idx, ef : ef + K]).to(DEVICE)

        h0 = warm_to_edit(model, o, ef)
        h_edit = (
            editor(h0, tgt)
            if editor is not None
            else readout_inject(h0, tgt, W, b_, W_pinv)
        )
        edit_loss = F.mse_loss(rollout(model, h_edit), gt)

        ret_loss = torch.zeros((), device=DEVICE)
        if args.retention > 0 and args.mode == "finetune":
            ridx = rng.integers(0, len(ret_obs), size=args.batch)
            ro = torch.from_numpy(ret_obs[ridx]).to(DEVICE)
            pred, _ = model(ro)
            ret_loss = F.mse_loss(pred, ro[:, 1:, :])

        loss = edit_loss + args.retention * ret_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()

        if step % 25 == 0 or step == 1:
            log.append(
                dict(
                    step=step,
                    edit_loss=edit_loss.item(),
                    retention_loss=ret_loss.item(),
                    total=loss.item(),
                )
            )
        if step % 250 == 0 or step == 1:
            el = time.perf_counter() - t0
            print(
                f"  step {step:>5}/{args.steps}  edit {edit_loss.item():.5f}  "
                f"retention {ret_loss.item():.5f}  ({el:.0f}s)"
            )

    # ── save in the standard checkpoint format so the whole eval suite just works ──
    ckpt = {
        "epoch": args.steps,
        "model_state": model.state_dict(),
        "model_config": dataclasses.asdict(model.cfg),
        "train_config": {**vars(args), "base_val_loss": info.val_loss},
        "val_loss": float("nan"),
    }
    torch.save(ckpt, out_dir / "best_model.pt")
    if editor is not None:
        torch.save(
            {"editor_state": editor.state_dict(), "hidden": model.hidden_size},
            out_dir / "amortized_editor.pt",
        )
    # the frozen probe travels with the run — the notebook must use the SAME editor it was trained for
    np.savez(
        out_dir / "frozen_probe.npz",
        W=W.cpu().numpy(),
        b=b_.cpu().numpy(),
        W_pinv=W_pinv.cpu().numpy(),
        probe_rmse=np.array([probe_rmse]),
    )
    (out_dir / "log.json").write_text(json.dumps(log, indent=1))
    (out_dir / "config.json").write_text(
        json.dumps({**vars(args), "n_train_edits": int(len(train_idx))}, indent=1)
    )
    print(f"\nDone in {time.perf_counter()-t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
