#!/usr/bin/env python3
"""Trained editors on exogenous-action world models — 3 models x 2 editors x 2 losses.

Extends `scripts/train_editable_gru.py` (which is fixed to a plain GRU on dataset 4, and whose
edit loss is hardcoded to a 15-step rollout) along the three axes this thread needs:

**Three world models, one interface.**
  `XG_A_H256`  action-conditioned GRU — teleport actions ARE an input (`datasets/7_cont_teleport`)
  `XG_C_H256`  the observer twin — identical data and recipe, action input removed
  `CTRL_H256`  `runs/controls/H256` — the standard GRU, no actions and no teleports in training.
               The control for everything introduced here.

**Two trained editors.**
  `finetune`   the *world model* is fine-tuned so a FIXED, untrained editor works. The editor is
               linear-pseudoinverse readout injection through a probe fit once on the BASE model
               and then frozen — nothing about the editor is learned. Carries the **retention**
               term (ordinary next-step MSE on non-edit sequences), which is what separates
               "the model became editable" from "the model was destroyed and now echoes the
               editor" — measured 2026-07-30: the no-retention arm's unsteered index ROSE from
               degraded prediction alone.
  `mlp`        the *world model is frozen* and an editor network is trained:
                   E_theta(h, start_pos, target_pos) -> dh
               **This is not the previously-published amortized editor**, which took
               `(h, target)` only. Giving it the *starting* positions as well means it can
               condition on the displacement it has to produce rather than having to infer the
               current world state from `h` — a strictly easier problem, and the point of the
               variant.

**Two losses**, so rollout consistency is a measured variable rather than an assumption:
  `--edit-k 1`  loss = next-step prediction RMSE at the edit frame only.
  `--edit-k 8`  loss = RMSE over the next 8 free-run steps (the edit must survive the dynamics).

Edit data is disjoint from everything reported on:
  * dataset 4  — `edits[HELD_OUT:]`; `edits[:HELD_OUT]` is the notebooks' reporting set.
  * teleport world — `datasets/16_teleport_edittrain_single` (base seed 300000), while evaluation
    uses `datasets/15_teleport_eval_single` (base seed 200000); both disjoint from the world models'
    training seeds 0-89999, and **both generated with `--p-action 0.0`** so an episode carries
    exactly one intervention: the single teleport synthesised by `eval_action_sweep.xg_data`. The
    earlier `13_`/`14_` splits inherited the training-time `p_action = 0.30` and are superseded --
    they put random teleports in the scored window and in the visible context.

Usage
-----
    python scripts/train_action_editors.py --model XG_A_H256 --editor mlp      --edit-k 8
    python scripts/train_action_editors.py --model CTRL_H256 --editor finetune --edit-k 1
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval_action_sweep as EAS  # the shared edit construction + model loaders
from pim.world_models import load_checkpoint, load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_OBJ, EF = 2, 20
HELD_OUT = 2000            # dataset-4 edits[:HELD_OUT] are the reporting set — never trained on
N_TRAIN_EPISODES = 6000    # dataset-7 synthesised training episodes
RUN_DIR = Path("runs/action_editors")

MODELS = {
    "XG_A_H256": dict(kind="xg", code="XG_A_H256", actions=True,
                      label="Exogenous teleport · actions given · 256 hidden"),
    "XG_C_H256": dict(kind="xg", code="XG_C_H256", actions=False,
                      label="Exogenous teleport · observer (actions withheld) · 256 hidden"),
    "CTRL_H256": dict(kind="ctrl", code="H256", actions=False,
                      label="Control · standard GRU · no actions, no teleports · 256 hidden"),
}


# ── the two editors ───────────────────────────────────────────────────────────
def fit_readout_probe(model, obs, pos, actions=None):
    """Least-squares linear position probe on the BASE model's states; frozen thereafter."""
    with torch.no_grad():
        o = torch.from_numpy(obs).float().to(DEVICE)
        H = (model.get_hidden_states(o, actions=torch.from_numpy(actions).float().to(DEVICE))
             if actions is not None else model.get_hidden_states(o)).cpu().numpy()
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
    """The fixed, untrained write mechanism. Nothing here is learned."""
    return h + (target - (h @ W + b)) @ W_pinv


def fit_state_covariance(model, obs, actions=None):
    """Σ_hh of the BASE model's visited states — frozen along with the probe.

    Needed by the metric-corrected write. `metric_corrected_edits` (2026-08-05) measured a
    condition number of 1.79e4 here, i.e. strongly anisotropic, which is what makes un-whitening
    change the direction at all.
    """
    with torch.no_grad():
        o = torch.from_numpy(obs).float().to(DEVICE)
        H = (model.get_hidden_states(o, actions=torch.from_numpy(actions).float().to(DEVICE))
             if actions is not None else model.get_hidden_states(o)).cpu().numpy()
    X = H.reshape(-1, H.shape[-1]).astype(np.float64)
    Xc = X - X.mean(0)
    S = (Xc.T @ Xc) / (len(Xc) - 1)
    lam, V = np.linalg.eigh(S)
    lam = np.clip(lam, lam.max() * 1e-12, None)
    cond = float(lam.max() / lam.min())
    return S, cond


def metric_inject_factory(W, b, Sigma, alpha: float = 1.0, eps_scale: float = 1e-6):
    """The **un-whitened** (metric-corrected) write from `metric_corrected_edits`, α = 1.

        Δ_α = Σ^α Wᵀ (W Σ^α Wᵀ + εI)⁻¹ δ ,   δ = target − (hW + b)

    A least-squares probe is `W = Σ_ph Σ_hh⁻¹`, so its row space is the true Jacobian **whitened
    by the inverse state covariance**. Multiplying by `Σ^α` undoes that. This is the
    constraint-satisfying form: it hits the readout target **exactly** at every α, and α = 0
    reduces to the ordinary Euclidean pseudoinverse, so this arm differs from the `pinv` arm in
    the metric alone. Published as the best training-free structural editor in the thread
    (Edit Index −0.51 vs Euclidean −0.65 at fidelity 0.98).

    Everything here is computed once on the BASE model and frozen — it is part of the fixed,
    untrained editor, exactly like the probe.
    """
    lam, V = np.linalg.eigh(Sigma)
    lam = np.clip(lam, lam.max() * 1e-12, None)
    S = (V * (lam ** alpha)) @ V.T                       # Σ^α
    Wn = W.detach().cpu().numpy().astype(np.float64)     # (H, 4) — the notebook's Wᵀ
    M = Wn.T @ S @ Wn                                    # (4, 4)
    M = M + eps_scale * np.trace(M) / M.shape[0] * np.eye(M.shape[0])
    right = torch.tensor(np.linalg.solve(M, Wn.T @ S), dtype=torch.float32, device=DEVICE)

    def metric_inject(h, target):
        return h + (target - (h @ W + b)) @ right

    return metric_inject


class StateTargetEditor(nn.Module):
    """E_theta(h, start_pos, target_pos) -> dh.

    Differs from the published amortized editor `E(h, target)` by also receiving the **starting**
    positions, so the displacement it must produce is given rather than inferred from `h`.
    """

    def __init__(self, hidden: int, pos_dim: int = N_OBJ * 2, width: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden + 2 * pos_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, hidden),
        )

    def forward(self, h, start_pos, target_pos):
        return h + self.net(torch.cat([h, start_pos, target_pos], -1))


# ── model ops that work for both the plain and the action-conditioned GRU ─────
def warm_to_edit(model, obs_t, actions_t=None, ef: int = EF):
    """Teacher-force obs[0..ef-1]; gradients flow. `actions_t` is the NO-OP-at-ef-1 sequence:
    the latent editors are never told about the teleport."""
    state = None
    for t in range(ef):
        state = (model.step(obs_t[:, t], state, action=actions_t[:, t])[1]
                 if actions_t is not None else model.step(obs_t[:, t], state)[1])
    return model.flat_state(state)


def rollout(model, h_flat, steps: int):
    """Free-run from an edited state; step 0 decodes the edit frame."""
    state = model.state_from_flat(h_flat)
    out = [model.decode(state)]
    for _ in range(steps - 1):
        p, state = model.predict_step(state)
        out.append(p)
    return torch.stack(out, 1)


# ── edit training pools ───────────────────────────────────────────────────────
def build_pool(spec: dict, edit_k: int):
    """(obs, actions_noop, start_pos, target_pos, gt_rollout) for the editor training set."""
    if spec["kind"] == "ctrl":
        bundle = load_dataset(Path("datasets/4_fixed_refl_inview"), n_obj_keep=N_OBJ)
        edits, test = bundle.edits, bundle.test
        sl = slice(HELD_OUT, len(edits.obs))
        pos_ef = edits.positions[sl, EF, :N_OBJ, :].reshape(-1, N_OBJ * 2).astype(np.float32)
        # "start" = where the objects are in the UNEDITED frame-ef world (what the model
        # would render); the edited object is put back on its own ballistic continuation.
        pre = edits.positions[sl, EF - 1, :N_OBJ, :].astype(np.float32)
        import h5py
        with h5py.File(edits.h5_path, "r") as f:
            vel = f["velocities"][sl, EF - 1, :N_OBJ, :].astype(np.float32)
        start = (pre + vel).reshape(-1, N_OBJ * 2)
        return dict(obs=edits.obs[sl].astype(np.float32), actions=None,
                    start=start, target=pos_ef,
                    gt=edits.clean_obs[sl, EF : EF + edit_k].astype(np.float32),
                    retention_obs=test.obs.astype(np.float32))

    E = EAS.xg_data(N_TRAIN_EPISODES, n_probe=1, seed=1,
                    h5_path=Path("datasets/16_teleport_edittrain_single/train.h5"),
                    n_gt_steps=edit_k)
    n = len(E["obs"])
    start = E["pos"][:, EF].reshape(n, N_OBJ * 2).astype(np.float32)   # un-teleported frame ef
    target = E["tgt_pos"].reshape(n, N_OBJ * 2).astype(np.float32)     # teleported frame ef
    return dict(obs=E["obs"].astype(np.float32),
                actions=E["act_noop"].astype(np.float32) if spec["actions"] else None,
                start=start, target=target, gt=E["gt_roll"].astype(np.float32),
                retention_obs=E["obs"].astype(np.float32))


def load_base(spec: dict):
    if spec["kind"] == "ctrl":
        model, _ = load_checkpoint(Path("runs/controls") / spec["code"] / "best_model.pt",
                                   device=DEVICE)
        return model
    model, _, _ = EAS.xg_load(spec["code"])
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--editor", choices=["finetune", "mlp"], required=True)
    ap.add_argument("--write", choices=["pinv", "metric"], default="pinv",
                    help="finetune only: which FIXED write the model must learn to honour. "
                         "'pinv' = Euclidean min-norm pseudoinverse; 'metric' = the un-whitened "
                         "Σ^1 write from metric_corrected_edits (same readout target, different "
                         "metric).")
    ap.add_argument("--edit-k", type=int, required=True, help="1 = next-step loss; 8 = rollout")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--retention", type=float, default=1.0,
                    help="weight on the prediction-retention loss (finetune only)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    spec = MODELS[a.model]
    wtag = "" if (a.editor != "finetune" or a.write == "pinv") else f"__{a.write}"
    name = f"{a.model}__{a.editor}{wtag}__k{a.edit_k}"
    out_dir = RUN_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {name} ===\n{spec['label']} | editor={a.editor} | write={a.write} | "
          f"edit-k={a.edit_k}")

    base = load_base(spec)
    pool = build_pool(spec, a.edit_k)
    n_pool = len(pool["obs"])
    print(f"training pool: {n_pool} edit episodes | GT horizon {pool['gt'].shape[1]} steps")

    # The frozen write mechanism, fit on the BASE model. It needs PER-FRAME positions, which the
    # training pool does not carry (it only stores the frame-`ef` start/target), so the probe gets
    # its own small slice of sequences.
    if spec["kind"] == "ctrl":
        bundle = load_dataset(Path("datasets/4_fixed_refl_inview"), n_obj_keep=N_OBJ)
        pobs = bundle.edits.obs[HELD_OUT : HELD_OUT + 2000].astype(np.float32)
        ppos = bundle.edits.positions[HELD_OUT : HELD_OUT + 2000, :, :N_OBJ, :]
        pact = None
    else:
        Ep = EAS.xg_data(2000, n_probe=1, seed=2,
                         h5_path=Path("datasets/16_teleport_edittrain_single/train.h5"),
                         n_gt_steps=1)
        pobs = Ep["obs"].astype(np.float32)
        ppos = Ep["pos"][:, :, :N_OBJ, :]
        pact = Ep["act_noop"].astype(np.float32) if spec["actions"] else None
    W, b_, W_pinv, probe_rmse = fit_readout_probe(
        base, pobs, ppos.reshape(len(pobs), -1, N_OBJ * 2), pact)
    print(f"frozen linear position probe on the BASE model: train RMSE {probe_rmse:.3f} sim units")

    if a.write == "metric":
        Sigma, cond = fit_state_covariance(base, pobs, pact)
        fixed_write = metric_inject_factory(W, b_, Sigma, alpha=1.0)
        print(f"frozen Σ_hh for the un-whitened write: condition number {cond:.3g} "
              f"(anisotropy is what makes un-whitening change the direction)")
    else:
        def fixed_write(h, target):
            return readout_inject(h, target, W, b_, W_pinv)


    model = copy.deepcopy(base).to(DEVICE)
    editor = None
    if a.editor == "finetune":
        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
        params = list(model.parameters())
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        # stay in train(): cuDNN cannot backprop through an RNN in eval mode, and this GRU has
        # no dropout, so train()/eval() are behaviourally identical. The freeze is requires_grad.
        model.train()
        editor = StateTargetEditor(model.hidden_size).to(DEVICE)
        params = list(editor.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)

    log, rng, t0 = [], np.random.default_rng(a.seed), time.perf_counter()
    for step in range(1, a.steps + 1):
        idx = rng.choice(n_pool, size=a.batch, replace=False)
        o = torch.from_numpy(pool["obs"][idx]).to(DEVICE)
        act = (torch.from_numpy(pool["actions"][idx]).to(DEVICE)
               if pool["actions"] is not None else None)
        start = torch.from_numpy(pool["start"][idx]).to(DEVICE)
        target = torch.from_numpy(pool["target"][idx]).to(DEVICE)
        gt = torch.from_numpy(pool["gt"][idx]).to(DEVICE)

        h0 = warm_to_edit(model, o, act)
        h_edit = editor(h0, start, target) if editor is not None else fixed_write(h0, target)
        edit_loss = F.mse_loss(rollout(model, h_edit, a.edit_k), gt)

        ret_loss = torch.zeros((), device=DEVICE)
        if a.retention > 0 and a.editor == "finetune":
            ridx = rng.integers(0, len(pool["retention_obs"]), size=a.batch)
            ro = torch.from_numpy(pool["retention_obs"][ridx]).to(DEVICE)
            pred = (model(ro, actions=torch.from_numpy(pool["actions"][ridx]).to(DEVICE))[0]
                    if pool["actions"] is not None else model(ro)[0])
            ret_loss = F.mse_loss(pred, ro[:, 1:, :])

        loss = edit_loss + a.retention * ret_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()

        if step % 25 == 0 or step == 1:
            log.append(dict(step=step, edit_loss=edit_loss.item(),
                            retention_loss=ret_loss.item(), total=loss.item()))
        if step % 500 == 0 or step == 1:
            print(f"  step {step:>5}/{a.steps}  edit {edit_loss.item():.5f}  "
                  f"retention {ret_loss.item():.5f}  ({time.perf_counter()-t0:.0f}s)")

    ckpt = {
        "arm": name, "model": a.model, "editor": a.editor, "write": a.write,
        "edit_k": a.edit_k,
        "label": spec["label"], "args": vars(a), "log": log,
        "probe": {"W": W.cpu().numpy(), "b": b_.cpu().numpy(), "W_pinv": W_pinv.cpu().numpy(),
                  "rmse": probe_rmse},
        "model_state": model.state_dict() if a.editor == "finetune" else None,
        "editor_state": editor.state_dict() if editor is not None else None,
    }
    torch.save(ckpt, out_dir / "ckpt.pt")
    (out_dir / "config.json").write_text(json.dumps(
        {k: v for k, v in ckpt.items() if k not in ("model_state", "editor_state", "probe", "log")},
        indent=1, default=str))
    print(f"saved → {out_dir/'ckpt.pt'}  ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
