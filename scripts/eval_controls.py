#!/usr/bin/env python3
"""Shared metric suite for the controls thread (branch `michael_controls`).

One pass per GRU checkpoint computes the four affordance families that
`research/directions/{hidden-size-sweep,noise-ablation}.md` ask for, so the
notebooks stay short and only load, plot and tabulate:

  1. predictive quality — next-step RMSE, free-run RMSE per horizon step, plus
     that model's OWN dataset baselines (copy-previous-frame / noise floor /
     random frame).  Per-model baselines matter: the noise-ablation cells have
     different noise floors, so raw RMSE is not comparable across them.
  2. recoverability   — position & velocity R², linear and MLP, held out.
  3. canonicality     — fiber residual ‖h − g(pos,vel)‖ / ‖h‖, linear and MLP.
  4. editability      — the canonical §4 scorecard from `scripts/editability_metrics.py`
     (Edit Index, plus Target / Ghost / Collateral / Edit-frame / GT-traj RMSE and the
     fidelity ratio) for the standard editor suite, bracketed by the true-state swap and
     the decoder-gradient oracle.

Metric names and formulas follow `notebooks/experiments/editability/METRICS_AND_EDITORS.md`
and are implemented once in `scripts/editability_metrics.py` — nothing is re-derived here.

Writes `runs/controls/eval/<code>.json` (scalars + curves) and
`runs/controls/eval/<code>_rollouts.npz` (the arrays the waterfalls need).

Usage
-----
    python scripts/eval_controls.py --runs H8 H32 H128 H256 H512
    python scripts/eval_controls.py --runs N_obs0_pos0 --n-edits 64
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # canonical metric module

import h5py
import numpy as np
import torch
import torch.nn as nn

from pim.editors import (
    fit_state_subspace,
    inject_state,
    manifold_steer,
    manifold_steer_local,
    probe_decomposition,
)
from editability_metrics import (
    build_edit_zones,
    edit_scorecard,
    fidelity_ratio,
)
from pim.eval.baselines import compute_obs_baselines
from pim.extractors import LinearExtractor, StateDefinition
from pim.world_models import load_checkpoint, load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("runs/controls")  # overridden by --root
OUT = ROOT / "eval"

N_OBJ = 2
K = 15  # post-edit rollout steps
HORIZON = 20  # free-run horizon for predictive quality
WARM = 10  # teacher-forced frames before the free run
LATE_T = 15  # "late" frames = t >= 15 (the filter-converged regime)

# Which dataset each run was trained on.  Kept here (not inferred) so the
# mapping is explicit and greppable; mirrors CONTROL_RUNS.md.
RUN_DATASET = {
    "H8": "4_fixed_refl_inview",
    "H32": "4_fixed_refl_inview",
    "H128": "4_fixed_refl_inview",
    "H256": "4_fixed_refl_inview",
    "H512": "4_fixed_refl_inview",
    "N_obs0_pos0": "9_obsnoise0_posnoise0",
    "N_obs0_pos004": "10_obsnoise0_posnoise004",
    "N_obs02_pos0": "11_obsnoise02_posnoise0",
}

EDITORS = [
    "Unsteered",
    "Oracle observation",
    "Readout injection",
    "Global-PCA projection",
    "PCA geodesic",
    "MLP-probe gradient",
    "Decoder gradient",
]


# ── data ──────────────────────────────────────────────────────────────────────


def read_velocities(h5_path: str, n: int | None = None) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        v = f["velocities"][:n, :, :N_OBJ, :]
    return v.astype(np.float32)


# ── model passes ──────────────────────────────────────────────────────────────


@torch.no_grad()
def hidden_states(model, obs: np.ndarray, batch: int = 512) -> np.ndarray:
    """Teacher-forced hidden states, (N, T-1, H).  h[:, t] follows obs[:, t]."""
    out = []
    for i in range(0, len(obs), batch):
        o = torch.from_numpy(obs[i : i + batch]).float().to(DEVICE)
        out.append(model.get_hidden_states(o).cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def next_step_rmse(
    model, obs: np.ndarray, clean: np.ndarray, batch: int = 512
) -> float:
    """Teacher-forced one-step prediction vs the CLEAN next frame."""
    se, n = 0.0, 0
    for i in range(0, len(obs), batch):
        o = torch.from_numpy(obs[i : i + batch]).float().to(DEVICE)
        pred, _ = model(o)  # (B, T-1, R) — pred[:, t] ≈ obs[:, t+1]
        gt = torch.from_numpy(clean[i : i + batch, 1:]).float().to(DEVICE)
        se += float(((pred - gt) ** 2).sum())
        n += gt.numel()
    return float(np.sqrt(se / n))


@torch.no_grad()
def freerun_rmse_by_step(model, obs, clean, steps=HORIZON, warm=WARM, batch=512):
    """Warm up on obs[0..warm-1], then free-run.  Rollout step s ↔ frame warm+s."""
    per_step = np.zeros(steps)
    n = 0
    for i in range(0, len(obs), batch):
        o = torch.from_numpy(obs[i : i + batch]).float().to(DEVICE)
        state = None
        for t in range(warm):
            _, state = model.step(o[:, t], state)
        preds = [model.decode(state)]
        for _ in range(steps - 1):
            p, state = model.predict_step(state)
            preds.append(p)
        roll = torch.stack(preds, 1).cpu().numpy()  # (B, steps, R)
        gt = clean[i : i + batch, warm : warm + steps]
        per_step += ((roll - gt) ** 2).mean(axis=(0, 2)) * len(gt)
        n += len(gt)
    return np.sqrt(per_step / n)


# ── probes ────────────────────────────────────────────────────────────────────


def _r2(pred: np.ndarray, gt: np.ndarray) -> float:
    ss_res = ((pred - gt) ** 2).sum()
    ss_tot = ((gt - gt.mean(axis=0)) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def _fit_mlp(x: np.ndarray, y: np.ndarray, *, epochs=60, width=128, lr=3e-3):
    """Small MLP regressor x → y, trained full-batch on the GPU."""
    net = nn.Sequential(
        nn.Linear(x.shape[1], width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, y.shape[1]),
    ).to(DEVICE)
    xt = torch.from_numpy(x).float().to(DEVICE)
    yt = torch.from_numpy(y).float().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(xt)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for j in range(0, n, 4096):
            idx = perm[j : j + 4096]
            opt.zero_grad()
            ((net(xt[idx]) - yt[idx]) ** 2).mean().backward()
            opt.step()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.no_grad()
def _apply(net, x: np.ndarray) -> np.ndarray:
    return net(torch.from_numpy(x).float().to(DEVICE)).cpu().numpy()


def _lstsq(x: np.ndarray, y: np.ndarray):
    A = np.concatenate([x, np.ones((len(x), 1), np.float32)], 1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return lambda z: np.concatenate([z, np.ones((len(z), 1), np.float32)], 1) @ sol


def recoverability_and_canonicality(H_flat, pos_flat, vel_flat, seed=0):
    """Held-out (70/30) probe read-out of (pos, vel) from h, and the reverse fiber map.

    Returns position/velocity R² (linear + MLP) and the fiber residual
    ‖h − g(pos,vel)‖ / ‖h‖ (linear + MLP), all on the held-out 30%.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(H_flat))
    cut = int(0.7 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    out = {}

    for name, Y in (("pos", pos_flat), ("vel", vel_flat)):
        lin = _lstsq(H_flat[tr], Y[tr])
        out[f"{name}_r2_linear"] = _r2(lin(H_flat[te]), Y[te])
        mlp = _fit_mlp(H_flat[tr], Y[tr])
        out[f"{name}_r2_mlp"] = _r2(_apply(mlp, H_flat[te]), Y[te])

    # fiber residual: how much of h is NOT a function of the physical state
    S = np.concatenate([pos_flat, vel_flat], 1)  # (M, 8)
    hnorm = np.linalg.norm(H_flat[te], axis=1).mean()
    lin = _lstsq(S[tr], H_flat[tr])
    out["fiber_resid_linear"] = float(
        np.linalg.norm(H_flat[te] - lin(S[te]), axis=1).mean() / hnorm
    )
    mlp = _fit_mlp(S[tr], H_flat[tr])
    out["fiber_resid_mlp"] = float(
        np.linalg.norm(H_flat[te] - _apply(mlp, S[te]), axis=1).mean() / hnorm
    )
    return out


# ── editability ───────────────────────────────────────────────────────────────


@torch.no_grad()
def tf_hidden_at(model, obs_seqs: np.ndarray, frame: int) -> np.ndarray:
    """Teacher-force obs[0..frame] and return the flat state (repo convention).

    Used for the **Oracle observation** reference: the model is given one extra
    teacher-forced frame, the REAL (noisy) post-edit observation `edits.obs[ef]`.
    Nothing about the state is swapped — it simply gets to see the teleport happen.

    NOTE the ±1 this implies: `warm_to_edit` stops at `ef-1` (so an ordinary rollout's
    step 0 decodes frame `ef`), while the oracle observation is fed `obs[ef]` and its
    rollout therefore leads by one frame.  Recorded in the JSON as `swap_frame_lead`.
    """
    o = torch.from_numpy(obs_seqs).float().to(DEVICE)
    state = None
    for t in range(frame + 1):
        _, state = model.step(o[:, t], state)
    return model.flat_state(state).cpu().numpy()


@torch.no_grad()
def warm_to_edit(model, obs_seqs: np.ndarray, ef: int) -> np.ndarray:
    """Teacher-force obs[0..ef-1]; the rollout's step 0 then decodes frame `ef`."""
    o = torch.from_numpy(obs_seqs).float().to(DEVICE)
    state = None
    for t in range(ef):
        _, state = model.step(o[:, t], state)
    return model.flat_state(state).cpu().numpy()


@torch.no_grad()
def rollout(model, h_flat: torch.Tensor, steps=K) -> np.ndarray:
    state = model.state_from_flat(h_flat)
    obs = [model.decode(state)]
    for _ in range(steps - 1):
        p, state = model.predict_step(state)
        obs.append(p)
    return torch.stack(obs, 1).cpu().numpy()


def build_editors(model, h0, target, states_bank, pos_bank, tgt_obs):
    """The canonical §4 write-mechanism suite, all targeting the same readout."""
    H = h0.shape[1]
    # flat (x0, y0, x1, y1) readout — keeps the probe output, the pseudoinverse
    # target and the MLP-probe target in one shape
    sdef = StateDefinition(
        name="positions_flat",
        state_shape=(N_OBJ * 2,),
        extract_fn=lambda b: b["positions"],
    )
    lin = LinearExtractor(H, sdef, use_lstsq=True)
    lin.fit(states_bank[None], pos_bank[None], device=DEVICE)
    lin = lin.to(DEVICE).eval()
    A, b, A_pinv = probe_decomposition(lin)

    def linear_inject(h, t):
        return inject_state(h, t, A, A_pinv, b)

    out = {"Readout injection": linear_inject(h0, target)}

    bank = torch.from_numpy(states_bank).float().to(DEVICE)
    sub = fit_state_subspace(bank, var_threshold=0.99)
    sub = replace(
        sub,
        mean=sub.mean.to(DEVICE),
        basis=sub.basis.to(DEVICE),
        explained_variance_ratio=sub.explained_variance_ratio.to(DEVICE),
    )
    out["Global-PCA projection"] = manifold_steer(
        h0, target, linear_inject, sub, n_iters=25
    )
    out["PCA geodesic"] = manifold_steer_local(
        h0, target, linear_inject, bank, k_neighbors=256, n_iters=50, bank_size=50_000
    )

    # MLP-probe gradient: freeze an MLP probe h → (pos of both objects), then
    # descend on h until the probe reads the target.
    probe = _fit_mlp(states_bank, pos_bank)
    h = h0.clone().requires_grad_(True)
    opt = torch.optim.Adam([h], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        ((probe(h) - target) ** 2).mean().backward()
        opt.step()
    out["MLP-probe gradient"] = h.detach()

    # Decoder gradient (oracle): match the true post-edit obs through the decoder.
    h = h0.clone().requires_grad_(True)
    opt = torch.optim.Adam([h], lr=0.05)
    for _ in range(250):
        opt.zero_grad()
        ((model.decode(model.state_from_flat(h)) - tgt_obs) ** 2).mean().backward()
        opt.step()
    out["Decoder gradient"] = h.detach()
    return out


def scorecard(ROLL, name, zones, gt_roll):
    """The canonical §4 scorecard — see `scripts/editability_metrics.py`.

    Adds `anti_reversion` (stickiness of the edit relative to the unsteered rollout),
    which is the one §4 metric that is genuinely about *change* rather than correctness
    and therefore still takes the unsteered rollout as its reference.
    """
    card = edit_scorecard(ROLL[name], zones, gt_roll)

    def rms_all(a, b):
        return float(np.sqrt(((a - b) ** 2).mean()))

    chg0 = rms_all(ROLL[name][:, 0], ROLL["Unsteered"][:, 0])
    chg_late = np.mean(
        [rms_all(ROLL[name][:, s], ROLL["Unsteered"][:, s]) for s in range(10, K)]
    )
    card["anti_reversion"] = chg_late / max(chg0, 1e-9)
    return card


def eval_editability(model, edits, sim, n_edit, seed=0):
    ef = edits.edit_frame
    N = min(n_edit, edits.n_samples)
    oe = edits.edit_object[:N].astype(int)

    obs = edits.obs[:N]
    with h5py.File(edits.h5_path, "r") as f:
        pre_vel = f["velocities"][:N, ef - 1, :N_OBJ, :].astype(np.float32)

    # the two ground-truth worlds at the edit frame + the ray zones (canonical module)
    gt_roll = edits.clean_obs[:N, ef : ef + K].astype(np.float32)
    zones = build_edit_zones(
        pre_pos=edits.positions[:N, ef - 1, :N_OBJ, :].astype(np.float32),
        tgt_pos=edits.positions[:N, ef, :N_OBJ, :].astype(np.float32),
        pre_vel=pre_vel,
        edit_object=oe,
        sim=sim,
        n_obj=N_OBJ,
        traj_pos=edits.positions[:N, ef : ef + K, :N_OBJ, :].astype(np.float32),
        gt_edited_traj=gt_roll,
    )
    tgt_pos = edits.positions[:N, ef, :N_OBJ, :].astype(np.float32)
    target = torch.from_numpy(tgt_pos.reshape(N, N_OBJ * 2)).float().to(DEVICE)
    tgt_obs = torch.from_numpy(gt_roll[:, 0]).float().to(DEVICE)

    h0 = torch.from_numpy(warm_to_edit(model, obs, ef)).float().to(DEVICE)
    # "Oracle observation": teacher-force one extra frame -- the REAL (noisy) post-edit
    # observation edits.obs[ef].  Not a state swap; the model simply gets to SEE the teleport.
    h_swap = torch.from_numpy(tf_hidden_at(model, obs, ef)).float().to(DEVICE)

    # probe/manifold bank: teacher-forced states over the edits sequences
    hb = hidden_states(model, obs)
    states_bank = hb.reshape(-1, hb.shape[-1])
    pos_bank = edits.positions[:N, : hb.shape[1], :N_OBJ, :].reshape(-1, N_OBJ * 2)

    Hs = {"Unsteered": h0, "Oracle observation": h_swap}
    Hs.update(build_editors(model, h0, target, states_bank, pos_bank, tgt_obs))
    ROLL = {k: rollout(model, v) for k, v in Hs.items()}

    cards = {ed: scorecard(ROLL, ed, zones, gt_roll) for ed in EDITORS}
    for ed in EDITORS:
        cards[ed]["fidelity_ratio"] = fidelity_ratio(cards[ed], cards["Unsteered"])

    # everything a notebook needs to draw the canonical waterfall without
    # re-opening the dataset: context frames, ray-centroid locators, teleport size
    def centroid(m):
        out = np.full(len(m), np.nan)
        for i in range(len(m)):
            idx = np.where(m[i])[0]
            if idx.size:
                out[i] = idx.mean()
        return out

    viz = dict(
        gt_roll=gt_roll,
        gt_after_ef=edits.clean_obs[:N, ef + 1 : ef + K].astype(np.float32),
        gt_unedited=zones.gt_unedited,
        ctx=edits.obs[:N, ef - 6 : ef].astype(np.float32),
        tgt_cx=centroid(zones.target),
        ghost_cx=centroid(zones.ghost),
        teleport=zones.teleport,
        n_ghost_rays=zones.ghost.sum(1),
        n_target_rays=zones.target.sum(1),
        n_differing_rays=zones.differing.sum(1),
        edit_frame=np.array([ef]),
    )
    return cards, ROLL, viz


# ── driver ────────────────────────────────────────────────────────────────────


def eval_run(code: str, n_probe: int, n_edit: int) -> dict:
    ckpt = ROOT / code / "best_model.pt"
    # runs not in the table (e.g. the trained-editability arms) are all on dataset 4
    data_dir = Path("datasets") / RUN_DATASET.get(code, "4_fixed_refl_inview")
    model, info = load_checkpoint(ckpt, device=DEVICE)
    bundle = load_dataset(data_dir, n_obj_keep=N_OBJ)
    test, edits = bundle.test, bundle.edits
    sim = test.config["dataset"]["sim"]

    obs = test.obs[:n_probe]
    clean = test.clean_obs[:n_probe]
    vis = test.is_visible[:n_probe, :-1, :N_OBJ].all(axis=2)
    pos = test.positions[:n_probe, :-1, :N_OBJ, :]
    vel = read_velocities(test.h5_path, n_probe)[:, :-1, :N_OBJ, :]

    res: dict = {
        "run": code,
        "dataset": RUN_DATASET.get(code, "4_fixed_refl_inview"),
        "hidden_size": int(info.model_config["hidden_size"]),
        "n_params": sum(p.numel() for p in model.parameters()),
        "obs_noise_std": float(sim["obs_noise_std"]),
        "position_noise_std": float(sim["position_noise_std"]),
        "epoch": int(info.epoch),
        "val_loss": float(info.val_loss),
        "swap_frame_lead": 1,
    }

    # 1 — predictive quality (each model against ITS OWN dataset baselines)
    bl = compute_obs_baselines(obs, clean, float(sim["obs_noise_std"]))
    res["baselines"] = {
        "identity_rmse": bl.identity_rmse,
        "noise_floor_rmse": bl.noise_floor_rmse,
        "random_rmse": bl.random_rmse,
    }
    res["nextstep_rmse_vs_clean"] = next_step_rmse(model, obs, clean)
    res["freerun_rmse_by_step"] = freerun_rmse_by_step(model, obs, clean).tolist()

    # 2/3 — recoverability + canonicality on frames where both objects are visible
    Hs = hidden_states(model, obs)
    T = Hs.shape[1]
    P = pos[:, :T].reshape(len(pos), T, N_OBJ * 2)  # (N, T, 4)
    V = vel[:, :T].reshape(len(vel), T, N_OBJ * 2)
    m = vis[:, :T]
    res.update(recoverability_and_canonicality(Hs[m], P[m], V[m]))

    late = np.zeros_like(m)
    late[:, LATE_T:] = True
    ml = m & late
    res.update(
        {
            f"late_{k}": v
            for k, v in recoverability_and_canonicality(Hs[ml], P[ml], V[ml]).items()
        }
    )

    # 4 — editability
    cards, ROLL, viz = eval_editability(model, edits, sim, n_edit)
    res["editability"] = cards

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / f"{code}_rollouts.npz",
        **viz,
        **{f"roll_{k}": v for k, v in ROLL.items()},
    )
    (OUT / f"{code}.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument(
        "--root",
        default="runs/controls",
        help="directory holding the run folders; also where eval/ is written",
    )
    ap.add_argument(
        "--n-probe", type=int, default=2000, help="test sequences for probes"
    )
    ap.add_argument("--n-edits", type=int, default=64)
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    global ROOT, OUT
    ROOT = Path(args.root)
    OUT = ROOT / "eval"

    for code in args.runs:
        if not (ROOT / code / "best_model.pt").exists():
            print(f"skip {code}: no checkpoint")
            continue
        r = eval_run(code, args.n_probe, args.n_edits)
        print(
            f"\n### {code}  H={r['hidden_size']}  obs_noise={r['obs_noise_std']}  "
            f"pos_noise={r['position_noise_std']}"
        )
        print(
            f"  next-step RMSE {r['nextstep_rmse_vs_clean']:.4f}  "
            f"(copy-prev {r['baselines']['identity_rmse']:.4f}, "
            f"noise floor {r['baselines']['noise_floor_rmse']:.4f})"
        )
        print(
            f"  pos R2 lin {r['pos_r2_linear']:.3f} mlp {r['pos_r2_mlp']:.3f} | "
            f"vel R2 lin {r['vel_r2_linear']:.3f} mlp {r['vel_r2_mlp']:.3f} | "
            f"fiber lin {r['fiber_resid_linear']:.3f} mlp {r['fiber_resid_mlp']:.3f}"
        )
        print(
            f"    {'editor':<24s}{'EditIdx':>9s}{'Target':>8s}{'Ghost':>8s}"
            f"{'Collat':>8s}{'EditFrm':>9s}{'GTtraj':>8s}{'fidel':>7s}"
        )
        for ed in EDITORS:
            c = r["editability"][ed]
            print(
                f"    {ed:<24s}{c['edit_index']:>+9.2f}{c['target_rmse']:>8.3f}"
                f"{c['ghost_rmse']:>8.3f}{c['collateral_rmse']:>8.3f}"
                f"{c['edit_frame_rmse']:>9.3f}{c['gt_traj_rmse']:>8.3f}"
                f"{c['fidelity_ratio']:>7.2f}"
            )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
