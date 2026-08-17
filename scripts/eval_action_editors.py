#!/usr/bin/env python3
"""Full editor ablation on the exogenous-action models + their control.

One pass per world model. For each, every editor is scored on the SAME held-out edit set with the
canonical §4 metrics (`scripts/editability_metrics.py`), so the families are directly comparable:

  **standard / training-free** — Unsteered · Pseudoinverse Injection · Metric-corrected Injection
      (the un-whitened Σ¹ write from `metric_corrected_edits`) · Global PCA Projection (POCS) ·
      Local PCA Geodesic · MLP Grad Steering (frozen 1×128) · Multistep Steering (PI) @8
  **oracle** — Oracle observation · Counterfactual Overwriting · Freeze-time TF ·
      Decoder Grad k=1 · Decoder Grad k=8 · Action interface (action-conditioned model only)
  **trained** (6 per model, from `scripts/train_action_editors.py`) —
      Fine-tune × {pinv, metric-corrected} × {k=1, k=8} · MLP editor × {k=1, k=8}

⚠ **A fine-tuned arm is a DIFFERENT world model**, so its Edit Index is meaningless against the
base model's unsteered row. Each fine-tuned arm therefore carries its **own** unsteered row and
its **own** next-step RMSE, and the notebook reports the gain over that. This is the exact trap
recorded on 2026-07-30: a no-retention fine-tune's unsteered index ROSE from degraded prediction
alone, so its apparent gain was scale movement. The MLP-editor arms leave the world model frozen
and therefore share the base model's unsteered row.

Writes `runs/action_editors/eval/<model>.json` + `<model>_rollouts.npz`.

Usage
-----
    python scripts/eval_action_editors.py --models XG_A_H256 XG_C_H256 CTRL_H256 --n-edits 128
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import eval_action_sweep as EAS
import train_action_editors as TAE
from editability_metrics import build_edit_zones, edit_scorecard, fidelity_ratio
from pim.editors import (
    fit_state_subspace,
    inject_state,
    manifold_steer,
    manifold_steer_local,
    probe_decomposition,
)
from pim.extractors import LinearExtractor, MLPExtractor, StateDefinition
from pim.world_models import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EF, K, N_OBJ, N_CTX = 20, 15, 2, 6
RUN_DIR = Path("runs/action_editors")

STANDARD = ["Unsteered", "Pseudoinverse Injection", "Metric-corrected Injection (un-whitened)",
            "Global PCA Projection", "Local PCA Geodesic", "MLP Grad Steering",
            "Multistep Steering @8"]
ORACLE = ["Oracle observation", "Counterfactual Overwriting", "Freeze-time TF @8",
          "Decoder Grad k=1", "Decoder Grad k=8", "Action interface"]
TRAINED = ["Fine-tune · pinv", "Fine-tune · metric-corrected", "MLP editor"]


# ── data ──────────────────────────────────────────────────────────────────────
def ctrl_data(n_edits: int):
    """Dataset-4 edits split — `edits[:HELD_OUT]` is the reporting set the trainers never saw."""
    bundle = load_dataset(Path("datasets/4_fixed_refl_inview"), n_obj_keep=N_OBJ)
    test, edits = bundle.test, bundle.edits
    sim = test.config["dataset"]["sim"]
    n = n_edits
    pos = edits.positions[:n, :, :N_OBJ, :].astype(np.float32)
    with h5py.File(edits.h5_path, "r") as f:
        vel = f["velocities"][:n, :, :N_OBJ, :].astype(np.float32)
    gt_roll = edits.clean_obs[:n, EF : EF + K].astype(np.float32)
    zones = build_edit_zones(pre_pos=pos[:, EF - 1], tgt_pos=pos[:, EF], pre_vel=vel[:, EF - 1],
                            edit_object=edits.edit_object[:n].astype(int), sim=sim, n_obj=N_OBJ,
                            traj_pos=pos[:, EF : EF + K], gt_edited_traj=gt_roll)
    start = (pos[:, EF - 1] + vel[:, EF - 1]).reshape(n, N_OBJ * 2)
    return dict(sim=sim, obs=edits.obs[:n].astype(np.float32), act_noop=None, act_edit=None,
                pos=pos, vel=vel, zones=zones, gt_roll=gt_roll,
                tgt_pos=pos[:, EF].reshape(n, N_OBJ * 2), start=start,
                edit_obj=edits.edit_object[:n].astype(int),
                probe_obs=test.obs[:600].astype(np.float32), probe_act=None,
                probe_pos=test.positions[:600, :, :N_OBJ, :].astype(np.float32))


def xg_eval_data(n_edits: int, use_actions: bool):
    E = EAS.xg_data(n_edits, n_probe=600, seed=0)
    n = len(E["obs"])
    E["start"] = E["pos"][:, EF].reshape(n, N_OBJ * 2).astype(np.float32)
    E["tgt_pos"] = E["tgt_pos"].reshape(n, N_OBJ * 2).astype(np.float32)
    if not use_actions:
        # the observer twin is a plain GRUModel — it has no action port at all, so every
        # action array must be dropped, including the probe-bank one
        E["act_noop"] = E["act_edit"] = E["probe_act"] = None
    return E


# ── model ops ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def warm(model, obs, act, upto=EF):
    o = torch.from_numpy(obs).float().to(DEVICE)
    a = torch.from_numpy(act).float().to(DEVICE) if act is not None else None
    state = None
    for t in range(upto):
        state = (model.step(o[:, t], state, action=a[:, t])[1] if a is not None
                 else model.step(o[:, t], state)[1])
    return model.flat_state(state).float()


@torch.no_grad()
def feed(model, h, frames, act=None):
    state = model.state_from_flat(h)
    f = torch.from_numpy(np.ascontiguousarray(frames)).float().to(DEVICE)
    for t in range(f.shape[1]):
        state = (model.step(f[:, t], state, action=act[:, t])[1] if act is not None
                 else model.step(f[:, t], state)[1])
    return model.flat_state(state).float()


@torch.no_grad()
def rollout(model, h, steps=K):
    h = h if torch.is_tensor(h) else torch.from_numpy(np.asarray(h))
    state = model.state_from_flat(h.float().to(DEVICE))
    out = [model.decode(state)]
    for _ in range(steps - 1):
        p, state = model.predict_step(state)
        out.append(p)
    return torch.stack(out, 1).cpu().numpy()


@torch.no_grad()
def nextstep_rmse(model, E, use_actions):
    o = torch.from_numpy(E["probe_obs"]).float().to(DEVICE)
    a = torch.from_numpy(E["probe_act"]).float().to(DEVICE) if E.get("probe_act") is not None else None
    pred = (model.observe_sequence(o, actions=a)[0] if a is not None
            else model.observe_sequence(o)[0]).cpu().numpy()
    T = pred.shape[1]
    sim_cfg = EAS.sim_from_dataset4()
    clean = np.stack([[EAS.clean_render(E["probe_pos"][i, t + 1], sim_cfg)[0]
                       for t in range(T)] for i in range(len(pred))])
    return float(np.sqrt(((pred - clean) ** 2).mean()))


# ── the editor suite ──────────────────────────────────────────────────────────
def build_standard(model, h0, target, bank, pos_bank):
    H = h0.shape[1]
    sdef = StateDefinition(name="positions_flat", state_shape=(N_OBJ * 2,),
                           extract_fn=lambda b: b["positions"])
    lin = LinearExtractor(H, sdef, use_lstsq=True)
    lin.fit(bank[None], pos_bank[None], device=DEVICE)
    lin = lin.to(DEVICE).eval()
    A, b_, A_pinv = probe_decomposition(lin)

    def inject(h, t):
        return inject_state(h, t, A, A_pinv, b_)

    out = {"Pseudoinverse Injection": inject(h0, target)}

    # the un-whitened Σ¹ write — same readout target, different metric
    W_t = A.T.contiguous()                       # (H, 4) to match the trainer's convention
    Sig = np.cov(bank.astype(np.float64).T)
    metric = TAE.metric_inject_factory(W_t, b_, Sig, alpha=1.0)
    out["Metric-corrected Injection (un-whitened)"] = metric(h0, target)

    bt = torch.from_numpy(bank).float().to(DEVICE)
    sub = fit_state_subspace(bt, var_threshold=0.99)
    sub = replace(sub, mean=sub.mean.to(DEVICE), basis=sub.basis.to(DEVICE),
                  explained_variance_ratio=sub.explained_variance_ratio.to(DEVICE))
    out["Global PCA Projection"] = manifold_steer(h0, target, inject, sub, n_iters=25)
    out["Local PCA Geodesic"] = manifold_steer_local(h0, target, inject, bt, k_neighbors=256,
                                                     n_iters=50, bank_size=50_000)

    probe = MLPExtractor(H, sdef).to(DEVICE)
    probe.fit(bank[None], pos_bank[None], device=DEVICE)
    probe = probe.to(DEVICE).eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    h = h0.clone().requires_grad_(True)
    opt = torch.optim.Adam([h], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        ((probe(h) - target) ** 2).mean().backward()
        opt.step()
    out["MLP Grad Steering"] = h.detach()
    return out, inject


def multistep_steering(model, h0, target, inject, steps=8, eta=1.0):
    """Push the readout a little, decode, feed the model's OWN decoded observation back, repeat.

    The model's own machinery only — never an externally rendered frame. That is exactly what
    separates it from freeze-time teacher forcing.
    """
    h = h0.clone()
    with torch.no_grad():
        for s in range(steps):
            cur = h.clone()
            h = cur + eta * (inject(cur, target) - cur) / max(steps - s, 1)
            state = model.state_from_flat(h)
            obs_hat = model.decode(state)
            h = model.flat_state(model.step(obs_hat, state)[1]).float()
    return h


def decoder_grad(model, h0, gt, iters=250, lr=0.05):
    """Adam on h so the model's own K-step rollout matches the GT post-edit observations.

    Runs the model in train() mode: cuDNN refuses to backprop through an RNN in eval mode,
    and this GRU has no dropout, so train()/eval() are behaviourally identical. The model
    weights are untouched — only `h` is optimised.
    """
    was_training = model.training
    model.train()
    h = h0.clone().requires_grad_(True)
    opt = torch.optim.Adam([h], lr=lr)
    steps = gt.shape[1]
    for _ in range(iters):
        opt.zero_grad()
        state = model.state_from_flat(h)
        outs = [model.decode(state)]
        for _ in range(steps - 1):
            p, state = model.predict_step(state)
            outs.append(p)
        F.mse_loss(torch.stack(outs, 1), gt).backward()
        opt.step()
    model.train(was_training)
    return h.detach()


def render_world(pos, sim_cfg):
    return EAS.clean_render(pos, sim_cfg)[0]


def build_oracles(model, E, use_actions, h0, sim_cfg):
    """Externally rendered / ground-truth-access editors."""
    n = len(E["obs"])
    out = {}
    # Oracle observation: teacher-force ONE extra frame — the real post-edit observation.
    # It therefore LEADS every other column by one frame; labelled, never re-aligned.
    edited_frame = E["gt_roll"][:, 0:1]
    out["Oracle observation"] = feed(model, h0, edited_frame)

    # Counterfactual Overwriting: a fabricated history in which the object always travelled
    # toward the target — rendered, then teacher-forced.
    n_hist = 8
    delta = (E["tgt_pos"] - E["start"]).reshape(n, N_OBJ, 2)
    hist = np.zeros((n, n_hist + 1, E["gt_roll"].shape[-1]), np.float32)
    for j in range(n_hist + 1):
        p = E["pos"][:, EF - 1 - j] + delta
        for i in range(n):
            hist[i, j] = render_world(p[i], sim_cfg)
    h_start = warm(model, E["obs"], E["act_noop"], EF - 1 - n_hist)
    out["Counterfactual Overwriting"] = feed(model, h_start, hist[:, : n_hist + 1][:, ::-1])

    # Freeze-time TF: the world is frozen and the edited object interpolates to the target over
    # N externally rendered frames, then time resumes.
    n_fr = 8
    frames = np.zeros((n, n_fr, E["gt_roll"].shape[-1]), np.float32)
    for j in range(n_fr):
        w = (j + 1) / n_fr
        p = ((1 - w) * E["start"] + w * E["tgt_pos"]).reshape(n, N_OBJ, 2)
        for i in range(n):
            frames[i, j] = render_world(p[i], sim_cfg)
    out["Freeze-time TF @8"] = feed(model, h0, frames)
    return out


def score(ROLL, zones, gt_roll, ref_unsteered=None):
    cards = {k: edit_scorecard(v, zones, gt_roll) for k, v in ROLL.items()}
    ref = cards[ref_unsteered] if ref_unsteered else cards["Unsteered"]
    for c in cards.values():
        c["fidelity_ratio"] = fidelity_ratio(c, ref)
    return cards


# ── per-model driver ──────────────────────────────────────────────────────────
def eval_model(model_key: str, n_edits: int) -> dict:
    spec = TAE.MODELS[model_key]
    use_actions = spec["actions"]
    base = TAE.load_base(spec)
    base.eval()
    E = ctrl_data(n_edits) if spec["kind"] == "ctrl" else xg_eval_data(n_edits, use_actions)
    sim_cfg = EAS.sim_from_dataset4()
    zones, gt_roll = E["zones"], E["gt_roll"]
    n = len(E["obs"])
    target = torch.from_numpy(E["tgt_pos"]).float().to(DEVICE)
    start = torch.from_numpy(E["start"]).float().to(DEVICE)
    gt_k1 = torch.from_numpy(gt_roll[:, :1]).float().to(DEVICE)
    gt_k8 = torch.from_numpy(gt_roll[:, :8]).float().to(DEVICE)

    res = {"model": model_key, "label": spec["label"], "n_edits": n,
           "hidden_size": int(base.hidden_size), "use_actions": use_actions}

    # probe bank on the base model
    with torch.no_grad():
        ob = torch.from_numpy(E["probe_obs"]).float().to(DEVICE)
        ab = (torch.from_numpy(E["probe_act"]).float().to(DEVICE)
              if E.get("probe_act") is not None else None)
        hb = (base.get_hidden_states(ob, actions=ab) if ab is not None
              else base.get_hidden_states(ob)).cpu().numpy()
    bank = hb.reshape(-1, hb.shape[-1])
    pos_bank = E["probe_pos"][:, : hb.shape[1], :N_OBJ, :].reshape(-1, N_OBJ * 2)

    h0 = warm(base, E["obs"], E["act_noop"])
    ROLL = {"Unsteered": rollout(base, h0)}
    std, inject = build_standard(base, h0, target, bank, pos_bank)
    for k, v in std.items():
        ROLL[k] = rollout(base, v)
    ROLL["Multistep Steering @8"] = rollout(base, multistep_steering(base, h0, target, inject))
    ROLL["Decoder Grad k=1"] = rollout(base, decoder_grad(base, h0, gt_k1))
    ROLL["Decoder Grad k=8"] = rollout(base, decoder_grad(base, h0, gt_k8))
    for k, v in build_oracles(base, E, use_actions, h0, sim_cfg).items():
        ROLL[k] = rollout(base, v)
    if use_actions:
        ROLL["Action interface"] = rollout(base, warm(base, E["obs"], E["act_edit"]))

    cards = score(ROLL, zones, gt_roll)
    for c in cards.values():
        c["own_unsteered"] = cards["Unsteered"]["edit_index"]
    res["base_nextstep_rmse"] = nextstep_rmse(base, E, use_actions)
    res["editors"] = dict(cards)

    # ── trained arms ──────────────────────────────────────────────────────────
    trained = {}
    for ck in sorted(RUN_DIR.glob(f"{model_key}__*/ckpt.pt")):
        d = torch.load(ck, map_location=DEVICE, weights_only=False)
        arm, ed_k = d["arm"], d["edit_k"]
        W = torch.tensor(d["probe"]["W"], device=DEVICE)
        b_ = torch.tensor(d["probe"]["b"], device=DEVICE)
        Wp = torch.tensor(d["probe"]["W_pinv"], device=DEVICE)
        if d["editor"] == "finetune":
            m = TAE.load_base(spec)
            m.load_state_dict(d["model_state"])
            m.to(DEVICE).eval()
            h0_ft = warm(m, E["obs"], E["act_noop"])
            if d.get("write", "pinv") == "metric":
                Sig, _ = TAE.fit_state_covariance(
                    m, E["probe_obs"][:400],
                    E["probe_act"][:400] if E.get("probe_act") is not None else None)
                wfn = TAE.metric_inject_factory(W, b_, Sig, alpha=1.0)
                h_ed = wfn(h0_ft, target)
            else:
                h_ed = TAE.readout_inject(h0_ft, target, W, b_, Wp)
            R = {"Unsteered": rollout(m, h0_ft), "Edited": rollout(m, h_ed)}
            c = score(R, zones, gt_roll)
            trained[arm] = dict(
                edit_k=ed_k, kind="finetune", write=d.get("write", "pinv"),
                own_unsteered=c["Unsteered"]["edit_index"],
                own_unsteered_by_step=c["Unsteered"]["edit_index_by_step"],
                nextstep_rmse=nextstep_rmse(m, E, use_actions),
                **c["Edited"])
            np.savez_compressed(RUN_DIR / "eval" / f"{arm}_roll.npz", **R)
        else:
            ed = TAE.StateTargetEditor(base.hidden_size).to(DEVICE)
            ed.load_state_dict(d["editor_state"])
            ed.eval()
            with torch.no_grad():
                h_ed = ed(h0, start, target)
            R = {"Unsteered": ROLL["Unsteered"], "Edited": rollout(base, h_ed)}
            c = score(R, zones, gt_roll)
            trained[arm] = dict(
                edit_k=ed_k, kind="mlp", write="learned",
                own_unsteered=c["Unsteered"]["edit_index"],
                own_unsteered_by_step=c["Unsteered"]["edit_index_by_step"],
                nextstep_rmse=res["base_nextstep_rmse"],
                **c["Edited"])
            np.savez_compressed(RUN_DIR / "eval" / f"{arm}_roll.npz", **R)
        ROLL[arm] = R["Edited"]
    res["trained"] = trained

    np.savez_compressed(RUN_DIR / "eval" / f"{model_key}_rollouts.npz",
                        **{f"roll::{k}": v for k, v in ROLL.items()},
                        ctx=E["obs"][:, EF - N_CTX : EF], gt_roll=gt_roll,
                        tgt_mask=zones.target, ghost_mask=zones.ghost, teleport=zones.teleport)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(TAE.MODELS))
    ap.add_argument("--n-edits", type=int, default=128)
    a = ap.parse_args()
    (RUN_DIR / "eval").mkdir(parents=True, exist_ok=True)
    for mk in a.models:
        print(f"\n=== {mk} ===")
        r = eval_model(mk, a.n_edits)
        (RUN_DIR / "eval" / f"{mk}.json").write_text(json.dumps(r, indent=1))
        print(f"  next-step RMSE {r['base_nextstep_rmse']:.4f}")
        for k, c in r["editors"].items():
            print(f"    {k:<42} index {c['edit_index']:+.3f}  fid {c['fidelity_ratio']:.2f}")
        for k, c in r["trained"].items():
            print(f"    [trained] {k:<32} index {c['edit_index']:+.3f}  "
                  f"own-unsteered {c['own_unsteered']:+.3f}  fid {c['fidelity_ratio']:.2f}  "
                  f"next-step {c['nextstep_rmse']:.4f}")


if __name__ == "__main__":
    main()
