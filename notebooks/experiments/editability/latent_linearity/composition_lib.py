"""Latent object-composition, trained vs randomly initialised — the shared implementation.

The question (Sevan, 2026-08-21): `delta_h_analysis` §7 found that `[move obj0] + [move obj1]`
recovers most of what `[move both]` does in latent space (cos +0.873, relative residual 0.39–0.69,
‖composed‖/‖direct‖ 1.13). Is that a **learned** property, or does a randomly initialised network
do it too?

Three things this implementation gets right that a first pass did not:

1. **Real teleport targets, from `edits.h5`.** Displacing every object in the same direction makes
   Δ_A and Δ_B point the same way and inflates every null.
2. **A triviality baseline.** `Δ_AB` is correlated with `Δ_A` by construction — the AB world
   *contains* the A displacement — so `cos(Δ_A + Δ_B, Δ_AB)` can look high without composition
   doing any work. `cos(Δ_A, Δ_AB)` is reported beside it to show how much is mere overlap.
3. **A displacement-magnitude sweep.** Additivity is a first-order Taylor property of *any* smooth
   map: `f(x+δ_A+δ_B) ≈ f(x) + J δ_A + J δ_B` with error second-order in ‖δ‖. Sweeping ‖δ‖
   separates "additive because everything is locally linear" from "additive because the
   representation is object-factored".

⚠ **What cannot be measured on a random model.** `delta_h_analysis` §7's decisive readout was the
composed state *applied and rolled out*, scored as "% of the direct edit's Edit-Index gain" — it
warned in those words that "vector agreement alone can mislead". A random decoder emits garbage, so
that readout is undefined here and only the vector metrics transfer. State this whenever these
numbers are quoted beside the original's 83–87%.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

# Absolute, resolved from this file — the notebook that imports this runs with its own directory
# as cwd, so repo-relative paths silently break.
REPO = Path(__file__).resolve().parents[4]
EF, N_OBJ = 20, 2
DATA = REPO / "datasets" / "4_fixed_refl_inview"
CKPT = {"linear enc+dec": str(REPO / "runs/controls/H256/best_model.pt"),
        "nonlinear enc+dec": str(REPO / "runs/nonlinear_gru/NL_enc2dec2_s0/best_model.pt")}
ARCH = {"linear enc+dec": dict(enc_hidden_layers=0, dec_hidden_layers=0),
        "nonlinear enc+dec": dict(enc_hidden_layers=2, dec_hidden_layers=2)}


def world():
    import sys
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    from editability_metrics import object_constants, sim_config_from
    sim = json.load(open(DATA / "dataset.json"))["sim"]
    cfg = sim_config_from(sim, N_OBJ)
    rad, refl = object_constants(sim, N_OBJ)
    return sim, cfg, rad, refl


def edit_set(n: int, seed: int = 0):
    """Real teleport targets for BOTH objects, plus the UN-teleported reference world.

    ⛔ On `edits.h5` the teleport is already **in the data**: `positions[ef]` IS the post-edit
    world. Measuring a displacement from `positions[ef]` therefore gives ZERO for the episode's
    own edit object. The un-teleported position has to be reconstructed ballistically from
    `ef-1` — the same correction `edit_directions.py` documents.

    Returns
    -------
    pos, vel : (n, T, n_obj, 2) the raw arrays.
    uned     : (n, n_obj, 2) where each object would be at `ef` had no teleport happened.
    tgt      : (n, n_obj, 2) a real teleport target for each object. The episode's own edit
               object uses its recorded `edit_value`; the other draws from the pool of recorded
               targets for that slot (the frustum is a fixed region, so in-frustum transfers).
    """
    with h5py.File(DATA / "edits.h5", "r") as f:
        pos = f["positions"][:n, :, :N_OBJ, :].astype(np.float32)
        vel = f["velocities"][:n, :, :N_OBJ, :].astype(np.float32)
        eobj_all = f["edit_object"][:].astype(int)
        eval_all = f["edit_value"][:].astype(np.float32)
    sim = json.load(open(DATA / "dataset.json"))["sim"]
    dt = float(sim["dt"])
    rng = np.random.default_rng(seed)
    pool = {o: eval_all[eobj_all == o] for o in range(N_OBJ)}

    uned = pos[:, EF].copy()
    tgt = pos[:, EF].copy()
    for i in range(n):
        o = eobj_all[i]
        # the world where THIS episode's teleport never happened
        uned[i, o] = pos[i, EF - 1, o] + vel[i, EF - 1, o] * dt
        tgt[i, o] = eval_all[i]
        oth = 1 - o
        tgt[i, oth] = pool[oth][rng.integers(len(pool[oth]))]
    return pos, vel, uned, tgt


def histories(pos, vel, uned, tgt, scale, sim, cfg, rad, refl):
    """The four rendered counterfactual histories, at `scale` x the real teleport.

    `base` is the UN-teleported world; a moved object follows a constant-velocity line that
    arrives at its (scaled) target at frame EF. Frames before EF are the true pre-teleport
    history in every world, so the four differ only in where each object is heading.
    """
    from pim.simulator.renderer import render_frame
    n, R, dt = len(pos), int(sim["obs_res"]), float(sim["dt"])
    t_idx = np.arange(EF + 1)
    H = {k: np.zeros((n, EF + 1, R), np.float32) for k in ("base", "A", "B", "AB")}
    for i in range(n):
        base_seq = pos[i, : EF + 1].copy()
        base_seq[EF] = uned[i]                       # undo the teleport that is in the data
        line = {}
        for o in range(N_OBJ):
            t = uned[i, o] + scale * (tgt[i, o] - uned[i, o])
            line[o] = t[None] - vel[i, EF, o][None] * (EF - t_idx)[:, None] * dt
        for key, moved in (("base", ()), ("A", (0,)), ("B", (1,)), ("AB", (0, 1))):
            h = base_seq.copy()
            for o in moved:
                h[:, o] = line[o]
            H[key][i] = np.stack([render_frame(q.astype(np.float32), rad, refl, cfg)[2] for q in h])
    return H


def render_ceiling(H):
    """The observation's OWN relative non-additivity — the floor no latent can beat.

    The two objects share rays, so `AB - (A + B - base)` is not zero even in the render.
    """
    r = H["AB"] - (H["A"] + H["B"] - H["base"])
    return float(np.sqrt((r ** 2).mean()) / np.sqrt(((H["AB"] - H["base"]) ** 2).mean()))


def models(device="cuda", seeds=(0, 1)):
    from pim.world_models import load_checkpoint
    from pim.world_models.gru.model import GRUModel, ModelConfig
    sim, *_ = world()
    R = int(sim["obs_res"])
    out = {}
    for name, kw in ARCH.items():
        m, _ = load_checkpoint(CKPT[name], device=device)
        out[f"TRAINED {name}"] = m.eval()
        for s in seeds:
            torch.manual_seed(s)
            out[f"RANDOM s{s} {name}"] = GRUModel(
                ModelConfig(input_dim=R, hidden_size=256, **kw)).to(device).eval()
    return out


@torch.no_grad()
def latents(model, H, device="cuda"):
    out = {}
    for k, arr in H.items():
        _, st = model(torch.from_numpy(arr).float().to(device))
        out[k] = model.flat_state(st).float().cpu().numpy()
    return out


def metrics(L, seed=0):
    """Every number reported. See the notebook's definitions table for formulas and units."""
    dA, dB, dAB = L["A"] - L["base"], L["B"] - L["base"], L["AB"] - L["base"]
    comp = dA + dB
    def nrm(v):
        return np.linalg.norm(v, axis=1)

    def cos(u, v):
        return (u * v).sum(1) / (nrm(u) * nrm(v) + 1e-12)

    rng = np.random.default_rng(seed)
    sA, sB = rng.permutation(len(dA)), rng.permutation(len(dB))
    return dict(
        cos=float(cos(comp, dAB).mean()),
        cos_std=float(cos(comp, dAB).std()),
        cos_trivial=float(cos(dA, dAB).mean()),
        resid=float((nrm(dAB - comp) / (nrm(dAB) + 1e-12)).mean()),
        resid_trivial=float((nrm(dAB - dA) / (nrm(dAB) + 1e-12)).mean()),
        norm_ratio=float((nrm(comp) / (nrm(dAB) + 1e-12)).mean()),
        # BOTH deltas shuffled. Shuffling only one leaves dA's overlap with dAB intact and the
        # "floor" is then bounded below by cos_trivial, which is not a floor at all.
        cos_shuffled=float(cos(dA[sA] + dB[sB], dAB).mean()),
    )
