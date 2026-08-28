"""Single-point pseudoinverse injection on `W16`, with the STEP SIZE swept.

⚠ Read this first, so the question is not mis-stated again.

**The single-point write is not new.** `transformers/transformer_world_state.ipynb` §4 (2026-08-04)
already applies `h <- h0 + (target - (W h0 + b)) W+` at **each residual point individually**, on
`W2`/`W4`/`W16`, and finds it inert at every one: Edit Index -0.65…-0.68, equal to each model's own
unsteered value, at fidelity ratio 1.00.

**What 2026-08-04 did not do is sweep alpha.** It took the full jump — alpha = 1, the exact
minimum-norm write that lands the read-out on the target. On Othello-GPT the same editor's
single-point optimum sits at **alpha = 1.5**, and the outcome varies ~50x across the alpha range
(2026-08-21). So `alpha = 1` being inert here does not by itself establish that no step size works.
This script closes that one gap and nothing else.

Everything else is held to 2026-08-04's configuration so the alpha = 1 column reproduces its
published number: same edits split, N = 192, K = 15, probe fit on 1200 TEST sequences by exact
`lstsq` per residual point, target = the true post-edit positions of both objects.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
for _p in (str(_HERE), str(_HERE.parent / "othello_gpt"), str(_REPO), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_REPO)  # `othello_gpt/pipeline.load` resolves runs/ and datasets/ against the CWD

import othello_probe as op  # noqa: E402
import pipeline as pl  # noqa: E402
from editability_metrics import (  # noqa: E402
    build_edit_zones,
    edit_index_by_step,
    edit_scorecard,
    fidelity_ratio,
)

# 2026-08-04's constants, unchanged — this is what makes the alpha = 1 column an anchor.
N_OBJ, K_ROLL, N_EVAL, N_PROBE, EF, SEED = 2, 15, 192, 1200, 20, 0
ALPHAS = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
REF_INERT = (-0.68, -0.65)  # the published band the alpha = 1 column must land in

torch.manual_seed(SEED)
np.random.seed(SEED)
DEV = pl.DEVICE
t0 = time.time()

b = pl.load("W16")
model, sim, test, edits = b.model, b.sim, b.test, b.edits
NP = model.cfg.n_layers + 1
print(f"W16: {model.cfg.n_layers} layers -> {NP} residual points · device {DEV}")

# ── probes: exact lstsq onto the 4 position outputs, one per residual point, fit on TEST ──
obs_p = test.obs[:N_PROBE].astype(np.float32)
pos_p = test.positions[:N_PROBE, :, :N_OBJ, :].astype(np.float32)
R = op.collect_residuals(model, obs_p, batch=128)          # (NP, N, T, d_model)
T = R.shape[2]
Y = pos_p[:, :T].reshape(N_PROBE, T, N_OBJ * 2)

PROBE = {}
print("\nlinear position probe by residual point (exact lstsq, all 1200 test sequences):")
for ell in range(NP):
    X = R[ell].reshape(-1, R.shape[-1]).astype(np.float64)
    A = np.concatenate([X, np.ones((len(X), 1))], 1)
    sol, *_ = np.linalg.lstsq(A, Y.reshape(-1, N_OBJ * 2).astype(np.float64), rcond=None)
    W = torch.tensor(sol[:-1], dtype=torch.float32, device=DEV)
    bb = torch.tensor(sol[-1], dtype=torch.float32, device=DEV)
    Wp = torch.tensor(np.linalg.pinv(sol[:-1]), dtype=torch.float32, device=DEV)
    rmse = float(np.sqrt(((A @ sol - Y.reshape(-1, N_OBJ * 2)) ** 2).mean()))
    PROBE[ell] = (W, bb, Wp)
    print(f"  point {ell}: fit RMSE {rmse:.4f}")
del R

# ── the edit set: the real teleports in edits.h5, exactly as 2026-08-04 read them ──
N = N_EVAL
obs_e = edits.obs[:N].astype(np.float32)
gt_roll = edits.clean_obs[:N, EF:EF + K_ROLL, :].astype(np.float32)
tgt_pos = edits.positions[:N, EF, :N_OBJ, :].astype(np.float32)
pre_pos = edits.positions[:N, EF - 1, :N_OBJ, :].astype(np.float32)
traj = edits.positions[:N, EF:EF + K_ROLL, :N_OBJ, :].astype(np.float32)
with h5py.File(edits.h5_path, "r") as f:
    vel_e = f["velocities"][:N, :, :N_OBJ, :].astype(np.float32)
ZONES = build_edit_zones(pre_pos=pre_pos, tgt_pos=tgt_pos, pre_vel=vel_e[:, EF - 1],
                         edit_object=edits.edit_object[:N].astype(int), sim=sim, n_obj=N_OBJ,
                         traj_pos=traj, gt_edited_traj=gt_roll)
tgt4 = torch.from_numpy(tgt_pos.reshape(N, N_OBJ * 2)).float().to(DEV)
state = model.state_from_obs(torch.from_numpy(obs_e[:, :EF]).float().to(DEV))
print(f"\nedit set: N={N} ef={EF} K={K_ROLL} · differing rays/sample "
      f"{ZONES.differing.sum(1).mean():.1f}")


@torch.no_grad()
def act_at(ell):
    model.state_view, model.probe_layer = "activations", ell
    return model.flat_state(state)


@torch.no_grad()
def run(ell, alpha):
    W, bb, Wp = PROBE[ell]
    h0 = act_at(ell)
    delta = alpha * ((tgt4 - (h0 @ W + bb)) @ Wp)
    h = h0 + delta
    roll = model.rollout_with_edit(state, ell, h, K_ROLL).cpu().numpy()
    c = edit_scorecard(roll, ZONES, gt_roll)
    c["write_ratio"] = float((delta.norm(dim=1) / h0.norm(dim=1)).mean())
    c["probe_err_after"] = float((h @ W + bb - tgt4).norm(dim=1).mean())
    c["step0"] = float(edit_index_by_step(roll, ZONES, gt_roll)[0])
    return c


with torch.no_grad():
    base = model.rollout_with_edit(state, NP - 1, act_at(NP - 1), K_ROLL).cpu().numpy()
UNS = edit_scorecard(base, ZONES, gt_roll)
UNS["step0"] = float(edit_index_by_step(base, ZONES, gt_roll)[0])
probe_err_before = {ell: float(((act_at(ell) @ PROBE[ell][0] + PROBE[ell][1]) - tgt4)
                               .norm(dim=1).mean()) for ell in range(NP)}
print(f"unsteered: Edit Index {UNS['edit_index']:+.4f} (step 0 {UNS['step0']:+.4f})   "
      f"target RMSE {UNS['target_rmse']:.4f}   collateral {UNS['collateral_rmse']:.4f}")

RES = {}
for ell in range(NP):
    print(f"\n=== residual point {ell} ===   probe error before the write "
          f"{probe_err_before[ell]:.3f} sim units")
    print(f"{'alpha':>6} {'|dh|/|h|':>9} {'probe err':>10} {'EditIdx K=15':>13} {'step 0':>9} "
          f"{'target':>8} {'ghost':>8} {'collat':>8} {'fidelity':>9}")
    for a in ALPHAS:
        c = run(ell, a)
        RES[(ell, a)] = c
        print(f"{a:>6} {c['write_ratio']:>9.3f} {c['probe_err_after']:>10.4f} "
              f"{c['edit_index']:>+13.4f} {c['step0']:>+9.4f} {c['target_rmse']:>8.4f} "
              f"{c['ghost_rmse']:>8.4f} {c['collateral_rmse']:>8.4f} "
              f"{fidelity_ratio(c, UNS):>9.3f}", flush=True)

print("\n" + "=" * 78)
a1 = {ell: RES[(ell, 1.0)]["edit_index"] for ell in range(NP)}
lo, hi = min(a1.values()), max(a1.values())
print(f"ANCHOR — alpha = 1.0 (the 2026-08-04 configuration): Edit Index {lo:+.3f}…{hi:+.3f} "
      f"across the {NP} points")
print(f"  published 2026-08-04 band for readout injection: {REF_INERT[0]:+.2f}…{REF_INERT[1]:+.2f}"
      f"   -> {'REPRODUCED' if lo > -0.75 and hi < -0.55 else 'DOES NOT MATCH — investigate'}")

best = max(RES, key=lambda k: RES[k]["edit_index"])
c = RES[best]
print(f"\nBEST over the whole (point x alpha) grid: point {best[0]}, alpha {best[1]}")
print(f"  Edit Index {c['edit_index']:+.4f} (unsteered {UNS['edit_index']:+.4f}), "
      f"step 0 {c['step0']:+.4f}, fidelity {fidelity_ratio(c, UNS):.3f}")
print(f"  target RMSE {c['target_rmse']:.4f} (unsteered {UNS['target_rmse']:.4f}), "
      f"collateral {c['collateral_rmse']:.4f} (unsteered {UNS['collateral_rmse']:.4f})")
print(f"  movement over the alpha axis at that point: "
      f"{min(RES[(best[0], a)]['edit_index'] for a in ALPHAS):+.4f} … "
      f"{max(RES[(best[0], a)]['edit_index'] for a in ALPHAS):+.4f}")
print("\nOthello-GPT reference, SAME editor, single point: Li error 2.723 -> 0.052, "
      "Edit Index -0.829 -> +0.697 at point 5, alpha 1.5")
print(f"\n{time.time() - t0:.0f}s")

np.save(_REPO / "runs" / "othello_transfer" / "pinv_alpha_discworld.npy",
        {"unsteered": UNS, "grid": {f"{e}|{a}": v for (e, a), v in RES.items()},
         "probe_err_before": probe_err_before,
         "config": {"N": N, "K": K_ROLL, "n_probe": N_PROBE, "ef": EF, "alphas": list(ALPHAS),
                    "model": "runs/transformers/W16", "dataset": "4_fixed_refl_inview"}},
        allow_pickle=True)
