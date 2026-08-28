"""Nanda's linear-direction addition, on OUR world model.

On Othello-GPT this beats every other write: x <- x + alpha * w, where w is the LINEAR probe's
weight row for the target readout, added at every residual point. 2.723 -> 0.062 Li error,
Edit Index -0.829 -> +0.603. Our own pseudoinverse injection, same probe, same model, never gets
past -0.275.

Here: the same mechanism on `W16`. The probe is a linear REGRESSION onto object positions, so the
"direction" is the weight row for one output. Kept as simple as possible per Sevan: target a
**pure X displacement of one object**, and score the **edit frame only** (step 0), no rollout.
"""
import sys, time
import h5py, numpy as np, torch

sys.path.insert(0, "notebooks/experiments/editability/othello_gpt")
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import othello_probe as op
import pipeline as pl
from editability_metrics import build_edit_zones, edit_scorecard

N, EF, N_OBJ, K, SEED = 512, 20, 2, 15, 0
DX = 1.0                     # pure X displacement, sim units (position std ~1.76)
H5 = "datasets/4_fixed_refl_inview/test.h5"
DEV = pl.DEVICE

bundle = pl.load("W16")
model, sim = bundle.model, bundle.sim
NP = model.cfg.n_layers + 1
print(f"W16: {model.cfg.n_layers} layers -> {NP} residual points "
      f"(Othello-GPT has 9; Nanda needed >=6 of them)")

with h5py.File(H5, "r") as f:
    obs = f["obs_intensity"][:N].astype(np.float32)
    pos = f["positions"][:N, :, :N_OBJ, :].astype(np.float32)
    vel = f["velocities"][:N, :, :N_OBJ, :].astype(np.float32)

# ── probes: linear regression onto the 4 position outputs, one per residual point ──
R = op.collect_residuals(model, obs, batch=128)          # (NP, N, T, d_model)
Y = pos.reshape(N, -1, 4)
rng = np.random.default_rng(SEED)
perm = rng.permutation(N); tr, te = perm[: int(0.8 * N)], perm[int(0.8 * N):]
probes = {}
print("\nlinear position probe by residual point (held out by sequence):")
for ell in range(NP):
    X = R[ell]
    p, s = op.fit_probe(X[tr].reshape(-1, X.shape[-1]), Y[tr].reshape(-1, 4),
                        X[te].reshape(-1, X.shape[-1]), Y[te].reshape(-1, 4),
                        hidden=None, device=DEV, seed=SEED)
    probes[ell] = p
    print(f"  point {ell}: R2 {s['r2']:+.4f}")
del R

# ── the edit: object 0, pure +/- X toward the frustum centre so the target stays in view ──
obj = np.zeros(N, int)
pre_pos = pos[:, EF - 1].copy()
pre_vel = vel[:, EF - 1].copy()
tgt_pos = pos[:, EF].copy()
sgn = -np.sign(tgt_pos[np.arange(N), obj, 0])
sgn[sgn == 0] = 1.0
tgt_pos[np.arange(N), obj, 0] += sgn * DX
zones = build_edit_zones(pre_pos=pre_pos, tgt_pos=tgt_pos, pre_vel=pre_vel,
                         edit_object=obj, sim=sim, n_obj=N_OBJ)
print(f"\npure-X edit of object 0, |dx| = {DX} sim units, N = {N}")
print(f"  differing rays: median {int(np.median(zones.differing.sum(1)))} of {obs.shape[2]}")

state = model.state_from_obs(torch.from_numpy(obs[:, :EF]).float().to(DEV))
gt = zones.gt_edited[:, None]                                     # (N, 1, R), step 0 only


def score(roll0):
    return edit_scorecard(roll0[:, None], zones, gt)


with torch.no_grad():
    uns = score(model.decode(state).cpu().numpy())
print(f"  unsteered: Edit Index {uns['edit_index']:+.4f}   target RMSE {uns['target_rmse']:.4f}")

# direction: the probe's weight row for object 0's X output, in RAW activation space
DIRS = {}
for ell, p in probes.items():
    w = p.net.weight.detach()[0]                                  # output 0 == obj0 x
    d = w / p.x_std
    DIRS[ell] = d / d.norm()

print(f"\n{'points written':>22} {'alpha':>7} {'|dx|/|x|':>9} {'EditIdx':>9} {'target':>8} "
      f"{'ghost':>8} {'collat':>8}")
rows = {}
for pts, lab in [(set(range(NP)), f"all {NP}")] + [(set(range(n)), f"first {n}") for n in range(1, NP)]:
    for a in (0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.75, 1.0):
        rec = []

        def hook(layer, x, _r=rec):
            if layer not in pts:
                return x
            cur = x[:, -1]
            delta = a * cur.norm(dim=1, keepdim=True) * DIRS[layer]
            _r.append(float((delta.norm(dim=1) / cur.norm(dim=1)).mean()))
            out = x.clone(); out[:, -1] = cur + delta
            return out

        with torch.no_grad():
            c = score(model.decode(state, edit=hook).cpu().numpy())
        rows[(lab, a)] = c
        if lab == f"all {NP}" or a in (0.12, 0.25):
            print(f"{lab:>22} {a:>7} {np.mean(rec):>9.3f} {c['edit_index']:>+9.4f} "
                  f"{c['target_rmse']:>8.4f} {c['ghost_rmse']:>8.4f} {c['collateral_rmse']:>8.4f}",
                  flush=True)
best = max(rows, key=lambda k: rows[k]["edit_index"])
print(f"\nBEST: {best[0]} points, alpha {best[1]}  ->  Edit Index {rows[best]['edit_index']:+.4f}"
      f"   (unsteered {uns['edit_index']:+.4f})")
print(f"Othello-GPT reference, same mechanism: -0.829 -> +0.603")
