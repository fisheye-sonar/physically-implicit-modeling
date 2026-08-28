"""Position decodability for discs the observation CANNOT show.

The controls so far said most of our NONLINEAR position decodability is already present in the
raw observation (MLP 0.851 on a 16-frame window vs 0.942 on the trained latent). The sharpest
way to separate "state" from "observation echo" is to ask about discs that contribute ZERO rays
to the current frame: the observation cannot supply their position, so anything readable must
have been carried.

Occlusion is defined from `obs_id` (which object each ray hit), NOT from `is_visible` — that
field means "overlaps the frustum" and is identically True on this always-in-frustum dataset.

Probes are fit on ALL frames (the standard probe) and evaluated split by visibility, so this
asks where the ordinary probe works, not whether a special probe can be trained.
"""
import sys, time
import h5py, numpy as np, torch

sys.path.insert(0, "notebooks/experiments/editability/othello_gpt")
sys.path.insert(0, ".")
import othello_probe as op
import pipeline as pl
from pim.world_models.transformer.model import TransformerModel

N_SEQ, N_OBJ, LATE_T, EPOCHS, HIDDEN, SEED, WIN = 20000, 2, 15, 200, 512, 0, 16
POINT = 3
H5 = "datasets/4_fixed_refl_inview/test.h5"

with h5py.File(H5, "r") as f:
    n = min(N_SEQ, f["obs_intensity"].shape[0])
    obs = f["obs_intensity"][:n].astype(np.float32)
    oid = f["obs_id"][:n]
    pos = f["positions"][:n, :, :N_OBJ, :].astype(np.float32)
N, T, R = obs.shape
rays = np.stack([(oid == j).sum(-1) for j in range(N_OBJ)], -1)     # (N, T, n_obj)
occ = rays == 0                                                      # fully occluded
# frames since this object last contributed a ray
since = np.zeros_like(rays, dtype=np.int16)
for t in range(1, T):
    since[:, t] = np.where(occ[:, t], since[:, t - 1] + 1, 0)

ts = np.arange(LATE_T, T)
rng = np.random.default_rng(SEED)
perm = rng.permutation(N)
tr, te = perm[: int(0.8 * N)], perm[int(0.8 * N):]
print(f"{N:,} sequences, frames t>={LATE_T}; occluded instances "
      f"{occ[:, ts].sum():,} of {occ[:, ts].size:,} ({occ[:, ts].mean()*100:.2f}%)")
print(f"position std per dim ~{pos.reshape(-1,4).std(0).mean():.2f} sim units "
      f"(a mean-predictor scores that RMSE)\n")


@torch.no_grad()
def latent(model):
    out = []
    dev = next(model.parameters()).device
    for i in range(0, N, 256):
        o = torch.from_numpy(obs[i:i+256]).float().to(dev)
        _, res = model._run(model.embed(o), model._seq_mask(T, dev), want_resid=True)
        out.append(res[POINT].float().cpu().numpy())
    return np.concatenate(out, 0)[:, ts]


trained = pl.load("W16").model
torch.manual_seed(0)
rnd = TransformerModel(trained.cfg).to(pl.DEVICE).eval()
REPS = {
    "obs, 1 frame": obs[:, ts],
    f"obs, {WIN}-frame window": np.stack([obs[:, t-WIN+1:t+1].reshape(N, WIN*R) for t in ts], 1),
    "random-init latent": latent(rnd),
    "TRAINED latent": latent(trained),
}
del rnd
torch.cuda.empty_cache()

print(f"{'representation':>24} {'obj':>4} {'probe':>7} | {'visible RMSE':>12} {'occluded RMSE':>13} "
      f"{'vis R2':>7} {'occ R2':>7} | {'n occ':>7}")
results = {}
for name, X in REPS.items():
    for j in range(N_OBJ):
        Y = pos[:, ts, j, :]
        o_te = occ[te][:, ts, j]
        for hid, fam in ((None, "linear"), (HIDDEN, "MLP")):
            probe, _ = op.fit_probe(X[tr].reshape(-1, X.shape[-1]), Y[tr].reshape(-1, 2),
                                    X[te].reshape(-1, X.shape[-1]), Y[te].reshape(-1, 2),
                                    hidden=hid, epochs=EPOCHS, device=pl.DEVICE, seed=SEED)
            with torch.no_grad():
                P = probe(torch.tensor(X[te].reshape(-1, X.shape[-1]), device=pl.DEVICE)).cpu().numpy()
            P = P.reshape(len(te), len(ts), 2)
            mu = Y[tr].reshape(-1, 2).mean(0)
            def score(m):
                e = ((P[m] - Y[te][m]) ** 2)
                ss = ((Y[te][m] - mu) ** 2).sum()
                return float(np.sqrt(e.mean())), float(1 - e.sum() / ss)
            vr, vr2 = score(~o_te)
            orm, or2 = score(o_te)
            results[(name, j, fam)] = (vr, orm, vr2, or2)
            print(f"{name:>24} {j:>4} {fam:>7} | {vr:>12.4f} {orm:>13.4f} {vr2:>7.4f} {or2:>7.4f} "
                  f"| {int(o_te.sum()):>7,}", flush=True)
            if name == "TRAINED latent" and fam == "MLP":
                s_te = since[te][:, ts, j]
                for lo, hi, lab in ((1,1,"1"), (2,2,"2"), (3,4,"3-4"), (5,8,"5-8"), (9,99,"9+")):
                    m = o_te & (s_te >= lo) & (s_te <= hi)
                    if m.sum() > 200:
                        r, _ = score(m)
                        print(f"{'':>24} {'':>4} {'':>7} |   frames hidden {lab:>4}: "
                              f"RMSE {r:.4f}  (n={int(m.sum()):,})")
