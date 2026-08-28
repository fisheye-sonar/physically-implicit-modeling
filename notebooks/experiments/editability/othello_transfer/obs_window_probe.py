"""How much of the latent's position decodability is just temporal integration?

2026-08-05 measured `clean_obs -> position` at ONE frame: linear R2 0.259, MLP 0.754.
The latent (W16, residual point 3) reads 0.765 linear / 0.960 MLP. But the latent sees the
whole history and that probe saw one frame, so the gap conflates two things:
  (a) the model inverting the renderer, and
  (b) plain temporal integration, which a linear probe on a WINDOW of raw frames also gets.

This sweeps the window length on the RAW observations the model actually receives (noisy
`obs_intensity`), same probe and fitting loop as everything else in the thread. Frames t >= 15
only (LATE_T, the filter-converged regime), held out by sequence 80/20.
"""
import sys, time
import h5py, numpy as np

sys.path.insert(0, "notebooks/experiments/editability/othello_gpt")
sys.path.insert(0, ".")
import othello_probe as op

N_SEQ, N_OBJ, LATE_T, SEED = 5000, 2, 15, 0
EPOCHS, HIDDEN = 200, 512
H5 = "datasets/4_fixed_refl_inview/test.h5"

with h5py.File(H5, "r") as f:
    obs = f["obs_intensity"][:N_SEQ].astype(np.float32)          # (N, T, 128) NOISY
    pos = f["positions"][:N_SEQ, :, :N_OBJ, :].astype(np.float32)
    vis = f["is_visible"][:N_SEQ, :, :N_OBJ].all(axis=2)
N, T, R = obs.shape
Y = pos.reshape(N, T, 4)
rng = np.random.default_rng(SEED)
perm = rng.permutation(N)
tr, te = perm[: int(0.8 * N)], perm[int(0.8 * N):]

print(f"{H5}  N={N}  T={T}  obs_res={R}  frames t>={LATE_T}")
print(f"reference (2026-08-05, clean_obs, ONE frame): linear 0.259 / MLP 0.754")
print(f"reference (W16 latent, residual point 3):     linear 0.765 / MLP 0.960\n")
print(f"{'window':>7} {'input dim':>10} {'rows':>9} {'linear R2':>10} {'MLP R2':>9} {'gap':>8}  time")

for W in (1, 2, 4, 8, 16, 40):
    t0 = time.time()
    ts = np.arange(max(LATE_T, W - 1), T)
    X = np.stack([obs[:, t - W + 1: t + 1].reshape(N, W * R) for t in ts], axis=1)
    Yw = Y[:, ts]
    m = vis[:, ts]
    out = {}
    for hid, fam in ((None, "linear"), (HIDDEN, "mlp")):
        _, s = op.fit_probe(X[tr][m[tr]], Yw[tr][m[tr]], X[te][m[te]], Yw[te][m[te]],
                            hidden=hid, epochs=EPOCHS, device="cuda", seed=SEED)
        out[fam] = s["r2"]
    print(f"{W:>7} {W*R:>10} {int(m[tr].sum()):>9,} {out['linear']:>10.4f} "
          f"{out['mlp']:>9.4f} {out['mlp']-out['linear']:>+8.4f}  {time.time()-t0:5.1f}s", flush=True)
    del X
