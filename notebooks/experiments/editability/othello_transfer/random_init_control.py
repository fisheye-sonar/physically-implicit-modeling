"""Random-weight control: how much position decodability is in the ARCHITECTURE alone?

A randomly initialised network is still a nonlinear random projection, and probes read a
surprising amount from random features. If a random-weight W16 already gives latent R2 near
the trained model's, then "the model learned a state" is not supported by decodability.

This is Li et al.'s own control (their `--random` arm, `model.apply(model._init_weights)`;
their Table 1/2 "randomized Othello-GPT" column). Same architecture, same data, same probe,
same split, same residual points — the only variable is trained vs random weights.
"""
import sys, time
import h5py, numpy as np, torch

sys.path.insert(0, "notebooks/experiments/editability/othello_gpt")
sys.path.insert(0, ".")
import othello_probe as op
import pipeline as pl
from pim.world_models.transformer.model import TransformerModel

N_SEQ, N_OBJ, EPOCHS, HIDDEN, SEED = 5000, 2, 200, 512, 0
H5 = "datasets/4_fixed_refl_inview/test.h5"


def fit_all_points(model, obs, Y, vis, tr, te, tag):
    R = op.collect_residuals(model, obs, batch=128)
    rows = []
    for ell in range(R.shape[0]):
        X = R[ell]
        r = {}
        for hid, fam in ((None, "linear"), (HIDDEN, "mlp")):
            _, s = op.fit_probe(X[tr][vis[tr]], Y[tr][vis[tr]], X[te][vis[te]], Y[te][vis[te]],
                                hidden=hid, epochs=EPOCHS, device=pl.DEVICE, seed=SEED)
            r[fam] = s["r2"]
        rows.append(r)
        print(f"  {tag:<18} point {ell}: linear {r['linear']:+.4f}   MLP {r['mlp']:+.4f}", flush=True)
    del R
    return rows


with h5py.File(H5, "r") as f:
    obs = f["obs_intensity"][:N_SEQ].astype(np.float32)
    pos = f["positions"][:N_SEQ, :, :N_OBJ, :].astype(np.float32)
    vis = f["is_visible"][:N_SEQ, :, :N_OBJ].all(axis=2)
N, T, _ = obs.shape
Y = pos.reshape(N, T, 4)
rng = np.random.default_rng(SEED)
perm = rng.permutation(N)
tr, te = perm[: int(0.8 * N)], perm[int(0.8 * N):]

trained = pl.load("W16").model
cfg = trained.cfg
print(f"W16 cfg: d_model={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads} window={cfg.window}")
print(f"N={N} sequences, probe hidden={HIDDEN}, {EPOCHS} epochs, held out by sequence 80/20\n")

t0 = time.time()
res = {"trained": fit_all_points(trained, obs, Y, vis, tr, te, "TRAINED")}
for s in (0, 1):
    torch.manual_seed(s)
    rnd = TransformerModel(cfg).to(pl.DEVICE).eval()
    res[f"random{s}"] = fit_all_points(rnd, obs, Y, vis, tr, te, f"RANDOM seed {s}")
    del rnd
    torch.cuda.empty_cache()

print(f"\n{'point':>6} {'trained lin':>12} {'random lin':>11} {'trained MLP':>12} {'random MLP':>11}")
for ell in range(len(res["trained"])):
    rl = np.mean([res[f"random{s}"][ell]["linear"] for s in (0, 1)])
    rm = np.mean([res[f"random{s}"][ell]["mlp"] for s in (0, 1)])
    print(f"{ell:>6} {res['trained'][ell]['linear']:>12.4f} {rl:>11.4f} "
          f"{res['trained'][ell]['mlp']:>12.4f} {rm:>11.4f}")
print(f"\nreference — raw observation, 16-frame window: linear 0.323 / MLP 0.851")
print(f"reference — raw observation, single frame:    linear 0.292 / MLP 0.837")
print(f"total {time.time()-t0:.0f}s")
