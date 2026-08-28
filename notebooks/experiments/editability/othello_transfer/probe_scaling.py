"""Does discworld probe quality improve with Othello-scale probe data?

Confound under test: our editability numbers fit probes on 1500 sequences (60k rows);
Li et al. fit theirs on ~140k games (8.4M rows) — a 140x gap. If discworld probe R2
climbs with more data, "our probes were under-fit" is a live alternative explanation
for the editability negative. If it is flat, that half of the data confound is dead.

Everything except the sequence count is the thread's own setup: `W16`, residual point 3
(the thread's best), `othello_probe.fit_probe` at hidden=512 / 200 epochs, held out
BY SEQUENCE 80/20, visible-frames-only mask.
"""
import sys, time
import h5py, numpy as np, torch

sys.path.insert(0, "notebooks/experiments/editability/othello_gpt")
sys.path.insert(0, ".")
import othello_probe as op
import pipeline as pl

POINT = 3          # residual point: "block 3 input" — the thread's best position probe
N_OBJ = 2
EPOCHS, HIDDEN, SEED = 200, 512, 0
H5 = "datasets/4_fixed_refl_inview/train.h5"


@torch.no_grad()
def residual_at(model, obs, point, batch=256):
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(obs), batch):
        o = torch.from_numpy(obs[i:i + batch]).float().to(dev)
        _, resids = model._run(model.embed(o), model._seq_mask(o.shape[1], dev),
                               want_resid=True)
        out.append(resids[point].float().cpu().numpy())
    return np.concatenate(out, 0)


def one(model, h5_path, n_seq):
    with h5py.File(h5_path, "r") as f:
        obs = f["obs_intensity"][:n_seq].astype(np.float32)
        pos = f["positions"][:n_seq, :, :N_OBJ, :].astype(np.float32)
        vis = f["is_visible"][:n_seq, :, :N_OBJ].all(axis=2)
    T = obs.shape[1]
    Y = pos[:, :T].reshape(n_seq, T, 4)
    X = residual_at(model, obs, POINT)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_seq)
    ntr = int(0.8 * n_seq)
    tr, te = perm[:ntr], perm[ntr:]
    res = {}
    for hid, fam in ((None, "linear"), (HIDDEN, f"MLP ({HIDDEN} hidden)")):
        _, s = op.fit_probe(X[tr][vis[tr]], Y[tr][vis[tr]], X[te][vis[te]], Y[te][vis[te]],
                            hidden=hid, epochs=EPOCHS, device=pl.DEVICE, seed=SEED)
        res[fam] = s["r2"]
    res["rows_train"] = int(vis[tr].sum())
    res["rows_test"] = int(vis[te].sum())
    del X
    return res


if __name__ == "__main__":
    model = pl.load("W16").model
    print(f"W16 loaded | residual point {POINT} | probe: hidden={HIDDEN}, {EPOCHS} epochs, "
          f"held out by sequence 80/20\n")
    print(f"{'source':>6} {'n_seq':>7} {'train rows':>11} {'linear R2':>10} {'MLP R2':>9} {'gap':>7}  time")
    # cross-check against the published number, which was fit on the TEST split
    for src, path, scales in (("test", "datasets/4_fixed_refl_inview/test.h5", [1500]),
                              ("train", H5, [1500, 5000, 15000, 45000, 90000])):
        for n in scales:
            t = time.time()
            r = one(model, path, n)
            lin, mlp = r["linear"], r[f"MLP ({HIDDEN} hidden)"]
            print(f"{src:>6} {n:>7,} {r['rows_train']:>11,} {lin:>10.4f} {mlp:>9.4f} "
                  f"{mlp - lin:>+7.4f}  {time.time() - t:5.1f}s", flush=True)
