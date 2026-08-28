"""The four probe controls on THIS repo's world model — shared implementation.

Asks, of `W16` on discworld: **is our probe reading a learned state, or is it reading the
observation?** If the latter, every downstream editability claim that leans on "high probe R²
means a meaningful representation" is weaker than it looks.

1. `probe_scaling`   — does probe quality improve at Li et al.'s probe-data scale?
2. `obs_window`      — how much of it is available from the raw observation, and from integrating
                       a window of frames?
3. `random_init`     — how much of it is the architecture rather than training? (Li et al.'s own
                       `--random` control.)
4. Interventions on their model live in `linear_intervention.py` / `single_layer.py` beside this.

Probes and fitting are the thread's own (`othello_gpt/othello_probe.fit_probe`), held out **by
sequence**, so every number is comparable to the rest of the editability thread.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
for _p in (str(REPO), str(REPO / "scripts"),
           str(REPO / "notebooks/experiments/editability/othello_gpt")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import othello_probe as op  # noqa: E402
import pipeline as pl  # noqa: E402

DATA = REPO / "datasets" / "4_fixed_refl_inview"
N_OBJ, HIDDEN, EPOCHS, SEED = 2, 512, 200, 0
POINT = 3  # "block 3 input" — the thread's best position probe on W16


def load(split: str, n: int):
    with h5py.File(DATA / f"{split}.h5", "r") as f:
        obs = f["obs_intensity"][:n].astype(np.float32)
        pos = f["positions"][:n, :, :N_OBJ, :].astype(np.float32)
    return obs, pos.reshape(len(obs), -1, 4)


def _split(n, holdout=0.2, seed=SEED):
    perm = np.random.default_rng(seed).permutation(n)
    cut = int((1 - holdout) * n)
    return perm[:cut], perm[cut:]


@torch.no_grad()
def residual_at(model, obs, point, batch=256):
    """(N, T, d_model) residual stream at ONE point — one at a time, since all five at 90k
    sequences is ~18 GB."""
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(obs), batch):
        o = torch.from_numpy(obs[i:i + batch]).float().to(dev)
        _, res = model._run(model.embed(o), model._seq_mask(o.shape[1], dev), want_resid=True)
        out.append(res[point].float().cpu().numpy())
    return np.concatenate(out, 0)


def _fit(X, Y, tr, te, hidden):
    _, s = op.fit_probe(X[tr].reshape(-1, X.shape[-1]), Y[tr].reshape(-1, Y.shape[-1]),
                        X[te].reshape(-1, X.shape[-1]), Y[te].reshape(-1, Y.shape[-1]),
                        hidden=hidden, epochs=EPOCHS, device=pl.DEVICE, seed=SEED)
    return s["r2"]


def probe_scaling(model, scales, split="train", log=print):
    """Held-out position R² against the number of sequences the probe is fit on."""
    rows = []
    for n in scales:
        obs, Y = load(split, n)
        X = residual_at(model, obs, POINT)
        tr, te = _split(n)
        r = {"n_seq": n, "rows": int(0.8 * n * obs.shape[1]),
             "linear": _fit(X, Y, tr, te, None), "mlp": _fit(X, Y, tr, te, HIDDEN)}
        rows.append(r)
        log(f"  {split:>5} n_seq {n:>6,}  rows {r['rows']:>9,}  linear {r['linear']:.4f}  "
            f"MLP {r['mlp']:.4f}")
        del X
    return rows


def obs_window(windows, n=5000, late_t=15, split="test", log=print):
    """Position R² from the RAW observation the model receives, by window length.

    Windows long enough that few timesteps qualify are degenerate (input dim outruns row count)
    and are reported with their row count so that is visible.
    """
    obs, Y = load(split, n)
    N, T, R = obs.shape
    tr, te = _split(N)
    rows = []
    for W in windows:
        ts = np.arange(max(late_t, W - 1), T)
        X = np.stack([obs[:, t - W + 1:t + 1].reshape(N, W * R) for t in ts], 1)
        Yw = Y[:, ts]
        r = {"window": W, "dim": W * R, "rows": int(len(tr) * len(ts)),
             "linear": _fit(X, Yw, tr, te, None), "mlp": _fit(X, Yw, tr, te, HIDDEN)}
        rows.append(r)
        log(f"  window {W:>3}  dim {r['dim']:>5}  rows {r['rows']:>8,}  "
            f"linear {r['linear']:+.4f}  MLP {r['mlp']:+.4f}")
        del X
    return rows


def random_init(trained, n=5000, seeds=(0, 1), split="test", log=print):
    """Li et al.'s `--random` control: same architecture and data, random weights."""
    from pim.world_models.transformer.model import TransformerModel
    obs, Y = load(split, n)
    tr, te = _split(n)
    n_pts = trained.cfg.n_layers + 1
    out = {"trained": [], "random": []}
    for ell in range(n_pts):
        X = residual_at(trained, obs, ell)
        out["trained"].append({"point": ell, "linear": _fit(X, Y, tr, te, None),
                               "mlp": _fit(X, Y, tr, te, HIDDEN)})
        del X
        acc = []
        for s in seeds:
            torch.manual_seed(s)
            rm = TransformerModel(trained.cfg).to(pl.DEVICE).eval()
            X = residual_at(rm, obs, ell)
            acc.append((_fit(X, Y, tr, te, None), _fit(X, Y, tr, te, HIDDEN)))
            del X, rm
            torch.cuda.empty_cache()
        out["random"].append({"point": ell, "linear": float(np.mean([a[0] for a in acc])),
                              "mlp": float(np.mean([a[1] for a in acc]))})
        log(f"  point {ell}: trained linear {out['trained'][-1]['linear']:+.4f} / MLP "
            f"{out['trained'][-1]['mlp']:+.4f}   random linear {out['random'][-1]['linear']:+.4f} "
            f"/ MLP {out['random'][-1]['mlp']:+.4f}")
    return out
