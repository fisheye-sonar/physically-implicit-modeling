"""INLP on discworld: the nullspace cascade at EVERY residual point, then edit through it.

Two questions, one run (Sevan, 2026-09-01):

  1. REDUNDANCY — how many orthogonal linear subspaces decode the state? The cascade
     deflates the residual stream probe by probe until R² < r2_stop; the profile of
     held-out R² per probe is the size of the linearly-readable code. If position is
     decodable from many orthogonal slices, a single-probe write moves one copy and
     leaves the rest saying "still here" — hypothesis A for the α=1 puzzle.
  2. Does writing to ALL of it edit? `multiprobe_delta` writes the first K probes at
     once (orthogonal row spaces, so exact for every one). Sweep K, α, uniform vs
     R²-shrunk targets; score with the canonical Edit Index + fidelity.

Fitted in the canonical probe's STANDARDISED space (z = (h−μ)/σ), so probe 1 of the
cascade is exactly PI[zspace] and K=1 must reproduce the canonical editor — the wiring
check printed at the end. Y stays raw: lstsq is per-output, so Y scaling is immaterial.
"""
import gc
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from pim.editors.nullspace import multiprobe_delta
from pim.editors.pinv import pinv_step
from pim.environments.discworld import bench as dwb
from pim.models import load_checkpoint
from pim.probes.base import collect_residuals
from pim.probes.cache import ProbeCache
from pim.probes.nullspace import fit_nullspace_cascade

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/initial_othello_comparison/L-dw-20m")
BASIS = sys.argv[2] if len(sys.argv) > 2 else "frustum"
N_SEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 20_000
OUT = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(f"experiments/inlp/scores/inlp_{RUN.name}_{BASIS}.json")
TARGET, MAX_ITER, R2_STOP = "full", 40, 0.02
KS = (1, 2, 4, 8, 16, 32)
ALPHAS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0,
          100.0, 175.0)

inst = json.loads((RUN / "config.json").read_text())["data"]["instance"]
root = Path("datasets/discworld") / inst
model, info = load_checkpoint(RUN / "best_model.pt", device=dwb.DEV)
NP, span = model.n_layers + 1, int(getattr(model, "state_span", 39))
print(f"{RUN}  basis={BASIS}  n_seq={N_SEQ}  val {info.val_loss:.5f}  points {NP}", flush=True)

# ── bench + the canonical PI reference (cache HIT; loaded BEFORE the big arrays) ─────
b = dwb.load_bench(model, n=192, target=TARGET, basis_name=BASIS, data_dir=root / "eval")
u = dwb.unsteered(model, b)
lin = dwb.fit_probes(model, target=TARGET, n_seq=30_000, family="linear", basis_name=BASIS,
                     data_dir=root / "probe", cache_dir=RUN / "probes", log=None)
ref = {}
for ell in range(NP):
    dwb.as_activations(model, ell)
    h0 = model.flat_state(b.state)
    ref[ell] = pinv_step(h0, b.tgt, lin[ell][0], space="zspace").cpu().numpy()
del lin

# ── probe corpus, exactly as fit_probes loads it ──────────────────────────────────────
with h5py.File(root / "probe" / "test.h5", "r") as f:
    obs = f["obs_intensity"][:N_SEQ].astype(np.float32)
    pos = f["positions"][:N_SEQ, :, :dwb.N_OBJ, :].astype(np.float32)
    vel = f["velocities"][:N_SEQ, :, :dwb.N_OBJ, :].astype(np.float32)
sim = json.load(open(root / "probe" / "dataset.json"))["sim"]
bp, bv = dwb._to_basis(pos, vel, sim, BASIS)
y = np.concatenate([bp.reshape(N_SEQ, bp.shape[1], -1), bv.reshape(N_SEQ, bv.shape[1], -1)], -1)
obs, y = obs[:, :span], y[:, :span]
T = obs.shape[1]
perm = np.random.default_rng(dwb.SEED).permutation(N_SEQ)          # the canonical split
tr_seq, te_seq = perm[: int(0.8 * N_SEQ)], perm[int(0.8 * N_SEQ):]
rows = lambda seqs: (seqs[:, None] * T + np.arange(T)[None, :]).ravel()
tr, te = rows(tr_seq), rows(te_seq)
Y = y.reshape(-1, y.shape[-1]).astype(np.float64)
mu_y = Y[tr].mean(0)

t0 = time.time()
STORE = ProbeCache(RUN / "probes")            # ⛔ every fitted cascade is PERSISTED here
_sdir = Path(".scratch"); _sdir.mkdir(exist_ok=True)
_mm = _sdir / f"inlp_resid_{RUN.name}_{BASIS}.npy"   # nvme, not /tmp (tmpfs)
R = collect_residuals(model, obs, batch=64, memmap=_mm)             # (NP, N, T, d) f32
print(f"residuals {R.shape} in {time.time()-t0:.0f}s", flush=True)
del obs, pos, vel, bp, bv
gc.collect()

tgt = b.tgt.cpu().numpy().astype(np.float64)
results = {"run": str(RUN), "basis": BASIS, "n_seq": N_SEQ, "target": TARGET,
           "unedited": {k: v for k, v in u.items() if np.isscalar(v)},
           "settings": {"max_iter": MAX_ITER, "r2_stop": R2_STOP, "Ks": KS, "alphas": ALPHAS},
           "points": {}}
for ell in range(NP):
    t0 = time.time()
    fname, prov = STORE.key(model, kind="nullspace_cascade", target=TARGET, n_seq=N_SEQ,
                            split="test", basis=BASIS, seed=dwb.SEED, point=ell,
                            max_iter=MAX_ITER, r2_stop=R2_STOP, space="zspace",
                            data=str((root / "probe").resolve()))
    hit = STORE.load(fname, prov, device="cpu")
    if hit is not None:
        casc, mu, sd = hit["cascade"], hit["mu"], hit["sd"]
        print(f"\npoint {ell}: cascade cache HIT {fname}", flush=True)
    else:
        H = R[ell].reshape(-1, R.shape[-1])
        mu, sd = H[tr].mean(0), H[tr].std(0) + 1e-6
        Z = ((H - mu) / sd).astype(np.float64)
        casc = fit_nullspace_cascade(Z, Y, tr, te, max_iter=MAX_ITER, r2_stop=R2_STOP, log=None)
        del Z, H
        gc.collect()
        # saved BEFORE any use: the fitted object is the result, the scores are derived
        STORE.store(fname, prov, {"cascade": casc, "mu": mu, "sd": sd, "mu_y": mu_y})
        print(f"\npoint {ell}: cascade WROTE {fname}", flush=True)
    prof = [(p["r2"], int(p["B"].shape[1])) for p in casc.probes]
    print(f"\npoint {ell}: {casc.n_probes} probes, total rank {casc.total_rank}, "
          f"R² profile {[round(r, 3) for r, _ in prof[:8]]}{'…' if len(prof) > 8 else ''}"
          f"  [{time.time()-t0:.0f}s]", flush=True)

    # ── edit through the first K probes ───────────────────────────────────────────
    dwb.as_activations(model, ell)
    h0_t = model.flat_state(b.state)
    h0 = h0_t.cpu().numpy()
    z0 = (h0 - mu) / sd
    arms = []
    for K in [k for k in KS if k <= casc.n_probes] + ([casc.n_probes] if casc.n_probes not in KS else []):
        for shrink in (False, True):
            dz = multiprobe_delta(casc, z0, tgt, K=K, shrink=shrink, mu=mu_y)
            dh = (dz * sd).astype(np.float32)
            if K == 1 and not shrink:                      # THE WIRING CHECK
                rel = float(np.linalg.norm(dh - ref[ell]) / (np.linalg.norm(ref[ell]) + 1e-9))
                results["points"].setdefault(str(ell), {})["k1_vs_canonical_pi_reldiff"] = rel
            dh_t = torch.from_numpy(dh).to(h0_t.device)
            for a in ALPHAS:
                roll = model.rollout_with_edit(b.state, ell, h0_t + a * dh_t, dwb.K_ROLL).cpu().numpy()
                rec = {"K": int(K), "shrink": bool(shrink), "alpha": float(a),
                       "write_ratio": float((a * dh_t).norm(dim=1).div(h0_t.norm(dim=1)).mean()),
                       **{k: v for k, v in dwb.score(model, b, roll).items() if np.isscalar(v)}}
                rec["fidelity_ratio"] = dwb.fidelity_ratio(rec, u)
                arms.append(rec)
    best = {}
    for K in sorted({r["K"] for r in arms}):
        for shrink in (False, True):
            sub = [r for r in arms if r["K"] == K and r["shrink"] == shrink]
            bb = max(sub, key=lambda r: r["edit_index"])
            best[f"K{K}{'s' if shrink else 'u'}"] = bb
    line = "  ".join(f"K{K}: {best[f'K{K}u']['edit_index']:+.3f}/{best[f'K{K}u']['fidelity_ratio']:.2f}"
                     f" (shr {best[f'K{K}s']['edit_index']:+.3f}/{best[f'K{K}s']['fidelity_ratio']:.2f})"
                     for K in sorted({r['K'] for r in arms}))
    print(f"  best EI/fid by K: {line}", flush=True)
    results["points"][str(ell)].update({
        "n_probes": casc.n_probes, "total_rank": casc.total_rank,
        "r2_profile": [r for r, _ in prof], "rank_profile": [k for _, k in prof],
        "best": best, "arms": arms})
    OUT.write_text(json.dumps(results, indent=1, default=float))     # checkpoint the JSON
    del casc, arms
    gc.collect()

print(f"\nunedited EI {u['edit_index']:+.4f}")
print("wiring check (K=1 uniform vs canonical PI[zspace] step, rel. diff):",
      {k: round(v['k1_vs_canonical_pi_reldiff'], 4) for k, v in results['points'].items()})
print(f"wrote {OUT}", flush=True)
del R
try:
    _mm.unlink()
except OSError:
    pass
