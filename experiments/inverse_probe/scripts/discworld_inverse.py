"""INVERSE PROBE on discworld (2026-09-14, Sevan): g: FULL STATE (frustum position + velocity,
8-d) → residual at point ℓ, fitted on the canonical probe corpus; write g(target state) into the
residual at the edit frame and score with the canonical ray-zone Edit Index and guard.

    overwrite   h' = g(s_post)                       s_post = the bench's pre-dynamics target state (dims all)
    delta       h' = h + α · (g(s_post) − g(s_pre))  s_pre = the frustum full state at the last context frame
    nn          both forms with g replaced by the mean residual of the k nearest training states
Controls: overwrite with the state-free mean residual; the run's canonical PI / GS rows.
Also: g's held-out R² per point; the canonical linear probe's read-out error at the written
residual ("landed", lower is better) against its value before the write.

    python experiments/inverse_probe/scripts/discworld_inverse.py --run noise_ablation/L-dw-noiseless-20m
    python experiments/inverse_probe/scripts/discworld_inverse.py --run ray_ablation/L-dw-8ray-20m --smoke
Contained: reads the run's cached probes and the probe corpus; writes experiments/inverse_probe/scores/.
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile, time
from pathlib import Path
import h5py, numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.editors.pinv import readout_error  # noqa: E402
from pim.environments import layout  # noqa: E402
from pim.environments.discworld import arms as dwa, bench as dwb  # noqa: E402
from pim.environments.discworld.bench import EF, K_ROLL, N_OBJ  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.base import FIT_BATCH, FIT_LR, collect_residuals, fit_probe  # noqa: E402

DEV = "cuda"
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="noise_ablation/L-dw-noiseless-20m")
ap.add_argument("--hidden", type=int, default=1024)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--alphas", type=float, nargs="+", default=(0.25, 0.5, 1.0, 1.5, 2.0, 3.0))
ap.add_argument("--points", type=int, nargs="*", default=None)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--tag", default="", help="suffix for the scores file (e.g. mirror128)")
a = ap.parse_args()
t0 = time.time()
run = REPO / "runs" / a.run
S = json.loads((run / "scores.json").read_text())["settings"]
inst = json.loads((run / "config.json").read_text())["data"]["instance"]
model, _ = load_checkpoint(run / "best_model.pt", device=DEV); model.eval(); NP = n_points(model)
span = int(getattr(model, "state_span", 39))
n_seq = 2000 if a.smoke else S["dw_probe_seqs"]
recipe = dwa.probe_recipe("full", inst, n_seq=n_seq)

# ── probe corpus: full frustum state per frame, residuals per point ────────────────────
with h5py.File(layout.probe_file("discworld", inst, "120k"), "r") as f:
    obs = f["obs_intensity"][:n_seq, :span].astype(np.float32)
    pos = f["positions"][:n_seq, :span, :N_OBJ, :].astype(np.float32)
    vel = f["velocities"][:n_seq, :span, :N_OBJ, :].astype(np.float32)
sim = json.load(open(layout.probe_manifest("discworld", inst, "120k")))["sim"]
Y, _ = dwa._targets("full", pos, vel, sim, "frustum")                    # (N, T, 8)
T = Y.shape[1]
perm = np.random.default_rng(0).permutation(n_seq)
tr_seq, te_seq = perm[: int(0.8 * n_seq)], perm[int(0.8 * n_seq):]
X_all = Y.reshape(-1, Y.shape[-1]).astype(np.float32)                    # (N·T, 8)
row_seq = np.repeat(np.arange(n_seq), T)
tr = np.isin(row_seq, tr_seq); te = ~tr
print(f"{a.run} ({inst}): {n_seq} sequences × {T} frames = {len(X_all):,} rows; {NP} points; "
      f"hidden {a.hidden}, epochs {a.epochs}, k {a.k}", flush=True)

# ── the bench: s_pre (frustum full state at EF−1), s_post (the pre-dynamics target, dims all)
b = dwb.load_bench(model, n=50 if a.smoke else S["dw_bench_n"], target="full", basis_name="frustum", instance=inst)
bp, bv = dwb._to_basis(b.pos[:, EF - 1], b.vel[:, EF - 1], b.sim, "frustum")
s_pre = np.concatenate([bp.reshape(b.n, -1), bv.reshape(b.n, -1)], 1).astype(np.float32)
s_post = b.tgt.cpu().numpy().astype(np.float32)
cm = b.change_mask.cpu().numpy()
assert np.allclose(s_pre[~cm], s_post[~cm], atol=1e-4), "unedited dims of s_pre and the target differ"
u = dwa.unsteered(model, b)
canon = json.loads((run / "scores.json").read_text())["bases"]["frustum"]["best"]
print(f"bench: {b.n} cases · unedited {u['edit_index']:+.3f} · canonical PI {canon['PI']['edit_index']:+.3f}/{canon['PI']['fidelity_ratio']:.2f}  "
      f"GS {canon['GS']['edit_index']:+.3f}/{canon['GS']['fidelity_ratio']:.2f}", flush=True)
lin = None if a.smoke else {e: p for e, (p, _) in dwa.fit_probes(model, target="full", family="linear", basis_name="frustum",
                                                                    cache_dir=run / "probes", log=None, require_cached=True, **recipe).items()}
Xpre_t, Xpost_t = (torch.from_numpy(x).to(DEV) for x in (s_pre, s_post))

def nn_mean(X_train_t, H_train_t, x_query_t, k):
    """(n, d): mean residual of the k nearest training states (Euclidean in standardised state units)."""
    mu, sd = X_train_t.mean(0), X_train_t.std(0).clamp_min(1e-6)
    A = (X_train_t - mu) / sd; Q = (x_query_t - mu) / sd
    a2 = (A * A).sum(1)
    out = torch.zeros(len(Q), H_train_t.shape[1], device=DEV)
    for i in range(0, len(Q), 256):
        q = Q[i:i + 256]
        d2 = a2[None, :] - 2 * q @ A.T + (q * q).sum(1)[:, None]         # (b, rows)
        idx = d2.topk(k, dim=1, largest=False).indices
        out[i:i + 256] = H_train_t[idx].mean(1)
    return out

def run_arm(ell, h0, h_new):
    roll = model.rollout_with_edit(b.state, ell, h_new, K_ROLL).cpu().numpy()
    c = dwa.score(model, b, roll, u)
    rec = {"edit_index": c["edit_index"], "fidelity_ratio": c["fidelity_ratio"],
           "write_ratio": float(((h_new - h0).norm(dim=1) / h0.norm(dim=1)).mean())}
    if lin is not None:
        rec["readout_err_before"] = readout_error(h0, b.tgt, lin[ell]); rec["readout_err_after"] = readout_error(h_new, b.tgt, lin[ell])
    return rec

out = {"run": a.run, "instance": inst, "n_seq": n_seq, "rows": int(len(X_all)), "hidden": a.hidden, "epochs": a.epochs, "k": a.k,
       "unedited": {k: v for k, v in u.items() if isinstance(v, (int, float))},
       "canonical": {e: {"edit_index": canon[e]["edit_index"], "fidelity_ratio": canon[e]["fidelity_ratio"]} for e in ("PI", "GS")},
       "points": {}}
points = a.points if a.points is not None else ([1] if a.smoke else list(range(NP)))
sdir = REPO / ".scratch"; sdir.mkdir(exist_ok=True)
for ell in points:
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False, dir=sdir); tmp.close()
    try:
        R = collect_residuals(model, obs, batch=64, memmap=tmp.name, points=[ell])[0]   # (N, T, d)
        H = np.ascontiguousarray(R.reshape(-1, R.shape[-1])); del R
    finally:
        os.unlink(tmp.name)
    g, st = fit_probe(X_all[tr], H[tr], X_all[te], H[te], hidden=a.hidden, epochs=a.epochs,
                      batch=FIT_BATCH, lr=FIT_LR, device=DEV, seed=0, n_classes=None)
    g.eval()
    for p_ in g.parameters():
        p_.requires_grad_(False)
    H_tr_t = torch.from_numpy(H[tr]).to(DEV); X_tr_t = torch.from_numpy(X_all[tr]).to(DEV)
    h_mean = H_tr_t.mean(0)
    with torch.no_grad():
        g_pre, g_post = g(Xpre_t), g(Xpost_t)
        nn_pre, nn_post = nn_mean(X_tr_t, H_tr_t, Xpre_t, a.k), nn_mean(X_tr_t, H_tr_t, Xpost_t, a.k)
        dwa.as_activations(model, ell)
        h0 = model.flat_state(b.state)
        arms = {"overwrite": run_arm(ell, h0, g_post), "mean_overwrite": run_arm(ell, h0, h_mean[None].expand_as(h0)),
                "nn_overwrite": run_arm(ell, h0, nn_post)}
        for al in a.alphas:
            arms[f"delta@{al:g}"] = run_arm(ell, h0, h0 + al * (g_post - g_pre))
            arms[f"nn_delta@{al:g}"] = run_arm(ell, h0, h0 + al * (nn_post - nn_pre))
    best_d = max((k for k in arms if k.startswith("delta@")), key=lambda k: arms[k]["edit_index"])
    best_n = max((k for k in arms if k.startswith("nn_delta@")), key=lambda k: arms[k]["edit_index"])
    out["points"][str(ell)] = {"g_r2_heldout": float(st["r2"]), "g_rmse": float(st["rmse"]), "arms": arms}
    f = lambda r: f"{r['edit_index']:+.3f}/{r['fidelity_ratio']:.2f}" + (f" err {r['readout_err_before']:.2f}→{r['readout_err_after']:.2f}" if "readout_err_after" in r else "")  # noqa: E731
    print(f"pt {ell}: g R² {st['r2']:+.3f} | overwrite {f(arms['overwrite'])} | mean-h {f(arms['mean_overwrite'])} | "
          f"nn-overwrite {f(arms['nn_overwrite'])} | {best_d} {f(arms[best_d])} | {best_n} {f(arms[best_n])}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    del H, H_tr_t, X_tr_t; torch.cuda.empty_cache()
tag = a.run.split("/")[-1] + (f"_{a.tag}" if a.tag else "") + ("_smoke" if a.smoke else "")
(REPO / "experiments/inverse_probe/scores" / f"discworld_{tag}.json").write_text(json.dumps(out, indent=1, default=float))
print("wrote", f"experiments/inverse_probe/scores/discworld_{tag}.json", f"[{(time.time() - t0) / 60:.1f} min]")
