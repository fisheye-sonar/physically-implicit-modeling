"""INVERSE PROBE on a frames-as-TOKENS discworld model (2026-09-14): the discworld variant on
the token path — residuals from token inputs, the write at the last context position through
the token bench's edit hook, the frame-set Edit Index (†) and move-fidelity guard, as the
canonical token scorer uses. g: full frustum state (8) → residual at point ℓ, mirrored MLP.
    python experiments/inverse_probe/scripts/discworld_tokens_inverse.py --run interface_ablation/L-dw-8ray-tok-20m --hidden 128 --epochs 200 --tag mirror128
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile, time
from pathlib import Path
import h5py, numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments import layout  # noqa: E402
from pim.environments.discworld import arms as dwa, bench as dwb, token_bench as tkb  # noqa: E402
from pim.environments.discworld.bench import EF, N_OBJ  # noqa: E402
from pim.environments.discworld.tokens import FrameVocab  # noqa: E402
from pim.metrics.set_editability import move_fidelity_ratio  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.base import FIT_BATCH, FIT_LR, collect_residuals, fit_probe  # noqa: E402

DEV = "cuda"
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="interface_ablation/L-dw-8ray-tok-20m")
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--alphas", type=float, nargs="+", default=(0.25, 0.5, 1.0, 1.5, 2.0, 3.0))
ap.add_argument("--points", type=int, nargs="*", default=None)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--tag", default="")
a = ap.parse_args()
t0 = time.time()
run = REPO / "runs" / a.run
S = json.loads((run / "scores.json").read_text())["settings"]
inst = json.loads((run / "config.json").read_text())["data"]["instance"]
model, _ = load_checkpoint(run / "best_model.pt", device=DEV); model.eval(); NP = n_points(model)
vocab = FrameVocab.load(run / "vocab.npz")
enc, _tag = tkb.token_encoder(vocab)
span = int(getattr(model, "state_span", 39))
n_seq = 2000 if a.smoke else S["dw_probe_seqs"]
with h5py.File(layout.probe_file("discworld", inst, "120k"), "r") as f:
    obs = f["obs_intensity"][:n_seq, :span].astype(np.float32)
    pos = f["positions"][:n_seq, :span, :N_OBJ, :].astype(np.float32)
    vel = f["velocities"][:n_seq, :span, :N_OBJ, :].astype(np.float32)
sim = json.load(open(layout.probe_manifest("discworld", inst, "120k")))["sim"]
Y, _ = dwa._targets("full", pos, vel, sim, "frustum")
T = Y.shape[1]
tok = enc(obs)                                                              # (N, T) token ids
perm = np.random.default_rng(0).permutation(n_seq)
tr = np.isin(np.repeat(np.arange(n_seq), T), perm[: int(0.8 * n_seq)]); te = ~tr
X_all = Y.reshape(-1, Y.shape[-1]).astype(np.float32)
print(f"{a.run} ({inst}, tokens V{vocab.size}): {n_seq} × {T} = {len(X_all):,} rows; {NP} points; hidden {a.hidden}, epochs {a.epochs}", flush=True)

n_bench = 50 if a.smoke else S["dw_bench_n"]
tb = tkb.load_token_bench(vocab, n=n_bench, target="full", basis_name="frustum", instance=inst)
arr = dwb.bench_arrays(n_bench, "full", "frustum", instance=inst)          # same cases: pos/vel for s_pre
bp, bv = dwb._to_basis(arr["pos"][:, EF - 1], arr["vel"][:, EF - 1], arr["sim"], "frustum")
s_pre = np.concatenate([bp.reshape(tb.n, -1), bv.reshape(tb.n, -1)], 1).astype(np.float32)
s_post = tb.tgt.cpu().numpy().astype(np.float32); cm = tb.change_mask.cpu().numpy()
assert np.allclose(s_pre[~cm], s_post[~cm], atol=1e-4)
uns, u = tkb.unsteered(model, tb)
H0 = tkb.residuals_last(model, tb)                                          # {point: (n, d)}
canon = json.loads((run / "scores.json").read_text())["bases"]["frustum"]["best"]
print(f"bench: {tb.n} cases ({int(tb.keep.sum())} kept) · unedited {u['edit_index']:+.3f} · canonical PI {canon['PI']['edit_index']:+.3f}/{canon['PI']['fidelity_ratio']:.2f}  GS {canon['GS']['edit_index']:+.3f}/{canon['GS']['fidelity_ratio']:.2f}", flush=True)
Xpre_t, Xpost_t = (torch.from_numpy(x).to(DEV) for x in (s_pre, s_post))

def nn_mean(X_train_t, H_train_t, x_query_t, k):
    mu, sd = X_train_t.mean(0), X_train_t.std(0).clamp_min(1e-6)
    A = (X_train_t - mu) / sd; Q = (x_query_t - mu) / sd; a2 = (A * A).sum(1)
    out = torch.zeros(len(Q), H_train_t.shape[1], device=DEV)
    for i in range(0, len(Q), 256):
        q = Q[i:i + 256]; d2 = a2[None, :] - 2 * q @ A.T + (q * q).sum(1)[:, None]
        out[i:i + 256] = H_train_t[d2.topk(k, dim=1, largest=False).indices].mean(1)
    return out

def run_arm(ell, h0, h_new):
    probs = tkb.probs_at_edit(model, tb, hook=tkb._write_hook(ell, h_new))
    c = tkb.scorecard(probs, tb, uns)
    return {"edit_index": c["edit_index"], "edit_index_symdiff": c.get("edit_index_symdiff"),
            "fidelity_ratio": c.get("fidelity_ratio", move_fidelity_ratio(probs, uns, tb.legal_post)),
            "p_post": c.get("p_post"), "write_ratio": float(((h_new - h0).norm(dim=1) / h0.norm(dim=1)).mean())}

out = {"run": a.run, "instance": inst, "ei_construction": "frame-set", "n_seq": n_seq, "rows": int(len(X_all)),
       "hidden": a.hidden, "epochs": a.epochs, "k": a.k, "unedited": {k: v for k, v in u.items() if isinstance(v, (int, float))},
       "canonical": {e: {"edit_index": canon[e]["edit_index"], "fidelity_ratio": canon[e]["fidelity_ratio"]} for e in ("PI", "GS")}, "points": {}}
points = a.points if a.points is not None else ([1] if a.smoke else list(range(NP)))
sdir = REPO / ".scratch"; sdir.mkdir(exist_ok=True)
for ell in points:
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False, dir=sdir); tmp.close()
    try:
        R = collect_residuals(model, tok, batch=64, memmap=tmp.name, points=[ell])[0]
        H = np.ascontiguousarray(R.reshape(-1, R.shape[-1])); del R
    finally:
        os.unlink(tmp.name)
    g, st = fit_probe(X_all[tr], H[tr], X_all[te], H[te], hidden=a.hidden, epochs=a.epochs, batch=FIT_BATCH, lr=FIT_LR, device=DEV, seed=0, n_classes=None)
    g.eval()
    for p_ in g.parameters():
        p_.requires_grad_(False)
    H_tr_t = torch.from_numpy(H[tr]).to(DEV); X_tr_t = torch.from_numpy(X_all[tr]).to(DEV); h_mean = H_tr_t.mean(0)
    with torch.no_grad():
        g_pre, g_post = g(Xpre_t), g(Xpost_t)
        nn_pre, nn_post = nn_mean(X_tr_t, H_tr_t, Xpre_t, a.k), nn_mean(X_tr_t, H_tr_t, Xpost_t, a.k)
        h0 = H0[ell]
        arms = {"overwrite": run_arm(ell, h0, g_post), "mean_overwrite": run_arm(ell, h0, h_mean[None].expand_as(h0)), "nn_overwrite": run_arm(ell, h0, nn_post)}
        for al in a.alphas:
            arms[f"delta@{al:g}"] = run_arm(ell, h0, h0 + al * (g_post - g_pre))
            arms[f"nn_delta@{al:g}"] = run_arm(ell, h0, h0 + al * (nn_post - nn_pre))
    bd = max((k for k in arms if k.startswith("delta@")), key=lambda k: arms[k]["edit_index"])
    bn = max((k for k in arms if k.startswith("nn_delta@")), key=lambda k: arms[k]["edit_index"])
    out["points"][str(ell)] = {"g_r2_heldout": float(st["r2"]), "g_rmse": float(st["rmse"]), "arms": arms}
    f = lambda r: f"{r['edit_index']:+.3f}/{r['fidelity_ratio']:.2f}"  # noqa: E731
    print(f"pt {ell}: g R² {st['r2']:+.3f} | overwrite {f(arms['overwrite'])} | mean-h {f(arms['mean_overwrite'])} | nn-overwrite {f(arms['nn_overwrite'])} | {bd} {f(arms[bd])} | {bn} {f(arms[bn])}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    del H, H_tr_t, X_tr_t; torch.cuda.empty_cache()
tag = a.run.split("/")[-1] + (f"_{a.tag}" if a.tag else "") + ("_smoke" if a.smoke else "")
(REPO / "experiments/inverse_probe/scores" / f"discworld_{tag}.json").write_text(json.dumps(out, indent=1, default=float))
print("wrote", f"experiments/inverse_probe/scores/discworld_{tag}.json", f"[{(time.time() - t0) / 60:.1f} min]")
