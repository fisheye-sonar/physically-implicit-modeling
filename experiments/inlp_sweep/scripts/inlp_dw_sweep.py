"""INLP SWEEP on discworld (2026-09-14 evening, Sevan): PER-VARIABLE iterative nullspace projection —
the construction of experiments/adjacent_flip_ablation/scripts/inlp_othello.py (one rank-1 cascade per
target variable, deflating that variable's own fitted direction, to exhaustion) on the discworld state,
plus the write through the first K orthogonal copies of every variable with each copy's target shrunk
toward the population mean by its own held-out R² (the `pim.editors.nullspace` shrink rule). The write is
solved JOINTLY across variables (2026-09-14 16:55, Sevan): all variables' first K directions are stacked
into one matrix A and Δz = A⁺ (t − r) — within a variable the rows are orthogonal, across variables they are
not, and a sum of independent per-variable steps lets each step disturb the others' read-outs. Because
multi-output least squares is separable per output, at K = 1 the stacked rows ARE the joint lstsq probe's
rows and the write is closed-form PI in z-space (the earlier INLP's wiring check: within 6–10 % of the
canonical PI step through the gradient-fitted probe).

Targets (--target):
  full             the canonical regression state — 8 variables: frustum position (x, y) and velocity
                   (vx, vy) of both objects; the edit writes the bench's pre-dynamics target for all 8
                   (dims "all": the edited object's position moves, everything else holds).
  appearance-fac   the factorised appearance target as 4 ORDINAL variables: run centre and run length
                   of each object (the local factor index); the edit writes the edited object's two
                   factors to their new indices (`Bench.moves`) and holds the other object's.

Per residual point: one closed-form min-norm least-squares cascade per variable on the standardised
residual (moment matrices, float64, GPU; the fitted direction is removed and the fit repeated until
held-out R² < 0.02 or max_iter), a matched random-direction control, the mean-over-variables R² curve
(the Othello figure's quantity), iterations to R² < 0.4 and < 0.05 on that curve, copies per variable;
then the K-copy write (K ∈ {1, 2, 4, 8, 16, 32, 64, 128, all}; exact at K = 1 as the reference, shrink
elsewhere) × α, scored on the canonical 1000-case bench (frame models: ray-zone Edit Index + guard; the
token model: the token bench's frame-set construction) against the run's canonical PI / ND / GS.

    python experiments/inlp_sweep/scripts/inlp_dw_sweep.py --run ray_ablation/L-dw-8ray-20m [--target appearance-fac] [--smoke]
Contained: reads the run's checkpoint and probe corpus; writes only under experiments/inlp_sweep/.
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile, time
from pathlib import Path
import h5py, numpy as np, torch

REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from pim.environments import layout  # noqa: E402
from pim.environments.discworld import arms as dwa, bench as dwb, token_bench as tkb  # noqa: E402
from pim.environments.discworld.bench import EF, K_ROLL, N_OBJ  # noqa: E402
from pim.environments.discworld.grid_target import FactorisedTarget  # noqa: E402
from pim.environments.discworld.tokens import FrameVocab  # noqa: E402
from pim.metrics.set_editability import move_fidelity_ratio  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.base import collect_residuals  # noqa: E402

DEV = "cuda"; D64 = torch.float64; EXP = REPO / "experiments/inlp_sweep"
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--target", choices=("full", "appearance-fac"), default="full")
ap.add_argument("--basis", default="frustum")
ap.add_argument("--n-seq", type=int, default=20000)
ap.add_argument("--points", type=int, nargs="*", default=None)
ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
ap.add_argument("--alphas", type=float, nargs="*", default=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0, 175.0])
ap.add_argument("--max-iter", type=int, default=480)
ap.add_argument("--r2-stop", type=float, default=0.02)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--solve", choices=("pinv", "trunc", "wridge"), default="wridge",
                help="joint step solver: pinv (unregularised), trunc (pseudo-inverse with singular values below --rtol × max dropped), wridge (rows weighted by R², ridge --ridge × mean diagonal)")
ap.add_argument("--rtol", type=float, default=1e-2)
ap.add_argument("--ridge", type=float, default=1e-2)
ap.add_argument("--writes-only", action="store_true", help="skip residual collection and cascade fitting; load probes/<run>/cascade_pt*.pt and re-run the write sweep")
a = ap.parse_args(); t0 = time.time()
run = REPO / "runs" / a.run; name = a.run.split("/")[-1]; FAC = a.target == "appearance-fac"
scores = json.loads((run / "scores.json").read_text()); S = scores["settings"]
canon = scores["bases"]["appearance-fac" if FAC else a.basis]["best"]
inst = json.loads((run / "config.json").read_text())["data"]["instance"]
model, _ = load_checkpoint(run / "best_model.pt", device=DEV); model.eval(); NP = n_points(model)
TOKENS = (run / "vocab.npz").exists()
span = int(getattr(model, "state_span", 39))
n_seq = 2000 if a.smoke else a.n_seq
alphas = [0.5, 1.0, 2.0] if a.smoke else list(a.alphas)
points = a.points if a.points is not None else ([4] if a.smoke else list(range(NP)))
max_iter = 40 if a.smoke else a.max_iter
n_bench = 50 if a.smoke else S["dw_bench_n"]
store = EXP / "probes" / (name + ("_fac" if FAC else "")); store.mkdir(parents=True, exist_ok=True)
fac = FactorisedTarget.parse("appearance-fac") if FAC else None

# ── probe corpus → per-frame VARIABLES (the canonical split by sequence, dwb.SEED) ────────
with h5py.File(layout.probe_file("discworld", inst, "120k"), "r") as f:
    obs = f["obs_intensity"][:n_seq, :span].astype(np.float32)
    pos = f["positions"][:n_seq, :span, :N_OBJ, :].astype(np.float32)
    vel = f["velocities"][:n_seq, :span, :N_OBJ, :].astype(np.float32)
sim = json.load(open(layout.probe_manifest("discworld", inst, "120k")))["sim"]
if FAC:
    F = fac.n_factors_on(sim); offs = np.tile(fac.class_offsets(sim), N_OBJ)                    # (N_OBJ·F,) per tile
    Y3 = (fac.factor_labels(pos, sim) - offs).astype(np.float64)                                  # (N, T, N_OBJ·F) local factor index
    VAR = [f"obj{j}_{fn}" for j in range(N_OBJ) for fn in fac.cat.factor_names[:F]]
else:
    Y3, _ = dwa._targets("full", pos, vel, sim, a.basis)
    VAR = [f"obj{j}_{c}" for j in range(N_OBJ) for c in ("x", "y")] + [f"obj{j}_{c}" for j in range(N_OBJ) for c in ("vx", "vy")]
T = Y3.shape[1]; Y = Y3.reshape(-1, Y3.shape[-1]).astype(np.float64); m = Y.shape[1]
perm = np.random.default_rng(dwb.SEED).permutation(n_seq)
tr = np.isin(np.repeat(np.arange(n_seq), T), perm[: int(0.8 * n_seq)])
tr_idx, te_idx = np.where(tr)[0], np.where(~tr)[0]
ybar = Y[tr_idx].mean(0); Y_t = torch.from_numpy(Y).to(DEV)
inp = obs
if TOKENS:
    vocab = FrameVocab.load(run / "vocab.npz"); enc, _ = tkb.token_encoder(vocab); inp = enc(obs)
del pos, vel, Y3
print(f"{a.run} ({inst}, {'tokens' if TOKENS else 'frames'}, target {a.target}, {m} variables {VAR}): n_seq {n_seq} × {T} = {len(Y):,} rows; "
      f"points {points}; Ks {a.ks}+all; α {alphas}; max_iter {max_iter}; r2_stop {a.r2_stop}", flush=True)

# ── bench: the per-variable TARGETS (raw units) ───────────────────────────────────────────
if TOKENS:
    tb = tkb.load_token_bench(vocab, n=n_bench, target=a.target, basis_name=a.basis, instance=inst)
    uns, u = tkb.unsteered(model, tb); H0 = tkb.residuals_last(model, tb); bt = tb
else:
    b = dwb.load_bench(model, n=n_bench, target=a.target, basis_name=a.basis, instance=inst)
    u = dwa.unsteered(model, b); bt = b
tgt = bt.tgt.cpu().numpy().astype(np.float64)
if FAC:
    tgt = tgt - offs[None, :]                     # shared class axis → local factor index (edited object's tiles at their NEW classes, the rest held)
n_cases = bt.n
print(f"bench: {n_cases} cases · unedited {u['edit_index']:+.3f} · canonical PI {canon['PI']['edit_index']:+.3f}/{canon['PI']['fidelity_ratio']:.2f}"
      + (f"  ND {canon['ND']['edit_index']:+.3f}/{canon['ND']['fidelity_ratio']:.2f}" if canon.get("ND") else "")
      + f"  GS {canon['GS']['edit_index']:+.3f}/{canon['GS']['fidelity_ratio']:.2f}", flush=True)


# ── per-variable rank-1 cascades from moments ─────────────────────────────────────────────
def moments(H, mu_t, sd_t, idx, chunk=131072):
    d = H.shape[1]; G = torch.zeros(d + 1, d + 1, dtype=D64, device=DEV); C = torch.zeros(d + 1, m, dtype=D64, device=DEV); yy = torch.zeros(m, dtype=D64, device=DEV)
    for i in range(0, len(idx), chunk):
        ii = idx[i:i + chunk]
        Z = (torch.from_numpy(H[ii]).to(DEV).to(D64) - mu_t) / sd_t
        Za = torch.cat([Z, torch.ones(len(ii), 1, dtype=D64, device=DEV)], 1); Yb = Y_t[torch.from_numpy(ii).to(DEV)]
        G += Za.T @ Za; C += Za.T @ Yb; yy += (Yb * Yb).sum(0); del Z, Za
    return G, C, yy, len(idx)


def solve_minnorm(G, c, B):
    """Min-norm least squares of ONE variable with the residual dims restricted to the complement of span(B): w (d+1), w[:d] ⊥ B."""
    d = G.shape[0] - 1; P = torch.eye(d + 1, dtype=D64, device=DEV)
    if B.shape[1]:
        P[:d, :d] -= B @ B.T
    return P @ torch.linalg.pinv(P @ G @ P, hermitian=True, rtol=1e-10) @ (P @ c)


def r2_of(w, G_te, c_te, yy_te, n_te, yb):
    sse = yy_te - 2 * float(w @ c_te) + float(w @ G_te @ w)
    sst = yy_te - 2 * yb * float(c_te[-1]) + n_te * yb ** 2
    return 1.0 - sse / max(sst, 1e-12)


def cascade(v, G_tr, C_tr, G_te, C_te, yy_te, n_te, *, random_dirs=False, n_iter=None, gen=None):
    d = G_tr.shape[0] - 1; B = torch.zeros(d, 0, dtype=D64, device=DEV); Ws, r2s = [], []
    for k in range(n_iter if n_iter else max_iter):
        w = solve_minnorm(G_tr, C_tr[:, v], B); r = r2_of(w, G_te, C_te[:, v], float(yy_te[v]), n_te, float(ybar[v])); r2s.append(r); Ws.append(w)
        if random_dirs:
            g = torch.randn(d, generator=gen, dtype=D64, device=DEV); g = g - B @ (B.T @ g); unew = g / g.norm()
        else:
            uu = w[:d] - B @ (B.T @ w[:d]); nu = uu.norm()
            if r < a.r2_stop or nu < 1e-9:
                break
            unew = uu / nu
        if B.shape[1]:
            assert float((B.T @ unew).abs().max()) < 1e-6, f"variable {v} iteration {k+1}: new direction not orthogonal"
        B = torch.cat([B, unew[:, None]], 1)
    return torch.stack(Ws), r2s, B


def iters_to(curve, thr):
    for k, r in enumerate(curve):
        if r < thr:
            return k
    return None


def score_write(ell, h_new):
    if TOKENS:
        probs = tkb.probs_at_edit(model, tb, hook=tkb._write_hook(ell, h_new)); c = tkb.scorecard(probs, tb, uns)
        return {"edit_index": c["edit_index"], "fidelity_ratio": c.get("fidelity_ratio", move_fidelity_ratio(probs, uns, tb.legal_post))}
    roll = model.rollout_with_edit(b.state, ell, h_new, K_ROLL).cpu().numpy(); c = dwa.score(model, b, roll, u)
    return {"edit_index": c["edit_index"], "fidelity_ratio": c["fidelity_ratio"]}


tagf = name + ("_fac" if FAC else "") + ("_smoke" if a.smoke else "")
out = {"run": a.run, "instance": inst, "model_kind": "tokens" if TOKENS else "frames", "target": a.target, "basis": a.basis, "variables": VAR, "n_seq": n_seq, "n_cases": int(n_cases),
       "settings": {"max_iter": max_iter, "r2_stop": a.r2_stop, "Ks": a.ks + ["all"], "alphas": alphas, "construction": "per-variable rank-1 deflation (inlp_othello); shrink target_k = μ_v + R²_k (t_v − μ_v); joint step over the stacked copies", "solve": a.solve, "rtol": a.rtol, "ridge": a.ridge},
       "unedited": {k: v for k, v in u.items() if isinstance(v, (int, float))},
       "canonical": {e: {"edit_index": canon[e]["edit_index"], "fidelity_ratio": canon[e]["fidelity_ratio"]} for e in ("PI", "ND", "GS") if canon.get(e)}, "points": {}}
sdir = REPO / ".scratch"; sdir.mkdir(exist_ok=True)
prev = json.load(open(EXP / "scores" / f"inlp_{tagf}.json")) if a.writes_only and (EXP / "scores" / f"inlp_{tagf}.json").exists() else None
if a.writes_only:
    out["settings"]["rescored_writes"] = f"joint step over the stacked copies, solver {a.solve} (rtol {a.rtol}, ridge {a.ridge})"
for ell in points:
    tp = time.time()
    if a.writes_only:
        ck = torch.load(store / f"cascade_pt{ell}.pt", weights_only=False); per_var = ck["per_var"]; mu, sd = ck["x_mean"], ck["x_std"]
        d = per_var[0]["W"].shape[1] - 1
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False, dir=sdir); tmp.close()
        try:
            R = collect_residuals(model, inp, batch=64, memmap=tmp.name, points=[ell])[0]; H = np.ascontiguousarray(R.reshape(-1, R.shape[-1])); del R
        finally:
            os.unlink(tmp.name)
        mu = H[tr_idx].mean(0).astype(np.float64); sd = (H[tr_idx].std(0) + 1e-6).astype(np.float64)
        mu_t, sd_t = torch.from_numpy(mu).to(DEV), torch.from_numpy(sd).to(DEV)
        G_tr, C_tr, _, _ = moments(H, mu_t, sd_t, tr_idx); G_te, C_te, yy_te, n_te = moments(H, mu_t, sd_t, te_idx); del H
        d = G_tr.shape[0] - 1; gen = torch.Generator(device=DEV).manual_seed(0)
        per_var = {}
        for v in range(m):
            W, r2s, B = cascade(v, G_tr, C_tr, G_te, C_te, yy_te, n_te)
            _, r2r, _ = cascade(v, G_tr, C_tr, G_te, C_te, yy_te, n_te, random_dirs=True, n_iter=len(r2s), gen=gen)
            per_var[v] = {"W": W.float().cpu(), "r2": r2s, "r2_random": r2r, "k_exhaust": len(r2s), "mu": float(ybar[v])}
    kmax = max(pv["k_exhaust"] for pv in per_var.values())
    curve = lambda key: [float(np.mean([pv[key][k] if k < len(pv[key]) else 0.0 for pv in per_var.values()])) for k in range(kmax)]  # noqa: E731
    mean_r2, mean_rand = curve("r2"), curve("r2_random")
    if not a.writes_only:
        torch.save({"per_var": per_var, "variables": VAR, "x_mean": mu, "x_std": sd, "point": ell, "run": a.run, "target": a.target}, store / f"cascade_pt{ell}.pt")
    summ = {"variables": VAR, "r2_first_by_var": [pv["r2"][0] for pv in per_var.values()], "k_exhaust_by_var": [pv["k_exhaust"] for pv in per_var.values()],
            "k_exhaust_mean": float(np.mean([pv["k_exhaust"] for pv in per_var.values()])), "k_exhaust_max": int(kmax),
            "mean_r2_curve": mean_r2, "mean_r2_random_curve": mean_rand, "r2_curve_by_var": [pv["r2"] for pv in per_var.values()],
            "iters_to_0.4": iters_to(mean_r2, 0.4), "iters_to_0.05": iters_to(mean_r2, 0.05), "iters_to_exhaust": iters_to(mean_r2, a.r2_stop)}
    if not a.writes_only:
        del G_tr, C_tr, G_te, C_te
    # ── the K-copy write: every variable through its first K orthogonal copies ────────────
    Wpad = torch.zeros(m, kmax, d + 1, device=DEV); R2 = torch.zeros(m, kmax, device=DEV); valid = torch.zeros(m, kmax, dtype=torch.bool, device=DEV)
    for v in range(m):
        k_v = per_var[v]["k_exhaust"]; Wpad[v, :k_v] = per_var[v]["W"].to(DEV); valid[v, :k_v] = True; R2[v, :k_v] = torch.tensor([max(r, 0.0) for r in per_var[v]["r2"]], device=DEV)
    MU = torch.tensor(ybar, device=DEV, dtype=torch.float32); TGT = torch.from_numpy(tgt.astype(np.float32)).to(DEV)                   # (n, m)
    mu32, sd32 = torch.from_numpy(mu.astype(np.float32)).to(DEV), torch.from_numpy(sd.astype(np.float32)).to(DEV)
    if TOKENS:
        h0 = H0[ell]
    else:
        dwa.as_activations(model, ell); h0 = model.flat_state(b.state)
    z = (h0 - mu32) / sd32                                                                                   # (n, d)
    w_all, b_all = Wpad[..., :-1], Wpad[..., -1]                                                             # (m, K, d), (m, K)
    r = torch.einsum("nd,mkd->nmk", z, w_all) + b_all[None]                                                  # every copy's read-out
    wn2 = (w_all * w_all).sum(-1).clamp_min(1e-12)[None]                                                     # (1, m, K)

    def delta(K, shrink):
        """Joint least-squares step: stack every variable's first K copies (rows of A), solve Δz = A⁺ (t − r)."""
        t_ex = TGT[:, :, None].expand_as(r)
        t_k = MU[None, :, None] + R2[None] * (TGT[:, :, None] - MU[None, :, None]) if shrink else t_ex
        msk = valid & (torch.arange(kmax, device=DEV)[None, :] < K)                                        # (m, K)
        A = w_all[msk].double()                                                                             # (R, d) the stacked copies
        rhs = (t_k - r)[:, msk].double()                                                                    # (n, R)
        if a.solve == "pinv":
            dz = rhs @ torch.linalg.pinv(A).T
        elif a.solve == "trunc":
            dz = rhs @ torch.linalg.pinv(A, rtol=a.rtol).T
        else:                                                                                               # weighted ridge: (AᵀΩA + λI)⁻¹ AᵀΩ δ, Ω = diag(R²)
            om = R2[msk].double().clamp_min(1e-6)
            M_ = A.T @ (om[:, None] * A); lam = a.ridge * float(M_.diagonal().mean())
            dz = rhs @ (om[:, None] * A) @ torch.linalg.inv(M_ + lam * torch.eye(A.shape[1], dtype=D64, device=DEV)).T
        return dz.float() * sd32                                                                            # back to raw residual units
    arms = []
    Ks = [K for K in a.ks if K <= kmax] + ([kmax] if kmax not in a.ks else [])
    for K in Ks:
        for shrink in ((False, True) if K == 1 else (True,)):
            dh = delta(K, shrink)
            for al in alphas:
                with torch.no_grad():
                    c = score_write(ell, h0 + al * dh)
                arms.append({"K": int(K), "K_is_all": K == kmax, "shrink": bool(shrink), "alpha": float(al), "write_ratio": float((al * dh).norm(dim=1).div(h0.norm(dim=1)).mean()), **c})
    shr = [x for x in arms if x["shrink"]]; best = max(shr, key=lambda x: x["edit_index"]); guarded = [x for x in shr if x["fidelity_ratio"] < 1]
    by_K = {}
    for K in Ks:
        sub = [x for x in shr if x["K"] == K]; g = [x for x in sub if x["fidelity_ratio"] < 1]
        by_K[str(K)] = {"best": max(sub, key=lambda x: x["edit_index"]), "best_guarded": max(g, key=lambda x: x["edit_index"]) if g else None}
    k1u = [x for x in arms if x["K"] == 1 and not x["shrink"]]
    summ.update({"arms": arms, "best_shrink": best, "best_shrink_guarded": max(guarded, key=lambda x: x["edit_index"]) if guarded else None, "best_by_K": by_K,
                 "k1_exact_best": max(k1u, key=lambda x: x["edit_index"]), "seconds": round(time.time() - tp, 1)})
    out["points"][str(ell)] = summ
    pr = lambda v: " ".join(f"{x:.2f}" for x in v[:12]) + (" …" if len(v) > 12 else "")  # noqa: E731
    bk = "  ".join(f"K{K}: {by_K[str(K)]['best']['edit_index']:+.2f}/{by_K[str(K)]['best']['fidelity_ratio']:.2f}" for K in Ks)
    print(f"pt {ell} [{summ['seconds']}s]: mean R² by iteration {pr(mean_r2)} | random {pr(mean_rand)} | iters to <0.4: {summ['iters_to_0.4']}, <0.05: {summ['iters_to_0.05']} | "
          f"copies per variable {summ['k_exhaust_by_var']} (mean {summ['k_exhaust_mean']:.1f}) | probe-1 R² by variable {[round(x, 2) for x in summ['r2_first_by_var']]}"
          f"\n      shrink best by K: {bk} | best {best['edit_index']:+.3f}/{best['fidelity_ratio']:.2f} (K{best['K']}, α{best['alpha']:g})", flush=True)
    (EXP / "scores" / f"inlp_{tagf}.json").write_text(json.dumps(out, indent=1, default=float))
    del Wpad, R2, valid, r, per_var; torch.cuda.empty_cache()
out["minutes"] = round((time.time() - t0) / 60, 1)
(EXP / "scores" / f"inlp_{tagf}.json").write_text(json.dumps(out, indent=1, default=float))
print(f"wrote experiments/inlp_sweep/scores/inlp_{tagf}.json [{out['minutes']} min]", flush=True)
