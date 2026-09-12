"""INLP on Othello colour — how many independent linear copies of each tile's colour does the
residual hold, and does writing all of them edit? (2026-09-11, Sevan's register test.)

Target: each tile's colour as ±1 (mine / theirs), fitted on the rows where that tile is OCCUPIED,
in the canonical probe standardisation (per-dim mean / std with the fit_probe floor). Closed-form
min-norm least squares from moment matrices, float64 on the GPU.

Cascades, per residual point:
  per-tile   64 independent deflation cascades, rank 1 per iteration (the fitted direction of that
             tile alone is removed for that tile), to exhaustion (held-out R² < 0.02) — the number
             of redundant linear copies of ONE tile's colour.
  random     the matched control: the same number of random directions removed per tile.
  whole      one shared cascade: every iteration removes the union of the 64 tiles' fitted
             directions (rank ≤ 64) — the total rank of the colour code.
On oth-adjacent-flip every iteration's probe is also scored on flipped-tile and parity-tile rows.

Editability: at the chosen points, write through the first K per-tile probes of the case's tile
(each orthogonal read-out stepped to the flipped sign, summed, × α) on the canonical 1001 cases;
canonical scorecard + fidelity guard. Everything persisted under probes/<run>/inlp/. Self-contained.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from flipped_tiles import replay_track  # noqa: E402
from pim.environments.othello import arms as oa, corpus as oc  # noqa: E402
from pim.environments.othello import case_targets, load_benchmark  # noqa: E402
from pim.environments.othello.data import BLANK, MINE, N_TILES, board_probs, canonical_vocab, harvest_point, tokens_and_labels  # noqa: E402
from pim.metrics.set_editability import move_fidelity_ratio, move_scorecard  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402

DEV = "cuda"; EXP = REPO / "experiments/adjacent_flip_ablation"; D64 = torch.float64
R2_STOP = 0.02


def moments(Zt, y, rows):
    """Augmented (d+1) moments over `rows`: G = Z̃ᵀZ̃, c = Z̃ᵀy, yy = yᵀy, n, ybar."""
    Z = Zt[rows]; yy_ = y[rows]
    Za = torch.cat([Z, torch.ones(len(Z), 1, dtype=D64, device=DEV)], 1)
    return Za.T @ Za, Za.T @ yy_, float(yy_ @ yy_), len(Z), float(yy_.mean())


def solve_minnorm(G, c, B):
    """Min-norm least squares with the residual dims restricted to the complement of span(B).
    G (d+1,d+1), c (d+1), B (d,k) orthonormal (k may be 0). Returns w (d+1) with w[:d] ⊥ B."""
    d = G.shape[0] - 1
    P = torch.eye(d + 1, dtype=D64, device=DEV)
    if B.shape[1]:
        P[:d, :d] -= B @ B.T
    Gp = P @ G @ P; cp = P @ c
    return P @ torch.linalg.pinv(Gp, hermitian=True, rtol=1e-10) @ cp


def r2_from_moments(w, G_te, c_te, yy_te, n_te, ybar_tr):
    sse = yy_te - 2 * float(w @ c_te) + float(w @ G_te @ w)
    sst = yy_te - 2 * ybar_tr * (c_te[-1].item()) + n_te * ybar_tr ** 2       # Σ(y - ȳ_tr)²  (c_te[-1] = Σ y_te)
    return 1.0 - sse / max(sst, 1e-12)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--n-games", type=int, default=20000)
    ap.add_argument("--points", type=int, nargs="+", default=list(range(9))); ap.add_argument("--edit-points", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128]); ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 5.0])
    ap.add_argument("--max-iter", type=int, default=480); a = ap.parse_args(); t0 = time.time()
    run_dir = REPO / a.run; inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]; rules = oc.rules_of(inst)
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV); model.eval(); NP = n_points(model)
    itos = {v: k for k, v in canonical_vocab().items()}
    ptok, pln = oc.load(oc.build(only=("probe",), instance=inst, log=lambda s: None)["probe"])
    data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(ptok[:a.n_games], pln[:a.n_games])], **rules)
    n_seq, T = data.mask.shape
    F = np.zeros(data.mine.shape, bool)
    if rules["flip"]:
        for i in range(n_seq):
            h = [int(itos[int(t)]) for t in ptok[i][: int(pln[i])]]; _, _, _, fl = replay_track(h, rules); Ti = min(len(fl), T); F[i, :Ti] = np.stack(fl[:Ti])
    lab = data.mine[data.mask]; F_all = F[data.mask]; occ = lab != BLANK
    y_col = np.where(lab == MINE, 1.0, -1.0)                                  # ±1, meaningful on occupied rows only
    seq_of_row = np.repeat(np.arange(n_seq), T)[data.mask.reshape(-1)]; tr, te = oa._split(n_seq, seq_of_row, "sequence", 0.2, 0)
    is_tr = np.zeros(len(lab), bool); is_tr[tr] = True
    store = EXP / "probes" / Path(a.run).name / "inlp"; store.mkdir(parents=True, exist_ok=True)
    out = {"run": a.run, "instance": inst, "n_games": n_seq, "target": "colour ±1 on occupied rows, canonical standardisation", "points": {}}
    print(f"{a.run}: {len(lab)} rows, occupied share {occ.mean():.3f}, flipped share of occupied {(F_all & occ).sum() / occ.sum():.3f}", flush=True)
    casc_store = {}
    for p in a.points:
        tp = time.time()
        R = harvest_point(model, data.tokens, p); R = R.reshape(-1, R.shape[-1]); X = R[data.mask.reshape(-1)]; del R
        xm, xs = X[tr].mean(0), X[tr].std(0); xs = np.maximum(xs, 1e-2 * np.median(xs)) + 1e-8
        Z = torch.tensor((X - xm) / xs, dtype=D64, device=DEV); del X; d = Z.shape[1]
        y = torch.tensor(y_col, dtype=D64, device=DEV)
        occ_t = torch.tensor(occ, device=DEV); tr_t = torch.tensor(is_tr, device=DEV); F_t = torch.tensor(F_all, device=DEV)
        # per-tile moments (train / test / test-flipped / test-parity)
        M = {}
        for t in range(N_TILES):
            rows_tr = occ_t[:, t] & tr_t; rows_te = occ_t[:, t] & ~tr_t
            M[t] = {"tr": moments(Z, y[:, t], rows_tr), "te": moments(Z, y[:, t], rows_te)}
            if rules["flip"]:
                M[t]["te_f"] = moments(Z, y[:, t], rows_te & F_t[:, t]); M[t]["te_p"] = moments(Z, y[:, t], rows_te & ~F_t[:, t])
        def r2s(w, t):
            G_tr, c_tr, _, _, ybar = M[t]["tr"]; res = {}
            for key in ("te", "te_f", "te_p"):
                if key in M[t] and M[t][key][3] > 20:
                    G_te, c_te, yy_te, n_te, _ = M[t][key]; res[key] = r2_from_moments(w, G_te, c_te, yy_te, n_te, ybar)
            return res
        # ── per-tile cascades + random control ─────────────────────────────────────
        per_tile = {}; rng = torch.Generator(device=DEV).manual_seed(0)
        for t in range(N_TILES):
            G_tr, c_tr = M[t]["tr"][0], M[t]["tr"][1]
            B = torch.zeros(d, 0, dtype=D64, device=DEV); Ws, r2_curve, r2_f, r2_p = [], [], [], []
            for k in range(a.max_iter):
                w = solve_minnorm(G_tr, c_tr, B); r = r2s(w, t); r2_curve.append(r["te"]); r2_f.append(r.get("te_f")); r2_p.append(r.get("te_p")); Ws.append(w)
                u = w[:d] - B @ (B.T @ w[:d]); nu = u.norm()
                if r["te"] < R2_STOP or nu < 1e-9: break
                B = torch.cat([B, (u / nu)[:, None]], 1)
            # random control: same number of random directions inside the remaining subspace
            k_real = len(r2_curve); Br = torch.zeros(d, 0, dtype=D64, device=DEV); r2_rand = []
            for k in range(k_real):
                w = solve_minnorm(G_tr, c_tr, Br); r2_rand.append(r2s(w, t)["te"])
                g = torch.randn(d, generator=rng, dtype=D64, device=DEV); g = g - Br @ (Br.T @ g); Br = torch.cat([Br, (g / g.norm())[:, None]], 1)
            per_tile[t] = {"r2": r2_curve, "r2_flipped": r2_f, "r2_parity": r2_p, "r2_random": r2_rand, "k_exhaust": k_real, "B": B.cpu(), "W": torch.stack(Ws).cpu(), "mu": M[t]["tr"][4]}
        # ── whole-subspace cascade ─────────────────────────────────────────────────
        Bw = torch.zeros(d, 0, dtype=D64, device=DEV); whole = []
        for k in range(12):
            ws = [solve_minnorm(M[t]["tr"][0], M[t]["tr"][1], Bw) for t in range(N_TILES)]
            r2_mean = float(np.mean([r2s(w, t)["te"] for t, w in enumerate(ws)])); whole.append({"k": k + 1, "r2_mean": r2_mean, "rank_removed": int(Bw.shape[1])})
            if r2_mean < R2_STOP or Bw.shape[1] >= d - N_TILES: break
            U = torch.stack([w[:d] for w in ws], 1); U = U - Bw @ (Bw.T @ U); Q, S = torch.linalg.qr(U); keep = (torch.linalg.svdvals(U) > 1e-8).sum().item()
            Uo, _, _ = torch.linalg.svd(U, full_matrices=False); Bw = torch.cat([Bw, Uo[:, :keep]], 1)
        # ── summaries ────────────────────────────────────────────────────────────────
        kmax = max(v["k_exhaust"] for v in per_tile.values())
        def curve(key):
            return [float(np.nanmean([v[key][k] if k < len(v[key]) and v[key][k] is not None else 0.0 for v in per_tile.values()])) for k in range(kmax)]
        mean_r2 = curve("r2"); mean_rand = curve("r2_random")
        k_half = int(np.argmax(np.array(mean_r2) < 0.5 * mean_r2[0])) if any(np.array(mean_r2) < 0.5 * mean_r2[0]) else kmax
        summ = {"k_exhaust_mean": float(np.mean([v["k_exhaust"] for v in per_tile.values()])), "k_exhaust_median": float(np.median([v["k_exhaust"] for v in per_tile.values()])),
                "k_half_mean_curve": k_half, "r2_first": mean_r2[0], "mean_r2_curve": mean_r2, "mean_r2_random_curve": mean_rand,
                "mean_r2_flipped_curve": curve("r2_flipped") if rules["flip"] else None, "mean_r2_parity_curve": curve("r2_parity") if rules["flip"] else None,
                "whole_subspace": whole, "seconds": round(time.time() - tp, 1)}
        out["points"][str(p)] = summ
        torch.save({"per_tile": per_tile, "x_mean": xm, "x_std": xs, "point": p, "run": a.run, "target": out["target"]}, store / f"cascade_pt{p}.pt")
        casc_store[p] = (per_tile, xm, xs)
        pr = lambda v: " ".join(f"{x:.2f}" for x in v[:10]) + (" …" if len(v) > 10 else "")
        print(f"pt {p} [{summ['seconds']}s]: per-tile R² by iteration (mean over tiles): {pr(mean_r2)} | random ctrl: {pr(mean_rand)} | k_half {k_half}, k_exhaust mean {summ['k_exhaust_mean']:.1f} median {summ['k_exhaust_median']:.0f}"
              + (f" | flipped: {pr(summ['mean_r2_flipped_curve'])} | parity: {pr(summ['mean_r2_parity_curve'])}" if rules["flip"] else "")
              + f" | whole-subspace R² by iteration: {' '.join(f'{w['r2_mean']:.2f}' for w in whole)} (rank {whole[-1]['rank_removed']})", flush=True)
        del Z, M; torch.cuda.empty_cache()
    # ── editability through the first K per-tile probes ─────────────────────────────
    bench = load_benchmark(inst); cur, tgt = case_targets(bench); uns = oa.unsteered_probs(model, bench); u = oa.unsteered(model, bench)
    y_star = np.where(tgt == MINE, 1.0, -1.0).astype(np.float32)
    out["edits"] = {"unedited": u["edit_index_union"], "arms": []}
    print(f"\nedits on the canonical {len(bench.pos_int)} cases (unedited {u['edit_index_union']:+.3f}):", flush=True)
    for p in a.edit_points:
        if p not in casc_store: continue
        per_tile, xm, xs = casc_store[p]; xm_t = torch.tensor(xm, device=DEV); xs_t = torch.tensor(xs, device=DEV)
        Kmax = max(per_tile[t]["W"].shape[0] for t in range(N_TILES)); d1 = per_tile[0]["W"].shape[1]
        Wpad = torch.zeros(N_TILES, Kmax, d1, device=DEV); R2 = torch.zeros(N_TILES, Kmax, device=DEV); valid = torch.zeros(N_TILES, Kmax, dtype=torch.bool, device=DEV)
        MU = torch.tensor([per_tile[t]["mu"] for t in range(N_TILES)], device=DEV, dtype=torch.float32)
        for t in range(N_TILES):
            W = per_tile[t]["W"].to(DEV).float(); k_t = W.shape[0]; Wpad[t, :k_t] = W; valid[t, :k_t] = True
            R2[t, :k_t] = torch.tensor([max(r, 0.0) for r in per_tile[t]["r2"]], device=DEV)
        for mode in ("exact", "shrink"):
            for K in a.ks:
                for al in a.alphas:
                    probs = np.zeros((len(bench.pos_int), N_TILES), np.float32)
                    for toks, ids in zip(bench.tokens, bench.case_ids):
                        idx = torch.from_numpy(toks).to(DEV); bsz = len(ids); sq = torch.from_numpy(bench.pos_int[ids]).to(DEV); ys = torch.from_numpy(y_star[ids]).to(DEV)
                        def hook(layer, x, sq=sq, ys=ys, bsz=bsz):
                            if layer != p: return x
                            c = x[:, -1]; z = (c - xm_t) / xs_t
                            W = Wpad[sq]; w = W[..., :-1]; b = W[..., -1]                          # (b, Kmax, d), (b, Kmax)
                            r = torch.einsum("bd,bkd->bk", z, w) + b
                            tgt_k = ys[:, None].expand_as(r) if mode == "exact" else MU[sq][:, None] + R2[sq] * (ys[:, None] - MU[sq][:, None])
                            m = valid[sq] & (torch.arange(Kmax, device=DEV)[None, :] < K)
                            coef = torch.where(m, (tgt_k - r) / (w * w).sum(-1).clamp_min(1e-12), torch.zeros_like(r))
                            dz = torch.einsum("bk,bkd->bd", coef, w)
                            o = x.clone(); o[:, -1] = c + al * dz * xs_t; return o
                        with torch.no_grad(): probs[ids] = board_probs(model.decode(idx, edit=hook), getattr(model, "output_kind", "logits"))
                    card = move_scorecard(probs, bench.legal_pre, bench.legal_post); fid = move_fidelity_ratio(probs, uns, bench.legal_post)
                    out["edits"]["arms"].append({"point": p, "mode": mode, "K": K, "alpha": al, "ei": card["edit_index_union"], "fid": fid, "legal_mass": card["legal_mass"]})
            for K in a.ks:
                arms = [r for r in out["edits"]["arms"] if r["point"] == p and r["K"] == K and r["mode"] == mode]; b = max(arms, key=lambda r: r["ei"]); g = [r for r in arms if r["fid"] <= 1.1]; bg = max(g, key=lambda r: r["ei"]) if g else None
                print(f"  pt {p} {mode:6s} K={K:3d}: best {b['ei']:+.3f}/fid {b['fid']:.2f} (α{b['alpha']:g})" + (f" | guarded {bg['ei']:+.3f}/{bg['fid']:.2f} (α{bg['alpha']:g})" if bg else " | no guarded arm"), flush=True)
        (EXP / "scores" / f"inlp_othello_{Path(a.run).name}.json").write_text(json.dumps(out, indent=1, default=float))
    out["minutes"] = round((time.time() - t0) / 60, 1)
    (EXP / "scores" / f"inlp_othello_{Path(a.run).name}.json").write_text(json.dumps(out, indent=1, default=float)); print("done", out["minutes"], "min", flush=True)


if __name__ == "__main__":
    main()
