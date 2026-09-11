"""Three cheap tests on L-oth-adjacent-flip-20m (2026-09-11), Sevan's questions:
 (a) does legality prediction degrade in prefixes with many recolourings so far?
 (b) is a FLIPPED tile (current colour != placement colour) more editable than a parity tile,
     searching all residual points (its register may live elsewhere)?
 (c) how accurate are the cached mine/theirs probes on flipped vs parity tiles, per point?
Self-contained; output scores/flipped_tiles_<run>.json.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from pim.environments.othello import arms as oa, corpus as oc  # noqa: E402
from pim.environments.othello import case_targets  # noqa: E402
from pim.environments.othello.bench import benchmark_from_cases, shipped_length_distribution  # noqa: E402
from pim.environments.othello.data import (BLANK, CENTRE, board_probs, canonical_vocab, harvest_point,  # noqa: E402
                                           tokens_and_labels)
from pim.environments.othello.vendor.othello import OthelloBoardState  # noqa: E402
from pim.metrics.set_editability import move_rmse_per_case  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402

DEV = "cuda"; EXP = REPO / "experiments/adjacent_flip_ablation"


def replay_track(h, rules):
    """Per position t (after move t): board state, flips-so-far count, and per-tile flipped flag
    (current colour != colour when first placed)."""
    b = OthelloBoardState(**rules); placed = np.zeros(64, int); flips = 0
    states, nflips, flipped = [], [], []
    for mv in h:
        before = b.state.copy(); b.umpire(mv); after = b.state
        for sq in range(64):
            r, c = divmod(sq, 8)
            if before[r, c] == 0 and after[r, c] != 0: placed[sq] = after[r, c]
            elif before[r, c] != 0 and after[r, c] != before[r, c]: flips += 1
        states.append(after.copy()); nflips.append(flips)
        flipped.append(np.array([after[sq // 8, sq % 8] != 0 and after[sq // 8, sq % 8] != placed[sq] for sq in range(64)]))
    return b, states, nflips, flipped


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="runs/adjacent_flip_ablation/L-oth-adjacent-flip-20m")
    ap.add_argument("--n-cases", type=int, default=300); a = ap.parse_args(); t0 = time.time()
    run_dir = REPO / a.run; inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]; rules = oc.rules_of(inst)
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV); model.eval(); NP = n_points(model)
    stoi = canonical_vocab(); itos = {v: k for k, v in stoi.items()}
    out = {"run": a.run, "instance": inst}

    def pred(hists):
        res = np.zeros((len(hists), 64), np.float32); by = {}
        for i, h in enumerate(hists): by.setdefault(len(h), []).append(i)
        with torch.no_grad():
            for L, ids in by.items():
                idx = torch.from_numpy(np.array([[stoi[x] for x in hists[i]] for i in ids])).to(DEV)
                res[ids] = board_probs(model.decode(idx), getattr(model, "output_kind", "logits"))
        return res

    # ── (a) legality quality vs recolourings so far ───────────────────────────────
    tok, ln = oc.load(oc.build(only=("test",), instance=inst, log=lambda s: None)["test"])
    games = [[int(itos[int(t)]) for t in row[:L]] for row, L in zip(tok[:3000], ln[:3000])]
    rng = np.random.default_rng(0); hs, nf, legal = [], [], []
    for g in games:
        L = int(rng.integers(8, min(45, len(g)))); h = g[:L]
        b, _, nflips, _ = replay_track(h, rules); lg = sorted(b.get_valid_moves())
        if not lg: continue
        hs.append(h); nf.append(nflips[-1]); legal.append(lg)
    P = pred(hs); nf = np.array(nf)
    mass = np.array([P[i, L].sum() for i, L in enumerate(legal)]); rmse = move_rmse_per_case(P, legal)
    bins = [(0, 0), (1, 3), (4, 8), (9, 15), (16, 99)]
    out["a_legality_by_flips"] = []
    print("(a) legality quality by recolourings so far in the prefix (test split, lengths 8-44):")
    for lo, hi in bins:
        m = (nf >= lo) & (nf <= hi)
        if m.sum() < 10: continue
        r = {"flips": f"{lo}-{hi}", "n": int(m.sum()), "legal_mass": float(mass[m].mean()), "rmse_to_uniform": float(rmse[m].mean()), "mean_len": float(np.mean([len(hs[i]) for i in np.where(m)[0]]))}
        out["a_legality_by_flips"].append(r); print(f"   flips {lo:>2}-{hi:<2} n={r['n']:4d} len {r['mean_len']:4.1f}: legal mass {r['legal_mass']:.4f}  rmse-to-uniform {r['rmse_to_uniform']:.4f}")

    # ── (c) probe accuracy on flipped vs parity tiles, per point ─────────────────
    ptok, pln = oc.load(oc.build(only=("probe",), instance=inst, log=lambda s: None)["probe"])
    data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(ptok[:20000], pln[:20000])], **rules)
    grid = oa.fit_probe_grid(model, data, cache_dir=run_dir / "probes", log=None)
    n_eval = 1500; sub = data.tokens[:n_eval]; mask = data.mask[:n_eval]; labels = data.mine[:n_eval]      # (n, T, 64)
    flipped = np.zeros(labels.shape, bool)
    for i in range(n_eval):
        h = [int(itos[int(t)]) for t in ptok[i][: int(pln[i])]]
        _, _, _, fl = replay_track(h, rules)
        T = min(len(fl), labels.shape[1])
        flipped[i, :T] = np.stack(fl[:T])
    occ = labels != BLANK
    print(f"(c) probe error on FLIPPED vs PARITY occupied tiles ({n_eval} probe games; flipped fraction of occupied tile-rows {flipped[mask & occ[:, :, 0][:, :, None].repeat(64, 2)].mean() if False else (flipped & occ)[mask].sum() / occ[mask].sum():.3f}):")
    out["c_probe_error"] = {}
    for p in range(NP):
        probe = grid.probes[("mine", "linear", "sequence", p)]
        R = harvest_point(model, sub, p); R = R.reshape(-1, R.shape[-1]) if R.ndim == 3 else R
        X = R[mask.reshape(-1)] if len(R) == mask.size else R
        with torch.no_grad(): pr = probe(torch.from_numpy(X).to(DEV)).view(len(X), 64, 3).argmax(-1).cpu().numpy()
        y = labels[mask]; fl = flipped[mask]; oc_ = occ[mask]
        err_f = float((pr[fl & oc_] != y[fl & oc_]).mean()); err_p = float((pr[~fl & oc_] != y[~fl & oc_]).mean()); err_all = float((pr != y).mean())
        out["c_probe_error"][str(p)] = {"flipped": err_f, "parity": err_p, "all_tiles": err_all}
        print(f"   pt {p}: error flipped {100*err_f:6.2f}%   parity {100*err_p:6.2f}%   all tiles {100*err_all:5.2f}%")

    # ── (b) editability: flipped-tile cases vs parity-tile cases, all points ─────
    def make_cases(kind, n):
        rng = np.random.default_rng(1 if kind == "flipped" else 2); lc = shipped_length_distribution(); tot = sum(lc.values())
        quota = {L: int(round(n * c / tot)) for L, c in sorted(lc.items())}; cases = []
        for L, want in quota.items():
            pool = [i for i, g in enumerate(games) if len(g) > L]; got = 0
            for gi in rng.permutation(pool):
                if got >= want: break
                h = games[gi][:L]; b, _, _, fl = replay_track(h, rules); pre = sorted(b.get_valid_moves())
                if not pre: continue
                cand = [sq for sq in range(64) if b.state[sq // 8, sq % 8] != 0 and sq not in CENTRE and (fl[-1][sq] == (kind == "flipped"))]
                for sq in rng.permutation(cand):
                    sq = int(sq); post = OthelloBoardState(**rules); post.update(h, prt=False); post.state[sq // 8, sq % 8] *= -1
                    lp = sorted(post.get_valid_moves())
                    if lp and lp != pre:
                        cases.append({"history": h, "pos_int": sq, "ori_color": 0.0 if b.state[sq // 8, sq % 8] < 0 else 2.0}); got += 1; break
        return cases
    lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(NP)}
    out["b_edits"] = {}
    for kind in ("flipped", "parity"):
        cases = make_cases(kind, a.n_cases); bench = benchmark_from_cases(cases, **rules); cur, tgt = case_targets(bench)
        uns = oa.unsteered_probs(model, bench); u = oa.unsteered(model, bench)
        from pim.metrics.set_editability import move_fidelity_ratio
        res = {"n": len(cases), "unedited": u["edit_index_union"], "arms": []}
        for ed, mode, grid_a in (("PI", "pinv", (0.5, 1, 2, 3, 5, 8)), ("ND", "add_sub", (0.2, 0.35, 0.7, 1, 2))):
            for p in range(NP):
                for al in grid_a:
                    pr, card = oa.linear_arm(model, bench, lin, tgt, cur, mode=mode, alpha=float(al), points={p})
                    res["arms"].append({"editor": ed, "point": p, "alpha": al, "ei": card["edit_index_union"], "fid": move_fidelity_ratio(pr, uns, bench.legal_post)})
        for ed in ("PI", "ND"):
            arms = [r for r in res["arms"] if r["editor"] == ed]; b = max(arms, key=lambda r: r["ei"]); g = [r for r in arms if r["fid"] <= 1.1]; bg = max(g, key=lambda r: r["ei"]) if g else None
            res[f"best_{ed}"] = b; res[f"guarded_{ed}"] = bg
            print(f"(b) {kind:8s} n={len(cases)} unedited {u['edit_index_union']:+.3f} | {ed} best {b['ei']:+.3f}/fid {b['fid']:.2f} (pt{b['point']} α{b['alpha']:g})" + (f" | guarded {bg['ei']:+.3f}/{bg['fid']:.2f} (pt{bg['point']} α{bg['alpha']:g})" if bg else ""), flush=True)
        # best-by-point (guarded) profile
        prof = {}
        for p in range(NP):
            for ed in ("PI", "ND"):
                g = [r for r in res["arms"] if r["editor"] == ed and r["point"] == p and r["fid"] <= 1.1]
                prof[f"{ed}{p}"] = max((r["ei"] for r in g), default=float("nan"))
        print("    guarded best by point  PI: " + " ".join(f"{prof[f'PI{p}']:+.2f}" for p in range(NP)) + " | ND: " + " ".join(f"{prof[f'ND{p}']:+.2f}" for p in range(NP)))
        res["guarded_by_point"] = prof; out["b_edits"][kind] = res
    out["minutes"] = round((time.time() - t0) / 60, 1)
    (EXP / "scores" / f"flipped_tiles_{Path(a.run).name}.json").write_text(json.dumps(out, indent=1, default=float)); print("done", out["minutes"], "min")


if __name__ == "__main__":
    main()
