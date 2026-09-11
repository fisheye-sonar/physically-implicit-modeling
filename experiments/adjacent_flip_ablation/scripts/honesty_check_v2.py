"""Honesty check v2 (2026-09-11): the legal-mass >= 0.99 filter is TOOTHLESS on the adjacency
instances (their models put mass 1.000 on everything), so swap-built counterfactual histories
slipped through and gave a false "low ceiling". A perfect predictor scores +1.000 here. Filter
instead on ORDINARINESS: the model's rmse to uniform-over-its-own-legal-set on the counterfactual
history must lie within the held-out distribution for prefixes of the same length (<= 95th pct).
Classifies each counterfactual as a single-move SUBSTITUTION or a SWAP; reports the ceiling, the
canonical best editors, and the probe alignment on the surviving cases. Self-contained.
"""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/edit_direction_alignment/scripts"))
from common import frac_in, haufe_patterns, orth, zspace  # noqa: E402
from othello_alignment import replay, search_cf  # noqa: E402
from pim.environments.othello import arms as oa, corpus as oc  # noqa: E402
from pim.environments.othello import case_targets  # noqa: E402
from pim.environments.othello.bench import benchmark_from_cases, cases_path  # noqa: E402
from pim.environments.othello.data import N_CLASSES, board_probs, canonical_vocab, tokens_and_labels  # noqa: E402
from pim.metrics.set_editability import edit_index_legal, move_rmse_per_case  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402

DEV = "cuda"; EXP = REPO / "experiments/adjacent_flip_ablation"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=900); ap.add_argument("--pct", type=float, default=95)
    a = ap.parse_args()
    run_dir = REPO / a.run; inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]; rules = oc.rules_of(inst)
    S = json.loads((run_dir / "scores.json").read_text()); best = S["best"]
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV); NP = n_points(model)
    stoi = canonical_vocab(); itos = {v: k for k, v in stoi.items()}

    def pred(hists):
        out = np.zeros((len(hists), 64), np.float32); by = {}
        for i, h in enumerate(hists): by.setdefault(len(h), []).append(i)
        with torch.no_grad():
            for L, ids in by.items():
                idx = torch.from_numpy(np.array([[stoi[x] for x in hists[i]] for i in ids])).to(DEV)
                out[ids] = board_probs(model.decode(idx), getattr(model, "output_kind", "logits"))
        return out

    own_legal = lambda hs: [sorted(replay(h, rules).get_valid_moves()) for h in hs]
    # held-out ordinariness by prefix length
    tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("test",), instance=inst)["test"])
    ho = {}
    r = 0
    for L in range(5, 31):
        hs = []
        while len(hs) < 60 and r < len(tok):
            h = [int(itos[int(t)]) for t in tok[r][: int(ln[r])]]; r += 1
            if len(h) > L: hs.append(h[:L])
        ho[L] = move_rmse_per_case(pred(hs), own_legal(hs))
    thr = {L: float(np.percentile(v, a.pct)) for L, v in ho.items()}
    ho_all = np.concatenate(list(ho.values()))
    print(f"{a.run}: held-out rmse-to-own-uniform mean {ho_all.mean():.4f}, p95 {np.percentile(ho_all, 95):.4f}", flush=True)

    cases = pickle.load(open(cases_path(inst), "rb")); rng = np.random.default_rng(0); raw = []
    for i in rng.permutation(len(cases))[: a.n]:
        c = cases[i]; h = [int(x) for x in c["history"]]; s = int(c["pos_int"]); bst, d, _ = search_cf(h, s, rules)
        if bst is not None and d == 0:
            hh = bst[0]; ndiff = sum(x != y for x, y in zip(h, hh))
            raw.append({"h": h, "hh": hh, "s": s, "ori": c["ori_color"], "kind": "sub" if ndiff == 1 else "swap"})
    p_cf = pred([f["hh"] for f in raw]); rm = move_rmse_per_case(p_cf, own_legal([f["hh"] for f in raw]))
    mass = np.array([p_cf[i, own_legal([f["hh"]])[0]].sum() for i, f in enumerate(raw)])
    for f, x, m in zip(raw, rm, mass): f["rmse"] = float(x); f["mass"] = float(m); f["ordinary"] = bool(x <= thr[len(f["hh"])])
    kinds = np.array([f["kind"] for f in raw]); ordn = np.array([f["ordinary"] for f in raw]); massok = mass >= 0.99
    print(f"  {len(raw)} exact boards: {int((kinds=='sub').sum())} substitutions, {int((kinds=='swap').sum())} swaps | pass legal-mass filter {int(massok.sum())} | pass ordinariness filter {int(ordn.sum())} "
          f"(subs {int((ordn & (kinds=='sub')).sum())}, swaps {int((ordn & (kinds=='swap')).sum())}) | mean rmse subs {rm[kinds=='sub'].mean() if (kinds=='sub').any() else float('nan'):.4f} swaps {rm[kinds=='swap'].mean() if (kinds=='swap').any() else float('nan'):.4f}", flush=True)

    out = {"run": a.run, "instance": inst, "heldout_rmse_mean": float(ho_all.mean()), "heldout_rmse_p95": float(np.percentile(ho_all, 95)),
           "n_exact": len(raw), "n_sub": int((kinds == "sub").sum()), "n_swap": int((kinds == "swap").sum()), "n_pass_mass": int(massok.sum()), "n_pass_ordinary": int(ordn.sum()),
           "subsets": {}}
    data = None
    for name, sel in (("mass_filter (old)", massok), ("ordinary (new)", ordn), ("ordinary & substitution", ordn & (kinds == "sub"))):
        idx = np.where(sel)[0]
        if len(idx) == 0: out["subsets"][name] = {"n": 0}; print(f"  [{name}] n=0"); continue
        found = [raw[i] for i in idx]; pc = p_cf[idx]
        bench = benchmark_from_cases([{"history": f["h"], "pos_int": f["s"], "ori_color": f["ori"]} for f in found], **rules)
        pre, post = bench.legal_pre, bench.legal_post
        ei_cf = float(np.nanmean(edit_index_legal(pc, pre, post, "union"))); ei_un = float(np.nanmean(edit_index_legal(pred([f["h"] for f in found]), pre, post, "union")))
        res = {"n": len(idx), "ceiling": ei_cf, "unedited": ei_un, "cf_rmse_mean": float(rm[idx].mean()), "editors": {}}
        cur, tgt = case_targets(bench)
        if data is None:
            ptok, pln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",), instance=inst)["probe"])
            data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(ptok[:20000], pln[:20000])], **rules)
            grid = oa.fit_probe_grid(model, data, cache_dir=run_dir / "probes", log=None)
            lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(NP)}
        line = f"  [{name}] n={len(idx)} ceiling {ei_cf:+.3f} unedited {ei_un:+.3f}"
        for ed, mode in (("PI", "pinv"), ("ND", "add_sub")):
            b = best[ed]; pr, card = oa.linear_arm(model, bench, lin, tgt, cur, mode=mode, alpha=float(b["alpha"]), points={int(b["point"])})
            res["editors"][ed] = {"point": b["point"], "alpha": b["alpha"], "edit_index_union": card["edit_index_union"]}
            line += f" | {ed} {card['edit_index_union']:+.3f}"
        # alignment on this subset, best point by rows fraction
        def resid(hists):
            outs = {p: [None] * len(hists) for p in range(NP)}; by = {}
            for i, h in enumerate(hists): by.setdefault(len(h), []).append(i)
            with torch.no_grad():
                for L, ids in by.items():
                    ix = torch.from_numpy(np.array([[stoi[x] for x in hists[i]] for i in ids])).to(DEV); rr = model.residual_stack(ix)[:, :, -1]
                    for p in range(NP):
                        for j, i in enumerate(ids): outs[p][i] = rr[p, j]
            return {p: torch.stack(v) for p, v in outs.items()}
        R = resid([f["h"] for f in found]); Rcf = resid([f["hh"] for f in found])
        with torch.no_grad():
            rs = model.residual_stack(torch.from_numpy(data.tokens[19000:19400]).to(DEV)); mk = torch.from_numpy(data.mask[19000:19400]).to(DEV)
        al = {}
        for p in range(NP):
            probe = lin[p]; W = probe.net.weight.detach(); cov = torch.cov(zspace(probe, rs[p][mk]).T); P = haufe_patterns(W, cov)
            dz = zspace(probe, Rcf[p]) - zspace(probe, R[p])
            rows = [torch.tensor([f["s"] * N_CLASSES + c for c in range(N_CLASSES)], device=DEV) for f in found]
            fr = float(torch.stack([frac_in(dz[i:i+1], orth(W[rows[i]]))[0] for i in range(len(found))]).mean())
            fh = float(torch.stack([frac_in(dz[i:i+1], orth(P[rows[i]]))[0] for i in range(len(found))]).mean())
            al[str(p)] = {"rows": fr, "haufe": fh}
        bp = max(al, key=lambda p: al[p]["rows"]); res["alignment_by_point"] = al; res["alignment_best"] = {"point": int(bp), **al[bp]}
        line += f" | alignment best pt{bp}: rows {al[bp]['rows']:.3f} haufe {al[bp]['haufe']:.3f}"
        print(line, flush=True); out["subsets"][name] = res
    (EXP / "scores" / f"honesty_check_v2_{Path(a.run).name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
