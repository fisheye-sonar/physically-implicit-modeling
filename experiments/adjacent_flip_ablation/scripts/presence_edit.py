"""PRESENCE edits (occupied <-> empty) on any Othello run, the standard way (2026-09-11): the
artificial post-edit board's uniform-over-legal under the INSTANCE'S OWN RULES vs the pre-edit one,
canonical scorecard + fidelity guard. Rule-aware re-run of `experiments/flip_ablation/scripts/
presence_probe_edit.py`, whose case builder and labeller only knew the flip flag (wrong legal sets
on the adjacency instances). Reuses its dedicated 2-class presence probe fit and its PI arm
unchanged. Cases: half REMOVE an occupied non-centre disc, half ADD a disc on an empty square with
the parity colour (as in the original, for comparability), rejected if the legal set is unchanged
or empty; Li's prefix-length mix; 400 cases from the instance's test split. Self-contained;
probes persist under experiments/adjacent_flip_ablation/probes; output scores/presence_edit_<run>.json.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/flip_ablation/scripts"))
from presence_pilot import PARITY_COLOUR, bench_from  # noqa: E402
from presence_probe_edit import fit_presence_probe, presence_pinv_arm  # noqa: E402
from pim.environments.othello import arms as oa, corpus as oc  # noqa: E402
from pim.environments.othello.bench import shipped_length_distribution  # noqa: E402
from pim.environments.othello.data import BLANK, CENTRE, MINE, THEIRS, canonical_vocab, tokens_and_labels  # noqa: E402
from pim.environments.othello.vendor.othello import OthelloBoardState  # noqa: E402
from pim.metrics.set_editability import move_fidelity_ratio  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402
import torch  # noqa: E402
from pim.environments.othello.data import N_TILES, board_probs  # noqa: E402
from pim.metrics.set_editability import move_scorecard  # noqa: E402

EXP = REPO / "experiments/adjacent_flip_ablation"; DEV = "cuda"


def presence_cases(hist, n, length_counts, seed, rules):
    """`presence_pilot.presence_cases` with the instance's full rules (flip AND placement)."""
    rng = np.random.default_rng(seed); tot = sum(length_counts.values())
    quota = {L: int(round(n * c / tot)) for L, c in sorted(length_counts.items())}
    cases, rej = [], {"same": 0, "empty": 0}
    for L, want in quota.items():
        pool = np.array([i for i, h in enumerate(hist) if len(h) > L]); got = 0
        for g in rng.permutation(pool):
            if got >= want: break
            h = list(hist[g][:L]); b = OthelloBoardState(**rules); b.update(h, prt=False); pre = sorted(b.get_valid_moves())
            if not pre: continue
            op = "remove" if got % 2 == 0 else "add"
            cand = [sq for sq in range(64) if (b.state[sq // 8, sq % 8] != 0 and sq not in CENTRE)] if op == "remove" else [sq for sq in range(64) if b.state[sq // 8, sq % 8] == 0]
            for sq in rng.permutation(cand):
                sq = int(sq); post = OthelloBoardState(**rules); post.update(h, prt=False)
                post.state[sq // 8, sq % 8] = 0 if op == "remove" else int(PARITY_COLOUR[sq // 8, sq % 8])
                lp = sorted(post.get_valid_moves())
                if not lp: rej["empty"] += 1; continue
                if lp == pre: rej["same"] += 1; continue
                nxt = b.next_hand_color
                colour = b.state[sq // 8, sq % 8] if op == "remove" else PARITY_COLOUR[sq // 8, sq % 8]
                occ_lab = MINE if colour == nxt else THEIRS
                cur, tgt = (occ_lab, BLANK) if op == "remove" else (BLANK, occ_lab)
                cases.append({"history": h, "pos_int": sq, "op": op, "cur": cur, "tgt": tgt, "legal_pre": pre, "legal_post": lp, "game": int(g)})
                got += 1; break
    return cases, rej


@torch.no_grad()
def presence_nd_arm(model, bench, probe, point, alpha):
    """ND through the 2-class presence probe: direction = W[tile, target] - W[tile, current], scaled to
    alpha * |z| (the canonical `add_sub` construction with 2 classes instead of 3)."""
    W = probe.net.weight.detach()
    probs = np.zeros((len(bench.pos_int), N_TILES), np.float32); landed = []
    for toks, ids in zip(bench.tokens, bench.case_ids):
        idx = torch.from_numpy(toks).to(DEV); bsz = len(ids)
        sq = torch.from_numpy(bench.pos_int[ids]).to(DEV)
        want_occ = torch.from_numpy((bench.tgt_lab[ids] != BLANK).astype(np.int64)).to(DEV)
        cur_occ = 1 - want_occ
        rec = {}

        def hook(layer, x, _rec=rec):
            if layer != point:
                return x
            cur = x[:, -1]; z = (cur - probe.x_mean) / probe.x_std
            dvec = W[sq * 2 + want_occ] - W[sq * 2 + cur_occ]
            dz = alpha * z.norm(dim=-1, keepdim=True) * dvec / dvec.norm(dim=-1, keepdim=True)
            out = x.clone(); out[:, -1] = cur + dz * probe.x_std
            lg2 = probe(out[:, -1]).view(bsz, N_TILES, 2)
            _rec["landed"] = float((lg2[torch.arange(bsz), sq].argmax(-1) == want_occ).float().mean())
            return out

        probs[ids] = board_probs(model.decode(idx, edit=hook), getattr(model, "output_kind", "logits"))
        landed.append(rec.get("landed", float("nan")))
    card = move_scorecard(probs, bench.legal_pre, bench.legal_post); card["readout_landed"] = float(np.nanmean(landed))
    return probs, card


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    ap.add_argument("--points", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6]); ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 5.0])
    ap.add_argument("--nd-alphas", type=float, nargs="+", default=[0.05, 0.2, 0.35, 0.7, 1.0, 2.0])
    ap.add_argument("--n", type=int, default=400); args = ap.parse_args(); t0 = time.time()
    run_dir = REPO / args.run; inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]; rules = oc.rules_of(inst)
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV); model.eval(); label = Path(args.run).name
    print(f"=== {label} on {inst} rules {rules} ===", flush=True)
    itos = {v: k for k, v in canonical_vocab().items()}
    ptok, pln = oc.load(oc.build(only=("probe",), instance=inst, log=lambda s: None)["probe"])
    data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(ptok[:20_000], pln[:20_000])], **rules)
    store = ProbeCache(EXP / "probes" / label)
    tok, ln = oc.load(oc.build(only=("test",), instance=inst, log=lambda s: None)["test"])
    hist = [[int(itos[int(t)]) for t in row[:L]] for row, L in zip(tok, ln)]
    cases, rej = presence_cases(hist, args.n, shipped_length_distribution(), 0, rules); bench = bench_from(cases)
    is_rm = np.array([c["op"] == "remove" for c in cases])
    uns_probs = oa.unsteered_probs(model, bench); u = oa.unsteered(model, bench)
    sym = np.mean([len(set(c["legal_pre"]) ^ set(c["legal_post"])) for c in cases])
    print(f"{len(cases)} cases ({int(is_rm.sum())} remove / {int((~is_rm).sum())} add; rejected same {rej['same']} empty {rej['empty']}); |legal_pre| {np.mean([len(c['legal_pre']) for c in cases]):.1f} symdiff {sym:.2f}; unedited EI {u['edit_index_union']:+.3f}", flush=True)
    out = {"run": args.run, "instance": inst, "rules": rules, "n_cases": len(cases), "n_remove": int(is_rm.sum()), "symdiff_mean": float(sym),
           "unedited": {k: v for k, v in u.items() if isinstance(v, (int, float))}, "probes": {}, "arms": []}
    print(f"{'pt':>3} {'alpha':>5} | {'EI':>7} {'fid':>5} {'legal':>6} {'landed':>6} | {'remove':>7} {'add':>7}")
    for p in args.points:
        probe, st = fit_presence_probe(model, data, p, store, log=lambda s: print(s, flush=True))
        out["probes"][p] = {k: v for k, v in st.items() if isinstance(v, (int, float))}
        for ed, fn, grid in (("PI", presence_pinv_arm, args.alphas), ("ND", presence_nd_arm, args.nd_alphas)):
          for a in grid:
            pr, card = fn(model, bench, probe, p, a); fid = move_fidelity_ratio(pr, uns_probs, bench.legal_post)
            ei = np.array(card["edit_index_union_per_case"], float)
            rec = {"editor": ed, "point": p, "alpha": a, "edit_index_union": card["edit_index_union"], "fidelity_ratio": fid, "li_error_vs_post": card["li_error_vs_post"],
                   "li_error_vs_pre": card["li_error_vs_pre"], "legal_mass": card["legal_mass"], "readout_landed": card["readout_landed"],
                   "ei_remove": float(np.nanmean(ei[is_rm])), "ei_add": float(np.nanmean(ei[~is_rm]))}
            out["arms"].append(rec)
            print(f"{ed} {p:>3} {a:>5g} | {rec['edit_index_union']:>+7.3f} {fid:>5.2f} {rec['legal_mass']:>6.3f} {rec['readout_landed']:>6.2f} | {rec['ei_remove']:>+7.3f} {rec['ei_add']:>+7.3f}", flush=True)
    out["best"] = {}; out["best_guarded"] = {}
    for ed in ("PI", "ND"):
        arms = [r for r in out["arms"] if r["editor"] == ed]
        b = max(arms, key=lambda r: r["edit_index_union"]); g = [r for r in arms if r["fidelity_ratio"] <= 1.1]; bg = max(g, key=lambda r: r["edit_index_union"]) if g else None
        out["best"][ed] = b; out["best_guarded"][ed] = bg
        print(f"BEST {label} {ed}: {b['edit_index_union']:+.3f} / fid {b['fidelity_ratio']:.2f} (pt{b['point']} α{b['alpha']:g}; remove {b['ei_remove']:+.3f} add {b['ei_add']:+.3f})" + (f" | guarded {bg['edit_index_union']:+.3f} / {bg['fidelity_ratio']:.2f} (pt{bg['point']} α{bg['alpha']:g})" if bg else ""), flush=True)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    (EXP / "scores" / f"presence_edit_{label}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
