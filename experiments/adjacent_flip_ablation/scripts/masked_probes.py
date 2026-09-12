"""Masked colour probes and a flip-bit probe on L-oth-adjacent-flip-20m (2026-09-11). Plan agreed
with Sevan: the canonical classification fit (per-tile CE, 200 epochs, Adam 1e-3, batch 4096,
held out by sequence, seed 0 — `pim.probes.base.fit_probe` copied line for line) with ONE change,
a per-(row, tile) loss weight:
    all            weight 1 everywhere                      (control: must reproduce the canonical probe)
    flipped+blank  blank tiles + tiles whose colour != placement colour   (mine/theirs learnt on FLIPPED rows only)
    parity+blank   blank tiles + never-recoloured occupied tiles          (mirror control)
plus a 2-class FLIP-BIT probe (flipped / not, on occupied tiles). Each probe is evaluated held out
on flipped, parity and blank tiles, then used for canonical PI / ND edits (`arms.linear_arm`, the
probe dict swapped in) on 300 flipped-tile and 300 parity-tile cases at every residual point; the
flip-bit probe through 2-class PI / ND arms (the presence-edit construction). Probes are persisted
under probes/<run>/masked/. Self-contained. Output scores/masked_probes_<run>_<data>.json.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from flipped_tiles import replay_track  # noqa: E402
from pim.editors.pinv import inject_state  # noqa: E402
from pim.environments.othello import arms as oa, corpus as oc  # noqa: E402
from pim.environments.othello import case_targets  # noqa: E402
from pim.environments.othello.bench import benchmark_from_cases, shipped_length_distribution  # noqa: E402
from pim.environments.othello.data import (BLANK, CENTRE, N_TILES, board_probs, canonical_vocab, harvest_point,  # noqa: E402
                                           tokens_and_labels)
from pim.environments.othello.vendor.othello import OthelloBoardState  # noqa: E402
from pim.metrics.set_editability import move_fidelity_ratio, move_scorecard  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.base import FIT_BATCH, FIT_EPOCHS, FIT_LR, WorldStateProbe  # noqa: E402

DEV = "cuda"; EXP = REPO / "experiments/adjacent_flip_ablation"
PI_ALPHAS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0); ND_ALPHAS = (0.2, 0.35, 0.7, 1.0, 2.0)


def fit_masked(x_tr, y_tr, w_tr, n_classes, seed=0, epochs=FIT_EPOCHS, lr=FIT_LR, batch=FIT_BATCH):
    """`fit_probe`'s classification path with a per-(row, tile) loss weight `w_tr` (N, d_out)."""
    torch.manual_seed(seed)
    xm, xs = x_tr.mean(0), x_tr.std(0); xs = np.maximum(xs, 1e-2 * np.median(xs)) + 1e-8
    probe = WorldStateProbe(x_tr.shape[1], y_tr.shape[1], None, x_mean=torch.tensor(xm, dtype=torch.float32), x_std=torch.tensor(xs, dtype=torch.float32),
                            y_mean=torch.zeros(y_tr.shape[1]), y_std=torch.ones(y_tr.shape[1]), n_classes=n_classes).to(DEV)
    xt = torch.tensor(x_tr, dtype=torch.float32, device=DEV); yt = torch.tensor(y_tr, dtype=torch.long, device=DEV); wt = torch.tensor(w_tr, dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(probe.parameters(), lr=lr); n = len(xt)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]; logits = probe(xt[idx])
            l = torch.nn.functional.cross_entropy(logits.reshape(-1, n_classes), yt[idx].reshape(-1), reduction="none").view(len(idx), -1)
            loss = (l * wt[idx]).sum() / wt[idx].sum().clamp_min(1.0)
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval(); return probe


@torch.no_grad()
def predict(probe, x):
    out = []
    for i in range(0, len(x), 8192): out.append(probe(torch.tensor(x[i:i + 8192], dtype=torch.float32, device=DEV)).argmax(-1).cpu().numpy())
    return np.concatenate(out, 0)


def errors(pred, y, F, occ):
    """held-out error on flipped / parity / blank tiles + all"""
    fl, pa, bl = F & occ, (~F) & occ, ~occ
    return {"flipped": float((pred[fl] != y[fl]).mean()), "parity": float((pred[pa] != y[pa]).mean()), "blank": float((pred[bl] != y[bl]).mean()), "all": float((pred != y).mean())}


@torch.no_grad()
def twoclass_arm(model, bench, probe, point, alpha, mode, cur_bit, tgt_bit):
    """PI (swap the tile's two logits, re-solve) / ND (W[tgt]-W[cur]) through a 2-class per-tile probe."""
    W = probe.net.weight.detach(); Wp, bv = torch.linalg.pinv(W), probe.net.bias.detach()
    probs = np.zeros((len(bench.pos_int), N_TILES), np.float32); landed = []
    for toks, ids in zip(bench.tokens, bench.case_ids):
        idx = torch.from_numpy(toks).to(DEV); bsz = len(ids); sq = torch.from_numpy(bench.pos_int[ids]).to(DEV)
        tb = torch.from_numpy(tgt_bit[ids]).to(DEV); cb = torch.from_numpy(cur_bit[ids]).to(DEV); rec = {}

        def hook(layer, x, _rec=rec):
            if layer != point: return x
            cur = x[:, -1]; z = (cur - probe.x_mean) / probe.x_std; ar = torch.arange(bsz, device=DEV)
            if mode == "pinv":
                lg = probe.net(z).view(bsz, N_TILES, 2).clone(); lg[ar, sq] = lg[ar, sq].flip(-1)
                z_new = inject_state(z, lg.view(bsz, -1), W, Wp, bv); dz = alpha * (z_new - z)
            else:
                dvec = W[sq * 2 + tb] - W[sq * 2 + cb]; dz = alpha * z.norm(dim=-1, keepdim=True) * dvec / dvec.norm(dim=-1, keepdim=True)
            out = x.clone(); out[:, -1] = cur + dz * probe.x_std
            lg2 = probe(out[:, -1]).view(bsz, N_TILES, 2); _rec["landed"] = float((lg2[ar, sq].argmax(-1) == tb).float().mean()); return out

        probs[ids] = board_probs(model.decode(idx, edit=hook), getattr(model, "output_kind", "logits")); landed.append(rec.get("landed", float("nan")))
    card = move_scorecard(probs, bench.legal_pre, bench.legal_post); card["readout_landed"] = float(np.nanmean(landed)); return probs, card


def make_cases(games, rules, kind, n):
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
                sq = int(sq); post = OthelloBoardState(**rules); post.update(h, prt=False); post.state[sq // 8, sq % 8] *= -1; lp = sorted(post.get_valid_moves())
                if lp and lp != pre:
                    cases.append({"history": h, "pos_int": sq, "ori_color": 0.0 if b.state[sq // 8, sq % 8] < 0 else 2.0, "flipped": bool(fl[-1][sq])}); got += 1; break
    return cases


def best_of(arms, key="ei"):
    b = max(arms, key=lambda r: r[key]); g = [r for r in arms if r["fid"] <= 1.1]; bg = max(g, key=lambda r: r[key]) if g else None
    return b, bg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="runs/adjacent_flip_ablation/L-oth-adjacent-flip-20m")
    ap.add_argument("--data", choices=("canonical", "large"), default="canonical"); ap.add_argument("--large-games", type=int, default=40000)
    ap.add_argument("--variants", nargs="+", default=["all", "flipped+blank", "parity+blank", "flipbit"]); ap.add_argument("--n-cases", type=int, default=300)
    ap.add_argument("--points", type=int, nargs="+", default=list(range(9))); a = ap.parse_args(); t0 = time.time()
    run_dir = REPO / a.run; inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]; rules = oc.rules_of(inst)
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV); model.eval(); NP = n_points(model)
    itos = {v: k for k, v in canonical_vocab().items()}
    split = "probe_large" if a.data == "large" else "probe"
    ptok, pln = oc.load(oc.build(only=(split,), instance=inst, log=lambda s: None)[split]); ng = a.large_games if a.data == "large" else 20000
    data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(ptok[:ng], pln[:ng])], **rules)
    n_seq, T = data.mask.shape
    F = np.zeros(data.mine.shape, bool)
    for i in range(n_seq):
        h = [int(itos[int(t)]) for t in ptok[i][: int(pln[i])]]; _, _, _, fl = replay_track(h, rules); Ti = min(len(fl), T); F[i, :Ti] = np.stack(fl[:Ti])
    y_all = data.mine[data.mask]; F_all = F[data.mask]; occ_all = y_all != BLANK
    seq_of_row = np.repeat(np.arange(n_seq), T)[data.mask.reshape(-1)]
    tr, te = oa._split(n_seq, seq_of_row, "sequence", 0.2, 0)
    print(f"{a.run} [{a.data}: {n_seq} games, {len(y_all)} rows; flipped share of occupied tile-rows {(F_all & occ_all).sum() / occ_all.sum():.3f}]", flush=True)
    weights = {"all": np.ones(y_all.shape, np.float32), "flipped+blank": ((~occ_all) | F_all).astype(np.float32), "parity+blank": ((~occ_all) | (~F_all & occ_all)).astype(np.float32),
               "flipbit": occ_all.astype(np.float32)}
    store = EXP / "probes" / Path(a.run).name / f"masked_{a.data}"; store.mkdir(parents=True, exist_ok=True)
    out = {"run": a.run, "instance": inst, "data": a.data, "n_games": n_seq, "variants": {}}
    probes = {v: {} for v in a.variants}
    for p in a.points:
        R = harvest_point(model, data.tokens, p); R = R.reshape(-1, R.shape[-1]); X = R[data.mask.reshape(-1)]; del R
        for v in a.variants:
            f = store / f"{v.replace('+', '_')}_pt{p}.pt"
            n_classes = 2 if v == "flipbit" else 3; y = F_all.astype(np.int64) if v == "flipbit" else y_all
            if f.exists():
                blob = torch.load(f, map_location=DEV, weights_only=False); probe = blob["probe"].to(DEV).eval(); st = blob["stats"]
            else:
                probe = fit_masked(X[tr], y[tr], weights[v][tr], n_classes); pred = predict(probe, X[te])
                if v == "flipbit":
                    o = occ_all[te]; yt_ = y[te]; st = {"error_occupied": float((pred[o] != yt_[o]).mean()), "error_on_flipped": float((pred[o & (yt_ == 1)] != 1).mean()), "error_on_parity": float((pred[o & (yt_ == 0)] != 0).mean())}
                else:
                    st = errors(pred, y[te], F_all[te], occ_all[te])
                torch.save({"probe": probe.cpu(), "stats": st, "variant": v, "point": p, "data": a.data, "run": a.run}, f); probe.to(DEV)
            probes[v][p] = probe; out["variants"].setdefault(v, {}).setdefault("error", {})[str(p)] = st
            print(f"  pt {p} {v:14s}: " + (f"occupied err {100*st['error_occupied']:.2f}%  on flipped {100*st['error_on_flipped']:.2f}%  on parity {100*st['error_on_parity']:.2f}%" if v == "flipbit"
                  else f"flipped {100*st['flipped']:6.2f}%  parity {100*st['parity']:5.2f}%  blank {100*st['blank']:5.2f}%  all {100*st['all']:5.2f}%"), flush=True)
        del X
    # ── edits ────────────────────────────────────────────────────────────────────
    tok, ln = oc.load(oc.build(only=("test",), instance=inst, log=lambda s: None)["test"])
    games = [[int(itos[int(t)]) for t in row[:L]] for row, L in zip(tok[:3000], ln[:3000])]
    for kind in ("flipped", "parity"):
        cases = make_cases(games, rules, kind, a.n_cases); bench = benchmark_from_cases(cases, **rules); cur, tgt = case_targets(bench)
        uns = oa.unsteered_probs(model, bench); u = oa.unsteered(model, bench)
        cur_bit = np.array([int(c["flipped"]) for c in cases]); tgt_bit = 1 - cur_bit
        print(f"\n== edits on {kind} tiles (n={len(cases)}, unedited {u['edit_index_union']:+.3f}) ==", flush=True)
        for v in a.variants:
            arms = []
            for p in a.points:
                if v == "flipbit":
                    for ed, mode, grid in (("PI", "pinv", PI_ALPHAS), ("ND", "add_sub", ND_ALPHAS)):
                        for al in grid:
                            pr, card = twoclass_arm(model, bench, probes[v][p], p, al, mode, cur_bit, tgt_bit)
                            arms.append({"editor": ed, "point": p, "alpha": al, "ei": card["edit_index_union"], "fid": move_fidelity_ratio(pr, uns, bench.legal_post), "landed": card["readout_landed"]})
                else:
                    for ed, mode, grid in (("PI", "pinv", PI_ALPHAS), ("ND", "add_sub", ND_ALPHAS)):
                        for al in grid:
                            pr, card = oa.linear_arm(model, bench, probes[v], tgt, cur, mode=mode, alpha=float(al), points={p})
                            arms.append({"editor": ed, "point": p, "alpha": al, "ei": card["edit_index_union"], "fid": move_fidelity_ratio(pr, uns, bench.legal_post)})
            res = {"n": len(cases), "unedited": u["edit_index_union"], "arms": arms}
            for ed in ("PI", "ND"):
                b, bg = best_of([r for r in arms if r["editor"] == ed]); res[f"best_{ed}"] = b; res[f"guarded_{ed}"] = bg
                prof = " ".join(f"{max((r['ei'] for r in arms if r['editor'] == ed and r['point'] == p and r['fid'] <= 1.1), default=float('nan')):+.2f}" for p in a.points)
                print(f"  {v:14s} {ed}: best {b['ei']:+.3f}/fid {b['fid']:.2f} (pt{b['point']} α{b['alpha']:g})" + (f" | guarded {bg['ei']:+.3f}/{bg['fid']:.2f} (pt{bg['point']} α{bg['alpha']:g})" if bg else " | guarded none") + f" | guarded by pt: {prof}", flush=True)
            out["variants"][v].setdefault("edits", {})[kind] = res
    out["minutes"] = round((time.time() - t0) / 60, 1)
    (EXP / "scores" / f"masked_probes_{Path(a.run).name}_{a.data}.json").write_text(json.dumps(out, indent=1, default=float)); print("done", out["minutes"], "min", flush=True)


if __name__ == "__main__":
    main()
