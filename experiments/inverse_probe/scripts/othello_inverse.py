"""INVERSE PROBE on Othello (2026-09-14, Sevan): learn g: state → residual at point ℓ from the
probe corpus, then edit by writing g(target state) into the residual — can a map from state
alone supply the write that the counterfactual history supplies?

Three write forms, every residual point, the canonical bench and scorecards:
    overwrite   h' = g(s_post)                        (the conditional mean of h given the state)
    delta       h' = h + α · (g(s_post) − g(s_pre))   (keeps what h carries beyond the state)
    nn          the same two with g replaced by the mean residual of the k training rows
                nearest to the state (Hamming over the 64 tiles) — retrieval, no training
Controls: overwrite with the state-free mean residual; the canonical PI / ND rows for scale;
and (2026-09-14 evening, Sevan) the RECONSTRUCTION control — overwrite with g(s_pre), the
conditional mean of the UNEDITED, reachable state: if that alone moves the output, the output
depends on what the state does not determine, independent of any off-manifold extrapolation.
Also reported: g's held-out R² per point (how much of the residual the board explains) and
whether the canonical linear probe reads the written residual as s_post ("landed").

Case sets (--cases): "canonical" = the instance's 1000 single-tile flips; "last-tile" = the
same 1000 histories with the disc JUST PLACED recoloured (Sevan's test: the only tile whose
square and colour both enter through the position the write touches), filtered to a legality
change; canonical PI / ND are re-run on that set with the run's own α grids for scale.

    python experiments/inverse_probe/scripts/othello_inverse.py --run initial_othello_comparison/L-oth-20m
    python experiments/inverse_probe/scripts/othello_inverse.py --run ... --smoke

Contained: reads the run's cached probes and the probe corpus, writes
experiments/inverse_probe/scores/<run>.json; nothing canonical changes.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments.othello import arms as oa, corpus as oc  # noqa: E402
from pim.environments.othello import case_targets, load_benchmark  # noqa: E402
from pim.environments.othello.bench import benchmark_from_cases, cases_path  # noqa: E402
from pim.environments.othello.vendor.othello import OthelloBoardState  # noqa: E402
from pim.environments.othello.data import (  # noqa: E402
    N_CLASSES, N_TILES, board_probs, canonical_vocab, flatten_rows, harvest_point, tokens_and_labels)
from pim.metrics.set_editability import move_fidelity_ratio, move_scorecard  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.base import FIT_BATCH, FIT_LR, fit_probe  # noqa: E402

DEV = "cuda"
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="initial_othello_comparison/L-oth-20m")
ap.add_argument("--hidden", type=int, default=1024)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--k", type=int, default=10, help="neighbours for the retrieval form")
ap.add_argument("--alphas", type=float, nargs="+", default=(0.25, 0.5, 1.0, 1.5, 2.0, 3.0))
ap.add_argument("--points", type=int, nargs="*", default=None)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--tag", default="", help="suffix for the scores file (e.g. mirror128)")
ap.add_argument("--cases", choices=("canonical", "last-tile"), default="canonical")
ap.add_argument("--recon-only", action="store_true", help="only the reconstruction control arms (g(s_pre), nn(s_pre))")
a = ap.parse_args()
t0 = time.time()
run = REPO / "runs" / a.run
S = json.loads((run / "scores.json").read_text())["settings"]
inst = json.loads((run / "config.json").read_text())["data"]["instance"]
rules = oc.rules_of(inst)
model, _ = load_checkpoint(run / "best_model.pt", device=DEV); model.eval(); NP = n_points(model)
okind = getattr(model, "output_kind", "logits")

# ── the probe corpus: states (mine/theirs, 64 tiles) and, per point, residuals ─────────
tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",), instance=inst)["probe"])
itos = {v: k for k, v in canonical_vocab().items()}
n_games = 300 if a.smoke else S["oth_probe_games"]
data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(tok[:n_games], ln[:n_games])], **rules)
seq_of_row, states = flatten_rows(data, "mine")                       # (rows,), (rows, 64) in {0,1,2}
rng = np.random.default_rng(0)
perm = rng.permutation(n_games); tr_games = set(perm[: int(0.8 * n_games)].tolist())
tr = np.array([s in tr_games for s in seq_of_row]); te = ~tr
onehot = lambda st: np.eye(N_CLASSES, dtype=np.float32)[st].reshape(len(st), -1)   # (n, 192)  # noqa: E731
X_all = onehot(states)
S_all_t = torch.from_numpy(states.astype(np.int8)).to(DEV)                          # for the NN search
print(f"{a.run}: {n_games} games → {len(states):,} rows ({tr.sum():,} train / {te.sum():,} held out); "
      f"{NP} points; hidden {a.hidden}, epochs {a.epochs}, k {a.k}", flush=True)

# ── the bench: per case the pre-edit board (mover's frame) and the post-edit board ───────
def last_tile_cases() -> list[dict]:
    """The canonical histories with the LAST-PLACED disc recoloured, kept iff the legal set changes
    (the same rule engine and rejection rule as `synthesise_cases`)."""
    import pickle
    base = pickle.load(open(cases_path(inst), "rb"))
    out_ = []
    for c in base:
        h = [int(t) for t in c["history"]]; sq = h[-1]
        b = OthelloBoardState(**rules); b.update(h, prt=False); pre = sorted(b.get_valid_moves())
        ori = 0.0 if b.state[sq // 8, sq % 8] < 0 else 2.0
        pb = OthelloBoardState(**rules); pb.update(h, prt=False); pb.state[sq // 8, sq % 8] = int(2 - ori) - 1
        post = sorted(pb.get_valid_moves())
        if post and post != pre:
            out_.append({"history": h, "pos_int": sq, "ori_color": ori, "game": c.get("game")})
    return out_

if a.cases == "last-tile":
    cases_lt = last_tile_cases()
    bench = benchmark_from_cases(cases_lt, **rules)
    print(f"last-tile cases: {len(cases_lt)} of 1000 histories change legality when the just-placed disc is recoloured", flush=True)
else:
    bench = load_benchmark(inst)
n_cases = bench.n_cases
if a.smoke:                                     # a few cases from each bucket
    keep = np.concatenate([ids[:8] for ids in bench.case_ids])
else:
    keep = np.arange(n_cases)
cur_lab, tgt_lab = case_targets(bench)
hist = [None] * n_cases
for toks, ids in zip(bench.tokens, bench.case_ids):
    for row, i in zip(toks, ids):
        hist[i] = [itos[int(t)] for t in row]
bd = tokens_and_labels([hist[i] for i in range(n_cases)], **rules)
s_pre = np.stack([bd.mine[i, len(hist[i]) - 1] for i in range(n_cases)])          # (N, 64)
s_post = s_pre.copy(); s_post[np.arange(n_cases), bench.pos_int] = tgt_lab
assert (s_pre[np.arange(n_cases), bench.pos_int] == cur_lab).all(), "pre-edit board disagrees with the bench's current label"
Xpre_t, Xpost_t = (torch.from_numpy(onehot(s)).to(DEV) for s in (s_pre, s_post))
uns = oa.unsteered_probs(model, bench)
u = move_scorecard(uns, bench.legal_pre, bench.legal_post)
canon = json.loads((run / "scores.json").read_text())["best"]
print(f"bench: {n_cases} cases ({len(keep)} scored) · unedited symdiff {u['edit_index_symdiff']:+.3f} · "
      f"canonical PI {canon['PI']['edit_index_symdiff']:+.3f}/{canon['PI']['fidelity_ratio']:.2f}  "
      f"ND {canon['ND']['edit_index_symdiff']:+.3f}/{canon['ND']['fidelity_ratio']:.2f}", flush=True)
grid = oa.fit_probe_grid(model, data if not a.smoke else data, cache_dir=run / "probes", log=None) if not a.smoke else None
canon_on_cases = None
if a.cases == "last-tile" and grid is not None and not a.recon_only:
    # the canonical linear editors on THIS case set, the run's own α grids, every point (GS omitted)
    lin_mine = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(NP)}
    canon_on_cases = {}
    for mode, label, al in (("add_sub", "ND", S["oth_alpha_nd"]), ("pinv", "PI", S["oth_alpha_pi"])):
        rows = []
        for ell in range(NP):
            for alpha in al:
                pr, card = oa.linear_arm(model, bench, lin_mine, tgt_lab, cur_lab, mode=mode, alpha=alpha, points={ell})
                rows.append({"point": ell, "alpha": alpha, "edit_index": card["edit_index_symdiff"],
                             "edit_index_union": card["edit_index_union"],
                             "fidelity_ratio": move_fidelity_ratio(pr, uns, bench.legal_post)})
        best = max(rows, key=lambda r: r["edit_index_union"])                     # the canonical selection rule
        guarded = [r for r in rows if r["fidelity_ratio"] < 1]
        canon_on_cases[label] = {"best": best, "best_guarded": max(guarded, key=lambda r: r["edit_index"]) if guarded else None, "arms": rows}
        print(f"canonical {label} on last-tile cases: best {best['edit_index']:+.3f}/{best['fidelity_ratio']:.2f} (pt {best['point']}, α {best['alpha']})"
              + (f"; best guarded {canon_on_cases[label]['best_guarded']['edit_index']:+.3f}/{canon_on_cases[label]['best_guarded']['fidelity_ratio']:.2f}" if guarded else "; nothing under the guard")
              + f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)

def nn_mean(S_train_t: torch.Tensor, H_train_t: torch.Tensor, s_query: np.ndarray, k: int) -> torch.Tensor:
    """(n, d): mean residual of the k training rows nearest each query board (Hamming).
    Hamming distance over 64 three-way tiles = 64 − (one-hot agreement), computed as a
    matmul against a (rows, 192) half-precision one-hot matrix — never a (b, rows, 64)
    broadcast (a 4 GB intermediate; its OOM path trips PyTorch's NVML assert under the
    2026-09-11 driver mismatch)."""
    S1 = torch.nn.functional.one_hot(S_train_t.long(), N_CLASSES).reshape(len(S_train_t), -1).half()   # (rows, 192)
    q = torch.from_numpy(onehot(s_query)).to(DEV).half()                                              # (n, 192)
    out = torch.zeros(len(q), H_train_t.shape[1], device=DEV)
    for i in range(0, len(q), 256):
        agree = q[i:i + 256] @ S1.T                                                # (b, rows) matches out of 64
        idx = agree.topk(k, dim=1, largest=True).indices                            # nearest = most agreement
        out[i:i + 256] = H_train_t[idx].mean(1)
    del S1
    return out

def run_arm(ell: int, make_new):
    """Score one write form at point ell: make_new(cur (B,d), case_ids) -> new (B,d)."""
    probs = np.zeros((n_cases, N_TILES), np.float32); ratios = []
    landed = []
    lin = grid.probes[("mine", "linear", "sequence", ell)] if grid is not None else None
    for toks, ids in zip(bench.tokens, bench.case_ids):
        ids_k = np.array([i for i in ids if i in set(keep.tolist())]) if a.smoke else ids
        if len(ids_k) == 0:
            continue
        idx = torch.from_numpy(toks[[list(ids).index(i) for i in ids_k]]).to(DEV)
        def hook(layer, x, _ids=ids_k):
            if layer != ell:
                return x
            cur = x[:, -1]
            new = make_new(cur, _ids)
            ratios.append(float(((new - cur).norm(dim=1) / cur.norm(dim=1)).mean()))
            if lin is not None:
                lab = lin(new).argmax(-1).cpu().numpy()                           # (B, 64)
                sq = bench.pos_int[_ids]
                landed.append((lab[np.arange(len(_ids)), sq] == tgt_lab[_ids]).mean())
                landed.append(-(lab == s_post[_ids]).mean())                     # board agreement, negative-tagged
            out = x.clone(); out[:, -1] = new
            return out
        probs[ids_k] = board_probs(model.decode(idx, edit=hook), okind)
    sel = keep
    card = move_scorecard(probs[sel], [bench.legal_pre[i] for i in sel], [bench.legal_post[i] for i in sel])
    rec = {"edit_index": card["edit_index_symdiff"], "edit_index_union": card["edit_index_union"],
           "fidelity_ratio": move_fidelity_ratio(probs[sel], uns[sel], [bench.legal_post[i] for i in sel]),
           # for a NULL write (the reconstruction control): distance to the PRE-edit truth relative
           # to the unsteered model's (1.0 = output preserved), and the raw drift from the unsteered output
           "pre_fidelity": move_fidelity_ratio(probs[sel], uns[sel], [bench.legal_pre[i] for i in sel]),
           "drift_rmse": float(np.sqrt(((probs[sel] - uns[sel]) ** 2).mean())),
           "write_ratio": float(np.mean(ratios)) if ratios else None}
    if landed:
        L = np.array(landed); rec["landed_tile"] = float(L[L >= 0].mean()); rec["board_agreement"] = float(-L[L < 0].mean())
    return rec

out = {"run": a.run, "n_games": n_games, "rows": int(len(states)), "hidden": a.hidden, "epochs": a.epochs, "k": a.k,
       "unedited": {k: v for k, v in u.items() if isinstance(v, (int, float))},
       "canonical": {e: {"edit_index": canon[e]["edit_index_symdiff"], "fidelity_ratio": canon[e]["fidelity_ratio"]} for e in ("PI", "ND", "GS")},
       "cases": a.cases, "n_cases": int(n_cases), "recon_only": a.recon_only,
       "canonical_on_cases": canon_on_cases,
       "points": {}}
points = a.points if a.points is not None else ([1] if a.smoke else list(range(NP)))
for ell in points:
    acts = harvest_point(model, data.tokens, ell)                          # (N, 59, d)
    H = acts[data.mask]; del acts
    g, st = fit_probe(X_all[tr], H[tr], X_all[te], H[te], hidden=a.hidden, epochs=a.epochs,
                      batch=FIT_BATCH, lr=FIT_LR, device=DEV, seed=0, n_classes=None)
    g.eval()
    for p_ in g.parameters():
        p_.requires_grad_(False)
    H_tr_t = torch.from_numpy(H[tr]).to(DEV); S_tr_t = S_all_t[torch.from_numpy(np.where(tr)[0]).to(DEV)]
    h_mean = H_tr_t.mean(0)
    with torch.no_grad():
        g_pre, g_post = g(Xpre_t), g(Xpost_t)                                  # (N, d) each
        nn_pre, nn_post = nn_mean(S_tr_t, H_tr_t, s_pre, a.k), nn_mean(S_tr_t, H_tr_t, s_post, a.k)
    arms = {}
    arms["recon_overwrite"] = run_arm(ell, lambda cur, ids: g_pre[torch.as_tensor(ids, device=DEV)])       # the control: g of the UNEDITED state
    arms["nn_recon_overwrite"] = run_arm(ell, lambda cur, ids: nn_pre[torch.as_tensor(ids, device=DEV)])
    if not a.recon_only:
        arms["overwrite"] = run_arm(ell, lambda cur, ids: g_post[torch.as_tensor(ids, device=DEV)])
        arms["mean_overwrite"] = run_arm(ell, lambda cur, ids: h_mean[None].expand_as(cur))
        arms["nn_overwrite"] = run_arm(ell, lambda cur, ids: nn_post[torch.as_tensor(ids, device=DEV)])
    for al in ([] if a.recon_only else a.alphas):
        arms[f"delta@{al:g}"] = run_arm(ell, lambda cur, ids, al=al: cur + al * (g_post - g_pre)[torch.as_tensor(ids, device=DEV)])
        arms[f"nn_delta@{al:g}"] = run_arm(ell, lambda cur, ids, al=al: cur + al * (nn_post - nn_pre)[torch.as_tensor(ids, device=DEV)])
    out["points"][str(ell)] = {"g_r2_heldout": float(st["r2"]), "g_r2_insample": float(st.get("r2_insample", np.nan)),
                               "g_rmse": float(st["rmse"]), "arms": arms}
    f = lambda r: f"{r['edit_index']:+.3f}/{r['fidelity_ratio']:.2f}" + (f" landed {r['landed_tile']:.2f}" if "landed_tile" in r else "")  # noqa: E731
    rc = arms["recon_overwrite"]
    line = f"pt {ell}: g R² {st['r2']:+.3f} | RECON g(s_pre) EI {rc['edit_index']:+.3f} pre-fid {rc['pre_fidelity']:.2f} drift {rc['drift_rmse']:.4f}"
    if not a.recon_only:
        best_d = max((k for k in arms if k.startswith("delta@")), key=lambda k: arms[k]["edit_index"])
        best_n = max((k for k in arms if k.startswith("nn_delta@")), key=lambda k: arms[k]["edit_index"])
        line += (f" | overwrite {f(arms['overwrite'])} | mean-h {f(arms['mean_overwrite'])} | "
                 f"nn-overwrite {f(arms['nn_overwrite'])} | {best_d} {f(arms[best_d])} | {best_n} {f(arms[best_n])}")
    print(line + f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    del H, H_tr_t; torch.cuda.empty_cache()
tag = a.run.split("/")[-1] + (f"_{a.tag}" if a.tag else "") + ("_lasttile" if a.cases == "last-tile" else "") + ("_recon" if a.recon_only else "") + ("_smoke" if a.smoke else "")
(REPO / "experiments/inverse_probe/scores" / f"othello_{tag}.json").write_text(json.dumps(out, indent=1, default=float))
print("wrote", f"experiments/inverse_probe/scores/othello_{tag}.json", f"[{(time.time() - t0) / 60:.1f} min]")
