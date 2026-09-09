"""Othello: the true edit direction from an ORACLE counterfactual history, for the standard
game (L-oth-20m) and the adjacency variant (L-oth-adjacent-20m).

For a bench case (history h, tile s), the counterfactual is a real history h' of the same
length, found by SEARCH with the instance's own simulator over single-move substitutions and
move swaps, whose final board has tile s flipped and is otherwise as close as possible to
the edited board (Hamming distance recorded; the mover must match). Δ = h'_residual −
h_residual at the last position. The probe subspace is the mine/theirs linear probe's rows
for the tiles that differ between the two boards (and, separately, for tile s alone); the
generic baseline is the displacement to an unrelated history of the same length.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import frac_in, orth, subspace_fracs, zspace  # noqa: E402

from pim.environments.othello import arms as oa  # noqa: E402
from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.bench import cases_path  # noqa: E402
from pim.environments.othello.data import N_CLASSES, N_TILES, canonical_vocab, tokens_and_labels  # noqa: E402
from pim.environments.othello.vendor.othello import OthelloBoardState  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402

EXP = REPO / "experiments" / "edit_direction_alignment"
DEV = "cuda"


def replay(h, rules):
    b = OthelloBoardState(**rules)
    try:
        b.update(h, prt=False)
    except AssertionError:
        return None
    return b


def mine_board(b):
    st = (b.state + 1).flatten()                          # white 0 / blank 1 / black 2
    nxt = 2 if b.next_hand_color > 0 else 0
    return np.where(st == 1, 0, np.where(st == nxt, 1, 2))   # blank 0 / mine 1 / theirs 2


def search_cf(h, s, rules, max_sub=400, subs_only=False):
    """Closest real history (same length, same mover) whose board has tile s flipped.

    ⛔ Move SWAPS produce legal histories the model handles BADLY (legal mass 0.845, against
    0.994 for single-move substitutions and 0.998 for ordinary held-out prefixes), and they
    contaminated the first run of this analysis. They are kept because substitutions alone
    never reach the flipped board exactly (0/900 cases); the caller MUST therefore filter on
    the model's legal mass on the returned history, which is what screens them."""
    b0 = replay(h, rules); m0 = mine_board(b0)
    want = m0.copy(); want[s] = 3 - want[s]                # flip mine<->theirs at s
    best, best_d, tried = None, 99, 0
    cands = []
    for k in range(len(h)):
        bk = replay(h[:k], rules)
        for mv in bk.get_valid_moves():
            if mv != h[k]:
                cands.append(h[:k] + [mv] + h[k + 1:])
    if not subs_only:
        for i in range(len(h)):
            for k in range(i + 1, len(h)):
                hh = list(h); hh[i], hh[k] = hh[k], hh[i]; cands.append(hh)
    rng = np.random.default_rng(s)
    for hh in [cands[j] for j in rng.permutation(len(cands))[:max_sub]]:
        b = replay(hh, rules)
        if b is None or b.next_hand_color != b0.next_hand_color:
            continue
        tried += 1
        m = mine_board(b)
        if m[s] != want[s]:
            continue
        d = int((m != want).sum())
        if d < best_d:
            best, best_d = (hh, m), d
            if d == 0:
                break
    return best, best_d, m0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/initial_othello_comparison/L-oth-20m")
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--min-mass", type=float, default=0.99,
                    help="keep only counterfactual histories the model handles normally")
    a = ap.parse_args()
    run_dir = REPO / a.run
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    rules = oc.rules_of(inst)
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV)
    NP = n_points(model)
    stoi = canonical_vocab(); itos = {v: k for k, v in stoi.items()}
    # the canonical mine/theirs linear probes (cache hit on the run's own grid)
    tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",), instance=inst)["probe"])
    data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(tok[:20000], ln[:20000])], **rules)
    grid = oa.fit_probe_grid(model, data, cache_dir=run_dir / "probes", log=None)
    lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(NP)}
    # residual covariance from held-out probe games (positions 5..58)
    with torch.no_grad():
        idx = torch.from_numpy(data.tokens[19000:19400]).to(DEV)      # (400, 59) int64, the model block
        rs = model.residual_stack(idx)                                        # (NP, 400, 59, d)
        m = torch.from_numpy(data.mask[19000:19400]).to(DEV)
        cov_rows = {p: rs[p][m] for p in range(NP)}
    with open(cases_path(inst), "rb") as f:
        cases = pickle.load(f)
    rng = np.random.default_rng(0)
    sel = rng.permutation(len(cases))[: a.n]
    raw = []
    for i in sel:
        c = cases[i]; h = [int(x) for x in c["history"]]; s = int(c["pos_int"])
        best, d, m0 = search_cf(h, s, rules)
        if best is None or d > 0:                      # EXACT counterfactual boards only
            continue
        hh, m1 = best
        raw.append({"h": h, "hh": hh, "s": s, "diff": np.where(m0 != m1)[0].tolist(), "ham": d, "L": len(h)})
    # validate: the model must treat the counterfactual history as an ordinary game
    from pim.metrics.set_editability import uniform_over_legal
    from pim.environments.othello.data import board_probs
    found = []
    by = {}
    for i, f in enumerate(raw):
        by.setdefault(len(f["hh"]), []).append(i)
    with torch.no_grad():
        for L, ids in by.items():
            idx = torch.from_numpy(np.array([[stoi[x] for x in raw[i]["hh"]] for i in ids])).to(DEV)
            pr = board_probs(model.decode(idx), getattr(model, "output_kind", "logits"))
            for j, i in enumerate(ids):
                lg = sorted(replay(raw[i]["hh"], rules).get_valid_moves())
                m = float(pr[j, lg].sum())
                if m >= a.min_mass:
                    found.append({**raw[i], "cf_legal_mass": m})
    print(f"  {len(raw)} exact counterfactual boards from {len(sel)} cases; "
          f"{len(found)} pass the model-normality filter (legal mass >= {a.min_mass})", flush=True)
    hams = np.array([f["ham"] for f in found])
    print(f"{a.run}: {len(found)} clean cases (exact board, substitution-only, model-normal); "
          f"mean cf legal mass {np.mean([f['cf_legal_mass'] for f in found]):.4f}", flush=True)
    # residuals at the last position, per case (bucket by length so the last index aligns)
    def resid(hists):
        outs = {p: [None] * len(hists) for p in range(NP)}
        by_len = {}
        for i, h in enumerate(hists):
            by_len.setdefault(len(h), []).append(i)
        with torch.no_grad():
            for L, ids in by_len.items():
                idx = torch.from_numpy(np.array([[stoi[x] for x in hists[i]] for i in ids])).to(DEV)
                r = model.residual_stack(idx)[:, :, -1]                        # (NP, B, d)
                for p in range(NP):
                    for j, i in enumerate(ids):
                        outs[p][i] = r[p, j]
        return {p: torch.stack(v) for p, v in outs.items()}
    R = resid([f["h"] for f in found]); Rcf = resid([f["hh"] for f in found])
    # generic: another found case of the same length
    gen_idx = []
    for i, f in enumerate(found):
        same = [j for j, g in enumerate(found) if g["L"] == f["L"] and j != i]
        gen_idx.append(int(rng.choice(same)) if same else i)
    out = {"run": a.run, "instance": inst, "n_cases": len(found), "construction": "substitution-only, exact board, model legal mass >= " + str(a.min_mass),
           "cf_legal_mass": float(np.mean([f["cf_legal_mass"] for f in found])), "layers": {}}
    for p in range(NP):
        probe = lin[p]
        W = probe.net.weight.detach()                                            # (192, d)
        z, zcf = zspace(probe, R[p]), zspace(probe, Rcf[p])
        dz = zcf - z
        dgen = zspace(probe, R[p][torch.tensor(gen_idx, device=DEV)]) - z
        cov_z = torch.cov(zspace(probe, cov_rows[p]).T)
        fr_diff, fr_s, fr_gen, fr_hf, fr_gen_hf, fr_pca16 = [], [], [], [], [], []
        for i, f in enumerate(found):
            rows_diff = [t * N_CLASSES + c for t in f["diff"] for c in range(N_CLASSES)]
            rows_s = [f["s"] * N_CLASSES + c for c in range(N_CLASSES)]
            fr = subspace_fracs(dz[i:i + 1], W[rows_diff], cov_z, ks=(16,))
            fr_diff.append(float(fr["rows"])); fr_hf.append(float(fr["haufe"])); fr_pca16.append(float(fr["pca16"]))
            fr_s.append(float(frac_in(dz[i:i + 1], orth(W[rows_s]))))
            fg = subspace_fracs(dgen[i:i + 1], W[rows_diff], cov_z, ks=(16,))
            fr_gen.append(float(fg["rows"])); fr_gen_hf.append(float(fg["haufe"]))
        res = {"rows_changed_tiles": [float(np.mean(fr_diff)), float(np.std(fr_diff))],
               "rows_tile_s": [float(np.mean(fr_s)), float(np.std(fr_s))],
               "haufe_changed_tiles": [float(np.mean(fr_hf)), float(np.std(fr_hf))],
               "pca16_changed_tiles": float(np.mean(fr_pca16)),
               "generic_rows_changed_tiles": float(np.mean(fr_gen)), "generic_haufe": float(np.mean(fr_gen_hf)),
               "delta_norm_over_h_norm": float((dz.norm(dim=1) / z.norm(dim=1)).mean()),
               "n_rows_mean": float(np.mean([3 * len(f["diff"]) for f in found]))}
        out["layers"][str(p)] = res
        print(f"pt {p}: frac(Δ in rows of changed tiles) {res['rows_changed_tiles'][0]:.3f}  tile-s rows {res['rows_tile_s'][0]:.3f}"
              f"  haufe {res['haufe_changed_tiles'][0]:.3f}  pca16 {res['pca16_changed_tiles']:.3f}"
              f"  | generic rows {res['generic_rows_changed_tiles']:.3f} haufe {res['generic_haufe']:.3f}"
              f"  | |Δ|/|h| {res['delta_norm_over_h_norm']:.2f}  rows/case {res['n_rows_mean']:.1f}", flush=True)
    (EXP / "scores" / f"othello_alignment_{Path(a.run).name}_clean.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
