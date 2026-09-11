"""Extras on top of `edit_direction_alignment/scripts/othello_alignment.py`, for any Othello run
(2026-09-11). Same oracle counterfactual (exact board by move substitution/swap, model-normal
filter, legal mass >= 0.99), same probes (the run's cached mine/theirs linear grid), and:

  * the single ND WRITE DIRECTION d = W[s,target] - W[s,current] against Δ: cos²(Δ, d), raw and
    with the Haufe pattern rows P in place of W; a random unit direction as the floor;
  * the 3-row subspace fractions again (rows / Haufe / generic), so the split below is on one footing;
  * the split of cases by whether tile s was actually RECOLOURED during the real history
    (its colour is then computed, not the placement-parity lookup) vs never recoloured;
  * the ceiling: the model run on the counterfactual history itself, scored with the
    canonical union Edit Index against the case's legal_pre / legal_post.

Self-contained; nothing canonical is touched. Output → scores/alignment_extras_<run>.json.
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/edit_direction_alignment/scripts"))
from common import frac_in, haufe_patterns, orth, zspace  # noqa: E402
from othello_alignment import replay, search_cf  # noqa: E402
from pim.environments.othello import arms as oa, corpus as oc  # noqa: E402
from pim.environments.othello.bench import cases_path  # noqa: E402
from pim.environments.othello.data import (
    N_CLASSES,
    board_probs,
    canonical_vocab,
    tokens_and_labels,
)  # noqa: E402
from pim.environments.othello.vendor.othello import OthelloBoardState  # noqa: E402
from pim.metrics.set_editability import move_scorecard  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402

DEV = "cuda"
EXP = REPO / "experiments/adjacent_flip_ablation"


def recoloured_during(h, s, rules):
    """Was tile s ever recoloured while replaying h? (placed colour != final colour)"""
    b = OthelloBoardState(**rules)
    for mv in h:
        before = b.state[s // 8, s % 8]
        b.umpire(mv)
        after = b.state[s // 8, s % 8]
        if before != 0 and after != before:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--min-mass", type=float, default=0.99)
    a = ap.parse_args()
    run_dir = REPO / a.run
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    rules = oc.rules_of(inst)
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV)
    NP = n_points(model)
    stoi = canonical_vocab()
    itos = {v: k for k, v in stoi.items()}
    tok, ln = oc.load(
        oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",), instance=inst)[
            "probe"
        ]
    )
    data = tokens_and_labels(
        [[itos[int(t)] for t in row[:L]] for row, L in zip(tok[:20000], ln[:20000])],
        **rules,
    )
    grid = oa.fit_probe_grid(model, data, cache_dir=run_dir / "probes", log=None)
    lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(NP)}
    with torch.no_grad():
        rs = model.residual_stack(torch.from_numpy(data.tokens[19000:19400]).to(DEV))
        m = torch.from_numpy(data.mask[19000:19400]).to(DEV)
        cov_z = {p: torch.cov(zspace(lin[p], rs[p][m]).T) for p in range(NP)}
    cases = pickle.load(open(cases_path(inst), "rb"))
    rng = np.random.default_rng(0)
    sel = rng.permutation(len(cases))[: a.n]
    raw = []
    for i in sel:
        c = cases[i]
        h = [int(x) for x in c["history"]]
        s = int(c["pos_int"])
        best, d, m0 = search_cf(h, s, rules)
        if best is None or d > 0:
            continue
        hh, m1 = best
        raw.append(
            {
                "h": h,
                "hh": hh,
                "s": s,
                "L": len(h),
                "cur": int(m0[s]),
                "tgt": int(m1[s]),
            }
        )

    def pred(hists):
        out = np.zeros((len(hists), 64), np.float32)
        by = {}
        for i, h in enumerate(hists):
            by.setdefault(len(h), []).append(i)
        with torch.no_grad():
            for L, ids in by.items():
                idx = torch.from_numpy(
                    np.array([[stoi[x] for x in hists[i]] for i in ids])
                ).to(DEV)
                out[ids] = board_probs(
                    model.decode(idx), getattr(model, "output_kind", "logits")
                )
        return out

    p_cf = pred([f["hh"] for f in raw])
    found = []
    for i, f in enumerate(raw):
        lg = sorted(replay(f["hh"], rules).get_valid_moves())
        if float(p_cf[i, lg].sum()) >= a.min_mass:
            found.append({**f, "cf_i": i})
    n = len(found)
    print(f"{a.run}: {len(raw)} exact boards, {n} model-normal", flush=True)
    for f in found:
        f["recoloured"] = recoloured_during(f["h"], f["s"], rules)
    rec = np.array([f["recoloured"] for f in found])
    Ls = np.array([f["L"] for f in found])
    print(
        f"  prefix length: mean {Ls.mean():.1f} (min {Ls.min()}, max {Ls.max()}); tile s recoloured during the game in {int(rec.sum())}/{n}",
        flush=True,
    )
    # ceiling + unedited on the same cases
    legal_pre = [sorted(replay(f["h"], rules).get_valid_moves()) for f in found]

    def legal_post(f):
        b = replay(f["h"], rules)
        b.state[f["s"] // 8, f["s"] % 8] *= -1
        return sorted(b.get_valid_moves())

    legal_post = [legal_post(f) for f in found]
    p_un = pred([f["h"] for f in found])
    p_cfc = p_cf[[f["cf_i"] for f in found]]
    card_un = move_scorecard(p_un, legal_pre, legal_post)
    card_cf = move_scorecard(p_cfc, legal_pre, legal_post)
    print(
        f"  ceiling (true counterfactual) EI {card_cf['edit_index_union']:+.3f} vs unedited {card_un['edit_index_union']:+.3f}",
        flush=True,
    )

    # residuals at the last position
    def resid(hists):
        outs = {p: [None] * len(hists) for p in range(NP)}
        by = {}
        for i, h in enumerate(hists):
            by.setdefault(len(h), []).append(i)
        with torch.no_grad():
            for L, ids in by.items():
                idx = torch.from_numpy(
                    np.array([[stoi[x] for x in hists[i]] for i in ids])
                ).to(DEV)
                r = model.residual_stack(idx)[:, :, -1]
                for p in range(NP):
                    for j, i in enumerate(ids):
                        outs[p][i] = r[p, j]
        return {p: torch.stack(v) for p, v in outs.items()}

    R = resid([f["h"] for f in found])
    Rcf = resid([f["hh"] for f in found])
    gen_idx = [
        int(
            rng.choice(
                [j for j, g in enumerate(found) if g["L"] == f["L"] and j != i] or [i]
            )
        )
        for i, f in enumerate(found)
    ]
    g = torch.Generator(device="cpu").manual_seed(0)
    out = {
        "run": a.run,
        "instance": inst,
        "n_cases": n,
        "n_exact_boards": len(raw),
        "prefix_len_mean": float(Ls.mean()),
        "n_recoloured": int(rec.sum()),
        "ceiling_edit_index": card_cf["edit_index_union"],
        "unedited_edit_index": card_un["edit_index_union"],
        "ceiling_recoloured": None,
        "ceiling_lookup": None,
        "layers": {},
    }
    if rec.any() and (~rec).any():
        out["ceiling_recoloured"] = move_scorecard(
            p_cfc[rec],
            [x for x, r in zip(legal_pre, rec) if r],
            [x for x, r in zip(legal_post, rec) if r],
        )["edit_index_union"]
        out["ceiling_lookup"] = move_scorecard(
            p_cfc[~rec],
            [x for x, r in zip(legal_pre, rec) if not r],
            [x for x, r in zip(legal_post, rec) if not r],
        )["edit_index_union"]

    def cos2(A, B):  # rowwise
        return (A * B).sum(1) ** 2 / (
            (A.norm(dim=1) ** 2) * (B.norm(dim=1) ** 2)
        ).clamp_min(1e-12)

    for p in range(NP):
        probe = lin[p]
        W = probe.net.weight.detach()
        P = haufe_patterns(W, cov_z[p])
        z, zcf = zspace(probe, R[p]), zspace(probe, Rcf[p])
        dz = zcf - z
        dgen = zspace(probe, R[p][torch.tensor(gen_idx, device=DEV)]) - z
        rows_i = torch.tensor(
            [[f["s"] * N_CLASSES + c for c in range(N_CLASSES)] for f in found],
            device=DEV,
        )
        d_raw = torch.stack(
            [
                W[f["s"] * N_CLASSES + f["tgt"]] - W[f["s"] * N_CLASSES + f["cur"]]
                for f in found
            ]
        )
        d_hf = torch.stack(
            [
                P[f["s"] * N_CLASSES + f["tgt"]] - P[f["s"] * N_CLASSES + f["cur"]]
                for f in found
            ]
        )
        rnd = torch.randn(n, W.shape[1], generator=g).to(DEV)
        c_raw, c_hf, c_rnd = cos2(dz, d_raw), cos2(dz, d_hf), cos2(dz, rnd)
        c_gen_raw = cos2(dgen, d_raw)
        fr_rows = torch.stack(
            [frac_in(dz[i : i + 1], orth(W[rows_i[i]]))[0] for i in range(n)]
        )
        fr_hf = torch.stack(
            [frac_in(dz[i : i + 1], orth(P[rows_i[i]]))[0] for i in range(n)]
        )
        fr_gen = torch.stack(
            [frac_in(dgen[i : i + 1], orth(W[rows_i[i]]))[0] for i in range(n)]
        )
        fr_gen_hf = torch.stack(
            [frac_in(dgen[i : i + 1], orth(P[rows_i[i]]))[0] for i in range(n)]
        )
        r_t = torch.from_numpy(rec).to(DEV)

        def split(v):
            return {
                "all": float(v.mean()),
                "recoloured": float(v[r_t].mean()) if r_t.any() else None,
                "lookup": float(v[~r_t].mean()) if (~r_t).any() else None,
            }

        L = {
            "nd_dir_cos2_raw": split(c_raw),
            "nd_dir_cos2_haufe": split(c_hf),
            "nd_dir_cos2_random": float(c_rnd.mean()),
            "nd_dir_cos2_generic_raw": float(c_gen_raw.mean()),
            "rows_frac": split(fr_rows),
            "haufe_frac": split(fr_hf),
            "generic_rows_frac": float(fr_gen.mean()),
            "generic_haufe_frac": float(fr_gen_hf.mean()),
            "delta_norm_over_h_norm": float((dz.norm(dim=1) / z.norm(dim=1)).mean()),
        }
        out["layers"][str(p)] = L
        print(
            f"pt {p}: ND-dir cos² raw {L['nd_dir_cos2_raw']['all']:.3f} (recol {L['nd_dir_cos2_raw']['recoloured'] or float('nan'):.3f} / lookup {L['nd_dir_cos2_raw']['lookup'] or float('nan'):.3f})"
            f"  haufe {L['nd_dir_cos2_haufe']['all']:.3f}  random {L['nd_dir_cos2_random']:.4f}  generic {L['nd_dir_cos2_generic_raw']:.4f}"
            f" | rows {L['rows_frac']['all']:.3f} (recol {L['rows_frac']['recoloured'] or float('nan'):.3f} / lookup {L['rows_frac']['lookup'] or float('nan'):.3f}) gen {L['generic_rows_frac']:.3f}"
            f" | haufe {L['haufe_frac']['all']:.3f} genH {L['generic_haufe_frac']:.3f}",
            flush=True,
        )
    (EXP / "scores").mkdir(exist_ok=True)
    (EXP / "scores" / f"alignment_extras_{Path(a.run).name}.json").write_text(
        json.dumps(out, indent=1)
    )


if __name__ == "__main__":
    main()
