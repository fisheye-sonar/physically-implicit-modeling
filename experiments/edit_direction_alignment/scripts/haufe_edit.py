"""Do HAUFE-CORRECTED directions edit? (2026-09-09)

The Haufe pattern matrix P = [Σ Wᵀ (W Σ Wᵀ)⁻¹]ᵀ satisfies W Pᵀ = I, so Pᵀ is a right inverse
of the probe's read-out map, exactly like the pseudo-inverse W⁺ that canonical PI uses — the
two land the read-out on the SAME target and differ only in the null-space component of the
write. So this is a like-for-like swap:

    PI-haufe : Δz = α · Pᵀ (target − current read-out)      vs  canonical Δz = α · W⁺ (…)
    ND-haufe : direction = P[target row] − P[current row]   vs  canonical W[…] / x_std

Run on the two inert models (L-oth-adjacent-20m, L-dw-noiseless-20m) and, as a positive
control, the editable one (L-oth-20m). Scored with the canonical Edit Index and fidelity
guard. Nothing outside this experiment directory is touched.
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import haufe_patterns  # noqa: E402

from pim.environments.discworld import arms as dwa, bench as dwb  # noqa: E402
from pim.environments.discworld.bench import K_ROLL, dim_idx, restrict_mask  # noqa: E402
from pim.environments.othello import arms as oa, corpus as oc  # noqa: E402
from pim.environments.othello import case_targets, load_benchmark  # noqa: E402
from pim.environments.othello.data import N_CLASSES, board_probs, canonical_vocab, tokens_and_labels  # noqa: E402
from pim.metrics.set_editability import move_fidelity_ratio, move_scorecard  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.base import collect_residuals  # noqa: E402

DEV = "cuda"
OUT = {}


def patterns(probe, cov_z):
    """P (d_out, d) with W Pᵀ = I — the forward directions of this probe's read-outs."""
    return haufe_patterns(probe.net.weight.detach(), cov_z)


# ── Othello ──────────────────────────────────────────────────────────────────
def othello(run, inst, alphas_pi=(0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0), alphas_nd=(0.05, 0.2, 0.35, 0.7, 1.0, 2.0)):
    model, _ = load_checkpoint(REPO / run / "best_model.pt", device=DEV)
    NP = n_points(model); rules = oc.rules_of(inst)
    tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",), instance=inst)["probe"])
    itos = {v: k for k, v in canonical_vocab().items()}
    data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(tok[:20000], ln[:20000])], **rules)
    grid = oa.fit_probe_grid(model, data, cache_dir=REPO / run / "probes", log=None)
    lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(NP)}
    with torch.no_grad():                                    # residual covariance per point
        rs = model.residual_stack(torch.from_numpy(data.tokens[:400]).to(DEV))
        m = torch.from_numpy(data.mask[:400]).to(DEV)
        cov = {p: torch.cov(((rs[p][m] - lin[p].x_mean) / lin[p].x_std).T) for p in range(NP)}
    P = {p: patterns(lin[p], cov[p]) for p in range(NP)}
    bench = load_benchmark(inst); cur, tgt = case_targets(bench)
    uns = oa.unsteered_probs(model, bench); u = oa.unsteered(model, bench)
    res = {"unedited": u["edit_index_union"], "arms": []}
    for mode in ("pinv", "add_sub"):
        for ell in range(NP):
            probe, Pl = lin[ell], P[ell]
            for a in (alphas_pi if mode == "pinv" else alphas_nd):
                probs = np.zeros((bench.n_cases, 64), np.float32)
                for toks, ids in zip(bench.tokens, bench.case_ids):
                    idx = torch.from_numpy(toks).to(DEV); bsz = len(ids)
                    sq = torch.from_numpy(bench.pos_int[ids]).to(DEV)
                    td = torch.from_numpy(tgt[ids]).to(DEV); cd = torch.from_numpy(cur[ids]).to(DEV)
                    def hook(layer, x, probe=probe, Pl=Pl, a=a, sq=sq, td=td, cd=cd, bsz=bsz):
                        if layer != ell: return x
                        c = x[:, -1]; z = (c - probe.x_mean) / probe.x_std
                        ar = torch.arange(bsz, device=DEV)
                        if mode == "pinv":
                            lg = probe.net(z).view(bsz, -1, N_CLASSES).clone()
                            sel = lg[ar, sq].clone(); new = sel.clone()
                            new[ar, td] = sel[ar, cd]; new[ar, cd] = sel[ar, td]
                            lg[ar, sq] = new
                            d_y = lg.view(bsz, -1) - probe.net(z)          # required read-out change
                            dz = a * (d_y @ Pl)                            # Pᵀ δy, batched
                        else:
                            dvec = Pl[sq * N_CLASSES + td] - Pl[sq * N_CLASSES + cd]
                            dz = a * z.norm(dim=-1, keepdim=True) * dvec / dvec.norm(dim=-1, keepdim=True)
                        o = x.clone(); o[:, -1] = c + dz * probe.x_std; return o
                    with torch.no_grad():
                        probs[ids] = board_probs(model.decode(idx, edit=hook), getattr(model, "output_kind", "logits"))
                card = move_scorecard(probs, bench.legal_pre, bench.legal_post)
                res["arms"].append({"editor": ("PI" if mode == "pinv" else "ND") + "-haufe", "point": ell, "alpha": a,
                                    "edit_index_union": card["edit_index_union"],
                                    "fidelity_ratio": move_fidelity_ratio(probs, uns, bench.legal_post),
                                    "li_error_vs_post": card["li_error_vs_post"]})
    for ed in ("PI-haufe", "ND-haufe"):
        sub = [r for r in res["arms"] if r["editor"] == ed]
        b = max(sub, key=lambda r: r["edit_index_union"])
        g = [r for r in sub if r["fidelity_ratio"] <= 1.1]
        bg = max(g, key=lambda r: r["edit_index_union"]) if g else None
        print(f"  {inst:13s} {ed}: best {b['edit_index_union']:+.3f}/fid {b['fidelity_ratio']:.2f} (pt{b['point']} α{b['alpha']}) | "
              f"guarded {bg['edit_index_union']:+.3f}/{bg['fidelity_ratio']:.2f} (pt{bg['point']} α{bg['alpha']})" if bg else "", flush=True)
        res[ed] = {"best": b, "guarded": bg}
    del model
    return res


# ── discworld ────────────────────────────────────────────────────────────────
def discworld(run="runs/noise_ablation/L-dw-noiseless-20m",
              alphas_pi=(0.25, 1.0, 3.0, 8.0, 20.0, 60.0, 175.0), alphas_nd=(0.05, 0.2, 0.5, 1.0, 2.0)):
    inst = json.loads((REPO / run / "config.json").read_text())["data"]["instance"]
    root = REPO / "datasets/discworld" / inst
    model, _ = load_checkpoint(REPO / run / "best_model.pt", device=DEV); NP = n_points(model)
    lin = dwa.fit_probes(model, target="full", n_seq=30_000, family="linear", basis_name="frustum",
                         data_dir=root / "probe", cache_dir=REPO / run / "probes", log=None)
    import h5py
    with h5py.File(root / "probe/test.h5", "r") as f:
        obs_cov = f["obs_intensity"][30_000:31_500, :39].astype(np.float32)
    Rc = collect_residuals(model, obs_cov, batch=64)
    b = dwb.load_bench(model, n=192, target="full", basis_name="frustum", data_dir=root / "eval")
    u = dwa.unsteered(model, b)
    res = {"unedited": u["edit_index"], "arms": []}
    for ell in range(NP):
        probe = lin[ell][0]
        z_all = (torch.from_numpy(Rc[ell].reshape(-1, Rc.shape[-1])).to(DEV) - probe.x_mean) / probe.x_std
        Pl = patterns(probe, torch.cov(z_all.T))
        dwa.as_activations(model, ell)
        h0 = model.flat_state(b.state)
        z0 = (h0 - probe.x_mean) / probe.x_std
        for dims in ("pos", "all"):
            cm = restrict_mask(b.change_mask, dims)
            y_star = torch.where(cm, b.tgt, probe(h0))                    # hold the rest
            d_y = (y_star - probe.y_mean) / probe.y_std - probe.net(z0)   # in standardised read-out units
            dz_pi = d_y @ Pl
            idx = dim_idx(dims)
            rows = list(range(probe.net.weight.shape[0])) if idx is None else list(idx)
            dvec = (torch.sign(d_y[:, rows]) @ Pl[rows])                   # ND: signed sum of pattern rows
            for a in alphas_pi:
                roll = model.rollout_with_edit(b.state, ell, h0 + a * dz_pi * probe.x_std, K_ROLL).cpu().numpy()
                c = dwa.score(model, b, roll, u)
                res["arms"].append({"editor": "PI-haufe", "point": ell, "alpha": a, "dims": dims,
                                    "edit_index": c["edit_index"], "fidelity_ratio": c["fidelity_ratio"]})
            for a in alphas_nd:
                d = a * h0.norm(dim=-1, keepdim=True) * dvec / dvec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                roll = model.rollout_with_edit(b.state, ell, h0 + d * probe.x_std, K_ROLL).cpu().numpy()
                c = dwa.score(model, b, roll, u)
                res["arms"].append({"editor": "ND-haufe", "point": ell, "alpha": a, "dims": dims,
                                    "edit_index": c["edit_index"], "fidelity_ratio": c["fidelity_ratio"]})
    for ed in ("PI-haufe", "ND-haufe"):
        sub = [r for r in res["arms"] if r["editor"] == ed]
        bb = max(sub, key=lambda r: r["edit_index"])
        g = [r for r in sub if r["fidelity_ratio"] <= 1.1]
        bg = max(g, key=lambda r: r["edit_index"]) if g else None
        print(f"  dw-noiseless  {ed}: best {bb['edit_index']:+.3f}/fid {bb['fidelity_ratio']:.2f} (pt{bb['point']} α{bb['alpha']} {bb['dims']})"
              + (f" | guarded {bg['edit_index']:+.3f}/{bg['fidelity_ratio']:.2f} (pt{bg['point']} α{bg['alpha']} {bg['dims']})" if bg else ""), flush=True)
        res[ed] = {"best": bb, "guarded": bg}
    del model
    return res


if __name__ == "__main__":
    print("discworld (unedited −0.92):", flush=True)
    OUT["dw-noiseless"] = discworld()
    print("othello (unedited −0.82 / −0.68):", flush=True)
    OUT["oth-adjacent"] = othello("runs/adjacency_ablation/L-oth-adjacent-20m", "oth-adjacent")
    OUT["oth-uniform"] = othello("runs/initial_othello_comparison/L-oth-20m", "oth-uniform")
    (REPO / "experiments/edit_direction_alignment/scores/haufe_edit.json").write_text(json.dumps(OUT, indent=1, default=float))
