"""Nanda et al. (2309.00941) §4.1 linear-direction intervention, and our pseudoinverse, on their model.

Their method: x' <- x + alpha * p_d, where p_d is the LINEAR probe's weight column for the target
direction d in {MINE, YOURS, EMPTY} at the intervened tile, added to the residual stream at EVERY
layer. One vector addition, no gradients. Their Table 2: null 2.723, non-linear (Li) 0.12,
linear addition 0.10.

Ours: the same target, reached by `pim.editors.probe_steering` — the minimum-norm write that makes
the linear read-out say the target. Solved in the probe's STANDARDISED space (where it is linear)
and mapped back to the raw activation.

Probes come from the run-5 cache: ("mine", "linear", "frame", point), 0.72% held-out error.
"""
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent / "othello_gpt"), str(_HERE.parents[3])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import othello_data as od  # noqa: E402
import othello_probe as op  # noqa: E402
import transfer_pipeline as tp  # noqa: E402
from pim.editors.probe_steering import inject_state, probe_decomposition  # noqa: E402


class _AsLinearExtractor:
    """The whole bridge to our editor: `probe_decomposition` reads `.linear.weight/.bias`.

    Our `WorldStateProbe` keeps that layer as `.net`. Nothing else about the editor changes —
    `inject_state` is called unmodified, so this is our canonical Pseudoinverse Injection, not a
    re-derivation of its formula.
    """

    def __init__(self, probe):
        self.linear = probe.net

DEV, N_POINTS = tp.DEVICE, tp.N_POINTS
BLANK, MINE, THEIRS = 0, 1, 2


def load_linear_probes():
    import glob
    blob = torch.load(sorted(glob.glob(str(tp.CACHE_DIR / "*.pt")))[0],
                      map_location=DEV, weights_only=False)
    out = {}
    for k, (d_in, d_out, hidden, sd) in blob["probes"].items():
        tgt, fam, split, pt = k.split("|")
        if (tgt, fam, split) != ("mine", "linear", "frame"):
            continue
        pr = op.WorldStateProbe(d_in, d_out, None, n_classes=od.N_CLASSES).to(DEV)
        pr.load_state_dict(sd)
        pr.eval()
        out[int(pt)] = pr
    err = [s["error_rate"] for s in blob["stats"]
           if (s["target"], s["family"], s["split"]) == ("mine", "linear", "frame")]
    print(f"loaded {len(out)} linear mine/theirs probes; held-out error "
          f"{min(err):.2f}%-{max(err):.2f}% across points")
    return out


def case_targets(bench):
    """Per case: the intervened tile's CURRENT and TARGET label in mine/theirs coordinates.

    The benchmark flips absolute colour. The player to move does not change, so flipping colour
    flips MINE<->THEIRS exactly.
    """
    from data.othello import OthelloBoardState
    import pickle
    with open(od.OTHELLO_ROOT / "intervention_benchmark.pkl", "rb") as f:
        ds = pickle.load(f)
    cur = np.zeros(bench.n_cases, np.int64)
    for i, c in enumerate(ds):
        b = OthelloBoardState()
        b.update(c["history"], prt=False)
        nxt = 2 if b.next_hand_color > 0 else 0
        cur[i] = MINE if c["ori_color"] == nxt else THEIRS
    tgt = np.where(cur == MINE, THEIRS, MINE)
    return cur, tgt


def run(shim, bench, probes, mode, alpha, points, tgt_lab, cur_lab, subtract=False):
    """One arm. `mode`: 'add' = Nanda's vector addition; 'pinv' = our pseudoinverse injection."""
    probs = np.zeros((bench.n_cases, od.N_TILES), np.float32)
    ratios = []
    pinv = {}
    if mode == "pinv":
        for ell, p in probes.items():
            # our editor's own decomposition, unmodified
            pinv[ell] = probe_decomposition(_AsLinearExtractor(p))   # (A, b, A_pinv)
    for toks, ids in zip(bench.tokens, bench.case_ids):
        idx = torch.from_numpy(toks).to(DEV)
        b = len(ids)
        sq = torch.from_numpy(bench.pos_int[ids]).to(DEV)
        td = torch.from_numpy(tgt_lab[ids]).to(DEV)
        cd = torch.from_numpy(cur_lab[ids]).to(DEV)
        rec = []

        def hook(layer, x, _rec=rec):
            if layer not in points:
                return x
            p = probes[layer]
            cur = x[:, -1]
            if mode in ("add", "add_raw"):
                W = p.net.weight.detach().view(od.N_TILES, od.N_CLASSES, -1)   # (64,3,512)
                # "add": the probe standardises its input, so the direction that raises this logit
                # in RAW activation space is w / x_std. "add_raw": the weight row as-is, which is
                # what Nanda's un-standardised probe would give.
                d = W[sq, td] / p.x_std if mode == "add" else W[sq, td]
                if subtract:
                    d = d - (W[sq, cd] / p.x_std if mode == "add" else W[sq, cd])
                # alpha is a FRACTION OF THE ACTIVATION NORM, so one value means the same size of
                # write at every residual point (their alpha is unstated and the scale differs
                # ~3x across our points)
                d = d / d.norm(dim=1, keepdim=True)
                delta = alpha * cur.norm(dim=1, keepdim=True) * d
            elif mode == "pinv_local":
                W = p.net.weight.detach().view(od.N_TILES, od.N_CLASSES, -1)   # (64,3,512)
                z = (cur - p.x_mean) / p.x_std
                lg = p.net(z).view(b, od.N_TILES, od.N_CLASSES)
                sel = lg[torch.arange(b), sq]                                  # (B,3)
                new = sel.clone()
                new[torch.arange(b), td] = sel[torch.arange(b), cd]
                new[torch.arange(b), cd] = sel[torch.arange(b), td]
                As = W[sq]                                                     # (B,3,512)
                dz = torch.linalg.lstsq(As, (new - sel).unsqueeze(-1)).solution.squeeze(-1) \
                     if False else torch.einsum("bij,bj->bi", torch.linalg.pinv(As), (new - sel))
                delta = alpha * dz * p.x_std
            else:
                # OUR editor, unmodified. The probe is linear in its STANDARDISED input, so the
                # injection is solved in z-space and mapped back — that mapping is the only
                # bridge, and it is exact.
                A, bvec, A_pinv = pinv[layer]
                z = (cur - p.x_mean) / p.x_std
                logits = p.net(z)                                              # (B, 192)
                lg = logits.view(b, od.N_TILES, od.N_CLASSES).clone()
                sel = lg[torch.arange(b), sq]                                  # (B, 3)
                new = sel.clone()
                new[torch.arange(b), td] = sel[torch.arange(b), cd]
                new[torch.arange(b), cd] = sel[torch.arange(b), td]
                lg[torch.arange(b), sq] = new
                z_new = inject_state(z, lg.view(b, -1), A, A_pinv, bvec)
                delta = alpha * (z_new - z) * p.x_std
            _rec.append(float((delta.norm(dim=1) / cur.norm(dim=1)).mean()))
            out = x.clone()
            out[:, -1] = cur + delta
            return out

        with torch.no_grad():
            h, _ = shim._run(shim.embed(idx), None, edit=hook)
            probs[ids] = od.board_probs(shim.decoder(shim.norm_out(h[:, -1])))
        ratios.append(np.mean(rec) if rec else 0.0)
    card = od.scorecard(probs, bench)
    card["write_ratio"] = float(np.mean(ratios))
    return probs, card


if __name__ == "__main__":
    shim = tp.load_model()
    bench = od.load_benchmark()
    probes = load_linear_probes()
    cur_lab, tgt_lab = case_targets(bench)
    uns = od.scorecard(tp.unsteered(shim, bench), bench)
    print(f"\nnull intervention: Li error {uns['li_error_vs_post']:.3f} "
          f"(Nanda Table 2 reports 2.723)   Edit Index {uns['edit_index_union']:+.3f}\n")
    ALL = set(range(N_POINTS))
    hdr = (f"{'arm':>34} {'alpha':>7} {'|dx|/|x|':>9} {'Li post':>8} {'Li pre':>8} "
           f"{'EI union':>9} {'EI symd':>8} {'legal':>6}")
    print(hdr)
    rows = {}
    FINE = (0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5)
    for mode, subtract, alphas, tag in (
        ("add", False, FINE, "Nanda add (dir = w/std)"),
        ("add_raw", False, FINE, "Nanda add (dir = w, raw)"),
        ("add", True, FINE, "Nanda add, tgt - cur"),
        ("pinv", False, (0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
         "OURS: inject_state (unmodified)"),
    ):
        for a in alphas:
            _, c = run(shim, bench, probes, mode, a, ALL, tgt_lab, cur_lab, subtract)
            rows[(tag, a, mode)] = c
            print(f"{tag:>34} {a:>7} {c['write_ratio']:>9.3f} {c['li_error_vs_post']:>8.3f} "
                  f"{c['li_error_vs_pre']:>8.3f} {c['edit_index_union']:>+9.3f} "
                  f"{c['edit_index_symdiff']:>+8.3f} {c['legal_mass']:>6.3f}", flush=True)
    # their Figure 7: how many layers must be intervened
    best = min([k for k in rows if k[0].startswith("Nanda add")], key=lambda k: rows[k]["li_error_vs_post"])
    print(f"\nbest alpha for Nanda addition: {best[1]}  ->  reproducing their Figure 7 (first N layers)")
    for n in range(1, N_POINTS + 1):
        _, c = run(shim, bench, probes, best[2] if len(best) > 2 else "add", best[1], set(range(n)), tgt_lab, cur_lab)
        print(f"{'first ' + str(n) + ' residual points':>34} {best[1]:>7} {c['write_ratio']:>9.3f} "
              f"{c['li_error_vs_post']:>8.3f} {c['li_error_vs_pre']:>8.3f} "
              f"{c['edit_index_union']:>+9.3f} {c['edit_index_symdiff']:>+8.3f} {c['legal_mass']:>6.3f}",
              flush=True)
