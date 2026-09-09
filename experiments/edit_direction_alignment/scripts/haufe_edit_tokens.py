"""Does the Haufe direction edit the FRAMES-AS-TOKENS discworld model? (2026-09-09)

L-dw-8ray-tok-20m has the largest absolute Haufe gain of any run (cos²θ 0.002 → 0.070, ×32),
so it is the sharpest test of "better-aligned write ⇒ editable". Scored with the run's own
canonical frame-set Edit Index and fidelity guard (`token_bench.scorecard`); canonical
editors on this run: PI +0.004, GS −0.097, unedited −0.755.

PI-haufe: Δz = α Pᵀ δy with P the pattern matrix (W Pᵀ = I, so it lands the SAME read-out
target as the pseudo-inverse and differs only in the null-space component).
ND-haufe: direction = signed combination of the pattern rows of the driven read-outs.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import haufe_patterns  # noqa: E402
from pim.environments.discworld import arms as dwa, bench as dwb, token_bench as tkb  # noqa: E402
from pim.environments.discworld.tokens import FrameVocab  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.base import collect_residuals  # noqa: E402

DEV = "cuda"
RUN = "runs/interface_ablation/L-dw-8ray-tok-20m"
run_dir = REPO / RUN
inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
root = REPO / "datasets/discworld" / inst
model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV); NP = n_points(model)
vocab = FrameVocab.load(run_dir / "vocab.npz"); enc, tag = tkb.token_encoder(vocab)
tb = tkb.load_token_bench(vocab, n=192, target="full", basis_name="frustum", data_dir=root / "eval")
lin = dwa.fit_probes(model, target="full", n_seq=30_000, family="linear", basis_name="frustum",
                     data_dir=root / "probe", cache_dir=run_dir / "probes", log=None,
                     encoder=enc, encoder_tag=tag)
uns, u = tkb.unsteered(model, tb)
print(f"{RUN}: {int(tb.keep.sum())}/{tb.n} scoreable cases | unedited EI {u['edit_index']:+.3f}", flush=True)

import h5py
with h5py.File(root / "probe/test.h5", "r") as f:
    obs_cov = f["obs_intensity"][30_000:31_500, :39].astype(np.float32)
Rc = collect_residuals(model, enc(obs_cov).astype(np.int64), batch=64)
x0 = tkb.residuals_last(model, tb)
ALPHA_PI = (0.25, 1.0, 3.0, 8.0, 20.0, 60.0, 175.0)
ALPHA_ND = (0.05, 0.2, 0.5, 1.0, 2.0, 4.0)
arms = []
for ell in range(NP):
    probe = lin[ell][0]
    z_all = (torch.from_numpy(Rc[ell].reshape(-1, Rc.shape[-1])).to(DEV) - probe.x_mean) / probe.x_std
    P = haufe_patterns(probe.net.weight.detach(), torch.cov(z_all.T))
    h0 = x0[ell]; z0 = (h0 - probe.x_mean) / probe.x_std
    for dims in ("pos", "all"):
        cm = dwb.restrict_mask(tb.change_mask, dims)
        y_star = torch.where(cm, tb.tgt, probe(h0))
        d_y = (y_star - probe.y_mean) / probe.y_std - probe.net(z0)
        dz_pi = d_y @ P
        idx = dwb.dim_idx(dims)
        rows = list(range(probe.net.weight.shape[0])) if idx is None else list(idx)
        dvec = torch.sign(d_y[:, rows]) @ P[rows]
        for a in ALPHA_PI:
            probs = tkb.probs_at_edit(model, tb, hook=tkb._write_hook(ell, h0 + a * dz_pi * probe.x_std))
            arms.append({"editor": "PI-haufe", "point": ell, "alpha": a, "dims": dims,
                         **{k: v for k, v in tkb.scorecard(probs, tb, uns).items() if np.isscalar(v)}})
        for a in ALPHA_ND:
            d = a * h0.norm(dim=-1, keepdim=True) * dvec / dvec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            probs = tkb.probs_at_edit(model, tb, hook=tkb._write_hook(ell, h0 + d * probe.x_std))
            arms.append({"editor": "ND-haufe", "point": ell, "alpha": a, "dims": dims,
                         **{k: v for k, v in tkb.scorecard(probs, tb, uns).items() if np.isscalar(v)}})
    b = max((r for r in arms if r["point"] == ell), key=lambda r: r["edit_index"])
    print(f"  pt {ell}: best {b['editor']} {b['edit_index']:+.3f}/fid {b['fidelity_ratio']:.2f} "
          f"(α {b['alpha']}, {b['dims']}) p_post {b['p_post']:.3f} p_pre {b['p_pre']:.3f}", flush=True)
out = {"run": RUN, "unedited": {k: v for k, v in u.items() if np.isscalar(v)}, "arms": arms}
for ed in ("PI-haufe", "ND-haufe"):
    sub = [r for r in arms if r["editor"] == ed]
    bb = max(sub, key=lambda r: r["edit_index"])
    g = [r for r in sub if r["fidelity_ratio"] <= 1.1]
    bg = max(g, key=lambda r: r["edit_index"]) if g else None
    out[ed] = {"best": bb, "guarded": bg}
    print(f"{ed}: BEST {bb['edit_index']:+.3f}/fid {bb['fidelity_ratio']:.2f} (pt{bb['point']} α{bb['alpha']} {bb['dims']})"
          + (f" | guarded {bg['edit_index']:+.3f}/{bg['fidelity_ratio']:.2f}" if bg else " | none within fid 1.1"), flush=True)
(REPO / "experiments/edit_direction_alignment/scores/haufe_edit_tokens.json").write_text(json.dumps(out, indent=1, default=float))
