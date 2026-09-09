"""Alignment for the FRAMES-AS-TOKENS discworld run (L-dw-8ray-tok-20m).

Same measurement as `discworld_alignment.py`, but the model consumes frame TOKEN ids, so the
counterfactual history is rendered as frames and then encoded through the run's own frame
vocabulary. Two extra validity numbers, because the encoding can fail where the float path
cannot: the fraction of counterfactual frames that fall OUTSIDE the vocabulary (UNK), and the
probability the model puts on the true edited frame from the counterfactual history vs the
real one.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import subspace_fracs, zspace  # noqa: E402

from pim.environments.discworld import arms as dwa, bench as dwb  # noqa: E402
from pim.environments.discworld.bench import EF, N_OBJ  # noqa: E402
from pim.environments.discworld.renderer import render_frame  # noqa: E402
from pim.environments.discworld.sim import fully_in_frustum  # noqa: E402
from pim.environments.discworld import token_bench as tkb  # noqa: E402
from pim.environments.discworld.tokens import UNK, FrameVocab, encode  # noqa: E402
from pim.metrics.zone_editability import sim_config_from  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.base import collect_residuals  # noqa: E402

EXP = REPO / "experiments" / "edit_direction_alignment"; DEV = "cuda"
ap = argparse.ArgumentParser(); ap.add_argument("--run", default="runs/interface_ablation/L-dw-8ray-tok-20m")
a_ = ap.parse_args(); run_dir = REPO / a_.run
inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
root = REPO / "datasets/discworld" / inst
model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV); NP = n_points(model)
vocab = FrameVocab.load(run_dir / "vocab.npz"); enc, tag = tkb.token_encoder(vocab)

arr = dwb.bench_arrays(n=192, target="full", basis_name="frustum", data_dir=root / "eval")
sim = arr["sim"]; cfg = sim_config_from(sim, N_OBJ)
pos, vel, eobj = arr["pos"], arr["vel"], arr["edit_object"]
n = len(eobj); ar = np.arange(n); dt = float(sim["dt"])
delta = pos[ar, EF, eobj] - (pos[ar, EF - 1, eobj] + vel[ar, EF - 1, eobj] * dt)
cf_pos = pos[:, :EF].copy(); cf_pos[ar, :, eobj] += delta[:, None, :]
ms = cfg.collision_margin * 2.0 * cfg.radius
keep = np.array([i for i in range(n) if fully_in_frustum(cf_pos[i], cfg.radius, cfg)
                 and (np.linalg.norm(cf_pos[i, :, 0] - cf_pos[i, :, 1], axis=-1) >= ms).all()])
refl = np.linspace(sim["refl_min"], sim["refl_max"], N_OBJ).astype(np.float32)
rad = np.full(N_OBJ, sim["radius"], np.float32)
obs_cf = np.stack([np.stack([render_frame(cf_pos[i, f].astype(np.float32), rad, refl, cfg)[2]
                             for f in range(EF)]) for i in keep]).astype(np.float32)
obs = arr["obs"][keep, :EF].astype(np.float32)
tok, tok_cf = enc(obs), enc(obs_cf)
unk_real, unk_cf = float((tok == UNK).mean()), float((tok_cf == UNK).mean())
# validity: probability on the TRUE edited frame token, from the cf history vs the real one
gt_tok = enc(arr["gt_roll"][keep, 0][:, None, :])[:, 0]
with torch.no_grad():
    lg_cf = model.decode(torch.from_numpy(tok_cf.astype(np.int64)).to(DEV))
    lg_or = model.decode(torch.from_numpy(tok.astype(np.int64)).to(DEV))
    p_cf = torch.softmax(lg_cf, -1)[torch.arange(len(keep)), torch.from_numpy(gt_tok.astype(np.int64)).to(DEV)].mean().item()
    p_or = torch.softmax(lg_or, -1)[torch.arange(len(keep)), torch.from_numpy(gt_tok.astype(np.int64)).to(DEV)].mean().item()
print(f"{a_.run}: {len(keep)}/{n} valid counterfactuals | UNK frames: real {unk_real:.3%}, cf {unk_cf:.3%}", flush=True)
print(f"  VALIDITY: p(true edited frame) from the cf history {p_cf:.3f} vs from the real history {p_or:.3f}", flush=True)

R = collect_residuals(model, tok.astype(np.int64), batch=64)[:, :, -1]
Rcf = collect_residuals(model, tok_cf.astype(np.int64), batch=64)[:, :, -1]
lin = dwa.fit_probes(model, target="full", n_seq=30_000, family="linear", basis_name="frustum",
                     data_dir=root / "probe", cache_dir=run_dir / "probes", log=None,
                     encoder=enc, encoder_tag=tag)
import h5py
with h5py.File(root / "probe/test.h5", "r") as f:
    obs_cov = f["obs_intensity"][30_000:31_500, :39].astype(np.float32)
Rc = collect_residuals(model, enc(obs_cov).astype(np.int64), batch=64)
out = {"run": a_.run, "instance": inst, "n_cases": int(len(keep)), "unk_cf": unk_cf, "unk_real": unk_real,
       "p_edited_frame_cf": p_cf, "p_edited_frame_real": p_or, "layers": {}}
rng = np.random.default_rng(0)
for ell in range(NP):
    probe = lin[ell][0]
    z, zcf = zspace(probe, torch.from_numpy(R[ell]).to(DEV)), zspace(probe, torch.from_numpy(Rcf[ell]).to(DEV))
    dz = zcf - z
    cov = torch.cov(zspace(probe, torch.from_numpy(Rc[ell].reshape(-1, Rc.shape[-1])).to(DEV)).T)
    W = probe.net.weight.detach()
    fr = subspace_fracs(dz, W[:4], cov)
    dgen = z[torch.from_numpy(rng.permutation(len(keep))).to(DEV)] - z
    fg = subspace_fracs(dgen, W[:4], cov)
    Y = dz.cpu().numpy(); s_ = np.linalg.svd(Y - Y.mean(0), compute_uv=False); ev = s_ ** 2 / (s_ ** 2).sum()
    out["layers"][str(ell)] = {"rows": float(fr["rows"].mean()), "generic_rows": float(fg["rows"].mean()),
                               "haufe": float(fr["haufe"].mean()), "generic_haufe": float(fg["haufe"].mean()),
                               "rank90": int(np.searchsorted(np.cumsum(ev), 0.9) + 1),
                               "delta_norm_over_h_norm": float((dz.norm(dim=1) / z.norm(dim=1)).mean())}
    L = out["layers"][str(ell)]
    print(f"pt {ell}: rows {L['rows']:.3f}/gen {L['generic_rows']:.3f}  haufe {L['haufe']:.3f}/gen {L['generic_haufe']:.3f}"
          f"  rank90 {L['rank90']}  |Δ|/|h| {L['delta_norm_over_h_norm']:.2f}", flush=True)
(EXP / "scores" / "discworld_alignment_dw-8ray-tokens.json").write_text(json.dumps(out, indent=1))
