"""Position decodability ON HIDDEN FRAMES, for all three probe sources (2026-09-10, Sevan).

The blink finding reports the TRAINED model's read of a hidden object (MLP 0.97–0.99, LIN
0.63–0.89) against the canonical floors — which are whole-split numbers. This script puts the
two floors on the same hidden frames: the OBSERVATION probes (right-aligned history — they
still see the frames before the blackout, so they can extrapolate — and left-aligned) and the
RANDOM-INIT model's probes, all read from the cache (nothing is refitted), all evaluated on
the same 4,000 held-out probe-split sequences and the same frame masks as the finding's table
(`subset_editability.py`): skill = 1 − SSE / SSE(mean) per object over the masked frames,
frustum basis, best residual point by whole-split R² for the model probes. Contained.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import h5py, numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments import layout  # noqa: E402
from pim.environments.discworld import arms as dwa  # noqa: E402
from pim.environments.discworld.bench import N_OBJ, _to_basis  # noqa: E402
from pim.metrics.decodability import probe_skill_regression  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.base import collect_residuals  # noqa: E402
from pim.probes.baselines import CausalHistory, random_init_model  # noqa: E402

DEV = "cuda"
RUN = REPO / "runs/blink_ablation/L-dw-blink-20m"
INST = "dw-blink"; BASIS = "frustum"; N_SEQ, START = 4000, 30_000     # the fits used [0, 30k)
model, info = load_checkpoint(RUN / "best_model.pt", device=DEV); model.eval()
span = int(getattr(model, "state_span", 39))
with h5py.File(layout.probe_file("discworld", INST, "120k"), "r") as f:
    obs = f["obs_intensity"][START:START + N_SEQ, :span].astype(np.float32)
    pos = f["positions"][START:START + N_SEQ, :span, :N_OBJ, :].astype(np.float32)
    vel = f["velocities"][START:START + N_SEQ, :span, :N_OBJ, :].astype(np.float32)
    vis = f["blink_visible"][START:START + N_SEQ, :span, :N_OBJ].astype(bool)
sim = json.load(open(layout.probe_manifest("discworld", INST, "120k")))["sim"]
bp, bv = _to_basis(pos, vel, sim, BASIS)
y = np.concatenate([bp.reshape(N_SEQ, span, -1), bv.reshape(N_SEQ, span, -1)], -1)   # (N, T, 8)
since = np.zeros(vis.shape, int)
for t in range(1, span):
    since[:, t] = np.where(vis[:, t], 0, since[:, t - 1] + 1)
print(f"{N_SEQ} held-out sequences × {span} frames; hidden frames per object: "
      f"{(~vis[:, :, 0]).sum()} / {(~vis[:, :, 1]).sum()}", flush=True)

def skills(pred):
    """pred (N, T, 8) → per object: visible / hidden / by frames-since-seen."""
    row = {}
    for j in range(N_OBJ):
        d = [2 * j, 2 * j + 1]
        yy, pp = y[:, :, d], pred[:, :, d]
        mean = yy.reshape(-1, 2).mean(0)
        sk = lambda m: (float(probe_skill_regression(pp[m], yy[m], mean)) if m.sum() >= 50 else None)  # noqa: E731
        row[f"obj{j}"] = {"visible": sk(vis[:, :, j]), "hidden": sk(~vis[:, :, j]),
                          "since": {int(s): sk(since[:, :, j] == s) for s in (1, 3, 6, 10)}}
    return row

def model_pred(m, probes):
    ell = max(probes, key=lambda e: probes[e][1]["r2"])
    R = collect_residuals(m, obs, batch=64, points=[ell])[0]                      # (N, T, d)
    with torch.no_grad():
        pred = np.concatenate([probes[ell][0](torch.from_numpy(R[i:i + 256]).to(DEV).reshape(-1, R.shape[-1]))
                               .cpu().numpy().reshape(-1, span, y.shape[-1]) for i in range(0, N_SEQ, 256)])
    return pred, int(ell), float(probes[ell][1]["r2"])

def obs_pred(probe, align):
    hist = CausalHistory(torch.from_numpy(obs).to(DEV), align=align)
    out = np.zeros((N_SEQ, span, y.shape[-1]), np.float32)
    fr = torch.arange(span, device=DEV)
    with torch.no_grad():
        for i in range(0, N_SEQ, 32):
            seq = torch.arange(i, min(i + 32, N_SEQ), device=DEV)
            S, Fm = torch.meshgrid(seq, fr, indexing="ij")
            X = hist.build(S.reshape(-1), Fm.reshape(-1))
            out[i:i + len(seq)] = probe(X).cpu().numpy().reshape(len(seq), span, -1)
    return out

out = {"run": str(RUN.relative_to(REPO)), "basis": BASIS, "n_seq": N_SEQ, "sources": {}}
probe_small = {"instance": INST, "size": "120k"}; probe_large = {"instance": INST, "size": "250k"}
rand = random_init_model(info.arch, info.model_config, seed=0, device=DEV); rand.eval()
bdir = REPO / "runs/_baselines" / INST / "probes"
for fam in ("linear", "mlp"):
    # trained and random-init model probes (cached: the run's probes/, the instance's baseline probes/)
    for name, m, cdir in (("trained", model, RUN / "probes"), ("random_init", rand, bdir)):
        pr = dwa.fit_probes(m, target="full", n_seq=30_000, family=fam, basis_name=BASIS, probe=probe_small,
                            cache_dir=cdir, log=None, require_cached=True)
        pred, ell, r2 = model_pred(m, pr)
        out["sources"][f"{name}/{fam}"] = {"point": ell, "whole_split_r2": r2, **skills(pred)}
    # observation floors: right-aligned (matched 30k and large 250k/50 epochs) and left-aligned large
    for name, align, n_seq, probe, epochs in (("observation_right", "right", 30_000, probe_small, None),
                                              ("observation_right_large", "right", 250_000, probe_large, 50),
                                              ("observation_large", "left", 250_000, probe_large, 50)):
        probe_, st = dwa.observation_probes(target="full", n_seq=n_seq, family=fam, basis_name=BASIS, span=span,
                                            probe=probe, cache_dir=bdir, log=None, epochs=epochs, align=align,
                                            require_cached=True)
        out["sources"][f"{name}/{fam}"] = {"whole_split_r2": float(st["r2"]), **skills(obs_pred(probe_, align))}
f = lambda v: "   —  " if v is None else f"{v:+.3f}"  # noqa: E731
print(f"\n{'source':28s} {'obj':>3} {'visible':>8} {'hidden':>8} | since 1 {'3':>7} {'6':>7} {'10':>7}")
for k, r in out["sources"].items():
    for j in range(N_OBJ):
        o = r[f"obj{j}"]
        print(f"{k:28s} {j:>3} {f(o['visible']):>8} {f(o['hidden']):>8} | {f(o['since'][1])} {f(o['since'][3])} "
              f"{f(o['since'][6])} {f(o['since'][10])}")
(REPO / "experiments/blink_ablation/scores/hidden_frame_floors.json").write_text(json.dumps(out, indent=1))
