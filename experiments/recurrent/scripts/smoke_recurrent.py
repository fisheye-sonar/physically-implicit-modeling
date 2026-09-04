"""Eval smoke for a NEW ARCHITECTURE: every path master_eval will take, on the 200-step
smoke checkpoint, at n_seq=500, in a scratch cache. Numbers are meaningless; crashes are not."""
import sys
from pathlib import Path

import torch

from pim.environments.discworld import bench as dwb
from pim.models import load_checkpoint, n_points
from pim.probes.baselines import random_init_model
from pim.probes.mlp import check_probe_sanity

RUN, CD, N = Path(sys.argv[1]), Path(sys.argv[2]), 500
root = Path("datasets/discworld/dw-pn04")
m, info = load_checkpoint(RUN / "best_model.pt", device=dwb.DEV)
print(f"arch {info.arch}  points {n_points(m)}  span {m.state_span}  "
      f"params {sum(p.numel() for p in m.parameters()):,}  val {info.val_loss:.5f}", flush=True)
b = dwb.load_bench(m, n=32, target="full", basis_name="frustum", data_dir=root / "eval")
lin = dwb.fit_probes(m, target="full", n_seq=N, family="linear", basis_name="frustum",
                     data_dir=root / "probe", cache_dir=CD, log=None)
mlp = dwb.fit_probes(m, target="full", n_seq=N, family="mlp", basis_name="frustum",
                     data_dir=root / "probe", cache_dir=CD, log=None)
check_probe_sanity(lin, mlp, strict=False, log=print, label="smoke")
u = dwb.unsteered(m, b)
print(f"unedited EI {u['edit_index']:+.3f}", flush=True)
arms = []
for dims in ("pos", "all"):
    arms += dwb.pinv_arm(m, b, lin, (1.0, 20.0), dims=dims)
    arms += dwb.nanda_arm(m, b, lin[1][0], 1, (0.5,), dims=dims)
    arms += dwb.grad_steer_arm(m, b, mlp, (0,), (0.1,), n_steps=5, dims=dims)
for r in arms:
    r["fidelity_ratio"] = dwb.fidelity_ratio(r, u)
for ed in ("PI", "ND", "GS"):
    bb = max((r for r in arms if r["editor"].startswith(ed)), key=lambda r: r["edit_index"])
    print(f"  {ed}: EI {bb['edit_index']:+.3f} fid {bb['fidelity_ratio']:.2f} dims {bb['dims']}", flush=True)
_, st = dwb.observation_probes(target="full", n_seq=N, family="linear", basis_name="frustum",
                               span=int(m.state_span), data_dir=root / "probe", cache_dir=CD, log=None)
print(f"observation floor (LIN, n=500): {st['r2']:+.3f}  d_in {st['d_in']}", flush=True)
rand = random_init_model(info.arch, info.model_config, seed=0, device=dwb.DEV)
f = dwb.fit_probes(rand, target="full", n_seq=N, family="linear", basis_name="frustum",
                   data_dir=root / "probe", cache_dir=CD, log=None)
print(f"random-init floor (LIN, n=500): {max(v[1]['r2'] for v in f.values()):+.3f}", flush=True)
roll = dwb.free_rollout(m, b.obs[:4], 20, 5)
print("free rollout", roll.shape, "| SMOKE OK", flush=True)
