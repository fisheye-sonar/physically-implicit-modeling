"""Tiny end-to-end smoke of both baseline floors — wiring only, not a result."""
import sys
from pathlib import Path

from pim.environments.discworld import bench as dwb
from pim.environments.othello import arms as oa
from pim.environments.othello import corpus as oc
from pim.environments.othello.data import canonical_vocab, tokens_and_labels
from pim.probes.baselines import random_init_model

N, CD = 500, Path(sys.argv[1])
print("--- discworld observation ---", flush=True)
for fam in ("linear", "mlp"):
    _, st = dwb.observation_probes(target="full", n_seq=N, family=fam,
                                   basis_name="frustum", span=39,
                                   data_dir="datasets/discworld/dw-pn04/probe",
                                   cache_dir=CD, log=print)
    print(f"  {fam}: skill {st['r2']:+.4f}  in-sample {st['r2_insample']:+.4f}  "
          f"gap {st['r2_insample']-st['r2']:+.4f}  d_in {st['d_in']}  rows {st['n_train_rows']}",
          flush=True)

print("--- discworld random-init ---", flush=True)
m, info = __import__("pim.models", fromlist=["load_checkpoint"]).load_checkpoint(
    "runs/initial_othello_comparison/L-dw-20m/best_model.pt", device="cpu")
rnd = random_init_model(info.arch, info.model_config, seed=0, device=dwb.DEV)
f = dwb.fit_probes(rnd, target="full", n_seq=N, family="linear", basis_name="frustum",
                   data_dir="datasets/discworld/dw-pn04/probe", cache_dir=CD, log=None)
print(f"  linear best-point skill {max(v[1]['r2'] for v in f.values()):+.4f}", flush=True)

print("--- othello observation ---", flush=True)
tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",))["probe"])
itos = {v: k for k, v in canonical_vocab().items()}
data = tokens_and_labels([[itos[int(t)] for t in r[:L]] for r, L in zip(tok[:N], ln[:N])])
for fam in ("linear", "mlp"):
    _, st = oa.observation_probes(data, family=fam, cache_dir=CD, log=print)
    print(f"  {fam}: err {st['error_rate']:.2f}%  majority {st['majority_class_error_rate']:.2f}%"
          f"  skill {1-st['error_rate']/st['majority_class_error_rate']:+.4f}"
          f"  d_in {st['d_in']}  rows {st['n_train_rows']}", flush=True)
print("SMOKE OK", flush=True)
