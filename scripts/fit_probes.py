#!/usr/bin/env python3
"""Fit a discworld run's probes for a PROBE TARGET, deliberately — the one place an extra
target's probes get fitted (2026-09-09).

    python scripts/fit_probes.py --run ray_ablation/L-dw-8ray-20m --target appearance
    python scripts/fit_probes.py --run interface_ablation/L-dw-8ray-tok-20m --target appearance
    python scripts/fit_probes.py --run ray_ablation/L-dw-8ray-20m --target appearance --random-init
    python scripts/fit_probes.py --run ray_ablation/L-dw-8ray-20m --target appearance --observation

The canonical scorer (``notebooks/master_eval.ipynb``) NEVER fits an extra target's probes:
it asks for them with ``require_cached`` and skips the block until they exist. This script
fits them with the target's recipe (``pim.environments.discworld.arms.probe_recipe``: the
large probe split, 200k sequences, 50 epochs for every categorical target) and persists
them where the scorer looks —

    the trained model's probes  → runs/<topic>/<run>/probes/
    --random-init               → runs/_baselines/<instance>/probes/  (the same architecture,
                                  seeded, untrained: the random-init floor)
    --observation               → runs/_baselines/<instance>/probes/  (the right-aligned
                                  causal-history probe on the same corpus: the observation floor)

A frames-as-tokens run is probed on TOKEN inputs through the run's own vocabulary, exactly
as the scorer does. Every fit is a cache hit the second time, so re-running is free.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from pim.environments.discworld import arms as dwa  # noqa: E402
from pim.environments.discworld.grid_target import categorical_target  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.baselines import random_init_model  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="runs/<topic>/<run>")
    p.add_argument("--target", required=True, help='probe target, e.g. "appearance", "grid-16x8"')
    p.add_argument("--basis", default="frustum")
    p.add_argument("--families", nargs="+", default=("linear", "mlp"))
    p.add_argument("--random-init", action="store_true", help="fit the random-init floor instead")
    p.add_argument("--observation", action="store_true", help="fit the observation floor instead")
    p.add_argument("--n-seq", type=int, default=None, help="override the recipe (smoke only)")
    p.add_argument("--epochs", type=int, default=None, help="override the recipe (smoke only)")
    p.add_argument("--cache-dir", default=None, help="override where the probes go (smoke only)")
    a = p.parse_args()

    run_dir = _REPO / "runs" / a.run
    cfg = json.loads((run_dir / "config.json").read_text())
    inst = cfg["data"]["instance"]
    inst_root = _REPO / "datasets" / "discworld" / inst
    recipe = dwa.probe_recipe(a.target, inst_root)
    if a.n_seq:
        recipe["n_seq"] = a.n_seq
    if a.epochs:
        recipe["epochs"] = a.epochs
    if categorical_target(a.target) is None:
        raise SystemExit(f"{a.target!r} is not a categorical target; the regression probes are "
                         f"fitted by the scorer itself")

    model, info = load_checkpoint(run_dir / "best_model.pt", device="cpu")
    enc = {}
    if info.arch.endswith("_tokens"):                       # probe a token model on token inputs
        from pim.environments.discworld import token_bench as tkb
        from pim.environments.discworld.tokens import FrameVocab

        _e, _tag = tkb.token_encoder(FrameVocab.load(run_dir / "vocab.npz"))
        enc = {"encoder": _e, "encoder_tag": _tag}
    if a.random_init:
        model = random_init_model(info.arch, info.model_config, seed=0, device=DEV)
        cache_dir = _REPO / "runs" / "_baselines" / inst / "probes"
        what = f"random-init {info.arch}"
    elif a.observation:
        model = None
        cache_dir = _REPO / "runs" / "_baselines" / inst / "probes"
        what = "observation floor (right-aligned)"
    else:
        model = model.to(DEV)
        cache_dir = run_dir / "probes"
        what = a.run
    if a.cache_dir:
        cache_dir = Path(a.cache_dir)
    print(f"{what} · target {a.target} · basis {a.basis} · recipe {recipe} → {cache_dir}", flush=True)

    for fam in a.families:
        t0 = time.time()
        if a.observation:
            span = int(getattr(load_checkpoint(run_dir / "best_model.pt", device="cpu")[0],
                               "state_span", 39))
            _, st = dwa.observation_probes(target=a.target, family=fam, basis_name=a.basis,
                                           span=span, cache_dir=cache_dir, align="right",
                                           log=print, **recipe)
            print(f"  {fam}: skill {1 - st['error_rate'] / st['majority_class_error_rate']:+.4f}"
                  f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        else:
            fits = dwa.fit_probes(model, target=a.target, family=fam, basis_name=a.basis,
                                  cache_dir=cache_dir, log=print, **recipe, **enc)
            best = max(fits, key=lambda e: 1 - fits[e][1]["error_rate"]
                       / fits[e][1]["majority_class_error_rate"])
            st = fits[best][1]
            print(f"  {fam}: best point {best} skill "
                  f"{1 - st['error_rate'] / st['majority_class_error_rate']:+.4f}"
                  f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
