#!/usr/bin/env python
"""The Bayes floor of an environment instance → runs/_baselines/<instance>/bayes_floor.json.

    .pim/bin/python scripts/bayes_floor.py                       # the paper's ten instances
    .pim/bin/python scripts/bayes_floor.py --instance dw-8ray    # one
    .pim/bin/python scripts/bayes_floor.py --instance dw-8ray --smoke   # 20 sequences, 64 particles (CPU-sized)

Othello: EXACT (``pim.environments.othello.bayes`` — seconds per instance, no GPU). Discworld:
ESTIMATED by posterior sampling over the initial state (``pim.environments.discworld.bayes`` —
GPU; minutes per instance at the defaults, 1000 sequences × 512 particles × 40 sweeps), reported
as a bracket with standard errors and the sampler's own diagnostics. Model-free: nothing here
loads a run. An existing file at the current version is skipped unless ``--force``; a replaced
file is first copied beside itself with its date (nothing under runs/ is deleted).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

PAPER_INSTANCES = ("oth-uniform", "oth-adjacent-flip", "oth-adjacent", "oth-noflip",
                   "dw-noiseless", "dw-blink", "dw-128ray", "dw-16ray", "dw-8ray", "dw-5ray")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--instance", nargs="*", default=list(PAPER_INSTANCES))
    ap.add_argument("--n-seq", type=int, default=None)
    ap.add_argument("--particles", type=int, default=512)
    ap.add_argument("--sweeps", type=int, default=40)
    ap.add_argument("--init-sweeps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny settings, written to bayes_floor.smoke.json")
    ap.add_argument("--no-sampler", action="store_true",
                    help="discworld: write ONLY the trivial predictors (CPU, seconds). The file is versioned "
                         "'<version>+trivial-only', so a later full run replaces it instead of skipping it")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for inst in a.instance:
        out = _REPO / "runs" / "_baselines" / inst / ("bayes_floor.smoke.json" if a.smoke else "bayes_floor.json")
        if inst.startswith("oth-"):
            from pim.environments.othello import bayes as ob

            version, run = ob.FLOOR_VERSION, lambda: ob.bayes_floor(inst, **({"n_games": 200} if a.smoke else {}))   # noqa: E731
        else:
            from pim.environments.discworld import bayes as db

            kw = ({"n_seq": 20, "particles": 64, "sweeps": 4, "init_sweeps": 50, "exact_draws": 1_000_000} if a.smoke else
                  {"n_seq": a.n_seq or db.N_FLOOR_SEQ, "particles": a.particles, "sweeps": a.sweeps,
                   "init_sweeps": a.init_sweeps})
            version, run = db.FLOOR_VERSION, lambda: db.bayes_floor(inst, seed=a.seed, device=a.device, **kw)   # noqa: E731
            if a.no_sampler:
                version = db.FLOOR_VERSION + "+trivial-only"
                run = lambda: {"instance": inst, "version": version, "split": "eval/test.h5",                  # noqa: E731
                               "method": "trivial predictors only — the floor itself has not been sampled yet",
                               "trivial": db.trivial_predictors(inst, n_seq=a.n_seq or db.N_FLOOR_SEQ)}
        if out.exists() and not a.force and json.loads(out.read_text()).get("version") == version:
            print(f"{inst}: {out.relative_to(_REPO)} is current — skipped", flush=True)
            continue
        res = run()
        res["created"] = time.strftime("%Y-%m-%d %H:%M")
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            shutil.copy2(out, out.with_name(f"{out.stem}.{time.strftime('%Y-%m-%d_%H%M')}.json"))
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(res, indent=1, default=float))
        os.replace(tmp, out)
        print(f"{inst}: wrote {out.relative_to(_REPO)}", flush=True)


if __name__ == "__main__":
    main()
