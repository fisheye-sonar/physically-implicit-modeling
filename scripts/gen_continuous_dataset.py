"""Generate one continuous-action dataset (train split) matched to dataset 4.

Base seeds match dataset-4 (train base_seed=0) so base trajectories are
byte-identical; only the continuous actions + resulting moved renders differ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pim.simulator.actions_continuous import (
    generate_continuous_action_dataset,
)  # noqa: E402
from pim.simulator.config import SimConfig  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["dxdy", "teleport", "axis_x"])
    p.add_argument("--out", required=True, help="output .h5 path")
    p.add_argument("--n-samples", type=int, default=90000)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--move-scale", type=float, default=4.0)
    p.add_argument("--p-action", type=float, default=0.30)
    p.add_argument("--n-workers", type=int, default=16)
    a = p.parse_args()

    d4 = json.load(
        open(
            Path(__file__).resolve().parent.parent
            / "datasets/4_fixed_refl_inview/dataset.json"
        )
    )
    # Take only the fields dataset 4 actually stored, and let the rest of SimConfig default.
    # Dataset 4 predates the soft-render (`soft_edge`, …) and omniscient-2D (`omni2d`, …) fields,
    # so iterating over every dataclass field raises KeyError on the newer ones. Those all
    # default to OFF, which is exactly what "matched to dataset 4" means.
    stored = {k: v for k, v in d4["sim"].items() if k in SimConfig.__dataclass_fields__}
    sim = SimConfig(**stored)
    missing = sorted(set(SimConfig.__dataclass_fields__) - set(stored))
    if missing:
        print(
            f"[gen] fields not stored by dataset 4, left at their defaults: {missing}"
        )

    generate_continuous_action_dataset(
        a.out,
        sim,
        mode=a.mode,
        n_samples=a.n_samples,
        base_seed=a.base_seed,
        move_scale=a.move_scale,
        p_action=a.p_action,
        n_workers=a.n_workers,
    )


if __name__ == "__main__":
    main()
