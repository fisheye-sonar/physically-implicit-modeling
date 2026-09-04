"""Noiseless discworld scenes rendered at a few ray counts — one animation per (seed, N).

Companion to the 2026-09-03 count of reachable observations (T(N) ≈ N⁴/6 for two
discs). World geometry is the CANONICAL generation geometry, untouched (SimConfig
defaults: y 3→12, x_near 1.5, x_far 6.0 — a pinhole frustum with slope x/y = 0.5, the
same value the ray-caster uses as its FOV). Sim config otherwise = dw-noiseless: 2
objects, 40 frames, fixed reflectivities, always_in_frustum, open boundary, no noise;
`simulate` rejects any trajectory with a collision. Pure caller of
`pim.environments.discworld` — nothing canonical is touched.

    # the original four: seed 7, N = 4 8 16 32
    PYTHONPATH=. .pim/bin/python experiments/ray_count_waterfalls/scripts/make_ray_waterfalls.py
    # bigger, faster discs (radius 1.0, speed 0.08-0.20) -> *_big.gif
    PYTHONPATH=. .pim/bin/python experiments/ray_count_waterfalls/scripts/make_ray_waterfalls.py --big
    # ten examples at ten rays, big discs -> waterfall_N10_big_seed<k>.gif
    PYTHONPATH=. .pim/bin/python experiments/ray_count_waterfalls/scripts/make_ray_waterfalls.py \
        --big --rays 10 --seeds 100 101 102 103 104 105 106 107 108 109
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from pim.environments.discworld.config import SimConfig  # noqa: E402
from pim.environments.discworld.renderer import render_scene  # noqa: E402
from pim.environments.discworld.sim import simulate  # noqa: E402
from pim.environments.discworld.viz import animate_scene, save_animation  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true",
                    help="double the disc radius (0.5 -> 1.0) and raise speeds (0.05-0.12 -> 0.08-0.20)")
    ap.add_argument("--rays", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    args = ap.parse_args()
    radius, speed = (1.0, (0.08, 0.20)) if args.big else (0.5, (0.05, 0.12))
    tag = "_big" if args.big else ""
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for n in args.rays:
            cfg = SimConfig(seed=seed, n_objects=2, n_frames=40, obs_res=n, boundary="open",
                            fixed_reflectivities=True, always_in_frustum=True,
                            obs_noise_std=0.0, position_noise_std=0.0,
                            radius=radius, speed_min=speed[0], speed_max=speed[1],
                            max_gen_attempts=5000)  # r=1.0 discs fail the 300 default
            scene = simulate(cfg)
            depth, ids, inten = render_scene(scene)
            distinct = np.unique(inten, axis=0).shape[0]
            anim = animate_scene(scene, depth, ids, inten, interval=100,
                                 title=f"N = {n} rays  ·  seed {seed}  ·  r={radius}  "
                                       f"v={speed[0]}-{speed[1]}  ·  noiseless  ·  "
                                       f"{distinct}/{cfg.n_frames} distinct frames")
            # the original single-seed files keep their short names
            name = (f"waterfall_N{n:02d}{tag}.gif" if args.seeds == [7]
                    else f"waterfall_N{n:02d}{tag}_seed{seed}.gif")
            save_animation(anim, str(OUT / name), fps=10, dpi=90)
            print(f"seed {seed}  N={n:2d}: {distinct} distinct frames of {cfg.n_frames}")


if __name__ == "__main__":
    main()
