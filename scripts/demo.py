#!/usr/bin/env python3
"""
Demo: generate and animate one toy-world scene.

Usage
-----
    python scripts/demo.py
    python scripts/demo.py --seed 7 --n-objects 4 --frames 80
    python scripts/demo.py --save outputs/demo.gif
    python scripts/demo.py --direction-noise 0.05 --speed-noise 0.03
    python scripts/demo.py --obs-res 256 --interval 30
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from pim.config import SimConfig
from pim.renderer import render_scene
from pim.sim import simulate
from pim.viz import animate_scene, save_animation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Physically-implicit-modeling demo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-objects", type=int, default=3)
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--obs-res", type=int, default=128)
    p.add_argument("--boundary", choices=["bounce", "open", "wrap"], default="bounce")
    p.add_argument(
        "--direction-noise",
        type=float,
        default=0.0,
        help="Velocity angle noise per step (radians); 0=straight, ~0.05=gentle curves",
    )
    p.add_argument(
        "--speed-noise",
        type=float,
        default=0.0,
        help="Fractional speed noise per step; 0=constant, ~0.05=varying",
    )
    p.add_argument(
        "--position-noise",
        type=float,
        default=0.0,
        help="Position diffusion std per step (world units); Brownian jitter on top of drift",
    )
    p.add_argument(
        "--obs-noise-std",
        type=float,
        default=0.04,
        help="Observation noise std (intensity units); 0 = no noise",
    )
    p.add_argument(
        "--waterfall-mode",
        choices=["model", "human"],
        default="model",
        help="Waterfall display: 'model'=grayscale intensity, 'human'=color+depth",
    )
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="Output path, e.g. outputs/demo.gif or outputs/demo.mp4",
    )
    p.add_argument("--fps", type=int, default=20)
    p.add_argument(
        "--interval", type=int, default=50, help="Milliseconds between animation frames"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = SimConfig(
        seed=args.seed,
        n_objects=args.n_objects,
        n_frames=args.frames,
        obs_res=args.obs_res,
        boundary=args.boundary,
        direction_noise_std=args.direction_noise,
        speed_noise_std=args.speed_noise,
        position_noise_std=args.position_noise,
        obs_noise_std=args.obs_noise_std,
    )

    print(
        f"simulating   seed={cfg.seed}  objects={cfg.n_objects}  frames={cfg.n_frames}"
    )
    scene = simulate(cfg)

    print("rendering 1D observations...")
    obs_depth, obs_id, obs_intensity = render_scene(scene)

    print("building animation...")
    anim = animate_scene(
        scene,
        obs_depth,
        obs_id,
        obs_intensity,
        interval=args.interval,
        title=f"seed {cfg.seed}  ·  {scene.positions.shape[1]} objects",
        waterfall_mode=args.waterfall_mode,
    )

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_animation(anim, str(out), fps=args.fps)
    else:
        plt.show()


if __name__ == "__main__":
    main()
