#!/usr/bin/env python3
"""
Generate a dataset of toy-world observation sequences.

Usage
-----
    python scripts/generate_dataset.py data/train.h5
    python scripts/generate_dataset.py data/train.h5 --n-samples 100000 --n-workers 8
    python scripts/generate_dataset.py data/small.h5 --n-samples 1000 --n-objects 3
    python scripts/generate_dataset.py data/wrap.h5  --boundary wrap --n-samples 50000
"""

import argparse

from pim.config import SimConfig
from pim.dataset import DatasetConfig, generate_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a pim dataset")
    p.add_argument("output_dir", help="Output directory (created if absent; must be empty if it exists)")

    # dataset scale
    p.add_argument("--n-samples",   type=int, default=100_000)
    p.add_argument("--seed",        type=int, default=0,   help="Base RNG seed")
    p.add_argument("--n-workers",   type=int, default=4,   help="Worker processes (0 = single-process)")

    # simulation
    p.add_argument("--n-objects",   type=int, default=3)
    p.add_argument("--frames",      type=int, default=100)
    p.add_argument("--obs-res",     type=int, default=128)
    p.add_argument("--boundary",        choices=["bounce", "open", "wrap"], default="bounce")
    p.add_argument("--direction-noise", type=float, default=0.0,
                   help="Velocity angle noise per step (radians); 0=straight, ~0.05=gentle curves")
    p.add_argument("--speed-noise",     type=float, default=0.0,
                   help="Fractional speed noise per step; 0=constant, ~0.05=varying")
    p.add_argument("--position-noise",  type=float, default=0.0,
                   help="Position diffusion std per step (world units); Brownian jitter on top of drift")
    p.add_argument("--obs-noise-std", type=float, default=0.04,
                   help="Observation noise std (intensity units); 0 = no noise")

    # storage
    p.add_argument("--compression-level", type=int, default=4,
                   help="gzip compression level 0–9 (default 4)")
    p.add_argument("--write-batch", type=int, default=512,
                   help="Samples buffered in RAM before each HDF5 flush")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sim = SimConfig(
        n_objects=args.n_objects,
        n_frames=args.frames,
        obs_res=args.obs_res,
        boundary=args.boundary,
        direction_noise_std=args.direction_noise,
        speed_noise_std=args.speed_noise,
        position_noise_std=args.position_noise,
        obs_noise_std=args.obs_noise_std,
    )
    dcfg = DatasetConfig(
        n_samples=args.n_samples,
        sim=sim,
        base_seed=args.seed,
        n_workers=args.n_workers,
        write_batch=args.write_batch,
        compression_level=args.compression_level,
    )

    generate_dataset(dcfg, args.output_dir)


if __name__ == "__main__":
    main()
