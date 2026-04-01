#!/usr/bin/env python3
"""
Generate a counterfactual edits dataset.

Sequences are identical to the normal dataset, except halfway through one object
is teleported to a new position.  The edit metadata (frame, object, new position)
is stored alongside the modified observations.

Usage
-----
    python scripts/generate_edits_dataset.py data/edits_train
    python scripts/generate_edits_dataset.py data/edits_train --n-samples 50000 --n-workers 8
    python scripts/generate_edits_dataset.py data/edits_small --n-samples 1000 --edit-frame 15
"""

import argparse

from pim.config import SimConfig
from pim.edits_dataset import EditDatasetConfig, generate_edits_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a pim edits dataset")
    p.add_argument(
        "output_dir",
        help="Output directory (created if absent; must be empty if it exists)",
    )

    # dataset scale
    p.add_argument("--n-samples", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    p.add_argument(
        "--n-workers", type=int, default=4, help="Worker processes (0 = single-process)"
    )

    # simulation
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
        "--fixed-reflectivities",
        action="store_true",
        default=False,
        help="Use uniformly spaced reflectivities (deterministic IDs) instead of random",
    )
    p.add_argument(
        "--always-in-frustum",
        action="store_true",
        default=False,
        help="Reject trajectories where any object ever touches a frustum edge",
    )

    # storage
    p.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="gzip compression level 0–9 (default 4)",
    )
    p.add_argument(
        "--write-batch",
        type=int,
        default=512,
        help="Samples buffered in RAM before each HDF5 flush",
    )

    # edit-specific
    p.add_argument(
        "--edit-frame",
        type=int,
        default=-1,
        help="Frame at which the position edit is applied (-1 = T//2)",
    )
    p.add_argument(
        "--edit-always-in-frustum",
        action="store_true",
        default=False,
        help="Reject edits that cause the moved object to leave the frustum",
    )
    p.add_argument(
        "--max-edit-attempts",
        type=int,
        default=50,
        help="Max retries to find a non-colliding edit position",
    )

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
        fixed_reflectivities=args.fixed_reflectivities,
        always_in_frustum=args.always_in_frustum,
    )
    dcfg = EditDatasetConfig(
        n_samples=args.n_samples,
        sim=sim,
        base_seed=args.seed,
        n_workers=args.n_workers,
        write_batch=args.write_batch,
        compression_level=args.compression_level,
        edit_frame=args.edit_frame,
        edit_always_in_frustum=args.edit_always_in_frustum,
        max_edit_attempts=args.max_edit_attempts,
    )

    generate_edits_dataset(dcfg, args.output_dir)


if __name__ == "__main__":
    main()
