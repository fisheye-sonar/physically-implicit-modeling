#!/usr/bin/env python3
"""
Generate a complete dataset suite: train, val, test, and edits splits.

All splits share the same simulator config.  Seeds are assigned
non-overlappingly: train starts at --seed, val follows, test follows,
edits follows.

Usage
-----
    python scripts/generate_dataset.py data/my_run

    python scripts/generate_dataset.py data/my_run \\
        --n-objects 2 --frames 40 --obs-noise-std 0.2 \\
        --fixed-reflectivities --always-in-frustum \\
        --n-train 100000 --n-val 10000 --n-test 10000 --n-edits 5000 \\
        --n-workers 8

Output
------
    <output_dir>/train.h5
    <output_dir>/val.h5
    <output_dir>/test.h5
    <output_dir>/edits.h5
    <output_dir>/dataset.json   ← master metadata for all splits
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

# self-locating, like every other entry point in scripts/ — this script used to depend
# on an ambient PYTHONPATH set by its caller, which broke it when run directly.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pim.environments.discworld.config import SimConfig  # noqa: E402
from pim.environments.discworld.dataset import DatasetConfig, generate_dataset  # noqa: E402
from pim.environments.discworld.edits_dataset import (  # noqa: E402
    EditDatasetConfig,
    generate_edits_dataset,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a pim dataset suite (train / val / test / edits)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "output_dir",
        help="Parent directory for the suite (created if absent; must be empty if it exists)",
    )

    # ── Split sizes ───────────────────────────────────────────────────────
    g = p.add_argument_group("split sizes")
    g.add_argument("--n-train", type=int, default=100_000, metavar="N")
    g.add_argument("--n-val",   type=int, default=10_000,  metavar="N")
    g.add_argument("--n-test",  type=int, default=10_000,  metavar="N")
    g.add_argument("--n-edits", type=int, default=5_000,   metavar="N")

    # ── Simulation (shared across all splits) ─────────────────────────────
    g = p.add_argument_group("simulation config (shared)")
    g.add_argument("--n-objects",  type=int,   default=2)
    g.add_argument("--frames",     type=int,   default=40)
    g.add_argument("--obs-res",    type=int,   default=128)
    g.add_argument("--boundary",   choices=["bounce", "open", "wrap"], default="open")
    g.add_argument("--direction-noise", type=float, default=0.0,
                   help="Velocity angle noise per step (radians)")
    g.add_argument("--speed-noise",     type=float, default=0.0,
                   help="Fractional speed noise per step")
    g.add_argument("--position-noise",  type=float, default=0.0,
                   help="Position diffusion std per step (world units)")
    g.add_argument("--obs-noise-std",   type=float, default=0.04,
                   help="Observation noise std (intensity units); 0 = no noise")
    g.add_argument("--fixed-reflectivities", action="store_true", default=False,
                   help="Uniformly spaced reflectivities (deterministic IDs)")
    g.add_argument("--soft-edge", type=float, default=0.0, metavar="W",
                   help="OPTIONAL soft rendering: silhouette softness in world units "
                        "(0 = the original hard ray-caster); see pim/simulator/soft_render.py")
    g.add_argument("--soft-shading", choices=["flat", "lambert"], default="flat",
                   help="OPTIONAL soft rendering: 'lambert' curves the object's image "
                        "instead of a constant-reflectivity plateau")
    g.add_argument("--soft-psf-sigma", type=float, default=0.0, metavar="RAYS",
                   help="OPTIONAL soft rendering: Gaussian sensor blur along the ray axis")
    g.add_argument("--soft-occlusion-temp", type=float, default=0.0, metavar="T",
                   help="OPTIONAL soft rendering: soft depth-ordering temperature "
                        "(0 = hard nearest-hit; >0 makes the renderer differentiable)")
    g.add_argument("--always-in-frustum",    action="store_true", default=False,
                   help="Reject trajectories where any object touches a frustum edge")
    g.add_argument("--omni2d", action="store_true", default=False,
                   help="OPTIONAL omniscient observation: replace the 1D perspective scan "
                        "with a top-down ORTHOGRAPHIC raster of the world rectangle — no "
                        "projection, no occlusion, no perspective. Frames are flattened "
                        "row-major and --obs-res is derived from the grid (any value passed "
                        "is overridden). See pim/simulator/render2d.py")
    g.add_argument("--omni2d-h", type=int, default=48, metavar="ROWS",
                   help="omni2d grid rows, spanning depth y in [y_near, y_far]; row 0 = near")
    g.add_argument("--omni2d-w", type=int, default=64, metavar="COLS",
                   help="omni2d grid columns, spanning x in [-x_far, x_far]")

    # ── Edit-split config ─────────────────────────────────────────────────
    g = p.add_argument_group("edits split config")
    g.add_argument("--edit-frame", type=int, default=-1,
                   help="Frame at which position edit is applied (-1 = T//2)")
    g.add_argument("--edit-always-in-frustum", action="store_true", default=False,
                   help="Reject edits that cause the moved object to leave the frustum")
    g.add_argument("--max-edit-attempts", type=int, default=50,
                   help="Max retries to find a non-colliding edit position")

    # ── Parallelism / storage ─────────────────────────────────────────────
    g = p.add_argument_group("parallelism / storage")
    g.add_argument("--seed",             type=int, default=0,   help="Base RNG seed for train split")
    g.add_argument("--seed-val",   type=int, default=None, metavar="S",
                   help="Base seed for the val split (default: --seed + n_train)")
    g.add_argument("--seed-test",  type=int, default=None, metavar="S",
                   help="Base seed for the test split (default: after val)")
    g.add_argument("--seed-edits", type=int, default=None, metavar="S",
                   help="Base seed for the edits split (default: after test). "
                        "Override all three to align a suite's splits with an EXISTING "
                        "dataset's seed ranges — scenes are a deterministic function of "
                        "the seed, so matching them makes two suites differ only in the "
                        "observation channel, and keeps a smaller suite's test/edits "
                        "scenes out of the larger one's TRAIN range.")
    g.add_argument("--n-workers",        type=int, default=4,   help="Worker processes (0 = single-process)")
    g.add_argument("--write-batch",      type=int, default=512, help="Samples buffered in RAM before each HDF5 flush")
    g.add_argument("--compression-level",type=int, default=4,   help="gzip compression level 0-9")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        print(
            f"Error: '{output_dir}' already exists and is not empty.  "
            "Halting to avoid overwriting data."
        )
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    # With --omni2d the flat observation dimension IS the grid size, and the whole
    # downstream stack sizes itself from `obs_res` — so derive it here rather than
    # asking the caller to keep two numbers in step (`render2d.validate` enforces it).
    obs_res = args.omni2d_h * args.omni2d_w if args.omni2d else args.obs_res

    sim = SimConfig(
        n_objects=args.n_objects,
        n_frames=args.frames,
        obs_res=obs_res,
        boundary=args.boundary,
        direction_noise_std=args.direction_noise,
        speed_noise_std=args.speed_noise,
        position_noise_std=args.position_noise,
        obs_noise_std=args.obs_noise_std,
        fixed_reflectivities=args.fixed_reflectivities,
        always_in_frustum=args.always_in_frustum,
        soft_edge=args.soft_edge,
        soft_shading=args.soft_shading,
        soft_psf_sigma=args.soft_psf_sigma,
        soft_occlusion_temp=args.soft_occlusion_temp,
        omni2d=args.omni2d,
        omni2d_h=args.omni2d_h,
        omni2d_w=args.omni2d_w,
    )

    if args.omni2d:
        dy = (sim.y_far - sim.y_near) / args.omni2d_h
        dx = (2.0 * sim.x_far) / args.omni2d_w
        print(
            f"omniscient 2D: {args.omni2d_h}x{args.omni2d_w} grid -> obs dim {obs_res}  |  "
            f"pixel {dy:.4f} x {dx:.4f} world units  |  "
            f"object diameter {2 * sim.radius / dy:.1f} x {2 * sim.radius / dx:.1f} px"
        )

    # Seeds are assigned non-overlappingly so no sample appears in two splits.
    # Each may be overridden to align with an existing suite (see --seed-edits).
    seed_train = args.seed
    seed_val   = args.seed + args.n_train        if args.seed_val   is None else args.seed_val
    seed_test  = seed_val + args.n_val           if args.seed_test  is None else args.seed_test
    seed_edits = seed_test + args.n_test         if args.seed_edits is None else args.seed_edits

    ranges = {
        "train": (seed_train, args.n_train),
        "val":   (seed_val,   args.n_val),
        "test":  (seed_test,  args.n_test),
        "edits": (seed_edits, args.n_edits),
    }
    for a in ranges:
        for b in ranges:
            if a >= b:
                continue
            (sa, na), (sb, nb) = ranges[a], ranges[b]
            if sa < sb + nb and sb < sa + na:
                print(
                    f"Error: seed ranges for '{a}' [{sa}, {sa + na}) and '{b}' "
                    f"[{sb}, {sb + nb}) overlap — the same scene would appear in both."
                )
                return
    print("seed ranges: " + "  ".join(f"{k}=[{s}, {s + n})" for k, (s, n) in ranges.items()))

    shared_storage = dict(
        n_workers=args.n_workers,
        write_batch=args.write_batch,
        compression_level=args.compression_level,
    )

    splits_meta: dict[str, dict] = {}
    suite_start = time.perf_counter()

    # ── Train ─────────────────────────────────────────────────────────────
    print("train")
    dcfg_train = DatasetConfig(
        n_samples=args.n_train, sim=sim, base_seed=seed_train, **shared_storage
    )
    meta_train = generate_dataset(dcfg_train, output_dir / "train.h5")
    splits_meta["train"] = {"n_samples": args.n_train, "base_seed": seed_train}

    # ── Val ───────────────────────────────────────────────────────────────
    print("val")
    dcfg_val = DatasetConfig(
        n_samples=args.n_val, sim=sim, base_seed=seed_val, **shared_storage
    )
    generate_dataset(dcfg_val, output_dir / "val.h5")
    splits_meta["val"] = {"n_samples": args.n_val, "base_seed": seed_val}

    # ── Test ──────────────────────────────────────────────────────────────
    print("test")
    dcfg_test = DatasetConfig(
        n_samples=args.n_test, sim=sim, base_seed=seed_test, **shared_storage
    )
    generate_dataset(dcfg_test, output_dir / "test.h5")
    splits_meta["test"] = {"n_samples": args.n_test, "base_seed": seed_test}

    # ── Edits ─────────────────────────────────────────────────────────────
    print("edits")
    dcfg_edits = EditDatasetConfig(
        n_samples=args.n_edits,
        sim=sim,
        base_seed=seed_edits,
        edit_frame=args.edit_frame,
        edit_always_in_frustum=args.edit_always_in_frustum,
        max_edit_attempts=args.max_edit_attempts,
        **shared_storage,
    )
    generate_edits_dataset(dcfg_edits, output_dir / "edits.h5")
    eff_edit_frame = args.edit_frame if args.edit_frame >= 0 else args.frames // 2
    splits_meta["edits"] = {
        "n_samples": args.n_edits,
        "base_seed": seed_edits,
        "edit_frame": eff_edit_frame,
        "edit_always_in_frustum": args.edit_always_in_frustum,
        "max_edit_attempts": args.max_edit_attempts,
    }

    # ── Master dataset.json ───────────────────────────────────────────────
    master = {
        "generated_at": meta_train["generated_at"],
        "sim": dataclasses.asdict(sim),
        "splits": splits_meta,
        "generation": {
            "n_workers": args.n_workers,
            "write_batch": args.write_batch,
            "compression": "gzip",
            "compression_level": args.compression_level,
        },
        "schema": meta_train["schema"],
    }
    json_path = output_dir / "dataset.json"
    json_path.write_text(json.dumps(master, indent=2))

    total_elapsed = time.perf_counter() - suite_start
    print(f"\ndone in {total_elapsed:.1f}s  →  {output_dir}")
    print(f"dataset.json  →  {json_path}")


if __name__ == "__main__":
    main()
