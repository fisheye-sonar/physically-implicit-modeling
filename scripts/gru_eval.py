"""Command-line GRU evaluation script.

Runs all four evaluation criteria (or a subset) for a trained GRU checkpoint
and saves figures + metrics to a timestamped output directory.

Usage:
    python scripts/gru_eval.py \\
        --checkpoint runs/my_run/best_model.pt \\
        --test-h5    datasets/3_fixed_refl_inview_brighter_eval/dataset.h5 \\
        --edits-h5   datasets/3_fixed_refl_inview_brighter_edits/dataset.h5 \\
        --output-dir outputs/eval_run1 \\
        --criteria 1 2 3 4 \\
        --device cuda

    # Run only predictive quality and recovery:
    python scripts/gru_eval.py ... --criteria 1 2
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from pim.world_models.gru.run_eval import (
    EvalConfig,
    plot_criterion1,
    plot_criterion2,
    plot_criterion3,
    plot_criterion4,
    plot_setup,
    run_all,
    save_config,
    save_figures,
    save_metrics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRU model evaluation")

    # Required paths
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--test-h5", required=True, dest="test_h5", help="Test dataset HDF5")
    p.add_argument(
        "--edits-h5", required=True, dest="edits_h5", help="Edits dataset HDF5"
    )

    # Output
    p.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Output directory (default: outputs/eval/<run_name>/<timestamp>)",
    )

    # Hardware
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", default=512, type=int, dest="batch_size")
    p.add_argument("--num-workers", default=6, type=int, dest="num_workers")

    # Which criteria to run
    p.add_argument(
        "--criteria",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4],
        metavar="{1,2,3,4}",
        help="Which eval criteria to run (default: all)",
    )

    # Criterion 1
    p.add_argument(
        "--n-context",
        default=10,
        type=int,
        dest="n_context",
        help="Warm-up frames before AR rollout",
    )

    # Criterion 2
    p.add_argument("--n-obj", default=2, type=int, dest="n_obj")
    p.add_argument("--use-hungarian", action="store_true", dest="use_hungarian")
    p.add_argument(
        "--no-lstsq",
        action="store_false",
        dest="use_lstsq",
        help="Use gradient descent instead of lstsq for LinearExtractor",
    )
    p.add_argument("--probe-epochs", default=30, type=int, dest="probe_n_epochs")
    p.add_argument("--probe-lr", default=5e-3, type=float, dest="probe_lr")
    p.add_argument("--probe-hidden", default=256, type=int, dest="probe_hidden_dim")
    p.set_defaults(use_lstsq=True)

    # Criterion 3
    p.add_argument(
        "--rollout-n-context", default=20, type=int, dest="rollout_n_context"
    )
    p.add_argument(
        "--rollout-n-rollout", default=20, type=int, dest="rollout_n_rollout"
    )
    p.add_argument("--coherence-n-eval", default=500, type=int, dest="coherence_n_eval")

    # Criterion 4
    p.add_argument("--ctrl-n-rollout", default=15, type=int, dest="ctrl_n_rollout")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = EvalConfig(
        checkpoint_path=args.checkpoint,
        test_h5_path=args.test_h5,
        edits_h5_path=args.edits_h5,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        criteria=tuple(sorted(set(args.criteria))),
        n_context=args.n_context,
        n_obj=args.n_obj,
        use_hungarian=args.use_hungarian,
        use_lstsq=args.use_lstsq,
        probe_n_epochs=args.probe_n_epochs,
        probe_lr=args.probe_lr,
        probe_hidden_dim=args.probe_hidden_dim,
        rollout_n_context=args.rollout_n_context,
        rollout_n_rollout=args.rollout_n_rollout,
        coherence_n_eval=args.coherence_n_eval,
        ctrl_n_rollout=args.ctrl_n_rollout,
    )

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_name = Path(args.checkpoint).parent.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / "eval" / run_name / timestamp
    cfg.output_dir = str(output_dir)

    print(f"\nOutput → {output_dir}\n")

    # Run all requested criteria
    results = run_all(cfg)
    s = results["setup"]

    # Collect and save figures
    all_figs = {}
    all_figs.update(plot_setup(cfg, s))

    if "c1" in results:
        all_figs.update(plot_criterion1(cfg, s, results["c1"]))
    if "c2" in results:
        all_figs.update(plot_criterion2(cfg, s, results["c1"], results["c2"]))
    if "c3" in results:
        all_figs.update(plot_criterion3(cfg, s, results["c3"]))
    if "c4" in results:
        all_figs.update(plot_criterion4(cfg, s, results["c2"], results["c4"]))

    save_figures(all_figs, output_dir)
    save_metrics(results, output_dir)
    save_config(cfg, s, output_dir)

    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
