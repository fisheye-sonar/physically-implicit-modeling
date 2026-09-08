# training_curve

**Question:** does editability emerge late in training (Othello) or trend (discworld)? **Status:** done 2026-09-02 — neither; both track decodability in time. `scripts/make_training_curve.py` lays each canonical run's own checkpoints out as run dirs under `runs/training_curve/`; `drivers/training_curve.sh` scores them through `master_eval`; Fig 1 in `build_full_table.ipynb`. Finding: `research/findings/training-curve.md`. `drivers/overnight_inlp_curve.sh` is the 2026-09-01 overnight chain (INLP, then the curve).
