# scripts/drivers — canonical run orchestration

Shell scripts that queue or chain training/evaluation runs live HERE and nowhere else —
committed, never ad-hoc — so even the automation that produced a batch of runs is part
of the record. (The old pattern of one-off `.sh` files scattered through `runs/` is what
this replaces; those are archived with their runs.)

Conventions:
- one driver per campaign, named after the runs/ topic dir it fills
  (e.g. `initial_othello_comparison.sh`)
- every launched run goes through `scripts/train.py`, so config.json + commit_sha +
  arch-stamped checkpoints are guaranteed
- notify (ntfy) at stage boundaries when a driver runs unattended
