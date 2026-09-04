# othello_decode_by_step — decodability by move number

**Question** (2026-09-02): how does the board's decodability vary with where we are in
the game? For L-oth-20m, the canonical LIN and MLP-128 probes read out per position.

**What it does** — `scripts/decode_by_step.py` loads the run's cached canonical probe
grid (the same key `fit_probe_grid` uses; a miss aborts, nothing is ever fitted here),
harvests each residual point on the held-out games of the grid's own seeded 80/20 split,
and computes at every position t: the tile error rate, the trivial baseline at that
position (majority class over all tiles of the train rows at t), and Probe Skill
`1 − err_t / majority_t`; the best residual point per step is the reported curve, the
other points are drawn faintly. The pooled-baseline skill (canonical majority over all
positions) is kept in the JSON alongside.

**Run** — `drivers/decode_by_step.sh` (transient unit `oth_decode_step`, MemoryMax 16G,
log in `logs/othello_decode_by_step/`). ~3 min on the local GPU (labelling the 20k
games is most of it).

**Results** — `scores/decode_by_step_<run>.json`; figures
`outputs/decode_by_step_linear.png` and `outputs/decode_by_step_mlp.png`.
