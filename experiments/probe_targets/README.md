# probe_targets — what the probe is asked to read (2026-09-09 → 10)

**Question.** Does editability depend on the probe TARGET rather than on the model? Othello's
probes read a categorical board; discworld's read continuous coordinates. Same models, same
editors, same bench — swap the target and rescore.

**What ran.** Overnight chains as systemd units (`scripts/drivers/probe_targets{,_2,_3,_4}.sh`,
logs in `logs/probe_targets*/`): the Othello signed regression target `mine_signed` on every
Othello run; the observation-exact `appearance` partition and the resolution sweep
(`appearance-lat`, `-d2`, `-d3`; product grids `grid-4x2` … `grid-32x16`; misaligned 30-cell
`grid-6x5`, `grid-10x3`) on `L-dw-8ray-20m` and `L-dw-8ray-tok-20m`; a noiseless sweep
(`grid-8x4`, `appearance-lat`, `grid-32x16`, `grid-64x32`) on `L-dw-noiseless-20m`. Probes
are fitted by `scripts/fit_probes.py` into each run's `probes/`, scored by `master_eval`
into `scores.json["bases"][<target>]`; nothing is computed here.

**Follow-up (2026-09-10 pm).** The SNAPPED regression target `pos@appearance` on `L-dw-8ray-20m` (unit `snapped_appearance`, driver `scripts/drivers/score_pending.sh`): the same 30 cells read as 4-D position regression — the alignment-vs-mechanism test.

**Here.** `scripts/sweep_figure.py` → `outputs/probe_target_sweep.png`: the sweep drawn
from the canonical blocks through `pim.figures.sweep_figure`. Finding:
`research/findings/probe-target-type.md`.
