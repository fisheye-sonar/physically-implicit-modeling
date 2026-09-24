# BRIEF — two clean metric plots: Edit Index across residual points, and across ray count with seed error bars

Read `paper/figs/FIGURE_BRIEF_COMMON.md` first. CPU only. Folder: `paper/figs/editability_trends/`, two scripts:
`by_point.py` and `by_rays.py`. Light academic theme (`pim/figures/theme.py::style_ax` look; white page,
`ps.EDITOR_COLORS`). Sevan: "Keep it simple, don't add too much text or detail. Just clean and straightforward.
Leave details to the captions."

## Figure A — `by_point`: Edit Index against residual point (appendix §"Inverse Mapping Editability Trends Across
Residual Points")
- Data: `runs/<topic>/<run>/scores.json`. Discworld: `s["bases"]["cartesian"]["arms"]` (arm dicts with `editor`,
  `point`, `alpha`, `edit_index`, `fidelity_ratio`; editor labels `PI[zspace]`, `GS@L<k>`, `IM`, `ND`, `IM-NN`) and
  `s["bases"]["cartesian"]["probe_skill_linear" | "probe_skill_mlp"]` (one value per point) and
  `["inverse_map"]["g_r2"]` (per point). Othello: `s["arms"]` with key `edit_index_symdiff`
  (`pim.figures.tables.OTH_EI`), `s["probe_skill"]["mine|mlp|sequence"]`, `s["inverse_map"]["g_r2"]`.
  Runs: standard Othello `initial_othello_comparison/L-oth-20m`; Rayworld standard `noise_ablation/L-dw-noiseless-20m`
  (and, as extra outputs, the 8-ray `ray_ablation/L-dw-8ray-20m` and adjacent-flip `L-oth-adjacent-flip-20m` — find
  with `pim.figures.tables.find_run(name)`).
- Selection per point (the only "logic" allowed, and it is a call): for each editor and each point, filter the arms
  to that point (`pim.metrics.selection.arms_of` then `a["point"] == p`) and take
  `pim.metrics.selection.best_arm(subset, editor, key)` — the best Edit Index INSIDE the fidelity guard at that
  point, unguarded only if none passes (`within_guard` tells you). GS arms carry their start layer in the label
  (`GS@L<k>`) and write from that layer onward: plot GS against its start point k.
- Draw: one row of small panels, one per run (Othello standard | Rayworld standard), x = residual point (0…8,
  integer ticks), y = Edit Index (−1…1, a light zero line), lines PI / GS / IM in `ps.EDITOR_COLORS` with small
  markers; a point whose reported arm is OUTSIDE the guard is drawn hollow (one legend entry: "outside fidelity
  guard"). Option B: a second thin row underneath with the inverse map's R² per point (IM colour) and the MLP probe
  skill (grey), y 0…1. Produce both the one-row (A1) and two-row (A2) variants. Legend once, few words. Full width
  5.5 in, height ≤ 2.2 in (A1) / 3.2 in (A2).

## Figure B — `by_rays`: Edit Index against ray count with seed spread (Results, beside Table 2, or appendix)
- Data via the tables' own reader, so the numbers are the paper's: `from pim.figures import tables as T`;
  `T.set_basis("cartesian")`; `F = T.collect([], ["L-dw-5ray-20m", "L-dw-8ray-20m", "L-dw-16ray-20m", "L-dw-128ray-20m"])`.
  `F.df` rows carry `run`, `basis` (block key), `canonical`, `PI EI`, `GS EI`, `IM EI`, `ND EI`, `<ed> fid`,
  `<ed> guarded`. Continuous = the row with `basis == "cartesian"`; categorical = `basis == "appearance-fac"`.
  Seed spread: `F.rep_sd[(run, block)]` with `"<ed> EI_mean"`, `"<ed> EI"` (SD over training seeds, n−1),
  `"<ed> EI_values"`, `n`, `steps`. All four families are at n = 3 (512k steps). Plot the pooled mean with ± SD
  error bars where n > 1 (no bar otherwise). Do not compute anything else.
- Draw: x = rays {5, 8, 16, 128} on a log axis with exactly those tick labels; y = Edit Index (−1…1, zero line).
  Editors PI / GS / IM in `ps.EDITOR_COLORS`; continuous solid, categorical dashed (legend: "continuous" /
  "categorical" style keys + the three editor colour keys — six short entries, or two small panels
  (continuous | categorical) sharing y with a three-entry legend: produce BOTH, B1 one panel and B2 two panels).
  Mark the reported arm outside the guard hollow, as in A. Half-width (2.65 in) version of B1 as well, for a
  wrap beside the table. Include ND in the categorical panel only if it does not clutter; say what you chose.
- Print the plotted table (rays × editor × basis: mean, SD, n, guarded) to stdout and into the README so the
  numbers can be checked against `experiments/paper_ci/dashboard/ledger.md`.

## Deliver
PDF + PNG for A1, A2, B1, B1-half, B2; `README.md` with regeneration commands, the printed table, caption facts
(guard rule, n, budget, which block is "categorical", that the categorical IM is the categorical inverse map).
