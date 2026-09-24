# editability_trends: Edit Index across residual points (`by_point`) and across ray count with seed spread (`by_rays`)

Built 2026-09-21 from `runs/<topic>/<run>/scores.json` only. Every number is read through the canonical
selection rule (`pim.metrics.selection.best_arm`) or the tables' own reader (`pim.figures.tables.collect`,
which pools seed replicates through `pim.metrics.replicates`). The scripts compute no metric. Style:
`paper/figs/paper_style.py` (Arial embedded as TrueType, Okabe-Ito editor colours `ps.EDITOR_COLORS`, zero
outer padding), axes in the `pim.figures.theme.style_ax` look (grey spines and tick marks, black text, top and
right spines hidden). Rounds 3 and 4 (Sevan): two figures at the top level, pieces under `pieces/` as PDF only,
a 0.18 in gap (`LEGEND_GAP`) between each x-axis label and the legend row beneath it, `by_rays` as one
full-width panel with PI, GS and IM only.

## Regenerate

    .pim/bin/python paper/figs/editability_trends/by_point.py     # by_point.pdf/.png, pieces/A_*.pdf, by_point_values.{md,json}
    .pim/bin/python paper/figs/editability_trends/by_rays.py      # by_rays.pdf/.png, pieces/B_*.pdf, by_rays_values.{md,json}
    .pim/bin/python paper/figs/editability_trends/by_point.py --all   # also the dropped variants, written under extra/ (see below)
    .pim/bin/python paper/figs/editability_trends/by_rays.py --all

CPU only, a few seconds each. `runs/` is read, never written. Both scripts print their value table.

## Files

| file | what it is | saved size |
|---|---|---|
| `by_point.pdf/.png` | two rows; columns Othello standard (left) and Rayworld standard (right). Top: Edit Index of each editor's reported arm at every residual point (PI, GS, IM; hollow = outside the fidelity guard). Bottom: the inverse map's R² per point (IM colour, diamonds) and the MLP probe skill (grey) | 5.46 x 3.18 in |
| `by_rays.pdf/.png` | one full-width panel: Edit Index against ray count (5, 8, 16, 128; log axis), continuous state solid and categorical state dashed, PI, GS, IM; mean over three seeds with +-1 SD bars; legend of three colour keys and two style keys one row below the x-axis label | 5.42 x 2.56 in |
| `pieces/A_ei_<run>.pdf`, `pieces/A_skill_<run>.pdf` | each `by_point` panel alone with its axis labels; runs `othello`, `rayworld`, `othello_adjacent_flip`, `rayworld_8ray` | 2.57 x 1.92 / 1.27 in |
| `pieces/A_legend.pdf`, `pieces/A_legend_editors.pdf` | the `by_point` legend (six keys) and the editors-only legend (PI, GS, IM, hollow key) | strips |
| `pieces/B_continuous.pdf`, `pieces/B_categorical.pdf` | each basis alone (PI, GS, IM) at half width | 2.57 x 2.12 in |
| `pieces/B_legend.pdf` | the `by_rays` legend (PI, GS, IM, continuous, categorical, hollow key) | strip |
| `by_point_values.md/.json`, `by_rays_values.md/.json` | the plotted values (the tables below) with the arm behind every point | |

Variants dropped from the top level (regenerate with `--all`; they land under `extra/`, which is not kept):
`by_point_one_row` (Edit Index row only), `by_point_extra` and `by_point_extra_one_row` (Othello adjacent-flip |
Rayworld 8-ray), `by_point_all` and `by_point_all_one_row` (all four runs in one row); `by_rays_two_panels`
(continuous | categorical side by side, the round-3 form) and `by_rays_half` (the one panel at half width with a
two-row legend). The extra runs' values stay in `by_point_values.md`. ND was dropped from `by_rays` in round 4
(figure, pieces and value table); its categorical-target numbers remain in `experiments/paper_ci/dashboard/ledger.md`.

## `by_point.py`

Runs: Othello standard `initial_othello_comparison/L-oth-20m`; Rayworld standard `noise_ablation/L-dw-noiseless-20m`
(the `cartesian` block); the `--all` extras `adjacent_flip_ablation/L-oth-adjacent-flip-20m` and
`ray_ablation/L-dw-8ray-20m` (`cartesian` block). Located with `pim.figures.tables.find_run`. Edit Index key:
`edit_index_symdiff` (`tables.OTH_EI`, the symmetric-difference construction) on Othello, `edit_index` on Rayworld.

Selection rule (the only logic, and it is a call): for each editor and residual point p, the arms at that point
(`pim.metrics.selection.arms_of`, then `a["point"] == p`) go through `pim.metrics.selection.best_arm`: the highest
Edit Index among the arms whose fidelity ratio is at most 1; if no arm at that point passes, the highest Edit
Index overall, and that point is drawn HOLLOW ("outside fidelity guard"). PI sweeps its step size alpha at every
point (16 values on Rayworld, 12 on Othello); GS sweeps its step size (10 / 6 values) at start layers 0, 2, 4, 6,
8, and since it writes from its start layer onward it is plotted at that start point; IM has one arm per point
(alpha 1, the full overwrite). Every arm is scored on the run's 1000-case bench. Second row: the inverse map's
held-out R² per point (`inverse_map.g_r2`, the fit of g from environment state to latent state) and the MLP probe
skill per point (`probe_skill_mlp` on Rayworld; `probe_skill["mine|mlp|sequence"]` on Othello), both held out by
sequence, both on a 0 to 1 axis labelled "Skill".

Caption facts: x = residual point 0 to 8 (0 = the embedding, k = the stream after block k); y = Edit Index of the
editor's reported arm at that point; hollow markers = the arm shown is outside the fidelity guard (fidelity ratio
above 1, the write degraded the prediction) because no arm at that point passed it; GS is placed at its start
layer; a light line marks Edit Index 0. Second row: inverse map R² and MLP probe skill per point, held out by
sequence. Say in the caption that the inverse map's R² is the held-out fit of g (how much of the residual the
environment state explains), which the registry keeps distinct from a decodability number; it shares the 0 to 1
axis only because both are skills of a held-out fit.

What the plotted values show (quantities only, from the table below): on Othello standard, PI peaks at point 4
(+0.818, fidelity 0.30), GS at point 4 (+0.828, 0.28), IM at point 5 (+0.806, 0.38); IM is outside the guard at
every point except 4 and 5. On Rayworld standard, IM rises from -0.227 at point 0 to +0.590 at point 6 and stays
above +0.54 through point 8, while PI never exceeds -0.096 and GS never exceeds -0.162, all inside the guard;
the inverse map's R² is 0.74 at point 0 and 0.28 to 0.34 at points 1 to 8, so the R² row does not track the
IM index on Rayworld. On adjacent-flip Othello every editor's arms at points 6 to 8 are outside the guard (PI
at alpha 100 with fidelity ratios above 8). On the 8-ray model IM is between +0.55 and +0.71 at every point.

### Plotted values (`by_point`; the extra runs are the `--all` variants)

| run | point | PI | GS | IM | inverse map R² | MLP probe skill |
|---|---|---|---|---|---|---|
| L-oth-20m | 0 | -0.933 (1.00, α0.5) | +0.785 (0.37, α0.2) | -0.228 (2.07, α1)* | 0.393 | 0.346 |
| L-oth-20m | 1 | -0.422 (0.93, α10) |  | -0.203 (1.99, α1)* | 0.682 | 0.810 |
| L-oth-20m | 2 | +0.412 (0.83, α10) | +0.804 (0.28, α0.2) | -0.031 (1.64, α1)* | 0.769 | 0.879 |
| L-oth-20m | 3 | +0.692 (0.43, α5) |  | +0.255 (1.14, α1)* | 0.796 | 0.927 |
| L-oth-20m | 4 | +0.818 (0.30, α3) | +0.828 (0.28, α0.2) | +0.567 (0.49, α1) | 0.818 | 0.952 |
| L-oth-20m | 5 | +0.783 (0.42, α3) |  | +0.806 (0.38, α1) | 0.827 | 0.966 |
| L-oth-20m | 6 | -0.133 (0.77, α3) | -0.276 (0.79, α0.2) | +0.385 (1.21, α1)* | 0.753 | 0.974 |
| L-oth-20m | 7 | -0.660 (0.98, α2) |  | +0.219 (2.03, α1)* | 0.728 | 0.975 |
| L-oth-20m | 8 | -0.626 (0.99, α1) | -0.712 (0.97, α0.05) | +0.162 (3.08, α1)* | 0.884 | 0.976 |
| L-dw-noiseless-20m | 0 | -0.930 (1.00, α0.75) | -0.162 (0.96, α0.35) | -0.227 (0.75, α1) | 0.741 | 0.940 |
| L-dw-noiseless-20m | 1 | -0.225 (0.98, α20) |  | +0.063 (0.57, α1) | 0.340 | 0.965 |
| L-dw-noiseless-20m | 2 | -0.294 (0.93, α12) | -0.209 (0.96, α0.35) | +0.293 (0.46, α1) | 0.310 | 0.968 |
| L-dw-noiseless-20m | 3 | -0.244 (0.92, α12) |  | +0.440 (0.39, α1) | 0.284 | 0.969 |
| L-dw-noiseless-20m | 4 | -0.166 (0.94, α12) | -0.249 (0.95, α0.35) | +0.542 (0.33, α1) | 0.298 | 0.970 |
| L-dw-noiseless-20m | 5 | -0.096 (0.96, α12) |  | +0.574 (0.34, α1) | 0.343 | 0.971 |
| L-dw-noiseless-20m | 6 | -0.203 (0.95, α8) | -0.326 (0.97, α0.35) | +0.590 (0.34, α1) | 0.339 | 0.972 |
| L-dw-noiseless-20m | 7 | -0.421 (0.93, α5) |  | +0.576 (0.36, α1) | 0.342 | 0.972 |
| L-dw-noiseless-20m | 8 | -0.689 (0.96, α2) | -0.582 (0.99, α0.2) | +0.546 (0.40, α1) | 0.344 | 0.973 |
| L-oth-adjacent-flip-20m | 0 | -0.964 (1.00, α0.5) | -0.058 (0.79, α0.2) | -0.733 (1.34, α1)* | 0.311 | 0.384 |
| L-oth-adjacent-flip-20m | 1 | +0.060 (0.80, α10) |  | -0.726 (1.08, α1)* | 0.657 | 0.895 |
| L-oth-adjacent-flip-20m | 2 | +0.348 (0.83, α5) | -0.237 (0.83, α0.2) | -0.529 (0.84, α1) | 0.888 | 0.937 |
| L-oth-adjacent-flip-20m | 3 | +0.235 (0.78, α3) |  | -0.083 (0.65, α1) | 0.949 | 0.958 |
| L-oth-adjacent-flip-20m | 4 | -0.571 (0.95, α2) | -0.629 (0.99, α0.2) | +0.558 (0.39, α1) | 0.958 | 0.967 |
| L-oth-adjacent-flip-20m | 5 | -0.961 (1.00, α0.25) |  | +0.664 (0.51, α1) | 0.959 | 0.970 |
| L-oth-adjacent-flip-20m | 6 | +0.152 (8.57, α100)* | -0.307 (5.89, α1.5)* | +0.479 (1.34, α1)* | 0.963 | 0.968 |
| L-oth-adjacent-flip-20m | 7 | +0.154 (8.51, α100)* |  | +0.369 (1.74, α1)* | 0.962 | 0.968 |
| L-oth-adjacent-flip-20m | 8 | +0.118 (8.41, α100)* | -0.326 (5.45, α1.5)* | +0.286 (1.93, α1)* | 0.968 | 0.968 |
| L-dw-8ray-20m | 0 | -0.897 (1.00, α0.75) | -0.070 (0.88, α0.7) | +0.671 (0.29, α1) | 0.461 | 0.867 |
| L-dw-8ray-20m | 1 | -0.242 (0.78, α175) |  | +0.550 (0.35, α1) | 0.373 | 0.918 |
| L-dw-8ray-20m | 2 | +0.180 (0.90, α175) | -0.295 (0.85, α0.7) | +0.613 (0.31, α1) | 0.474 | 0.928 |
| L-dw-8ray-20m | 3 | +0.212 (1.00, α175) |  | +0.667 (0.28, α1) | 0.525 | 0.931 |
| L-dw-8ray-20m | 4 | +0.184 (0.95, α100) | -0.381 (0.84, α0.7) | +0.707 (0.26, α1) | 0.566 | 0.932 |
| L-dw-8ray-20m | 5 | +0.160 (0.98, α100) |  | +0.702 (0.27, α1) | 0.572 | 0.932 |
| L-dw-8ray-20m | 6 | +0.110 (0.89, α60) | -0.468 (0.89, α0.7) | +0.711 (0.27, α1) | 0.580 | 0.932 |
| L-dw-8ray-20m | 7 | +0.085 (0.96, α60) |  | +0.709 (0.27, α1) | 0.585 | 0.932 |
| L-dw-8ray-20m | 8 | -0.220 (0.95, α20) | -0.582 (0.91, α0.7) | +0.651 (0.31, α1) | 0.579 | 0.932 |

Cell = Edit Index of the reported arm (fidelity ratio, step size α); * = that arm is outside the fidelity guard (no arm at that point has fidelity ratio ≤ 1). Blank = no arm at that point.

## `by_rays.py`

Runs: `ray_ablation/L-dw-5ray-20m`, `L-dw-8ray-20m`, `L-dw-16ray-20m`, `L-dw-128ray-20m` (x = 5, 8, 16, 128 rays,
log axis with exactly those ticks). Read with `T.set_basis("cartesian"); F = T.collect([], runs)`.
"continuous" = the `cartesian` block (the continuous full-state regression target in the Cartesian basis);
"categorical" = the `appearance-fac` block (the factorized appearance target: per disc one categorical variable
for the centre of its ray run and one for its length). The categorical block's IM is the CATEGORICAL inverse map
(deployed 2026-09-20; per `research/REGISTRY.md` its input is the target's own one-hot labels plus the discs'
Cartesian velocity, fitted with the target's forward-probe recipe on 200k sequences; IM only, no retrieval
form), not the continuous full-state map that these blocks carried before that date. Editors: PI, GS, IM
(ND dropped in round 4).

Per (run, block, editor) the parent run's reported arm is `best_arm` (the guard rule above; `<ed> guarded` in
`F.df`), and the seed spread is `F.rep_sd[(run, block)]`: the pooled mean (`<ed> EI_mean`) and SD with n - 1
(`<ed> EI`) over the run's seed replicates at a matched budget (`__seed0_s512000`, `__seed1`, `__seed2`; all four
families n = 3 at 512k steps). Plotted: the pooled mean with a +-1 SD bar where n > 1 (the parent's value would be
used otherwise; it never is here). The hollow mark follows the PARENT's reported arm; no reported arm in this
figure is outside the guard, so no hollow marker is drawn and the figure carries no hollow legend key (the
legend piece does). Legend: three colour keys (PI, GS, IM) plus two line-style keys (continuous, categorical) in
one row, chosen over six per-line keys because five short entries read cleaner at 5.5 in. SDs are 0.002 to
0.036, so most bars are hidden inside the 3.2 pt markers; the largest are 5-ray PI (0.036), 8-ray GS (0.030) and
128-ray GS (0.027) on the continuous target. Where continuous and categorical values coincide (5 rays: PI
categorical +0.538 and GS categorical +0.559) the markers overlap; the table below separates them.

Caption facts: y = Edit Index of each editor's reported arm (the highest Edit Index among arms with fidelity
ratio at most 1), mean over three training seeds at 512k steps, bars +-1 SD (mostly smaller than the marker);
solid = continuous state target, dashed = categorical (factorized appearance) target; the categorical IM is the
categorical inverse map; every plotted arm is inside the fidelity guard.

Ledger check: every mean and SD below equals the "mean +- SD" column of `experiments/paper_ci/dashboard/ledger.md`
(2026-09-21 10:32) for the same (run, block, editor); the "parent" column equals its "canonical" column.

### Plotted values (`by_rays`)

| rays | basis | editor | plotted (mean) | SD | n | budget | members | parent | parent fid | parent arm | inside guard | member fids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | continuous | PI | +0.132 | 0.036 | 3 | 512k | +0.171 +0.123 +0.101 | +0.141 | 0.85 | all·pt4·α175 | yes | 0.93 0.92 0.84 |
| 8 | continuous | PI | +0.207 | 0.022 | 3 | 512k | +0.184 +0.227 +0.210 | +0.212 | 1.00 | all·pt3·α175 | yes | 0.89 0.90 0.97 |
| 16 | continuous | PI | +0.153 | 0.021 | 3 | 512k | +0.162 +0.168 +0.129 | +0.164 | 0.96 | all·pt2·α100 | yes | 0.92 0.91 0.95 |
| 128 | continuous | PI | -0.003 | 0.009 | 3 | 512k | +0.007 -0.006 -0.010 | -0.019 | 0.98 | all·pt1·α35 | yes | 0.97 0.99 0.99 |
| 5 | continuous | GS | -0.093 | 0.015 | 3 | 512k | -0.095 -0.107 -0.078 | -0.102 | 0.86 | all·pt0·α0.7 | yes | 0.86 0.86 0.83 |
| 8 | continuous | GS | -0.074 | 0.030 | 3 | 512k | -0.057 -0.108 -0.057 | -0.070 | 0.88 | all·pt0·α0.7 | yes | 0.89 0.90 0.87 |
| 16 | continuous | GS | -0.121 | 0.010 | 3 | 512k | -0.113 -0.117 -0.131 | -0.095 | 0.90 | all·pt0·α0.7 | yes | 0.89 0.89 0.91 |
| 128 | continuous | GS | -0.097 | 0.027 | 3 | 512k | -0.110 -0.066 -0.115 | -0.104 | 0.97 | all·pt0·α0.35 | yes | 0.96 1.00 0.95 |
| 5 | continuous | IM | +0.808 | 0.009 | 3 | 512k | +0.807 +0.817 +0.799 | +0.810 | 0.23 | all·pt0·α1 | yes | 0.24 0.23 0.22 |
| 8 | continuous | IM | +0.720 | 0.006 | 3 | 512k | +0.721 +0.714 +0.726 | +0.711 | 0.27 | all·pt6·α1 | yes | 0.27 0.27 0.26 |
| 16 | continuous | IM | +0.663 | 0.007 | 3 | 512k | +0.661 +0.656 +0.670 | +0.660 | 0.29 | all·pt6·α1 | yes | 0.29 0.29 0.29 |
| 128 | continuous | IM | +0.574 | 0.006 | 3 | 512k | +0.568 +0.579 +0.576 | +0.566 | 0.32 | all·pt6·α1 | yes | 0.32 0.31 0.32 |
| 5 | categorical | PI | +0.538 | 0.017 | 3 | 512k | +0.520 +0.541 +0.553 | +0.513 | 0.70 | all·pt1·α20 | yes | 0.85 0.81 0.79 |
| 8 | categorical | PI | +0.392 | 0.006 | 3 | 512k | +0.393 +0.398 +0.386 | +0.380 | 0.95 | all·pt1·α60 | yes | 0.94 0.93 0.89 |
| 16 | categorical | PI | -0.050 | 0.014 | 3 | 512k | -0.063 -0.052 -0.035 | -0.061 | 0.96 | all·pt1·α20 | yes | 0.93 0.93 0.96 |
| 128 | categorical | PI | -0.331 | 0.024 | 3 | 512k | -0.304 -0.350 -0.338 | -0.308 | 0.98 | all·pt3·α0.5 | yes | 1.00 0.96 0.98 |
| 5 | categorical | GS | +0.559 | 0.012 | 3 | 512k | +0.568 +0.563 +0.546 | +0.598 | 0.46 | all·pt0·α0.35 | yes | 0.45 0.48 0.48 |
| 8 | categorical | GS | +0.445 | 0.014 | 3 | 512k | +0.462 +0.438 +0.436 | +0.458 | 0.54 | all·pt0·α0.35 | yes | 0.54 0.56 0.55 |
| 16 | categorical | GS | +0.255 | 0.006 | 3 | 512k | +0.258 +0.258 +0.248 | +0.280 | 0.85 | all·pt0·α1.5 | yes | 0.90 0.77 0.79 |
| 128 | categorical | GS | +0.286 | 0.014 | 3 | 512k | +0.301 +0.282 +0.273 | +0.308 | 0.79 | all·pt0·α0.35 | yes | 0.78 0.82 0.81 |
| 5 | categorical | IM | +0.912 | 0.003 | 3 | 512k | +0.909 +0.913 +0.915 | +0.913 | 0.25 | all·pt5·α1 | yes | 0.25 0.25 0.25 |
| 8 | categorical | IM | +0.874 | 0.005 | 3 | 512k | +0.869 +0.874 +0.879 | +0.866 | 0.27 | all·pt6·α1 | yes | 0.27 0.27 0.27 |
| 16 | categorical | IM | +0.821 | 0.002 | 3 | 512k | +0.821 +0.823 +0.819 | +0.823 | 0.29 | all·pt6·α1 | yes | 0.29 0.28 0.29 |
| 128 | categorical | IM | +0.637 | 0.003 | 3 | 512k | +0.638 +0.640 +0.634 | +0.637 | 0.29 | all·pt6·α1 | yes | 0.28 0.28 0.29 |

plotted = pooled mean over the seed replicates (n > 1) else the parent's value; SD with n − 1; parent = the parent run's reported arm (best Edit Index inside the fidelity guard) with its fidelity ratio and arm (point · step size); inside guard = that arm's fidelity ratio ≤ 1; member fids = the replicates' fidelity ratios at their own reported arms.

## Notes

- `pim.figures.tables` applies a seaborn theme at import; both scripts call `matplotlib.rcdefaults()` and then
  `ps.apply()` after importing it, so the paper style wins.
- `style_ax` colours tick labels grey and sets them to 9 pt; the scripts keep its spines and tick marks and set
  the labels back to black 8 pt (the paper's rule is black text).
- The legend row sits on the page's bottom edge; constrained layout lays the axes out in the region above it
  (`legend_below`: the layout `rect` starts `LEGEND_GAP` = 0.18 in above the legend), so the gap between the
  x-axis label and the legend is exact and nothing overlaps at the saved size. Text inside the figures is 8 to 9 pt.
- Pieces are written through `ps.save` and their PNG preview is removed, so their PDF options always match the
  composites'.
- The `by_point` second row puts two quantities (R² and probe skill) on one 0 to 1 axis as the brief asked; they
  are both skills of a held-out fit, not a dual-axis chart.
