# Brief — Table 3 alignment quantities at the best PI point (2026-09-12)

**You are a WORKER.** Execute this brief and report. Do not orchestrate, do not read
`research/PROGRESS.md`, `research/README.md` or `harness/ORCHESTRATION.md`. Read
`harness/WORKER.md`, `CLAUDE.md`, `harness/ANALYSIS.md`, then this file.

## Goal

For every (run, probe target) below, at ONE residual point — the best PI arm's point from the
run's `scores.json` — compute how much of the TRUE counterfactual edit displacement Δ lies in
the probe's row space, against a generic baseline, before and after the Haufe correction.
These fill Table 3 of the paper tables. Do NOT compute the Haufe-corrected editability (that
is queued for the overnight run); do NOT touch `pim/`, `notebooks/` or any `scores.json`.

## Runs and targets

Othello (target `mine/theirs`, the canonical categorical block; point = `scores["best"]["PI"]["point"]`):
`initial_othello_comparison/L-oth-20m`, `objective_ablation/L-oth-20m-mse`, `flip_ablation/L-oth-noflip-20m`,
`adjacency_ablation/L-oth-adjacent-20m`, `adjacent_flip_ablation/L-oth-adjacent-flip-20m`.

Discworld (targets: the `frustum` regression block — probe target `full` — and the
`appearance-fac` block where the run has one; point = `scores["bases"][block]["best"]["PI"]["point"]`):
`initial_othello_comparison/L-dw-20m`, `noise_ablation/L-dw-noiseless-20m`, `ray_ablation/L-dw-8ray-20m`,
`interface_ablation/L-dw-8ray-tok-20m` (token model — probes read TOKEN inputs; see `pilot.py`),
`ray_ablation/L-dw-5ray-20m`, `blink_ablation/L-dw-blink-20m`.

## Definitions (reuse the existing code; do not re-derive)

- **Probes.** The run's cached probes, exactly as the scorer loads them:
  `dwa.fit_probes(model, target, family="linear", basis_name="frustum", cache_dir=run/"probes",
  require_cached=True, **dwa.probe_recipe(target, inst))` (+ the token encoder for the token
  run, as in `experiments/edit_index_v2_pilot/scripts/pilot.py`). Othello:
  `oa.fit_probe_grid(model, oc.probe_data(paths["probe"], 20_000, **rules), cache_dir=run/"probes")`
  → `grid.probes[("mine","linear","sequence",point)]`. Cache hits only; never fit.
- **Counterfactual Δ = h_cf − h** at the LAST context position, at `point`, in the probe's
  z-space (`common.zspace`). Discworld: the canonical bench (`dwb.load_bench(model, n=192,
  target, "frustum", instance=inst)`; token model: `dwb.bench_arrays` wrapped as in `pilot.py`),
  counterfactual frames `dwa.counterfactual_history(b)`, validity = in-frustum and
  collision-free at every frame (`discworld_alignment.py` lines 55–65). Othello: substitution
  pairs from `pilot.make_pairs(hists, rules, K_BACK=4, POOL=900, rng)` on the instance's bench
  histories, keep pairs with legal mass ≥ 0.98 on BOTH histories. Residuals via
  `model.residual_stack(...)[point][:, -1]`.
- **Row subspace S = the probe rows the edit CHANGES.** Regression `full` (8 outputs:
  o1·x, o1·y, o2·x, o2·y, then velocities): the edited object's two position rows.
  `appearance-fac`: for each changed factor tile, the (tile, old-class) and (tile, new-class)
  rows (`bench.moves`, row = tile·C + class). Othello mine/theirs: for each changed tile, the
  (tile, cur) and (tile, tgt) rows (row = tile·3 + class). Fraction = ‖Q Qᵀ Δ‖² / ‖Δ‖² with Q an
  orthonormal basis of S (`common.orth`, `common.frac_in`), averaged over cases.
- **Generic baseline.** The same fraction with Δ replaced by the displacement to a DIFFERENT
  case's residual at the same point (a random permutation, `rng = default_rng(0)`).
- **Haufe.** Patterns `A = Σ Wᵀ (W Σ Wᵀ)⁻¹` (`common.haufe_patterns(W, cov_z)`), Σ = the
  z-space residual covariance at `point` over 2000 sequences of the instance's probe corpus
  (`layout.probe_file("discworld", inst, "120k")`, sequences 30000–31999 — outside the fit
  rows; Othello: 2000 games of `oc.probe_data(...)`, beyond the first 20000 if available,
  else the last 2000 of the 20000). Haufe fraction and Haufe generic use the SAME rows of A
  instead of W.

## Output contract (the notebooks read this)

`experiments/edit_direction_alignment/scores/table3_alignment.json` — a list of dicts:
`{"run": "<topic>/<run>", "env", "instance", "target": "frustum"|"appearance-fac"|"mine/theirs",
"point", "n_cases", "n_rows", "rows_frac", "rows_generic", "rows_ratio", "haufe_frac",
"haufe_generic", "haufe_ratio", "counterfactual": "<one line>", "notes": "<one line>"}`.
Also `scores/table3_summary.md` with the same numbers as a table. Do not include editability
columns. Script: `scripts/table3_alignment.py` (one file, `--only` for a subset, prints per row).

## Constraints

- Run with `.pim/bin/python`; GPU budget ≤ 45 min total; one model in memory at a time.
- Paths only through `pim.environments.layout`. Never spell a `datasets/` path.
- Scratch note `research/scratch/2026-09-12-table3-alignment.md`: what ran, the table, the
  case counts per row, anything odd. Then report: the table, the JSON path, the note path.
