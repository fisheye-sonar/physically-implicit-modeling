# 2026-09-12 — Table 3: alignment of the true edit displacement with the probe rows, at the best PI point

Worker note for `experiments/edit_direction_alignment/BRIEF_table3.md`. Alignment quantities only —
no editability was computed, nothing canonical was touched (`pim/`, `notebooks/`, `scores.json` untouched).

## What ran

`experiments/edit_direction_alignment/scripts/table3_alignment.py` (new; `--only <substr>` for a subset),
`.pim/bin/python`, 16 rows in 0.3 GPU-minutes total (one model in memory at a time; every probe a cache
hit — the Othello grid is preflighted through `ProbeCache.key`/`load` so a miss raises instead of fitting).
Outputs:

- `experiments/edit_direction_alignment/scores/table3_alignment.json` — the contract, exactly the 15 keys
- `experiments/edit_direction_alignment/scores/table3_summary.md` — the same numbers as a table + per-row counts
- `experiments/edit_direction_alignment/scores/table3_alignment_cases.npz` — per-case fractions, |Δ|/|h|, rows/case

Definitions (all imported from the experiment's `common.py`; nothing re-derived):

| quantity | formula |
|---|---|
| Δ | h_cf − h at the last context position, at `point`, in the probe's z-space (`common.zspace`) |
| rows frac | mean_i ‖Q_i Q_iᵀ Δ_i‖² / ‖Δ_i‖², Q_i = `common.orth` of the probe rows the edit changes in case i (`common.frac_in`) |
| rows generic | the same with Δ_i → z_{π(i)} − z_i, π a fixed-point-free permutation of the kept cases (`default_rng(0)`) |
| Haufe frac / generic | the same two with rows of A = Σ Wᵀ (W Σ Wᵀ)⁻¹ (`common.haufe_patterns`) in place of W |
| Σ | z-space residual covariance at `point` over 2000 held-out sequences (discworld: probe_120k seqs 30000–31999; Othello: see "odd" below) |
| ratio | frac / generic |
| point | `scores["best"]["PI"]["point"]` (Othello) / `scores["bases"][block]["best"]["PI"]["point"]` (discworld) |

Row subspaces: regression `full` → the edited object's two position rows (2 rows); `appearance-fac` → (tile, old)
and (tile, new) for every factor tile of the edited object whose class changes (`bench.moves`; 2–4 rows);
Othello mine/theirs → (tile, cur) and (tile, tgt) for every changed tile (4–22 rows). Chance for a random vector
is r/d = rows/512: 0.004 (2 rows), ~0.006–0.008 (3–4 rows), ~0.02 (10 rows).

## The table

| run | target | pt | n | rows | rows frac | generic | ratio | Haufe frac | Haufe generic | ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial_othello_comparison/L-oth-20m | mine/theirs | 4 | 576 | 10.1 | 0.303 | 0.048 | 6.3x | 0.356 | 0.067 | 5.3x |
| objective_ablation/L-oth-20m-mse | mine/theirs | 4 | 528 | 10.1 | 0.350 | 0.055 | 6.4x | 0.380 | 0.067 | 5.6x |
| flip_ablation/L-oth-noflip-20m | mine/theirs | 8 | 689 | 4.0 | 0.074 | 0.009 | 8.1x | 0.142 | 0.033 | 4.4x |
| adjacency_ablation/L-oth-adjacent-20m | mine/theirs | 8 | 699 | 4.0 | 0.119 | 0.006 | 18.3x | 0.367 | 0.043 | 8.6x |
| adjacent_flip_ablation/L-oth-adjacent-flip-20m | mine/theirs | 2 | 641 | 4.7 | 0.337 | 0.022 | 15.4x | 0.266 | 0.029 | 9.2x |
| initial_othello_comparison/L-dw-20m | frustum | 2 | 99 | 2.0 | 0.010 | 0.009 | 1.1x | 0.036 | 0.028 | 1.3x |
| noise_ablation/L-dw-noiseless-20m | frustum | 1 | 110 | 2.0 | 0.006 | 0.005 | 1.1x | 0.025 | 0.022 | 1.1x |
| noise_ablation/L-dw-noiseless-20m | appearance-fac | 1 | 110 | 3.7 | 0.048 | 0.020 | 2.4x | 0.017 | 0.016 | 1.0x |
| ray_ablation/L-dw-8ray-20m | frustum | 3 | 66 | 2.0 | 0.002 | 0.001 | 1.1x | 0.056 | 0.033 | 1.7x |
| ray_ablation/L-dw-8ray-20m | appearance-fac | 1 | 82 | 3.0 | 0.066 | 0.031 | 2.2x | 0.043 | 0.035 | 1.2x |
| interface_ablation/L-dw-8ray-tok-20m | frustum | 5 | 66 | 2.0 | 0.001 | 0.002 | 0.9x | 0.036 | 0.028 | 1.3x |
| interface_ablation/L-dw-8ray-tok-20m | appearance-fac | 2 | 82 | 3.0 | 0.074 | 0.022 | 3.4x | 0.098 | 0.048 | 2.0x |
| ray_ablation/L-dw-5ray-20m | frustum | 4 | 60 | 2.0 | 0.001 | 0.001 | 1.1x | 0.097 | 0.059 | 1.6x |
| ray_ablation/L-dw-5ray-20m | appearance-fac | 2 | 79 | 3.1 | 0.074 | 0.033 | 2.2x | 0.085 | 0.045 | 1.9x |
| blink_ablation/L-dw-blink-20m | frustum | 2 | 105 | 2.0 | 0.011 | 0.007 | 1.5x | 0.026 | 0.018 | 1.4x |
| blink_ablation/L-dw-blink-20m | appearance-fac | 7 | 100 | 3.7 | 0.075 | 0.029 | 2.6x | 0.009 | 0.009 | 1.0x |

Per-case spread (from the npz): the discworld regression rows have median ≈ mean (0.001–0.007) and q75 ≤ 0.017;
the fac rows median 0.05–0.07, q75 ≤ 0.10; Othello medians 0.055–0.32 with q75 up to 0.47 — the Othello means
are not carried by a few cases. |Δ|/|h| in z-space ≈ 0.8–1.2 on discworld, 0.5–0.76 on Othello.

## Case counts per row

- Othello (pairs from `pilot.make_pairs`, 900 attempts, K_BACK=4; kept if legal mass ≥ 0.98 on both):
  L-oth-20m 615 pairs → 576 kept (rejected: 199 illegal replay, 76 mover changed, 10 same legal set);
  L-oth-20m-mse same 615 pairs → 528 kept (the MSE model treats more substituted histories as abnormal);
  L-oth-noflip 689 → 689; L-oth-adjacent 705 → 699; L-oth-adjacent-flip 652 → 641.
  Tiles changed per case: 5.06 mean (uniform), exactly 2 (noflip, adjacent — no flips, so only the two placed
  tiles differ), 2.35 (adjacent-flip).
- Discworld `frustum` (canonical `full` bench, 192 cases, the instance's selection where it has one; kept if the
  shifted trajectory is in-frustum and collision-free at every frame): L-dw-20m 99, noiseless 110, 8ray 66,
  8ray-tok 66 (the 11 cases with an UNK frame were already invalid), 5ray 60, blink 105.
- Discworld `appearance-fac` (the fac bench = `grid_selection` cases whose teleport changes a cell, so a
  DIFFERENT case list from the frustum rows): noiseless 110, 8ray 82, 8ray-tok 82 (9 UNK, all already invalid),
  5ray 79, blink 100. Every valid case had ≥ 1 changed factor tile — none dropped for "no rows".

## Reading (quantities, not verdicts)

1. **Regression rows carry essentially none of Δ.** With the two position rows, the fraction is 0.001–0.011 on
   every discworld run — at or BELOW the r/d chance level of 0.004 on the 8-ray, 5-ray and token runs — and the
   generic displacement is indistinguishable (ratios 0.9–1.5x). The Haufe patterns raise it to 0.025–0.097 but
   the generic rises with it (ratios 1.1–1.7x). The true edit is orthogonal to what the position probe reads.
2. **Factorised categorical rows carry a little, and it is specific.** Raw fac rows hold 0.05–0.075 of Δ at
   2.2–3.4x the generic. The Haufe correction does NOT help here: it erases the enrichment on the 128-ray
   instances (noiseless 1.0x at 0.017, blink 1.0x at 0.009) and keeps only 1.2–2.0x on 8-ray / 5-ray / tokens.
   This is the opposite of the pattern that motivated Haufe for the regression probe — worth a look before
   the overnight Haufe editability run is read: on the fac probe, the forward patterns point mostly AWAY from Δ.
3. **Othello is a different regime.** 0.07–0.35 of Δ in the changed-tile rows, 6–18x the generic; Haufe
   0.14–0.38 at 4–9x. The two adjacency variants have the largest raw ratios because their generic is tiny
   (0.006, 0.022): unrelated games' boards barely project on the changed tiles' rows. L-oth-adjacent is the one
   run where Haufe triples the absolute fraction (0.119 → 0.367).

## Odd / caveats

- **Othello covariance corpus:** the canonical probe split holds exactly 20 000 games, so "beyond the first
  20 000" was unavailable; Σ uses games 18000–19999 — the brief's fallback, but these are INSIDE the fit corpus
  (the probe held out 20% of games by sequence, so ~1600 of the 2000 were fit rows). Discworld Σ is genuinely
  outside the 30k fit rows (seqs 30000–31999).
- **dw-blink:** `arms.counterfactual_history` has no blink path (it renders every object visible, no markers),
  so the blink rows use `discworld_alignment.py`'s construction — same shift, the case's own visibility schedule
  and toggle markers, noise-matched via the renderer's rng. The other five discworld runs use
  `arms.counterfactual_history(b, noise_matched=True)` verbatim.
- **Bench wrapping:** all discworld rows use `bench_arrays` wrapped in a namespace (the pilot's token-model
  form) rather than `load_bench`; the arrays are identical, only the warmed model state is skipped (unused here).
- **Different case sets across targets** (frustum vs fac rows of one run) — by construction of the two benches;
  the two rows of a run are not a paired comparison.
- **Generic partner is a derangement, not length-matched** on Othello (the brief asked for a permutation; the
  old `othello_alignment.py` matched history length). With 4–22 rows the chance floor differs per case
  (r/512); ratios are the comparable column, per ANALYSIS.md §8.
- Points: noflip and adjacent sit at point 8 (the final residual), blink-fac at 7; all other rows at 1–5.
- Not done here, by design: editability at these points, waterfalls, any figure.
