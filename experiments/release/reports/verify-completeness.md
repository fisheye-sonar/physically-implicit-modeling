# Verifier report: completeness

**Scope.** Three checks:
1. Every paper item has a release producer that reads a shipped artifact.
2. The paper's Experimental Setup and Implementation Details agree with the release code.
3. The README's scripts are wired, and the STAGING artifacts are all read and all shipped.

**Setup.**
- Work dir: `experiments/release/work/completeness/`.
  - `tree/` is a copy of RELEASE `pim/`, `scripts/`, `notebooks/` and `README.md`, with `runs/` and `datasets/` symlinked into STAGING.
  - `tree_flip/` is a copy with a writable `runs/_baselines/othello`.
  - `checks/` holds `dump_tables.py`, `compare_paper.py`, `trace_files.py` and their JSON outputs.
  - `trace/` holds the strace logs and executed notebooks; `logs/` holds the figure-script logs.
- Nothing was written to RELEASE, STAGING or PRIVATE `runs/`, `datasets/`, `logs/` or `outputs/`.

**Result.**
- No blockers.
- The release reproduces every paper table (855 of 855 printed cells, apart from 2 stale paper cells) and every figure that has a script:
  - 9 of the 15 figure PNGs are pixel-identical to PRIVATE's renders;
  - the by-point figure differs only in its legend text;
  - each of the 5 Othello grids differs in 3 cells.
- **7 major findings:**
  - 5 where the paper's text does not describe what the code does (categorical Probe Skill baseline, which checkpoint is evaluated, the GS optimizer, the edit-selection rule, the Othello appendix figures);
  - 1 paper item with no release producer (the reconstruction test);
  - 1 README rescoring instruction that silently drops the categorical blocks.
- Plus 15 minor findings and 6 nits.

## 1. Paper item → release producer → STAGING artifact

All 13 table producers are called in the notebooks. `checks/compare_paper.py` parses every labeled tabular in `paper_draft.tex` and compares it string for string with `Table.text`: **855 cells, 2 mismatches**. Both are in `tab:im_by_point`, Othello IM Index, where the paper is stale (finding m9).

| paper item | release producer (called at) | artifact read (all present) | status |
|---|---|---|---|
| tab:decodability | `tables.table_decodability` (paper_tables [5]) | `runs/<id>/scores.json`, `runs/_baselines/<env>/<inst>/baselines.json` | 80/80 |
| tab:editability + daggers | `table_editability` ([7]) | scores.json and the `__seed*` scores.json / config.json | 98/98 + 4 daggers |
| 53.6% | `im_vs_nn_gain` ([9]) | scores.json IM / IM-NN arms | 0.53633 |
| ≤0.004, <0.04, 4 daggers | `seed_sd_maxima` ([11]) | replicate scores.json | 0.0037 / 0.034 / 4 |
| 0.27 vs 2.2 flips per move | `scripts/othello_flip_rates.py` | `runs/_baselines/othello/*/corpus_stats.json` | reproduced exactly in `tree_flip`; **no notebook shows it** (m2) |
| "PI at α = 1 reads the target exactly" | none | scores.json `readout_err_after` | **no renderer, and false at point 0** (m1) |
| MLP ≥ 0.88, within 0.03 except blink, +0.35, +0.51 / +0.60 | Tables 1 / 2 | – | ok |
| fig:qualitative_edits | `scripts/figures/qualitative_overview.py` | `edits.h5`, `eval/test.json`, `cases_1000.pkl`, `probe_20000.npz`, weights and probes | pixel-identical to PRIVATE `composite_final_sidebyside.png` |
| fig:othello_and_variants, fig:rayworld_and_variants | **none** | (PRIVATE `paper/figs/environments_overview/*`) | missing (m3) |
| fig:teaser, fig:experimental_setup | diagrams, no code needed | `assets/teaser.png` | ok |
| tab:predictive_skill, 99% | `table_predictive_skill`, `gap_closed_min` (appendix [3], [5]) | scores.json `prediction`, `bayes_floor.json` | 108/108, min 0.9914 |
| fig:othello_predictions, fig:rayworld_predictions | `scripts/figures/predictions.py` | `edits.h5`, `selection.json`, `cases_1000.pkl` | pixel-identical |
| tab:seed_spread, 2.48, "six SD", daggers, fixed setting, 0.05 / 0.26, 0.48 vs 0.70 | `table_seed_spread`, `ci_multiplier`, `im_steps`, `dagger_cells`, `fixed_setting_check`, `seed_means_vs_main`, `replicate_arms` ([8]–[22]) | replicate scores.json and config.json | all match |
| probe refits (0.0007 / 0.016 / 0.004 / 0.036) | `probe_refit_spread` ([24]) | `variance.json` of othello/standard, adjacent-flip__seed0/1/2, rayworld/8-ray | match; selection caveat (m6) |
| tab:legal_illegal, counts 441 / 555 / 4 … | `reachability_counts`, `table_legal_illegal` ([26], [27]) | `editability_by_reachability.json`, `reachability.json` | match |
| tab:two_flip, SE 0.11, 480 | `table_two_flip`, `two_flip_numbers` ([29], [30]) | `two_flip_editability.json` | match |
| tab:tokens_* , 0.013 | `table_tokens_decodability` / `_editability`, `tokens_numbers` ([33]–[36]) | 8-ray and 8-ray-tokens scores.json, baselines | match (case-count omission, m8) |
| tab:additional_rw | `table_additional_rw` ([39]) | smooth / obs5 / standard scores.json and baselines | match |
| tab:im_by_point | `table_im_by_point` ([41]) | othello/standard, rayworld/standard | 2 stale paper cells (m9) |
| IM reconstruction test (≈2×, 5–16×) | **none** | – | missing (M5) |
| fig:editability_over_res | `scripts/figures/editability_by_point.py` | scores.json | data identical; legend text differs |
| tab:categorical | `table_categorical` ([43]) | rayworld/8-ray scores.json | match |
| tab:im_vs_nn and its prose | `table_im_vs_nn`, `im_vs_nn_gain` ([45], [46]) | scores.json | match |
| fig:history_rewriting and its numbers | `scripts/figures/history_rewrite.py` | rayworld/standard weights, the cached g, bench | image pixel-identical; every number equals the paper (+0.61 / 0.64, −0.63, −0.74, +0.63 / 0.64, +0.59, +0.23, +0.65 / 0.52, 0.114 / 0.264, 32 cases) |
| tab:fidelity_selected, "at most 0.10" | `table_fidelity_selected`, `fidelity_rule_shift` ([49], [51]) | scores.json | table matches; the true maximum shift is 0.106 (m10) |
| fig:more_qualitative_edits_othello_1–5 | `qualitative_othello.py` | `cases_1000.pkl`, `probe_20000.npz` | **3 cells per figure differ from the paper** (M6) |
| fig:more_qualitative_edits_rayworld_1–2 | `qualitative_rayworld.py` | `edits.h5`, `eval/test.json` | pixel-identical |
| training time 8–12 h / 19–26 h | none (`metrics.jsonl` `elapsed_s`) | metrics.jsonl | consistent (7.88–12.15 h, 19.21–25.73 h) |

## 2. Implementation Details and Experimental Setup against the code

**These match the paper:**
- **Model:** 8 blocks, 8 heads, d 512, learned `pos_emb`, dropout 0.1, 25.25–25.67M params (checked on all 13 main checkpoints), block 39 / 59, vocab 61.
- **Training:** AdamW 1e-3, wd 1e-4, clip 1.0, linear warmup of 2000 steps then constant, batch 256, 20M sequences with 2M held out, seeds 0 / 1 / 2.
- **Probes:**
  - 80/20 split by sequence;
  - linear regression by `lstsq`;
  - Adam 1e-3, batch 4096, 200 epochs;
  - categorical probes and categorical IM at 50 epochs on 200k sequences of `probe_250k` (checked in the cache provenance);
  - 30k Rayworld sequences, 20k Othello games (n_rows 1,179,476);
  - MLP width 128;
  - IM inputs: 8-d state, one-hot board, one-hot labels plus velocity.
- **Editors:**
  - PI and GS α / η grids equal `master_eval` SETTINGS and the stored arms;
  - GS takes 100 steps with hold weight 0.2 and starts at 0 / 2 / 4 / 6 / 8, writing at every later point;
  - PI and IM run at all 9 points;
  - every write is at the last position only.
- **Metrics:** Edit Fidelity is 1 − (pooled RMSE ratio); Edit Index is the mean of per-case values.
- **NN control:** k = 10, standardized Euclidean distance, and one-hot top-k, which is equivalent to Hamming distance.
- **Rayworld:**
  - frustum depths 3–12 with half-width depth/2 (x 1.5 → 6);
  - depth, then lateral position, then direction, drawn uniformly;
  - speed 0.05–0.12; radius 0.5 / 1.0;
  - N-ray instances cast N + 2 rays and drop the edge rays;
  - blink probability 0.05, warm-up 3, Geometric(1/7) capped at 12;
  - fixed reflectivities 0.4 / 0.8; 40 frames; 128 rays;
  - rays uniform in tan, so they are equally spaced on the far edge;
  - edit frame 20; teleport clear of walls and of the other disc;
  - disjoint seed ranges;
  - smooth profile (1 − (d/r)²)²; obs5 = 5 observers × 8 rays = 40 values;
  - 15 + 5 classes on 8-ray (`n_classes_on` = 20); vocabulary 421 + UNK.
- **Othello:**
  - adjacency placement over 8 neighbors;
  - pass only without a legal move; the game ends when neither player can move;
  - edits recolor an occupied non-center square after a 20-move prefix, and a case is rejected if the legal set is unchanged or empty;
  - disjoint index ranges.

**Mismatches:** findings M1–M4, m1, m4–m6, m8 and m11–m13 below.

## 3. README, scripts and artifacts

- **Scripts:**
  - Every script the README lists exists and imports: `--help` exits 0 for all 22.
  - I ran all 6 figure scripts (rc 0) and all 3 notebooks (0 errors; master_eval reports "all runs scored" and "all baselines present").
  - I reran `othello_flip_rates.py`, and its output is byte-equal to STAGING.
- **Files the notebooks open** (strace): scores / config / baselines / bayes_floor / reachability / two_flip / variance / editability_by_reachability, plus the 4 N-ray `edits.h5` headers. All are in the core bundle.
- **Dead artifacts:** see m7.
- **Missing artifacts:** see m5. The Othello edit-game split and the training corpora are also absent, but they are regenerated in place.

## Findings

### Major

**M1. The categorical Probe Skill baseline is the majority class pooled over all cells, not per cell.**
- **Where:** paper line 226 against `pim/probes/base.py:192`.
- **Evidence:**
  - The paper says "the most common class of each cell as the trivial predictor".
  - The code computes `1 - np.bincount(y_tr.reshape(-1)).max()/y_tr.size`, i.e. one class for all cells.
  - The cached stats of othello/standard give 53.12% for every point. Recomputing 1 − err/53.12 reproduces `scores.json` exactly.
  - On 4000 standard probe games, the per-cell majority error is 46.13% against 53.12% pooled.
  - Under the paper's stated definition, Table 1 standard Lin / MLP would read 0.971 / 0.972, not 0.975 / 0.976. Lower cells would move more.
  - This affects every Othello and every categorical Probe Skill.
- **Fix:** change the paper, since the code produced every number. Line 226 should read "with the most common class over all cells of the fit split as the trivial predictor".

**M2. Evaluated checkpoints are the lowest-validation-loss checkpoints, not "at 780,000 / 512,000 steps".**
- **Where:** paper lines 200, 465, 469, 519 and 1058, and README step 2; code at `pim/training/train.py:201` and `scripts/make_replicate_member.py:33`.
- **Evidence:**
  - `best_model.pt` steps of the main runs: 8-ray 555,000; 5-ray 450,000; standard-noflip 745,000; others 755k–780k.
  - Replicate steps: 8-ray__seed1 390,000; 5-ray__seed1 / seed2 450,000; adjacent-noflip__seed2 460,000.
  - Every seed-0 member is exactly 512,000, because `make_replicate_member.py` copies `ckpt/step_000512000.pt`, not the best checkpoint at or below 512k.
  - So line 469's "Table 2 … reports the same runs at 780,000 steps" is false for 8-ray and 5-ray (whose main model is earlier than its own seed-0 member).
  - The three seed members are selected by different rules.
- **Fix:**
  - Paper line 200 / 1058: add "and we evaluate the checkpoint with the lowest validation loss (checked every 5,000 steps on 16,384 held-out sequences)". The Rayworld validation loss uses 64 batches of 256.
  - Line 465 / 519: "the two new models (each at its best checkpoint within 512,000 steps) and the main run's checkpoint at exactly 512,000 steps".
  - Line 469: drop "at 780,000 steps".
  - README step 2: add the same sentence.

**M3. GS is Adam on the activation with learning rate η × (median activation std), not gradient descent with step η.**
- **Where:** paper lines 209, 213 and 1064 against `pim/editors/grad_steer.py:77` and `:105`.
- **Evidence:**
  - `_descend` runs `torch.optim.Adam([v], lr=alpha)` for 100 steps.
  - The hook passes `alpha * probe.act_scale`, where `act_scale` is the median per-dimension std of the fit activations.
  - The paper's equation `z ← z − η∇L` and "100 gradient steps of size η" describe plain gradient descent in absolute units. The η grid in line 1064 is meaningless without the scale.
- **Fix:** paper line 213 should read "takes 100 Adam steps on the latent state, with learning rate η times the median standard deviation of the latent state at that point". Note the scaling in line 1064. Optionally, write the update in the equation as an Adam step.

**M4. Rayworld edit cases must change at least 2 rays, not merely change the observation.**
- **Where:** paper line 221 against `scripts/make_edit_selection.py:35` (`--min-rays` default 2) and every shipped `edits/selection.json` (`"min_rays": 2`).
- **Evidence:**
  - 5-ray pool: 6000 cases, 1501 identical frames, 2489 kept. About 2010 one-ray edits are excluded.
  - 8-ray pool: 4000 cases, 693 identical, 2414 kept.
  - The selected teleports are longer than the pool's (5-ray 2.71 against 2.16; 8-ray 2.69 against 2.16).
- **Fix:** paper line 221 should read "We skip cases whose edit changes fewer than two rays of the next frame (Rayworld) or leaves the legal moves unchanged (Othello)".

**M5. The IM reconstruction test has no release producer.**
- **Where:** paper line 825 ("keeps the output's error within about twice the model's own only at points 4 and 5 … five to sixteen times … elsewhere").
- **Evidence:**
  - `grep -rni recon pim scripts notebooks` finds only Rayworld `reconstruct_clean_obs`.
  - The numbers come from PRIVATE `experiments/inverse_probe/scores/othello_L-oth-20m_mirror128_recon.json`, which is not shipped and not in SPEC scope.
- **Fix, one of:**
  - Add `scripts/im_reconstruction.py`: write g(unedited board) at each point with `pim.editors.inverse`, using the shipped othello/standard IM cache. Store the error against the pre-edit legal set, relative to the model's own, in `runs/othello/standard/im_reconstruction.json`. Add `tables.im_reconstruction()` and an appendix cell.
  - Or delete the two reconstruction-test sentences from line 825.

**M6. The Othello appendix figures cannot be reproduced as printed.**
- **Where:** paper line 1075 and Figs. `more_qualitative_edits_othello_1–5`, against `scripts/figures/qualitative_othello.py`.
- **Evidence:** my renders differ from PRIVATE's `othello_edits_seed{1..5}_cols.png` by 95,139 / 108,572 / 96,176 / 102,011 / 90,871 pixels. The differences are exactly the cells below:

  | cell | release draws | paper draws |
  |---|---|---|
  | adjacent-noflip GS | pt2 α0.05 | pt2 α1.5 |
  | standard-noflip GS | pt4 α0.05 | pt0 α1.5 |
  | standard-noflip IM | pt1 | pt7 |

  - The release draws Table 2's highest-fidelity fallback.
  - The paper's sentence "Where no setting reaches an Edit Fidelity of 0 … the figures show the setting with the highest Edit Index instead of the table's fallback" describes the old boards.
- **Fix:** regenerate the five paper figures with the release script and delete that sentence from line 1075. The release behavior matches the paper's selection rule.

**M7. The README's rescoring recipe silently drops the categorical blocks.**
- **Where:** `README.md:84` ("delete its `scores.json` (and its `probes/` to refit them) and run it again").
- **Evidence:**
  - `pim/scoring/rayworld.py:66-75` loads categorical probes with `require_cached=True`. On a miss it prints `SKIPPED` and writes `scores.json` without that block.
  - After deleting `probes/`, the appearance-fac, appearance and grid blocks would vanish.
  - Table 2's categorical rows and tab:categorical would then show "—" for that run.
- **Fix:** README:84 should read: "…and its `probes/` to refit them; then first rerun that run's `scripts/fit_probes.py` lines from step 3 of From scratch (the scorer reads categorical probes and their floors from the cache and skips a block whose probes are missing)".

### Minor

**m1. "At α = 1 the linear probe reads the edited state exactly" is not rendered, and it is false at point 0.**
- **Where:** paper lines 213 and 348.
- **Evidence:**
  - At point 0, `readout_err_after` of the PI α = 1 arms (cartesian block) is 1.62 on standard, 1.70 on blink, 1.66 on 128-ray, 2.53 on 16-ray, 2.64 on 8-ray and 2.44 on 5-ray.
  - At points 1–8 it is 1e-6 to 5e-6.
  - No notebook prints this.
- **Fix:** paper: "…lands exactly at every point after the embedding". Release: add `tables.pi_landing(F)`, which prints `readout_err_after` of the α = 1 arms per point, as a paper_tables cell.

**m2. The flip rates (0.27 vs 2.2) are not displayed.**
- **Where:** paper line 170.
- **Evidence:** `corpus_stats.json` (2.2449 / 0.2687, which my rerun reproduces) is read by no notebook or module.
- **Fix:** add `tables.flip_rates(runs)`, which reads `runs/_baselines/othello/<inst>/corpus_stats.json` `flips_per_move`, and a paper_tables cell "Flips per move (0.27 vs 2.2)".

**m3. Paper Figures 3 and 4 (`othello_overview.pdf`, `rayworld_overview.pdf`) have no release producer.**
- **Evidence:** they are drawn from real bench games and held-out frames by PRIVATE `paper/figs/environments_overview/{othello,rayworld}/make_figure.py`. There is no counterpart in `scripts/figures/`.
- **Fix:** port them as `scripts/figures/environments.py`, or say in the README that Figures 1–4 are illustrations with no script.

**m4. The Table 1 caption says the observation baseline is fit "on the same sequences"; it is not.**
- **Where:** paper line 290 against `tables.floor_cells` and `pim/scoring/baselines.py:30`.
- **Evidence:**
  - The observation baseline is `observation_right_large`: history aligned to the present, fit on Rayworld 250k sequences or Othello 170k games, 50 epochs.
  - The trained probes use 30k / 20k at 200 epochs.
- **Fix:** caption: "…and an observation baseline fit to the history aligned at the present on a larger corpus (250,000 Rayworld sequences, 170,000 Othello games)". Mention it in app:implementation.

**m5. Four observation floors cannot be refit from any bundle.**
- **Evidence:**
  - The `observation_right_large` floors of standard, blink, smooth and obs5 (Table 1, tab:additional_rw) were fit on `probe_250k.h5`.
  - STAGING ships `probe_250k` for 128/16/8/5-ray only.
  - The README bundle table says corpora are what is "needed for … refitting its probes".
- **Fix:** add to the README: "the large observation floors of standard, blink, smooth and obs5 need `generate_dataset.py --role probe --size 250k` first". Or ship those four corpora (about 8 GB).

**m6. The probe-refit PI spread uses the unguarded highest-Edit-Index setting, not the paper's rule.**
- **Where:** paper line 471; `scripts/probe_refit_variance.py:92` and `:109`; `tables.probe_refit_spread`.
- **Evidence:**
  - `PI@canonical_edit_best` takes `max(arms, key=edit_index)` over α.
  - On the adjacent-flip seeds this picks α 10–35, with fidelity ratios of 1.94, 2.11 and 2.67 (Edit Fidelity −0.9 to −1.7). Table 2's arms are α 5–10 and within the cutoff.
- **Fix:** paper: "…PI's Edit Index at its highest-index step size at the main run's point". Or change the script to `pim.metrics.selection.best_arm`, which would need a rerun.

**m7. Dead artifacts.** These are read by no code:
- `runs/othello/adjacent-flip/variance.json` and `runs/rayworld/standard/variance.json` (README step 5 does not regenerate them either);
- the `full` entry of `runs/rayworld/8-ray/variance.json`;
- the four `corpus_stats.json` (see m2). README step 5's default regenerates only standard and adjacent-flip.

**Fix:** drop the two variance files and the `full` entry from STAGING, or render them. Pass `--instance standard adjacent-flip adjacent-noflip standard-noflip` in README step 5, or ship only the two quoted files.

**m8. The token model's categorical rows are scored on 951 cases, not 1000.**
- **Where:** paper line 695 and Table 2 caption ("averaged over 1000 edit cases", "Columns as in …").
- **Evidence:** `bases.appearance-fac.n_cases_kept` = 951. The cartesian token block keeps 1000.
- **Fix:** tab:tokens_editability caption should add: "The categorical token rows use the 951 cases whose edit changes the next frame's token."

**m9. Two stale cells in tab:im_by_point.**
- **Where:** paper lines 852 and 855, Othello IM Index at points 3 and 6.
- **Evidence:** `scores.json` gives 0.2549 and 0.3849; the paper prints +0.26 and +0.39.
- **Fix:** change them to +0.25 and +0.38.

**m10. "IM … moves by at most 0.10".**
- **Where:** paper line 996.
- **Evidence:** `fidelity_rule_shift` gives 0.106 on adjacent-flip (+0.664 → +0.558).
- **Fix:** "by at most 0.11".

**m11. "Both probe-derived editors rise steadily" as rays coarsen on the categorical target.**
- **Where:** paper line 348.
- **Evidence:** GS goes +0.31 (128-ray) → +0.28 (16-ray) → +0.46 → +0.60. The first step is a drop of 0.03, above the seed SDs of 0.014 / 0.006.
- **Fix:** "…PI rises steadily and GS rises from 16 rays on, to +0.51 and +0.60 at 5 rays".

**m12. The Bayes-floor sampler is described as simple rejection.**
- **Where:** paper line 406.
- **Evidence:** `pim/environments/rayworld/bayes.py` runs SMC over frames with Metropolis–Hastings rejuvenation (512 particles, 500 initial sweeps, 40 per frame) and an exact frame-0 term, on the first 1000 eval sequences.
- **Fix:** add these settings to app:implementation. For example: "sequential Monte Carlo with 512 particles and Metropolis–Hastings rejuvenation, on the first 1000 held-out sequences".

**m13. `othello/standard` and `__seed0` `config.json` carry a legacy `train` block that contradicts the model.**
- **Evidence:** the block contains `d_model 256`, `n_layers 4`, `n_heads 4`, `warmup_frac 0.05`, `run_name "BIG20M_othello_L"`, `window 16`, `rung "D"`, `w16_reference_steps`, and `"arch": "theirs"` inside `train`. The real architecture is in `model` and the checkpoint: 8 / 8 / 512.
- **Fix:** in `export_artifacts.py`, drop those keys from those two configs, or rewrite the `train` block to the `TrainConfig` schema the other 41 configs use.

### Nits

- **n1. The paper omits two blink constraints** (lines 197, 1055). Only one disc may be hidden at a time, and a disc cannot start a new blackout on the frame after it reappears (`blink.py:41-52`). Fix: add "one disc at a time".
- **n2. The categorical PI target is not described in the paper.** It swaps the two classes' logits at the edited cell (`pinv.swap_class_logits`). Fix: add one sentence to app:implementation.
- **n3. `print_summaries` in master_eval prints the unguarded `best` arm with the union Edit Index.** For example, adjacent-flip IM shows pt4 +0.558 where Table 2 has +0.66. The docstring says so, but the notebook does not. Fix: add to master_eval's Summaries header "the best arm shown is the unguarded top arm; the paper's rule is in the table notebooks".
- **n4. "Fit on about 1.2M pairs"** (lines 202, 923). 1.18M is the fit plus held-out total; about 0.94M are fit. Fix: "about 1.2M pairs, 80% of them for fitting".
- **n5. `scripts/figures/editability_by_point.py` has no argument parser.** `--help` runs the script.
- **n6. `scripts/othello_flip_rates.py:37` spells its output path by hand.** It builds `REPO/"runs"/"_baselines"/…` instead of calling `layout.baselines_dir("othello", inst)`, against layout's stated contract. README step 5's `probe_refit_variance.py --run rayworld/8-ray` also runs the unreported frustum `full` refit (10 seeds). Pass `--targets appearance-fac --seeds 6`.

## Checks run

1. Both table notebooks executed (0 errors). My own parser compared 855 printed cells with the paper: 2 mismatches, both paper-side.
2. master_eval executed: a no-op, with 0 writes.
3. strace of the notebooks: every opened artifact exists; a category-level map is in `checks/`.
4. All 6 figure scripts run (rc 0). Pixel comparison with PRIVATE:

   | figure | result |
   |---|---|
   | overview | 0 differing pixels |
   | history rewrite | 0 |
   | 2 prediction figures | 0 |
   | 5 Rayworld grids | 0 |
   | by-point | legend only |
   | 5 Othello grids | 3 cells each (M6) |

   The history numbers equal the paper.
5. All 22 scripts pass `--help`. `othello_flip_rates.py` rerun is byte-equal to STAGING.
6. All 13 main checkpoints loaded and their architecture checked. All 43 config.json files diffed. The `best_model.pt` step of every run listed. Training time checked against `metrics.jsonl`.
7. Grids and points of the probe-cache provenance and the stored arms checked. Recipe constants read against every Implementation Details claim.
8. The categorical Probe Skill denominator recomputed against the cached stats, and the per-cell alternative estimated.
9. Edit selections inspected for all 8 instances.
