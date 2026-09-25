# Verifier report: rescore (GPU)

**Verdict: the release scorer reproduces the shipped Rayworld scores from scratch.** Nothing
differs by more than 1.2e-7, apart from the `minutes` field and the documented extra fields.
Every table cell, every selected arm and every rounded value is unchanged. The four findings are
all documentation or metadata issues (2 minor, 2 nit). No blocker, no major.

Work dir: `experiments/release/work/rescore/`
- trees: `treeA` (8-ray), `treeB` (standard), `treeC` (replicate), `treeD` (whole bundle, dry run only)
- results: `results/` (comparison JSON/TXT, the fresh `scores.json` files, the executed notebooks)
- helpers: `compare_scores.py`, `gs_spot.py`

Every tree is a copy of RELEASE `pim/ scripts/ notebooks/`. Each run dir and its baselines were
copied into the tree, and the datasets were symlinked read-only into STAGING. The scorer ran as
`jupyter-nbconvert --execute notebooks/master_eval.ipynb`, with the notebook's own SETTINGS and
`PYTHONPATH=<tree>`. `pim.__file__` was under the tree.

**No writes to RELEASE, STAGING or PRIVATE `runs/` / `datasets/`:**
- `find -newer` finds nothing in STAGING.
- There is no `__pycache__` in RELEASE.
- `sha256sum -c` passes for the touched STAGING files.
- RELEASE `.git` has mtime 10:04:09, during my session. That was not me: I ran no git command.

## (1) rayworld/8-ray: scores.json deleted, probes/ kept

- **Run.** The notebook ran in 23.8 min. Baselines skipped; the run was scored in full, all 8 blocks.
- **Files.** No probe file was written: all 79 were cache hits. The only new file is `scores.json`.
- **Numbers.** 18,740 numeric leaves were compared, with arms matched by (editor, point, alpha, dims).

| group | max abs diff | where |
|---|---|---|
| PI arms, all 8 blocks (1296–1728 leaves each) | **0** | – |
| GS arms, all 8 blocks, including every categorical block | **0** | – |
| Probe Skill, per-dim, sanity, unedited, alphas, recipe | **0** | – |
| IM arms | 1.19e-7 | grid-16x8 IM pt1 `write_ratio` 0.517227352 vs 0.517227471 |
| IM-NN arms | 4.4e-8 | pos@appearance IM-NN pt6 `fidelity_ratio` |
| inverse_map (g_r2, g_rmse, nn_r2) | 5.3e-9 | cartesian `nn_r2[2]` |
| best / best_by_dims | 2.2e-8 | appearance `best.IM.fidelity_ratio` |

- **Selection (table path).** The table path (`pim.figures.tables.block_row`, both `select="index"` and `"fidelity"`) picks the same arm for every block and editor, and no value changes at 2 or 3 decimals.
  - Cartesian: PI pt3 α175, GS pt0 α0.7, IM pt6, IM-NN pt7.
  - Rounded values: LIN 0.886 / MLP 0.932; PI +0.21 / 0.00; GS −0.07 / 0.12; IM +0.71 / 0.73. These match paper Tables 1 and 2.
- **Expected key differences only:**
  - the fresh file adds the per-case spread fields (`*_case_sd/se`, `*_ci95_*`, `*_n_cases`);
  - `prediction` is dropped;
  - `frustum.bench_selection` goes from null to a record (finding 2);
  - categorical `inverse_map` gains `nn_r2 = [NaN]*9` (finding 4).
- **Prediction block.** `scripts/score_prediction.py` adds the prediction block back, and it matches the shipped block to 3e-11.
- **Replicate skip (treeC).** `rayworld/8-ray__seed1` was copied with its `scores.json`. The notebook printed `skip  rayworld/8-ray__seed1  (scored at 1.0)`, and the baselines were skipped. `score_all` returned `[]` and all 83 file hashes are unchanged.
- **Whole bundle (treeD).** A dry run over the STAGING bundle skips 43/43 runs and 12/12 baselines.

### Private-scorer spot-check of categorical GS (`gs_spot.py`)
I ran GS@L0 at the six categorical alphas on appearance-fac, standalone, with PRIVATE `pim.environments.discworld` and with the release package. Private probes were copied into the work dir first.

- **8-ray main run.** Private, release standalone, the fresh release scorer and the shipped value are all bitwise equal (0 on every leaf). So the categorical GS drift reported in `scoring.md` does not occur on the main run.
- **8-ray__seed0.** Private and release are bitwise equal on this GPU (0 on every leaf). Both differ from the shipped value, which equals PRIVATE's file:
  - Edit Index by up to 7.50e-3 (α 0.7: +0.4172 vs +0.4097);
  - the fidelity ratio by up to 1.79e-3;
  - any leaf by up to 8.98e-3 (`edit_index_ci95_hi`).
- **Conclusion.** This confirms the scoring worker's claim: the release code equals the private code, and the drift comes from the compute environment the shipped replicate blocks were made on.

## (2) rayworld/standard: scores.json AND probes/ deleted (everything refit)

- **Run.** The notebook ran in 28.4 min. It refit 22 probe files: LIN + MLP for frustum and cartesian, and IM at 9 points per basis. Their filenames equal the shipped ones.
- **Probe files.** All tensors are bitwise equal to the shipped files: 18 inverse maps with 8 tensors each, and 4 forward probes with 54–72 tensors each. Only the forward probes' stored stats differ, by up to 1.6e-7. The inverse maps' 518 stats each are bitwise equal.

| group | max abs diff |
|---|---|
| PI arms (1728 per basis), GS arms (500 per basis), unedited | **0** |
| Probe Skill LIN / MLP | 5.8e-8 |
| probe per-dim / sanity | 1.4e-7 |
| IM arms | 3.0e-8 |
| IM-NN arms | 1.2e-7 |
| inverse_map | 2.3e-9 |

- **Selection.** The selected arms are unchanged, and so are all rounded cells.
  - Cartesian: PI pt5 α12, GS pt0 α0.35, IM pt6, IM-NN pt8.
  - Rounded values: LIN 0.872 / MLP 0.973; PI −0.10 / 0.04; GS −0.16 / 0.04; IM +0.59 / 0.66; g R² at IM 0.339. These equal paper Tables 1 and 2.
- **The appearance-fac block could not be refit.**
  - It was `SKIPPED (no cached probes …)`: the scorer never fits categorical probes, and `probe_250k` is not shipped for standard.
  - A follow-up dry run prints `WOULD add to rayworld/standard: blocks ['appearance-fac']` (finding 1).
- **Add-back.** I then copied only the 2 shipped appearance-fac probe files into the tree and ran the notebook again (2.6 min).
  - The block was added with 0 diff on 1296 PI, 300 GS and 84 skill / sanity leaves, and on the unedited card.
  - The final file therefore matches the shipped file on every block.
- **Prediction.** `score_prediction.py` reproduces the prediction block to 6e-12.

## Findings

1. **minor — README.md line 84–86.** The rescore instructions drop the categorical blocks.
   - The README says "To rescore a run, delete its `scores.json` (and its `probes/` to refit them) and run it again with the corpora bundle in place."
   - But the scorer only reads categorical probes from the cache. On standard and blink their corpus (`probe_250k`) is not in the corpora bundle.
   - Following the instruction silently loses the appearance-fac block: the scorer prints SKIPPED, then WOULD add on every later run. `scripts/figures/qualitative_rayworld.py` then leaves the Standard categorical row blank.
   - Fix: after "…to refit them)", add: "The scorer never fits categorical-target probes: keep their files, or refit them first with `scripts/fit_probes.py` (From scratch, step 3). For `standard` and `blink` this needs `scripts/generate_dataset.py --instance I --role probe --size 250k`, which is not in the corpora bundle."
2. **minor — STAGING `scores.json`: 7 blocks carry `bench_selection: null`.**
   - The blocks are 16-ray, 5-ray, 8-ray, blink, smooth and standard `frustum`, and obs5 `cartesian`.
   - This contradicts `pim/scoring/blocks.py:111` ("None = the first n cases"). A fresh score of the same block writes the selection record, and every arm matches bitwise, so these blocks were scored on the selected cases.
   - Fix (export): set each null to the fresh-score record. For the frustum blocks this is exactly the same run's `cartesian.bench_selection`; I checked this for 8-ray and standard. For obs5, use the record the scoring worker's obs5 rescore wrote.
3. **nit — `pim/scoring/summary.py:18,24` (the notebook's summaries).**
   - The column headed `fid` prints `fidelity_ratio`, not Edit Fidelity (1 − ratio). The arm shown is the unguarded top arm.
   - A reader comparing the output with the paper therefore sees, for example:
     - 8-ray cartesian IM "0.265" where the paper has 0.73;
     - standard PI "+0.2310 / 1.536" where the paper has −0.10 / 0.04.
   - Fix: change `'fid'` to `'ratio'`, and add to the notebook's "Summaries" markdown: "the unguarded top arm and its fidelity ratio; the tables apply the selection rule and report Edit Fidelity = 1 − ratio."
4. **nit — `pim/scoring/blocks.py` `attach_inverse`.** A fresh categorical `inverse_map` gains `"nn_r2": [NaN]*9`, which the shipped blocks lack. It is harmless, because the tables read NaN either way. It also adds non-standard `NaN` tokens to the JSON; the shipped 8-ray and 8-ray-tokens files already contain 36 and 9. Fix (optional): write `nn_r2` only when a retrieval bank exists.

Caveat to document, not a code defect: rescoring a replicate or token-model categorical block reproduces GS only to about 7.5e-3 in Edit Index (seed0 above). The selected arm and the two-decimal values are unchanged. A fresh seed-SD would move by about that much. The README does not say so. Suggested sentence for the README Scoring paragraph: "Rescoring reproduces `scores.json` to about 1e-7, except GS on the categorical blocks of the seed replicates and the token model, which were scored on another GPU and shift in the third decimal."

**Not verified here:** Othello, blink, tokens, obs5 and smooth rescoring; baseline rebuilds; categorical probe refits (`fit_probes.py`).
