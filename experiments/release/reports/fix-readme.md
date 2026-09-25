# fix-readme

Scope: RELEASE `README.md`. `.gitignore` needed no change: it already covers `.scratch/`, `.ruff_cache/`, `outputs/`, `runs/*`, `datasets/*`, `/MANIFEST.json`, `/SHA256SUMS` and `/.cache/`.

Work dir: `experiments/release/work/fix-readme/`.
- `README.before.md`: the README before my edits.
- `quick.sh` and `quick1.log`: the published quick check, run exactly as written, and its log.
- `tree_quick1/`, `tree_help/`: throwaway trees.

## Changes (all 10 items)

1. **Placeholders.** `<ARTIFACTS_URL>` in prose is now in backticks. `<ANON_HF_REPO>` appears only inside code blocks. After rendering with mistune, no bare `<...>` is left in prose.
2. **Terms table**, placed after Repository layout. It covers run and run id, instance and variant, block, arm, bench, basis, floors (with their directory), fidelity cutoff and Transformer-L.
   - The arm row says "a residual point and a step size". IM arms also store `alpha` 1.0, so this holds for IM too.
3. **"Reading `scores.json`"** paragraph, under Reproduce the paper. It covers:
   - `best_arm`: the highest Edit Index within the cutoff, else the highest Edit Fidelity. The fidelity-selected table uses `best_arm_by_fidelity`, which I checked in tables.py.
   - `best` is the top arm without the cutoff (`blocks.top_arm`).
   - Edit Fidelity = `1 - fidelity_ratio`.
   - Othello reports `edit_index_symdiff`.
   - In a Rayworld run, `bases` maps each block to its scores. Othello keeps its arms at the top level, and its `bases` is `{}`.
4. **Scoring paragraph** now says:
   - The scorer reads categorical probes only from the cache, so step 3's `fit_probes.py` lines come first.
   - The corpora bundle has `probe_250k` only for 128/16/8/5-ray, which I checked in STAGING. The categorical probes of standard and blink, and the large observation floors of standard, blink, smooth and obs5, need `generate_dataset.py --role probe --size 250k` first.
   - `score_prediction.py --runs` follows.
   - The three qualitative scripts need `--recompute`. They are the only figure scripts with that flag. `history_rewrite`, `predictions` and `editability_by_point` keep no cache.
5. **Checkpoints** (step 2).
   - What the code does: `pim/training/train.py` validates every `val_every` = 5,000 steps and at the last step, and saves `best_model.pt` whenever the validation loss improves.
   - Seeds 1 and 2 train for 512,000 steps. Their best checkpoint is therefore within 512,000 steps: validation runs at 510,000 and at the final step.
   - `make_replicate_member.py` copies `ckpt/step_000512000.pt` to seed 0's `best_model.pt`. 512,000 is on the log-spaced checkpoint schedule (1,000 × 2^9).
   - The README states only this.
6. **Resources.**
   - **Corpora.** The README says about 2.1 TB, not the brief's 1.9 TB, which counts frames only.
     - Frames: 20M × 40 × rays × 4 B gives 409.6 GB per 128-ray instance, 128 GB for obs5 (40 rays), and 51.2 / 25.6 / 16 GB for 16/8/5-ray. That totals 1.86 TB.
     - Metadata: each `meta.h5` adds 26.1 GB (measured on the PRIVATE corpora; 16-ray is not on this host, so I computed it).
     - The README gives about 2.1 TB in total and about 440 GB per 128-ray instance (measured 435.7 GB).
   - **Scratch.** A 30k-sequence residual stack is 22 GB (`collect_residuals` docstring). The README says "up to about 22 GB into `.scratch/`, so keep about 25 GB free".
   - **Figures.** The measured 8 GB is peak RSS, that is RAM, from figures-recheck. GPU memory was not measured. So the README says "up to about 8 GB of RAM", not GPU memory.
   - **250k probe corpora:** see item 4.
7. **Othello probe games.** The README says they are regenerated "within about a minute". Measurements disagree: figures-recheck saw 6 s per variant timed alone (65 s vs 58 s for the whole script), and the pipeline verifier saw 82 s vs 24 s. "Within" covers both.
8. **Figures, Demos and Quick check.**
   - Figures 1 to 4 (teaser, setup, the two environment overviews) are illustrations and are not regenerated. I checked the figure order in `paper_draft.tex`.
   - New Demos section: `demo.py --seed 8 --n-objects 4 --fixed-reflectivities` and `play.py`, with their keys, and `--save` (play.py then falls back to a scripted driver).
   - New "Quick check" subsection: the 13 toy commands verified by fix-scripts.
     - Run them in a fresh clone without the download, because they write datasets where the full ones go. `generate_dataset.py` refuses to overwrite, and `make_rayworld_tokens.py` does not check.
     - I kept the 120k and 250k probe lines. Categorical fits read the 250k corpus, and the tokenizer reads the probe split.
9. **macOS checksums.** `shasum -a 256 -c --ignore-missing SHA256SUMS` is in the comment beside the `sha256sum` line. `shasum` 6.x has `--ignore-missing`.
10. **IM reconstruction.** `python scripts/im_reconstruction.py` is added to step 5, which now also says the table notebooks read the flip rates and the reconstruction test. The paper's word is "reconstruction test", and "two-disc edits" became "two-token edits".

**Length.** The README is 326 lines, up from 263.
- The added material is about 75 lines.
- To compensate, I dropped step 4's duplicate notebook command, dropped the Variants intro (now covered by the Terms rows), and moved the Othello probe-split note into Figures.
- I left the Highlights and the intro unchanged, because they are the paper's claims.

## Verification

| check | result |
|---|---|
| Quick check, exactly as published, in a fresh tree (current RELEASE `pim/`, `scripts/`), plus both demo lines with `--save` | 15/15 rc 0; 52 s for the 13 quick-check commands, 74 s with the demos (`quick1.log`) |
| Every `python scripts/...` line in the README (52 commands, 24 scripts): script `--help` exits 0 and lists every flag used | 0 problems |
| `--instance` values vs `bigcorpus.INSTANCES` / `othello.corpus.INSTANCES`; run ids in the loops vs `runs/` | all present |
| Every referenced file (`assets/teaser.png`, notebooks, scripts, `LICENSE`, `setup.sh`, vendor dir) | all exist; `waterfall.py` or other removed files are not referenced |
| mistune render | 3 tables with consistent column counts, 13 code blocks, image `assets/teaser.png`, 11 h2 + 1 h3; placeholders render as code |
| Writing rules | no em or en dashes, no semicolons in prose, no dates, names, paths or `discworld`/`dw-`, no British spellings. The two history-like words ("was", "since") were reworded |
| Bundle sizes from `MANIFEST.json` | core 2.94, corpora 9.35, replicates 3.52 GB, which match the table |

Nothing was written to RELEASE except `README.md`. No `__pycache__` was created (`PYTHONDONTWRITEBYTECODE=1`), and all runs used my throwaway trees.

## Requests

- **export:** ship `runs/othello/standard/im_reconstruction.json`. README step 5 and the appendix notebook refer to it, and fix-scripts and fix-tables requested the same.
- **export:** if the final export changes the bundle contents (for example by adding the reconstruction file or dropping unread `variance.json` files), recheck the three bundle sizes in the README table. The current MANIFEST gives 2.9 / 9.4 / 3.5 GB.

## Open issues

- `<ARTIFACTS_URL>` and `<ANON_HF_REPO>` are still placeholders. Filling them needs a human step with an anonymous host account.
- Step 5's `probe_refit_variance.py --run rayworld/8-ray` uses the defaults, so it also refits the unreported frustum `full` target. That matches the shipped `variance.json`. If export drops that key, pass `--targets appearance-fac`.
