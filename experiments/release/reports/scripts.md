# scripts worker report

I own `RELEASE/scripts/*.py` and `scripts/demos/**`. There are now 17 scripts: 15 top-level plus the two demos.

- **Rewritten:** every script, to the SPEC writing rules. Each has a 2–5 line module docstring with one usage line and short help strings. There are no dates, names, history, research, driver or queue references, and no absolute paths.
- **Renamed:** `layout_checkpoint_replicate.py` → `make_replicate_member.py`, and `othello_corpus_stats.py` → `othello_flip_rates.py`. The old files are deleted.
- **New:** `build_rayworld_corpus.py` and `probe_refit_variance.py`.
- **Where outputs go:** paths are repo-relative. Analyses write to `runs/<id>/…` or `runs/_baselines/<env>/<inst>/…`, where the notebooks read them. Every JSON write is atomic and ends in a newline, like the shipped files.

Helpers, logs and the throwaway trees are in `experiments/release/work/scripts/`:
- `tree/` and `tree_tok/`: copies of RELEASE `pim/`, `scripts/` and `notebooks/`. STAGING is linked in read-only; outputs are real files.
- `helpers/`: the comparison scripts.
- `logs/`.
- `regen/`: the regenerated test data.

Large memmaps and checkpoints were deleted afterwards.

## What changed, per script

- **`train.py`**
  - Stage-A requests: removed `--arch` (Transformer-L only), `--objective`/`mse_onehot` and the `output_kind="raw"` line, `--dropout`, `--smoke`, `--topic`/`--run-name`, and the dated comments.
  - The run is now `--run <env>/<variant>`, written to `runs/<run>/`.
  - Paths in `config.json` are repo-relative: `corpus` uses `bc.OUT.relative_to(REPO)`, as do the token `corpus`/`vocab` and the Othello `corpus`.
  - The default instance is `standard` for both environments.
  - The recipe defaults are read from `TrainConfig` rather than restated. `--limit` is kept as "quick checks only".
- **`generate_dataset.py`**
  - Removed `--omni2d/-h/-w`, `--soft-edge`, `--soft-psf-sigma`, `--soft-occlusion-temp`, the `lambert` choice, `--direction-noise`/`--speed-noise` (never passed; SimConfig default 0), the `--seed-val/-test/-edits` overrides, `ensure_marker` and the render2d text.
  - `generated_at`/`layout` are no longer in the manifests.
  - Kept the positional four-split mode and every flag `bigcorpus.generate_shard` passes.
  - **New:** `--role` with a registered `--instance` takes its simulator flags, seeds and size from the registry. The seeds are `SEED_RANGES[inst]["eval suite"][0]` + 200M (eval) or + 300M (edits), and `probe`/`probe_large` `[0]` + 200 (probe 120k / 250k). The sizes are 10k (blink edits 20k), 120k and 250k; edits also get frame 20 and in-frustum edits.
  - So `generate_dataset.py --instance X --role eval|edits|probe --size S` regenerates each released split with no seeds on the command line. Any explicit flag still overrides.
  - `--obs-noise-std` now defaults to 0.0. Every in-scope instance passes 0.0 explicitly.
- **`build_rayworld_corpus.py`** (new): a thin loop over `bigcorpus.use_instance`, `generate_shard`, `strip_shard` and `verify`. It writes `corpus.json` with the same fields as `bigcorpus.__main__`, and adds `--concurrent`/`--workers`.
- **`make_othello_corpus.py`**: argparse (`--instance`, `--splits`, `--n-train`) over `corpus.build`, followed by `verify_splits`. This replaces the positional `runpy` shim.
- **`make_othello_edits.py`**: `--length` defaults to 20 and must be positive. The Li-mix `0` path, `shipped_length_distribution` and `ensure_marker` are removed. Output goes to `bench.cases_path(instance)` (plus `.json`).
- **`make_edit_selection.py`**
  - `vocab` is repo-relative and there is no `created` timestamp.
  - It no longer moves old files to `_unused/`; `--force` overwrites instead.
  - Per-instance default pools reproduce the released files: 6000 for blink and 5-ray, 4000 elsewhere.
- **`make_rayworld_tokens.py`**: docstring and help only. The tokenizer (rayworld-env) now writes relative `sources` and no timestamp.
- **`fit_probes.py`**
  - `--run` takes a run id. The floors go to `runs/_baselines/rayworld/<inst>/probes`.
  - `--random-init` and `--observation` are mutually exclusive.
  - It refuses a non-Rayworld run.
- **`make_replicate_member.py`** (renamed)
  - Takes `--run <id> --step 512000 [--force]` and writes `runs/<run>__seed0/`.
  - Removed: the `nearest:` mode, the `±10%` member reuse, the `commit_sha` copy and the dated note.
- **`score_prediction.py`**
  - Scans `runs/{othello,rayworld}/*/scores.json`; `--runs` takes run ids.
  - Removed: the dated `scores_backup/` copies, the `PIM_SKIP_TOPICS` filter and the ⛔ banner.
- **`bayes_floor.py`**
  - `--instance` takes `<env>/<instance>`, so the environment is explicit and not inferred from an `oth-` prefix. The output is `runs/_baselines/<env>/<inst>/bayes_floor.json`.
  - Removed: `--smoke`, `--no-sampler`, the `created` field and the dated backup copies.
  - Added `--exact-draws` and `--out` for small checks.
- **`reachability_table.py`, `two_flip_editability.py`**
  - They use run ids, write `VERSION = "1.0"` and no `created`, and step 1 writes `runs/_baselines/othello/<inst>/reachability.json`.
  - `scores.json` no longer carries `settings`. So the scorer's Othello settings are constants: `PROBE_GAMES=20000`, `GS_STEPS=100`, `GS_BETA=0.2`, as in the master_eval SETTINGS.
  - `two_flip` takes the PI and GS alpha grids and the GS start layers from the run's own scored arms. These equal `ALPHA_CAT` and `GS_LAYERS`, in the same order.
  - The default run is `othello/adjacent-noflip`.
- **`othello_flip_rates.py`** (renamed): defaults to the two instances the paper quotes, standard and adjacent-flip. The output is unchanged apart from dropping `created`: `runs/_baselines/othello/<inst>/corpus_stats.json`.
- **`probe_refit_variance.py`** (new)
  - One script replaces `probe_seeds.py` and `probe_seeds_othello.py`, with ND removed and the environment read from `config.json`.
  - It writes `runs/<run>/variance.json["probe_seeds"]` in the shipped format: Rayworld per-target entries, Othello `mine` and `inverse_map`. The same summary, seed-record keys and key order; no `written` timestamp.
  - Settings are constants (`RW_PROBE_SEQS=30000`, `RW_BENCH_N=1000`, `OTH_PROBE_GAMES=20000`). The PI alpha grid comes from the run's scored arms.
  - It merges into an existing `variance.json`. `--out` names another file.
- **`demos/demo.py`**: finds the repo itself (it failed to import without `PYTHONPATH`), and the usage line was corrected. 4 objects need `--fixed-reflectivities`; PRIVATE fails identically without it.
- **`demos/play.py`**
  - Removed the dead recurrent-model drivers and predictors: `ModelDriver`, `AutoregressiveModelDriver`, `AutoregressivePredictor` and the predictor panels. The "endogenous-action" wording is gone.
  - The human, random and avoid drivers, every key toggle and GIF saving are unchanged.

## API (CLI) changes

| script | before → now |
|---|---|
| `train.py` | `--topic T --run-name N` → `--run <env>/<variant>`. `--arch`, `--objective`, `--dropout` and `--smoke` are removed. |
| `generate_dataset.py` | The role mode no longer needs seeds or flags for a registered instance. |
| `bayes_floor.py` | `--instance` takes `<env>/<inst>`. |
| `score_prediction.py` | `--only <names>` → `--runs <ids>`. |
| `make_replicate_member.py` | `<topic/run> <step\|nearest:N>` (positional) → `--run --step`. |
| `make_othello_corpus.py` | positional arguments → `--instance --splits --n-train`. |
| `make_edit_selection.py` | `--pool` has a per-instance default; `--force` added. |
| `reachability_table.py` / `two_flip_editability.py` / `fit_probes.py` | Take run ids instead of `topic/run` paths. |

## Requests to other owners

1. **core, `pim/training/train.py`**
   - `train()` still writes a `commit_sha` file and a `"commit_sha"` field in `config.json`. The shipped configs have neither, and the value leaks the git hash of whatever tree it runs in; in my test it was PRIVATE's `bffd10f…`.
   - It also writes `"at": time.strftime(...)` into `resumed` records.
   - Change: delete `_commit_sha`, the `(run_dir / "commit_sha").write_text(...)` line, the `"commit_sha": sha` entry and the `"at"` key, and drop `commit_sha` from the docstring.
2. **scoring, `pim/scoring/othello.py`**: keep `_probe_games(n, instance)` importable under that name, or tell me the new public name. `reachability_table.py`, `two_flip_editability.py` and `probe_refit_variance.py` import it so the probe data is the scorer's.
3. **scoring, `pim/environments/layout.py`**: keep these call forms, which my scripts use:
   - `edits_file("rayworld", inst)`, `edits_manifest("rayworld", inst)`, `edits_selection("rayworld", inst)`;
   - `eval_file`/`eval_manifest(cls, inst)`, `probe_file`/`probe_manifest(cls, inst, size)`;
   - `tokens_dir(inst)`, `instance_root(cls, inst)`;
   - `othello_cases_file(inst)` (through `bench.cases_path`).

   The tree snapshot I tested already had the flattened `edits/`.
4. **scoring (optional)**: if SETTINGS move into an importable module, say so, and the three Othello scripts and `probe_refit_variance.py` will import `oth_probe_games`, `oth_gs_steps`, `oth_gs_beta`, `dw_probe_seqs` and `dw_bench_n` instead of restating them.
5. **rayworld-env** (`pim/environments/rayworld/dataset.py:227`, `edits_dataset.py:163`)
   - Every regenerated HDF5 still gets a `"generated_at"` timestamp in `config_json`. Nothing reads it now: I removed the script-side read.
   - Delete both lines if generated data should carry no timestamps. The shipped files keep theirs (export's open issue).
   - Also optional: `generate_h5` prints absolute output paths to stdout. That is display only.
6. **rayworld-env** (`bigcorpus.py`)
   - The module docstring and `__main__` duplicate `scripts/build_rayworld_corpus.py`. Point the docstring at the script, or drop `__main__`.
   - Keep the signatures of `generate_shard(k, workers)`, `strip_shard(k)`, `verify()`, `use_instance`, `obs_path` and `SEED_RANGES`, which the scripts use.
7. **infra (`README.md`)**: use the commands listed in my structured result. Also:
   - `outputs/`, `.scratch/` and `runs/*/`, `datasets/*/` data must stay gitignored;
   - `make_rayworld_tokens.py` needs the 8-ray corpus plus its eval, probe and edits splits, and must run before `make_edit_selection.py --instance 8-ray` (the selection filters on the vocabulary).

## Verification (throwaway tree; STAGING and PRIVATE data read-only)

Results for checks a–i; the counts are exact.
- **(a) `--help`:** all 17 scripts, 0 failures. Rerun on the final versions.
- **(b) Rayworld data**
  - `generate_dataset.py --instance X --role …` with registry defaults, first rows (24 for 8-ray and blink, 8 elsewhere): all 28 released splits (8 instances × eval, edits, probe_120k, plus probe_250k on 128/16/8/5-ray) are **bitwise identical** to STAGING in every HDF5 field.
  - For all 28, the stored `dataset` config equals STAGING's `n_samples`, `base_seed`, edit parameters and every `sim` key. The extra keys newer SimConfig versions add hold their defaults.
  - The registry default `(n, seed)` matches the stored value for all 28, including blink edits at 20,000.
  - `build_rayworld_corpus.py`, run in full with the shard size patched to 4: 40 shards → 160 rows, which equal PRIVATE `obs.f32` rows `k·500000 + i`, plus `meta.h5` positions, velocities, seeds, reflectivities and radii. **IDENTICAL for 8-ray and blink.**
  - A full real shard 0 of 8-ray (500,000 sequences, 231 s): **identical** to PRIVATE rows 0–499,999.
- **(c) Othello data**
  - `make_othello_corpus.py`, all 4 variants: train rows 0–1999 equal PRIVATE `train_20000000.npz`; test (10k) equals STAGING `test_10000.npz`; edits (10k) equals PRIVATE `edits_10000.npz`. **IDENTICAL**, and `verify_splits` passes.
  - `make_othello_edits.py`, all 4 variants: `cases_1000.pkl` is **byte-identical** to STAGING. `cases_1000.json` is byte-identical except the `minutes` value (0.0 vs 0.1). Rerun on the final version for standard.
- **(d) `make_edit_selection.py`**: `selection.json` is **byte-identical** to STAGING for 8-ray (with the vocabulary filter), 5-ray, standard and blink, using the default pools.
- **(e) `train.py`**, 50 steps each:
  - Rayworld 8-ray: frames, `--limit 10000` on the regenerated shard 0; val 0.009746.
  - Othello adjacent-flip: `--limit 20000`, `--seed 1 --replicate-of othello/adjacent-flip`; val 3.6985.
  - The 8-ray token model: a 10k-sequence corpus tokenized by `make_rayworld_tokens.py`, vocab 417 on that subset; val 4.3297.
  - All three wrote `config.json` with repo-relative `corpus`/`vocab` and the right `replicate` block. The one unclean field is `commit_sha` (request 1).
- **(f) `fit_probes.py`**
  - Tiny subset (2000 sequences, 2 epochs) into a scratch cache runs for appearance-fac (model, `--random-init`, `--observation`) and `pos@appearance`.
  - With the full recipe on a copy of the shipped probes, every call is a **pure cache hit**: 79 files before and after for the run, 44 before and after for the baselines, no refit. The printed skills equal `scores.json`:
    - appearance-fac 0.9351/0.9427, grid-16x8 0.1560/0.2123, `pos@appearance` 0.9587/0.9907;
    - floors: observation 0.4769/0.9349, random-init 0.9217/0.9349, appearance random-init 0.8773/0.8877, tokens random-init 0.9211/0.9249.
  - `make_replicate_member.py --run rayworld/8-ray --step 512000`, from PRIVATE's `ckpt/step_000512000.pt` (read-only): `config.json` is **byte-identical** to STAGING `8-ray__seed0`. `best_model.pt` has the same sha256 (`5a1c8437…`) and fingerprint `4c793847406e`.
- **(g) Analyses**
  - `score_prediction.py`, with the block stripped first:
    - othello/standard `prediction` is **identical**;
    - rayworld/8-ray: 4 float leaves differ by ≤ 1.2e-8 relative (loss 0.005760833761 vs 0.005760833777). This is float32 GPU forward-pass noise; the script only calls `score_run`. No other block changed.
  - `bayes_floor.py`:
    - othello/standard: **byte-identical** to STAGING, and a rerun skips as current;
    - rayworld/8-ray and blink, tiny (3 sequences, 32 particles, CPU): run, version "1.0", same key set as the shipped files.
  - `reachability_table.py`:
    - step 1 in full for adjacent-noflip and standard-noflip: equal to the shipped `reachability.json` except `minutes` (0/1000 reachable);
    - the script's own `classify` on the first 100 standard cases (budget 5M): records (verdict, nodes, witness) identical to the shipped file (46/53/1);
    - step 2 for othello/standard and othello/adjacent-noflip: `editability_by_reachability.json` **IDENTICAL**, every float included.
  - `two_flip_editability.py`: adjacent-noflip (default), standard, and standard-noflip `--no-legal` are **identical to the shipped JSON except `minutes`**.
  - `othello_flip_rates.py`: **2.2449** (standard) and **0.2687** (adjacent-flip) at 10,000 games. `corpus_stats.json` is **byte-identical** for both.
  - `probe_refit_variance.py`, 2 seeds (the full 6–10 seed runs take about 3.5 min per Othello linear seed plus the IM refits and much longer for the 200k-sequence categorical Rayworld fits, so they were not repeated):
    - **rayworld/8-ray, `full`:** writes the shipped format (same summary keys, recipe and seed-record keys).
      - Seed 0 skills are exact; PI Edit Index differs by 3e-8.
      - Seed 1 points 1–8 are equal to 6 decimals. Point 0 (the rank-deficient embedding point, whose shipped value swings from −0.91 to +0.17 across seeds) gives −0.6758 vs the shipped +0.1724.
      - **PRIVATE's own `fit_probes(seed=1)` on this GPU gives −0.675807 as well**, so the port is exact and the shipped number came from other hardware.
    - **othello/standard, `mine` + `inverse_map`:** same format.
      - Seeds 0/1 differ from the shipped file by ≤ 5.3e-7 relative. IM seed 0 (a cache hit on g) differs by ≤ 2.4e-7.
      - IM seed 1 (a fresh g fit): g R² 0.82625 vs 0.82794 shipped, IM Edit Index 0.8147 vs 0.8124. **PRIVATE's `inverse_arms(seed=1)` on this GPU gives exactly the release values** (g R² 0.8262487365667386, Edit Index 0.814748903309029).
      - IM-NN matches to 1e-11, so the split is identical and only the GPU optimization differs.
- **(h) Demos, headless (`MPLBACKEND=Agg`):**
  - `demo.py` with `--save` (3 objects; and 4 objects with `--fixed-reflectivities`) and without `--save` (`plt.show` under Agg);
  - `play.py` with `--driver avoid`/`random`/`human`, each with `--save`, and live `--dynamics shift`.
  - All exit 0. I inspected a frame of each GIF and they render correctly.
- **(i) ruff:** `ruff check --isolated --select F,E9` on every script: "All checks passed!". Also passes with PRIVATE's E/W/F config.
- **No writes into PRIVATE or STAGING:** 0 files in STAGING newer than my first helper, 0 writable STAGING entries, 0 files in PRIVATE `runs/` or `datasets/` newer. The PRIVATE check fits wrote their probe caches to `work/scripts/private_check_cache/`, and their `.scratch/` memmap was removed by the fit itself. I created no `__pycache__` in RELEASE (every run used `PYTHONDONTWRITEBYTECODE=1`).

## Open issues

- **Re-running `probe_refit_variance.py` on other hardware** reproduces the seed-0 rows and the per-seed skills at points 1–8. It does not reproduce, bit for bit, the fresh inverse-map fits or the rank-deficient point-0 skill. The quoted SDs (≤ 0.0007 skill, IM refit 0.004 / 0.036) are over seeds, so they are only affected at that noise level. It was not rerun at full size.
- **The cases-manifest `recipe` string** ("one occupied non-centre tile recoloured …", in `make_othello_edits.py`) keeps its British spelling so the regenerated `cases_1000.json` matches the shipped one. To change it, change both the script and the shipped manifest.
- **The scorer's settings are restated** in three scripts as constants (request 4).
- **Rayworld training corpora** are 20M sequences; shard 0 alone takes about 4 min on 16 cores. The tests covered shard 0 in full and all 40 shards at 4 rows each, not a full corpus build.
- **`train.py` still writes `commit_sha`** until core acts on request 1.
