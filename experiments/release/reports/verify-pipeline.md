# Verifier report: pipeline

**Verdict: issues.** Every "Reproduce the paper" command works on the staging artifacts and gives the paper's numbers and figures. The full-scale "From scratch" commands are consistent and in the right order. The paper's small-split generators regenerate the shipped data bit for bit.

The from-scratch path cannot be run at small scale as shipped:
- There is no size flag for the Rayworld corpus, and `train.py` requires a 20M-sequence corpus file.
- The `--limit` quick-check flag of `train.py` can hang forever.
- Several full-scale sizes and scorer settings are hard-coded in the scorer and the analysis scripts. On smaller data they crash with cryptic errors, or silently skip blocks.

No blocker was found in this scope. No number is wrong and no core path is broken at full scale.

Work dir: `experiments/release/work/pipeline/`
- `repro/`: a copy of the release code with symlinks to STAGING (part a).
- `fresh/`: empty `runs/` and `datasets/`, the tiny pipeline (part b).
- `core/`: `datasets/` without any `probe/` dirs, standing in for the core bundle alone.
- `rescore/`: `othello/standard` copied without `scores.json`.
- `tiny.py`: the corpus-size wrapper, described below.
- `logs/`.

Nothing was written to RELEASE, STAGING or PRIVATE `runs/`, `datasets/`, `logs/` or `outputs/`. No git commands were run.

## Part (a): Reproduce the paper, run exactly as written against STAGING

| command (README) | result |
|---|---|
| `nbconvert … paper_tables.ipynb --output paper_tables.executed.ipynb` | rc 0, 2.4 s. Table 2 is cell for cell the paper's `tab:editability`, daggers included. IM vs IM-NN mean gain 0.53633. Seed-SD maxima 0.0037 / 0.0344, 4 cells > 0.1 |
| `nbconvert … appendix_tables.ipynb --output …` | rc 0, 3.3 s, no error outputs |
| both notebooks with `CUDA_VISIBLE_DEVICES=""` | rc 0; text outputs identical to the GPU run ("tables need only a CPU" holds) |
| `nbconvert --ExecutePreprocessor.timeout=-1 … master_eval.ipynb` | rc 0, 4.4 s: 43 runs `skip (scored at 1.0)`, 12 baselines skip (a true no-op) |
| rescore (README): `othello/standard` with no `scores.json`, `probes/` kept, then master_eval, then `score_prediction.py --runs othello/standard` | rc 0, 116 s. PI/GS arms are bit-identical to the shipped `scores.json`; IM/IM-NN arms agree within 8e-7. The new file only adds per-case CI fields (`*_ci95_*`, `*_case_se`, `*_n_cases`) that the shipped (and PRIVATE) files lack, and lacks the out-of-scope `gates.output_kind/out_sum_mean/out_neg_mass_mean` |
| the 6 figure scripts | all rc 0: overview 57 s, rayworld 34 s, othello 24 s, editability_by_point 1 s, history_rewrite 3 s, predictions 11 s. History numbers equal the paper (+0.61/+0.63/+0.65; step 1 +0.59, step 14 +0.23, single write −0.63/−0.74; RMSE 0.114 vs 0.264) |
| same 6 scripts in `core/` (no `probe/` dirs) | all rc 0. All 16 PNGs are pixel-identical to the full-bundle run. The 4 regenerated `probe_20000.npz` are bit-identical to the shipped ones |
| `sha256sum -c --ignore-missing SHA256SUMS` (in STAGING) | 1494/1494 OK |

## Part (c): `--help`

All 23 scripts (`scripts/*.py`, `scripts/figures/*.py`, `scripts/demos/*.py`) exit with rc 0.

## Part (b): From scratch at tiny scale (`fresh/`)

What ran, in README order:
- **8-ray splits.** eval (full 10k), edits, probe_120k and probe_250k (`--n 3000`).
- **Tokens.** `make_rayworld_tokens.py`.
- **Edit selection.** `make_edit_selection.py`. It gives 1000 cases, identical to the shipped `selection.json`.
- **Othello data.** `make_othello_corpus.py --n-train 20000` and `make_othello_edits.py`. test, probe, probe_large and `cases_1000.pkl` are identical to the shipped files.
- **Training, 200 steps.** 8-ray frames, 8-ray tokens, Othello (`--limit 20000`), `make_replicate_member.py --step 124` and a seed-1 replicate.
- **Extra probes.** `fit_probes.py` (appearance-fac / appearance / grid-6x5 / pos@appearance, `--random-init`, `--observation`, token run) with `--n-seq 2000 --epochs 2`.
- **Scoring.** master_eval with reduced SETTINGS in MY copy only (`rw_probe_seqs` 2000, `oth_probe_games` 2000, `rw_bench_n` 200, GS steps 10, `oth_gates_games` 1000), then `score_prediction.py`.
- **Analyses.** `bayes_floor.py` (Othello and tiny Rayworld), `othello_flip_rates.py`, `reachability_table.py`, `two_flip_editability.py`, `probe_refit_variance.py`.
- **Tables.** `tables.collect` / `table_decodability` / `table_editability` on the tiny runs work.

Bit-identity checks:
- The regenerated 8-ray eval split and the edits/probe prefixes equal STAGING.
- The standard probe_250k prefix equals PRIVATE.
- The 8-ray training corpus shard prefixes equal PRIVATE (`obs.f32` and seeds).
- The Othello Bayes floors (standard, adjacent-flip, adjacent-noflip) and `corpus_stats.json` (standard) are JSON-identical to STAGING.

The Rayworld corpus and frame training need the wrapper `work/pipeline/tiny.py` (see M1). It sets `bigcorpus.SHARD_N` / `N_SHARDS` to 10000 × 2 and then runs the script. No release file was modified.

## Findings

### Major

**M1. There is no small-scale path to a Rayworld training corpus or frame model.**
- Where: `scripts/build_rayworld_corpus.py:29`, `pim/environments/rayworld/bigcorpus.py:18,223`, `scripts/train.py:73`.
- `build_rayworld_corpus.py` has no size flag. `SHARD_N, N_SHARDS = 500_000, 40` is fixed, and `open_obs` always maps `(N_TOTAL=20M, 40, R)`.
- `train.py` on any smaller `obs.f32` fails with `ValueError: mmap length is greater than file size`. `--limit` does not help, because the map is opened first.
- `tokens.tokenize_instance` already reads `n` from `train/corpus.json`, so the two readers disagree.
- Fix:
  - Add `--shards N` and `--shard-n N` to `build_rayworld_corpus.py`. Pass them to `bc.use_instance` and record them in `corpus.json`, which already stores `shard_n` and `n_shards`.
  - Make `bigcorpus.open_obs` and `verify` take `N` from `train/corpus.json` when it exists, instead of `N_TOTAL`.

**M2. `train.py --limit N` hangs forever for Rayworld frames when N < 6144.**
- Where: `pim/training/sources.py:27-28`, `pim/training/stream.py:25,31`.
- `n_val = max(2*block, 0.1 n)` with block 2048 leaves `n_train < block`. `BlockStream.starts` is then empty, the worker spins in `while True: for s in []`, and `batches()` blocks on `q.get()`.
- `--limit 5000 --steps 20` printed nothing in 90 s and had to be killed.
- Fix:
  - In `rayworld_source`, `if n_train < block: raise ValueError(f"--limit {n_total} leaves {n_train} training sequences, fewer than one {block}-sequence block; use at least {3*block}")`.
  - In `BlockStream.__init__`, `assert len(self.starts), "range shorter than one block"`.

**M3. The categorical probe recipe is fixed, so `fit_probes.py --n-seq/--epochs` fits are never read and the categorical blocks are silently skipped.**
- Where: `pim/environments/rayworld/arms.py:54`, `scripts/fit_probes.py:45-46`, `pim/scoring/rayworld.py` `_fit`.
- The scorer and the floors look probes up with `GRID_PROBE_RECIPE` (200k sequences of probe_250k, 50 epochs). No SETTINGS key changes it.
- In the tiny run, every categorical block of every run printed `SKIPPED (no cached probes …)` and so did every categorical floor. The `scores.json` files silently lack those blocks.
- Fix:
  - Add SETTINGS `rw_cat_probe_seqs` and `rw_cat_probe_epochs`. Thread them through `probe_recipe(target, inst, n_seq, cat_n_seq=…, cat_epochs=…)` in `rayworld.py` / `baselines.py`.
  - Give `fit_probes.py` the same defaults, or at least have it print that the scorer ignores overridden fits.

**M4. Hard-coded corpus sizes crash with cryptic errors on smaller corpora.**
- Where: `pim/environments/rayworld/arms.py:121,168,251`, `pim/scoring/baselines.py:30`.
- `_read_corpus` slices `[:n_seq]` and then indexes `permutation(n_seq)` without checking the file size. The failures:
  - master_eval on a 3000-sequence `probe_250k.h5`, with the hard-coded `LARGE["rw_n_seq"]=250_000`: `AcceleratorError: CUDA error: unspecified launch failure` (an out-of-bounds GPU gather).
  - `fit_probes.py` with its default recipe: `IndexError: index 3000 is out of bounds`.
  - `probe_refit_variance.py --run rayworld/8-ray`: `IndexError: index 6333 … size 3000`.
- The Othello large floors are also fixed at 170k games and 50 epochs. They cost 12 min even in the tiny run.
- Fix:
  - In `_read_corpus`, `if len(obs) < n_seq: raise ValueError(f"{h5_path.name} holds {len(obs):,} sequences; this fit needs {n_seq:,}")`.
  - Expose `LARGE` as SETTINGS keys (`rw_large_seqs`, `oth_large_split`, `large_epochs`).

**M5. The analysis scripts hard-code the scorer's SETTINGS instead of reading what the run was scored with.**
- Where: `scripts/reachability_table.py:35`, `scripts/two_flip_editability.py:36`, `scripts/probe_refit_variance.py:28`.
- These constants hold `PROBE_GAMES=20_000`, `GS_STEPS=100` and `RW_PROBE_SEQS, RW_BENCH_N = 30_000, 1000`. On a run scored with other SETTINGS:
  - `reachability_table.py` refits the probes at 20k games (6 min), then stops with `GS: … MISMATCH … nothing written`.
  - `two_flip_editability.py` raises `AssertionError: GS: the single-tile path does not reproduce scores.json`.
  - `probe_refit_variance.py` raises an `IndexError` (M4).
- Fix:
  - In `pim/scoring/driver.py`, write `scores["settings"] = {k: s[k] for k in ("oth_probe_games", "oth_gs_steps", "oth_gs_beta", "rw_probe_seqs", "rw_bench_n")}`.
  - Have the three scripts read it, with the current constants as the fallback for the shipped files.

**M6. The documented demo command fails.**
- Where: `scripts/demos/demo.py:4` (also its `--help` text and the stage-B command list).
- `python scripts/demos/demo.py --seed 7 --n-objects 4 --fixed-reflectivities --save outputs/demo.gif` raises `RuntimeError: Could not generate a collision-free scene after 300 attempts`. PRIVATE fails the same way. Without `--fixed-reflectivities` it raises `ValueError: refl_min_sep … exceeds refl range`.
- `--seed 0 --n-objects 4 --fixed-reflectivities` works, and so does `--seed 7 --n-objects 3`.
- Fix: change the usage line to `--seed 0 --n-objects 4 --fixed-reflectivities`.

**M7 (confirms verify-anon-lexical M2). All 28 shipped Rayworld `.h5` files keep `config_json["generated_at"]`, e.g. `"2026-08-31T17:38:23"`.**
- The JSON sidecars dropped it. Newly generated files do not carry it.
- Fix: rewrite the attribute into fresh copies, then regenerate SHA256SUMS and MANIFEST.

### Minor

**m1. Disk and time for a full rebuild are missing or understated (README:107).**
- "a 128-ray corpus takes about 410 GB" counts `obs.f32` only. With `meta.h5` it is about 436 GB (26 GB for 8-ray).
- The eight corpora total about 2.1 TB: 4 × 436 GB at 128 rays, plus 16-ray, 8-ray, 5-ray and obs5 (R=40). The 8-ray build took about 2 h on 32 cores. The `.scratch/` residual stacks are also "tens of GB".
- Fix: give the total, for example "about 2.1 TB for the eight Rayworld corpora (a 128-ray one about 440 GB), and hours per corpus on 32 cores".

**m2. Othello `train.py` without `--limit` silently regenerates the full 20M-game corpus when only a smaller `train_<n>.npz` exists (`scripts/train.py:97`).**
- It printed nothing for over 15 s (`oc.build` logs every 500k games).
- Fix: in `_othello_tokens`, if neither `train_{n}.npz` nor a larger file exists, `raise SystemExit("no train split of n games; run scripts/make_othello_corpus.py --instance … [--n-train N] and pass --limit N")`.

**m3. The README "Scoring" paragraph (README:84) is incomplete.**
- Deleting a Rayworld run's `probes/` also drops its categorical probes. The scorer then silently skips those blocks unless step 3's `fit_probes.py` lines are rerun.
- The figure caches `outputs/cache/qualitative_*.pkl` are keyed by (instance, seed) only, so after a rescore the figures need `--recompute`.
- Fix: add both sentences.

**m4. README:71 says the Othello probe games are regenerated "in a few seconds".**
- Measured: about 58 s (`qualitative_othello.py` took 82 s without the corpora bundle and 24 s with it).
- Fix: "in about a minute".

**m5. The shipped JSON sidecars do not match what the scripts write.**
- The shipped `eval/test.json`, `edits/edits.json` and `probe/probe_*.json` are four-split-suite manifests (`splits: train 100 / val / test / edits`, no `role`, an older `sim` without the blink/`n_observers`/`region` fields). `generate_dataset.py --role …` writes single-role manifests.
- The data are bit-identical, and nothing reads `splits`.
- Fix: rewrite the sidecars in the regenerated format during export, or note that only the sim block is read.

**m6. The corpora bundle has no `probe_250k.h5` for standard, blink, smooth and obs5, although their shipped floors use it.**
- The shipped `observation_right_large` floors of these four instances were fit at `n_seq` 250000.
- Refitting their floors from the bundle skips that floor (`obs_right_large +nan`).
- Fix: ship the four files (about 1.5 GB each at 128 rays), or say in the bundle table that refitting those floors needs `generate_dataset.py --role probe --size 250k`.

### Nit

- **n1.** `scripts/figures/editability_by_point.py` has no argparse, so `--help` runs it and writes the figure. Add an `argparse.ArgumentParser(description=__doc__).parse_args()`.
- **n2.** `scripts/make_edit_selection.py:66` records `"pool": a.pool` (4000) when the split holds 3000 cases. Use `int(len(ok))`.
- **n3.** `scripts/othello_flip_rates.py:27` defaults to standard and adjacent-flip, while STAGING ships `corpus_stats.json` for all four instances. Default to `sorted(oc.INSTANCES)`, or keep it and note that the paper quotes two.
- **n4.** `pim/environments/rayworld/viz.py:167` warns on every demo frame (`Setting the 'color' property will override the edgecolor or facecolor`). Pass `facecolor=`/`edgecolor=` instead of `color=`.
- **n5.** `bayes_floor.py` with no `--instance` iterates all ten paper instances. In a partial tree it silently generates other instances' Othello test splits. This is fine at full scale.
