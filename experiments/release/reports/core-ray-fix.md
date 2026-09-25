# core-ray-fix worker report

I edited five RELEASE files, and only those:
- `pim/training/train.py`
- `pim/environments/rayworld/{dataset.py, edits_dataset.py, bigcorpus.py, arms.py}`

Nothing changes the numbers. Backups of the originals, the check scripts and their outputs are in `experiments/release/work/core-ray-fix/`:
- `orig/`: the original files;
- `train_check.py`, `gen_check.py`, `im_check.py`, `im_compare.py`: the checks;
- `out/`: results, including `out/im_compare.txt`;
- `tree_before`, `tree_after`, `tree_nocorp`: the test trees.

## Changes

- **`train.py`**
  - Removed `_commit_sha`, the `import subprocess`, the `commit_sha` file write and the `"commit_sha"` key in `config.json`.
  - Removed the `"at"` timestamp from the `resumed` records.
  - Removed `commit_sha` from the docstring.
  - The "already holds a resumable state" message now says `--run` instead of `--run-name`, matching `scripts/train.py`.
- **`dataset.py`, `edits_dataset.py`**
  - `config_json` no longer gets a `"generated_at"` key.
  - `edits_dataset.py` no longer imports `time`. `dataset.py` still uses `time.perf_counter`.
  - A grep of RELEASE code, notebooks and JSON finds no reader of `generated_at`.
- **`bigcorpus.py`**
  - The module docstring now points at `python scripts/build_rayworld_corpus.py --instance <inst>`.
  - Removed the duplicate `__main__` block and the `json` import only it used.
  - `generate_shard`, `strip_shard`, `verify`, `use_instance`, `obs_path`, `SEED_RANGES` and `INSTANCES[...]['sim_flags']` are untouched.
- **`arms.py`**
  - **API addition:** `iter_inverse_maps(..., bank: bool = True)`. The default is unchanged. With `bank=False`:
    - it yields `(point, g, None, stats)` with `nn_r2 = NaN`;
    - on a cache hit it reads no probe corpus and computes no residuals;
    - on a miss it reads the corpus, fits and caches g exactly as before.
  - **Continuous path:** the corpus is read lazily, once. Each point's residuals are computed only when the bank is wanted or the map misses.
  - **Categorical path:** the corpus and residuals are read only on a miss.
  - **`n_classes` on a hit** comes from a new `_corpus_sim(probe)`:
    - the probe manifest when it exists, so the scorer uses the same source as before;
    - otherwise the instance's `eval/test.json`.
  - **New guard:** the categorical path checks the cached `stats["d_in"]` against `n_tiles * n_classes + 2 * N_OBJ` and raises `ValueError` on a mismatch.
  - **New private helpers:** `_probe_instance` (split out of `_probe_corpus`) and `_corpus_sim`.
  - **`.scratch/`** is now created only when residuals are computed.

## Verification

All checks passed. Every comparison is bitwise; NaN equals NaN.

1. **Training.** 30 CPU steps on a small Transformer-L with dropout 0.1 and synthetic data, run in PRIVATE, in RELEASE before the edit and in RELEASE after it.
   - Two variants: 30 steps straight, and 15 steps then `--resume` to 30.
   - The losses, the learning rates and all 84 final weight arrays are **identical** in all three trees.
   - A resumed run equals a straight run.
   - The new run directory has no `commit_sha` file, `config.json` has no `commit_sha` key, and `resumed` is `[{from_step, to_steps, batch_order_exact}]`.
2. **Data.** The first 3 sequences of 8-ray `eval` (`dataset.py`) and `edits` (`edits_dataset.py`), generated from each split's stored config, before and after the edit.
   - All 27 arrays are **identical**, and both match STAGING's first 3 rows.
   - The stored `dataset` config is equal. The `config_json` keys go from `[generated_at, dataset, schema]` to `[dataset, schema]`.
3. **`bigcorpus`.** Loaded side by side with the original: `INSTANCES` (with `sim_flags`), `SEED_RANGES`, `RESERVED`, 13 constants and the `use_instance` state for all 8 instances are **identical**. So are the signature and bytecode of 12 functions. Only `json` is gone. `scripts/build_rayworld_corpus.py --help` runs.
4. **(a) Scorer path, rayworld/8-ray, 50 cases, all 9 points, cache writes refused.** Old code (`tree_before`) against new code (`tree_after`):

   | model | block | records | result |
   |---|---|---|---|
   | frame | `full`/frustum (IM + IM-NN) | 18 | **identical** |
   | frame | `full`/cartesian (IM + IM-NN) | 18 | **identical** |
   | frame | `appearance-fac` (IM) | 9 | **identical** |
   | token | `full`/cartesian | 18 | **identical** |
   | token | `appearance-fac` | 9 | **identical** |

   The inverse-map stats are identical too. At point 4 the yielded g weights and the retrieval-bank tensors (H, A, mu, sd, a2, k) are **identical**.
5. **(b) Figure path with the probe corpora hidden** (`tree_nocorp`: no `datasets/rayworld/*/probe/`, `bank=False`).
   - The maps the qualitative and history figures read were checked: continuous on standard, blink, 16-ray, 8-ray and 5-ray, and categorical on 16-ray, 8-ray and 5-ray, each at its `scores.json` IM point.
   - g is **identical** to the old code with the corpora present, and so are the stats apart from `nn_r2`. `n_classes` matches: 34, 20 and 13.
   - With the bank on and the corpora present, the new code is identical to the old in everything, `nn_r2` included.
   - This check takes 1 s instead of 29 s and creates no `.scratch/`.
   - All 189 shipped categorical IM caches pass the `d_in` guard when the sim comes from the eval manifest.
   - **End to end:** I ran `history_rewrite.py` and `qualitative_rayworld.py --seeds 5` from a throwaway copy of `scripts/`, patched to pass `bank=False`, in the corpora-free tree.
     - The outputs (JSON, cache pickle, PNG pixels, PDF bytes) are **identical** to the unpatched scripts with the corpora present.
     - They also match the figures worker's `tree_ext` seed-5 grid pixel for pixel, and its `history_rewrite.json`.
     - `history_rewrite.py`: 2.6 s and 1.3 GB peak RSS, against 7.2 s and 6.0 GB.
     - `qualitative_rayworld.py`: 9.3 s and 2.7 GB, against 29.3 s and 6.7 GB.
6. **(c) Cache miss.** Tiny fresh fits (n_seq 300, points 0 and 5) into throwaway cache dirs, for continuous cartesian and categorical `appearance-fac` (3 epochs), on the frame and token models.
   - g, the stats and the bank hashes are **identical** between old and new code, and a second call is a hit that returns the same.
   - A `bank=False` miss fits a g **identical** to the default miss, with bank None.
   - If the corpus is needed and absent, h5py raises `FileNotFoundError` naming `probe_120k.h5`.

I deleted no `__pycache__` in RELEASE because none was created; every run used `PYTHONDONTWRITEBYTECODE=1`.

## Requests

1. **figures**
   - `scripts/figures/qualitative_rayworld.py`, in `inverse_map()`: add `bank=False` to the `rwa.iter_inverse_maps(...)` call, and drop the "frees the retrieval bank" comment on `gen.close()`.
   - `scripts/figures/history_rewrite.py:75`: add `bank=False` to the `rwa.iter_inverse_maps(...)` call.
   - With these two changes, the Rayworld qualitative and history figures run from the core bundle alone (verified in check 5).
2. **infra (`README.md`)**
   - Once request 1 lands, the Rayworld qualitative and history-rewrite figures need only the core bundle.
   - `qualitative_othello.py` still reads `probe_20000.npz` (corpora bundle, 46 MB) through Othello's `inverse_arms`.
   - Keep `.scratch/` gitignored: scoring and fits still use it, the figure path no longer creates it.
3. **rayworld-env (optional):** at `pim/environments/rayworld/tokens.py:108`, change the comment `# written by bigcorpus` to `# written by scripts/build_rayworld_corpus.py`.

## Open issues

- The shipped STAGING HDF5 files still carry `generated_at` in their `config_json` attributes. Newly generated data does not. This is an export matter, not a code one.
