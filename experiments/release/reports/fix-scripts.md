# fix-scripts

Scope: RELEASE `scripts/*.py` and `scripts/demos/**`. All 10 assigned findings are applied. Work
files are in `experiments/release/work/fix-scripts/`:
- `before/` is the pre-edit copy and `fix-scripts.diff` the full diff.
- `e2e.sh` is the quick check, `readme_quick.txt` the README command list, and `logs/` the run logs.
- The `tree_*` directories are throwaway trees.

No shipped number, cache key, probe provenance, data generation or scores.json schema changed.

## Changes

1. **`demos/demo.py`**: the usage line is now `--seed 8 --n-objects 4 --fixed-reflectivities [--save outputs/demo.gif]`. Seed 7 fails at 100 frames (no collision-free scene); seeds 0 and 8 work. It runs headless with `MPLBACKEND=Agg`, and `--help` shows this line.
2. **`im_reconstruction.py`** (new; default `--run othello/standard`)
   - Calls `othello.arms.inverse_arms(..., post_boards=<pre-edit boards>, return_probs=True)`, so IM writes g(pre-edit board) at every residual point. g is the run's cached inverse map, and the write goes into the latent state at the last position.
   - Metrics: `move_rmse` against `bench.legal_pre` for the edited and the unedited outputs; the ratio is `move_fidelity_ratio(probs, uns, legal_pre)`.
   - Writes `runs/<run>/im_reconstruction.json` with keys `{run, points, ratio, model_error, g_error, n_cases, version "1.0"}`.
   - If an inverse map is missing from the cache, it is fit with the scorer's recipe and the script says so. On othello/standard every map was a cache hit.
3. **`train.py`**
   - An Othello run with no train split of at least n games exits at once. The message names `make_othello_corpus.py --instance I --splits train [--n-train N]`, and when a smaller split exists, the `--limit` that uses it.
   - Clear exits were also added for a missing Rayworld corpus (`build_rayworld_corpus.py`), a missing frame vocabulary (`make_rayworld_tokens.py`), and a corpus too small for one training block. The last one is fix-lib's ValueError, re-raised without a traceback and with the corpus size.
   - `DEFAULT_INSTANCE` comes from `pim.environments.layout`.
   - `n_total` comes from the memmap's shape (`bc.open_obs`), which reads `corpus.json`.
4. **`build_rayworld_corpus.py`**: new `--shards N` and `--shard-n N`, passed to `bc.use_instance(inst, n_shards=, shard_n=)` only when given. `corpus.json` records them through `bc.SHARD_N` / `bc.N_SHARDS`. The help notes that a frame model needs 6,144 sequences. The usage line shows `--shards 2 --shard-n 4000`.
5. **`fit_probes.py`**
   - New `--cat-n-seq` (default 200000) and `--cat-epochs` (default 50). `--n-seq` now defaults to 30000.
   - All three go to `rwa.probe_recipe(target, inst, n_seq=, cat_n_seq=, cat_epochs=)`.
   - The old `--epochs` override is removed: it only made cache keys the scorer never reads.
   - It prints: `note: notebooks/master_eval.ipynb reads these probes only when its SETTINGS have rw_cat_probe_seqs = N and rw_cat_probe_epochs = E` (or `rw_probe_seqs = N` for `pos@<partition>`).
   - The floors directory comes from `layout.baselines_dir`.
6. **`othello_flip_rates.py`**: defaults to all four instances, writes through `layout.baselines_dir`, and prints "tokens flipped per move".
7. **"Token" wording**
   - `two_flip_editability.py`: "Two-token Othello edits", "flipped token s ... token t", "each color's count is preserved", "single-token path", "a partner cell is empty".
   - `othello_flip_rates.py`: "Tokens flipped per move".
   - `make_othello_edits.py`: "one token recolored".
   - JSON keys are unchanged, and "cell" follows the paper's word for a board position.
8. **`make_othello_edits.py` recipe string**, exactly: `one occupied non-center token recolored; rejected if the legal set is unchanged or empty (bench.synthesize_cases)`. "synthesise" became "synthesize" as well, to match fix-lib's rename and American spelling. fix-lib's report quotes the same string.
9. **fix-lib's renames**: `oc.LADDER["D"]` → `oc.N_TRAIN_GAMES` (`train.py`, `make_othello_corpus.py`), and `synthesise_cases` → `synthesize_cases`. No `CENTRE`, `FactorisedTarget`, `space=`, `mode=`, `optimizer=` or `frame_probs(kind)` remains in my files. fix-lib's request to change play.py's panel title to "2D world  (simulator state)" is applied.
10. **SETTINGS constants and paths**
    - Each restated constant has a one-line `SETTINGS["key"]` comment:
      - `reachability_table.py` and `two_flip_editability.py`: `oth_probe_games`, `oth_gs_steps`, `oth_gs_beta`;
      - `probe_refit_variance.py`: `rw_probe_seqs`, `rw_bench_n`, `oth_probe_games`, and `rw_bases[0]` for `BASIS`;
      - `fit_probes.py`: `rw_probe_seqs`, `rw_cat_probe_seqs`, `rw_cat_probe_epochs`;
      - `im_reconstruction.py`: `oth_probe_games`.
    - The hand-spelled `runs/_baselines` paths now use `layout.baselines_dir`: `bayes_floor.py`, `reachability_table.py`, `fit_probes.py`, `othello_flip_rates.py`. `train.py` uses `layout.train_dir` and `layout.tokens_dir`.

## IM reconstruction (othello/standard, 1000 cases)

The model's own error against the pre-edit legal set is 0.003078.

| point | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| ratio | 9.48 | 9.42 | 7.99 | 5.42 | 1.95 | 1.80 | 6.16 | 10.13 | 15.69 |

- **Paper:** about twice the model's own at points 4–5, and five to sixteen times elsewhere. **They agree:** 1.95 and 1.80 at points 4–5, and 5.42–15.69 at points 0–3 and 6–8.
- **PRIVATE record:** all nine ratios are bit-identical to `pre_fidelity` of `recon_overwrite` in `experiments/inverse_probe/scores/othello_L-oth-20m_mirror128_recon.json`.
- Rerun on the final library, the output was byte-identical. It takes 0.7 min on the GPU.
- The copy is at `experiments/release/generated/runs/othello/standard/im_reconstruction.json` (sha256 `d13cbfeb…88bd6`).

## Verification

- **`--help`**: all 18 scripts exit 0 on the final library.
- **ruff**: `ruff check --no-cache scripts/*.py scripts/demos/` passes.
- **Othello cases**: `make_othello_edits.py` regenerates `cases_1000.pkl` **byte-identical** to STAGING for all four variants. Each `.json` differs only in `recipe`; `minutes` also matched on these runs.
- **Flip rates**: `othello_flip_rates.py` (no flags) regenerates all four `corpus_stats.json` **byte-identical**: 2.2449 / 0.2687 / 0 / 0 flipped tokens per move.
- **Rayworld corpus**
  - `build_rayworld_corpus.py --instance 8-ray --shards 1 --shard-n 3000` builds and verifies in 4 s. `obs.f32` and every `meta.h5` field equal PRIVATE rows 0–2,999.
  - With `--shards 2 --shard-n 4000`, they equal PRIVATE rows 0–3,999 and 500,000–503,999.
- **Small splits**: eval 500, edits 1000, probe 120k/250k at 2000 are bitwise prefixes of the STAGING files in every field.
- **fit_probes defaults**: on the shipped 8-ray run, appearance-fac (model, `--random-init`, `--observation`) and `pos@appearance` are pure cache hits. The skills are 0.9351/0.9427, 0.9217/0.9349, 0.4769/0.9349 and 0.9587/0.9907.
- **Analysis scripts**, after the constant and path edits:
  - `bayes_floor.py --instance othello/standard`: byte-identical;
  - `reachability_table.py --runs othello/standard`: `editability_by_reachability.json` byte-identical;
  - `two_flip_editability.py --run othello/adjacent-noflip`: identical except `minutes`.
- **Throwaway tree** (empty `runs/` and `datasets/`): `e2e.sh` runs 25 commands and passes, including the expected clean exits. The README list below also runs in order on a fresh tree in 89 s.

### Quick-check commands that work (for the README)

```
python scripts/make_othello_corpus.py --instance standard --splits train --n-train 5000
python scripts/make_othello_edits.py --instance standard
python scripts/train.py --env othello --instance standard --run othello/quick --steps 100 --limit 5000
python scripts/build_rayworld_corpus.py --instance 8-ray --shards 2 --shard-n 4000
python scripts/generate_dataset.py --instance 8-ray --role eval --n 500
python scripts/generate_dataset.py --instance 8-ray --role edits --n 1000
python scripts/generate_dataset.py --instance 8-ray --role probe --size 120k --n 2000
python scripts/generate_dataset.py --instance 8-ray --role probe --size 250k --n 2000
python scripts/make_rayworld_tokens.py --instance 8-ray
python scripts/make_edit_selection.py --instance 8-ray --pool 1000 --n 200
python scripts/train.py --env rayworld --instance 8-ray --run rayworld/quick --steps 100
python scripts/train.py --env rayworld --instance 8-ray --repr tokens --run rayworld/quick-tokens --steps 100
python scripts/fit_probes.py --run rayworld/quick --target appearance-fac --cat-n-seq 1500 --cat-epochs 2 [--random-init | --observation]
python scripts/fit_probes.py --run rayworld/quick --target pos@appearance --n-seq 1500
python scripts/demos/demo.py --seed 8 --n-objects 4 --fixed-reflectivities --save outputs/demo.gif
python scripts/demos/play.py --driver avoid --save outputs/play.gif
```

- `--shards 1 --shard-n 3000` builds, but a frame model needs at least 6,144 sequences; on it, `train.py` exits with that message. The token model trains on 3,000.
- `--limit` below 6,144 exits the same way.
- The scorer reads small-recipe probes only if master_eval's `rw_cat_probe_seqs` / `rw_cat_probe_epochs` / `rw_probe_seqs` are set to the same values.

## Requests

- **export**
  - Ship `runs/othello/standard/im_reconstruction.json` from `experiments/release/generated/`, and add it to `SHA256SUMS` and `MANIFEST.json`.
  - Set the `recipe` of the four `datasets/othello/*/edits/cases_1000.json` to the string in item 8.
- **tables** (`pim/figures/tables.py`, `notebooks/appendix_tables.ipynb`): add `tables.im_reconstruction(run="othello/standard")`, which reads `runs/<run>/im_reconstruction.json` and prints `ratio` per point with `model_error`. Add an appendix cell "IM reconstruction control (Othello)" so the paper's 2x / 5–16x sentence has a renderer.
- **infra** (`README.md`)
  - Add `python scripts/im_reconstruction.py` to the analyses.
  - Add the demo line above (seed 8).
  - `othello_flip_rates.py` now regenerates all four files.
  - Say "two-token edits" where the README says "two-disc edits".
  - List the quick-check commands above.

## Open issues

- `generate_dataset.py` prints the absolute output path. That comes from rayworld-env's `generate_h5` and is stdout only, not shipped.
- Two phrases are left as shipped because matching artifacts carry them:
  - `make_replicate_member.py` still writes "the canonical run's own checkpoint …" into `__seed0/config.json` (verifier nit);
  - `two_flip_editability.py` still writes the `editor` string "IM (canonical inverse_arms, post_boards)".

  Changing either needs the export worker to patch the shipped files the same way.
