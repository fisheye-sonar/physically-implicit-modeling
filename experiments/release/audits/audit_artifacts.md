# Release-artifact audit — runs, probes, baselines, datasets (2026-09-23, read-only)

Method: I read `pim/scoring/*`, `pim/probes/cache.py`, `pim/environments/layout.py`, the discworld and Othello `arms.py` cache-key code, `pim/figures/tables.py`, both table notebooks, both `make_figure.py` scripts and `scripts/layout_checkpoint_replicate.py`. Scratchpad scripts (all read-only) then:
- inventoried every in-scope run directory (sizes via `os.walk`);
- read each probe file's stored provenance (mmap `torch.load`) and checked it against the fingerprint of that run's `best_model.pt`, computed on CPU with `pim.probes.cache.fingerprint`;
- checked what the scorer would do with each shipped `scores.json`, by calling the pure functions `driver.missing_blocks` / `missing_inverse` with the SETTINGS taken from `master_eval.ipynb` cells [2]/[2b]. No scoring, fitting or notebook execution happened.

Nothing in the repo, `runs/` or `datasets/` was modified.
Sizes are MB = 10^6 bytes.

## 0. Headline findings

1. **No shipped run triggers a refit under the current SETTINGS.**
   - 35 of 45 run dirs are skipped outright (eval_version matches, no missing block, no missing IM).
   - The other 10 are replicates: the 4 noiseless dirs (including `s421875`), 3 of the 8-ray, and 3 of the 5-ray. For these the scorer goes into its "add" branch. It loads the checkpoint, finds no cached categorical probes (`require_cached`), prints `SKIPPED`, and writes nothing. No dataset file is read.
   - The cause: these replicates inherit their parent's extra targets through `config.json` `replicate.of`, but only `appearance-fac` / `pos@appearance` were ever fitted for them.
   - Two opt-in env flags would start real work:
     - `PIM_ADD_CAT_IM=1`: 16-ray and 5-ray main runs, about 30 min per block, needs `probe_250k.h5`.
     - `PIM_ADD_NN_R2=1`: 33 run dirs, needs `probe_120k.h5`.
2. **Shipping fitted probes does NOT make the probe corpora unnecessary** wherever an inverse map is used. `iter_inverse_maps` opens `probe_120k.h5` (30k sequences) and collects residuals *before* the cache lookup; the IM-NN retrieval bank is the residuals themselves. The categorical inverse map reads 200k sequences of `probe_250k.h5` before its lookup.
   - So the discworld qualitative figure needs `probe_120k.h5` for all 5 of its instances, and `probe_250k.h5` for 16/8/5-ray.
   - The main-text figure (`paper/figs/qualitative_main`) also needs both corpora for dw-128ray (3.8 GB).
   - A from-scratch rescore needs `probe_120k.h5` for every discworld instance, and `probe_250k.h5` for the ray family (the instances listed in `dw_cat_im`).
   - Forward LIN/MLP probes are the exception: on a cache hit they never open the corpus.
3. **Discworld probe keys contain the instance name** (`data="discworld/<inst>"`), and the class name must be one of `("discworld", "othello")`.
   - Renaming an instance (for example `dw-noiseless` → `rayworld/standard`) makes **every** discworld cache miss: run probes, inverse maps, and the `_baselines` observation and random-init caches.
   - Othello keys have no instance field, but the rules lookup (`corpus.INSTANCES`) is keyed by name.
   - Renaming run directories keeps every cache hit, because keys carry no path and no run name. But the tables find runs by **bare directory name** through the glob `*/{name}/scores.json`, and replicates through `*/{name}__seed*`. Run names must therefore stay globally unique (`runs/rayworld/standard` next to `runs/othello/standard` would mix them up).
4. **Do not "clean" `scores.json`.**
   - Removing the `frustum` blocks makes `missing_blocks` list `frustum`, which is a regression block and is refit (needs `probe_120k.h5` and a GPU).
   - Removing the Othello `mine_signed` block causes an inline refit.
   - Removing IM arms from any regression block causes an IM recompute.
   - The paper notebook's Table 3 (`table_gridified`) draws *every* non-canonical discworld block in `scores.json`: 27 rows today, including grid-8x4/-32x16/appearance-d2/-d3/-lat.
5. **SETTINGS and `scores.json` disagree on two runs.** The 16-ray main run carries `appearance`, `grid-8x4` and `grid-16x8` blocks; the 5-ray main run carries `grid-8x4` and `grid-16x8`. Their probes exist, but `dw_extra_targets` no longer asks for them. The skip check is unaffected, but a from-scratch rescore would drop those Table 3 rows.
6. **The token run forces a dataset file.** `datasets/discworld/dw-8ray/tokens/vocab.npz` is required by the master_eval baselines cell whenever `L-dw-8ray-tok-20m` ships: `_dw_encoder` loads it outside the `try`, so it would crash if missing. It is byte-identical to the run's own `vocab.npz`. `tokens/train.i16` (1.6 GB) is training-only.
7. **Size is dominated by `ckpt/`**: 70.1 GB of 76.7 GB across the in-scope run dirs, and it is read by nothing downstream. Minimal bundle per run: about 116 MB for Othello, 119–226 MB for Rayworld.

## 1. Scorer skip decision (task 2)

**How the decision is made.** `scan_runs` lists every `runs/<topic>/<run>` that has both `config.json` and `best_model.pt`. It excludes `archive/` and `_`-prefixed topics (a `_`-prefixed *run* such as `ray_ablation/_R-dw-8ray-20m` is NOT excluded, so do not ship it). It also drops runs still training (last `metrics.jsonl` step < `train.steps`; a missing `metrics.jsonl` counts as complete).

`score_all` then skips a run iff all three hold:
- `scores.json["eval_version"] == EVAL_VERSION_BY_ENV[env]` (currently discworld `2026-09-12.2`, othello `2026-09-12.1`);
- `missing_blocks` is empty: every block key from `discworld_blocks` (bases `frustum,cartesian`, or `cartesian` for obs5, plus `dw_extra_targets[topic/run]`, which replicates take from their parent) or from `othello_blocks` (`mine_signed`) is present;
- `missing_inverse` is empty: every regression block and the Othello top level has an `IM` arm. Categorical blocks are owed IM only under `PIM_ADD_CAT_IM=1`; blocks lacking `inverse_map.nn_r2` are owed only under `PIM_ADD_NN_R2=1`.

A missing block is added through the scorer with `only=missing`. Categorical probes are `require_cached` (a miss is skipped); regression probes are refit.

**Result for every in-scope run.** All 45 dirs have a matching eval_version and `training_complete=True`.

| run (topic dir) | default | `PIM_ADD_CAT_IM=1` | `PIM_ADD_NN_R2=1` |
|---|---|---|---|
| 4 Othello main runs | skip | skip | skip |
| 13 Othello replicates (incl. adj-flip `s421875`) | skip | skip | 12 re-add IM on `mine/theirs`, `mine_signed` (`L-oth-noflip-20m__seed2` has `nn_r2` and still skips) |
| noiseless main | skip | skip | skip |
| noiseless `s421875`, `s512000`, seed1, seed2 | **no-op add**: blocks grid-16x8, grid-8x4, appearance-lat, grid-32x16 → SKIPPED (no probes) | same | + IM on frustum, cartesian |
| blink / 128-ray / 16-ray / 8-ray / 5-ray main | skip | 16-ray and 5-ray: **IM fits** on `appearance`, `grid-16x8` | skip |
| 8-ray replicates ×3 | **no-op add**: 10 categorical blocks → SKIPPED | same | + IM on frustum, cartesian, pos@appearance |
| 5-ray replicates ×3 | **no-op add**: `appearance` → SKIPPED | same | + IM |
| blink / 128-ray / 16-ray replicates | skip | skip | + IM on frustum, cartesian |
| smooth, obs5 | skip | skip | + IM (frustum/cartesian; obs5 cartesian) |
| 8-ray-tok | skip | skip | skip |

**Baselines cell (`score_all_baselines`).** It loads one `best_model.pt` per (instance, arch) to learn the arch. Every in-scope `baselines.json` is at `2026-09-06.b4`.
- dw-noiseless: re-attempts grid-8x4, appearance-lat, grid-32x16.
- dw-8ray: re-attempts appearance-d2/-d3/-lat, grid-16x8/8x4/32x16/6x5/10x3/4x2, for both archs.
- Each attempt fails `require_cached` and prints "nothing new". No write, no dataset read, **except** `tokens/vocab.npz` for the token arch.

**Other things the scorer never writes.** The `prediction` block (appendix Table A1) is written by `scripts/score_prediction.py`, not by master_eval. All 45 shipped files have it; a from-scratch rescore loses it until that script is run (needs discworld `eval/test.h5`).

## 2. Per-run inventory and what is read (task 1)

**What reads each file.**
- `best_model.pt`: `scan_runs` (existence), the baselines cell, the scorer, the figures. Tables: no.
- `config.json`: `scan_runs` (arch/env/instance/`train.steps`), the scorer (`data.instance`, `replicate.of`), the tables (`replicate.steps` / `seed` for pooling), the Othello figure.
- `scores.json`: skip logic, tables, figures.
- `probes/`: only on add/rescore, and in the figures (`require_cached`).
- `vocab.npz` (token run): token scorer, `score_prediction`.
- `metrics.jsonl`: optional `training_complete` check.

**Read by nothing in scorer, tables or figures:**
- `ckpt/`, `latest.pt` (training resume, and `layout_checkpoint_replicate.py` reads the parent's `ckpt/`);
- `scores_backup/`, `scores.pre-*.json`, `scores.s390000.json` (backups);
- `variance*.json` (read only by `experiments/seed_variance`, `experiments/paper_ci`);
- `figures/`, `commit_sha`;
- `editability_by_reachability.json`, `two_flip_editability.json`, `index_ceiling.json` (script outputs).

**Per-run sizes (MB).**
- "p.req" = probe files the current SETTINGS would request (fingerprint verified).
- "p.paper" = p.req minus the brief's out-of-scope set: frustum regression and IM, mine_signed, and categorical targets outside {appearance-fac, appearance, grid-16x8, grid-6x5, grid-10x3, pos@appearance}.
- "p.dead" = never requested.
- "MIN" = best_model + config + scores + p.req (+ vocab, metrics, commit_sha).

| run | total | model | ckpt | probes | p.req | p.paper | p.dead | scores | old scores | MIN |
|---|---|---|---|---|---|---|---|---|---|---|
| L-oth-20m | 1408 | 101 | 1217 | 88.7 | 14.4 | 10.4 | 74.3 | 0.30 | 0.54 | 116 |
| L-oth-adjacent-flip-20m | 2387 | 101 | 2231 | 53.9 | 14.4 | 10.4 | 39.6 | 0.30 | 0.54 | 116 |
| L-oth-adjacent-20m | 2348 | 101 | 2231 | 14.4 | 14.4 | 10.4 | 0 | 0.30 | 0.54 | 116 |
| L-oth-noflip-20m | 2348 | 101 | 2231 | 14.4 | 14.4 | 10.4 | 0 | 0.30 | 0.54 | 116 |
| L-dw-noiseless-20m | 2559 | 102 | 2235 | 216.5 | 110.2 | 43.4 | 106.3 | 0.96 | 4.13 | 213 |
| L-dw-blink-20m | 2377 | 102 | 2235 | 37.2 | 37.2 | 31.9 | 0 | 0.45 | 2.02 | 139 |
| L-dw-128ray-20m | 2687 | 102 | 2540 | 43.9 | 43.9 | 38.7 | 0 | 0.46 | 1.20 | 146 |
| L-dw-16ray-20m | 2676 | 101 | 2528 | 43.2 | 19.2 | 14.0 | 24.0 | 0.85 | 2.13 | 121 |
| L-dw-8ray-20m | 2528 | 101 | 2224 | 191.8 | 123.3 | 53.2 | 68.4 | 1.94 | 8.52 | 226 |
| L-dw-5ray-20m | 2365 | 101 | 2224 | 36.5 | 20.4 | 15.1 | 16.1 | 0.85 | 3.11 | 122 |
| L-dw-smooth-20m | 2680 | 102 | 2540 | 37.2 | 37.2 | 32.0 | 0 | 0.44 | 1.29 | 139 |
| L-dw-8ray-obs5-20m | 2692 | 101 | 2531 | 58.6 | 58.6 | 58.6 | 0 | 0.28 | 0.85 | 160 |
| L-dw-8ray-tok-20m | 2496 | 103 | 2261 | 120.8 | 120.8 | 50.7 | 0 | 2.57 | 8.55 | 226 |

**Group totals (MB).**

| group | n | total | model | ckpt | probes | p.req | p.dead | scores | MIN |
|---|---|---|---|---|---|---|---|---|---|
| main (10 paper runs) | 10 | 23,683 | 1,014 | 21,896 | 741 | 412 | 329 | 6.7 | **1,432** |
| smooth + obs5 + tok | 3 | 7,868 | 306 | 7,332 | 217 | 217 | 0 | 3.3 | **526** |
| replicates | 32 | 45,195 | 3,244 | 40,851 | 1,036 | 748 | 288 | 20.6 | **4,013** (3,758 without the two `s421875`) |

Replicate MIN is 116 MB (Othello) and 119–146 MB (Rayworld) each. Their `scores.json` + `config.json` alone (all the tables need) total 20.6 MB.

**Probe breakdown (fingerprint-verified; all probes that the settings request match their `best_model.pt`).**

*Required, per run type:*
- Othello: `othello_grid` mine (LIN+MLP, 7.0), `othello_grid` mine_signed (4.0), inverse_map mine-onehot × 9 points (3.4).
- Rayworld: forward full/cartesian and full/frustum (2.7 each), IM full × 9 per basis (2.55 each), and each requested categorical target (LIN+MLP, 250k recipe), plus categorical IM × 9 on in-scope ray instances.

*Dead (never requested):*
- **L-oth-20m** (74 MB): linear-only grids, seeds 0–9 (probe-seed experiment); IM seeds 1–9; a 10k-game grid; the retired state+frame 72-fit grid (27.8 MB).
- **Othello adjacent-flip** (main, `s421875`, `s512000`, seed1, seed2): about 40 MB each of probe-seed replicates.
- **Stale probes** fitted to the pre-512k checkpoints (fingerprint mismatch): adjacent-flip seed1 54 MB, noiseless seed1 38 MB, noiseless seed2 37 MB.
- **noiseless**: 24 probe-seed replicates (seed ≠ 0, 101 MB), retired `pos` target (5.2 MB), `_superseded/`.
- **8-ray**: `nullspace_cascade` (58.8 MB), probe-seed replicates (9.6 MB).
- **16-ray / 5-ray**: grid-8x4, grid-16x8 (and appearance on 16-ray). These back existing `scores.json` blocks but are orphaned by SETTINGS (see finding 5).

*Required by the scorer but out of scope in the brief:* frustum regression+IM (5.2 MB per Rayworld run), mine_signed (4.0 MB per Othello run), and noiseless / 8-ray / tok grid-32x16 (38 MB each), grid-8x4, grid-4x2, appearance-d2/-d3/-lat.

Dropping these is safe for the skip path. On a from-scratch rescore, frustum and mine_signed would be refit (minutes, needs the probe corpus), and the dropped categorical blocks would silently go missing. Keeping them costs about 174 MB for the main runs.

**What the qualitative figures read from runs.**
- Rayworld (`VARIANTS` noiseless / blink / 16 / 8 / 5-ray; plus 128-ray in `qualitative_main`): `best_model.pt`, `scores.json` arms, forward full/cartesian and appearance-fac/frustum, and IM full/cartesian plus categorical IM appearance-fac at the chosen point. That is 12–39 MB of probes per run.
- Othello (4 main runs): `best_model.pt`, `scores.json`, `config.json`, grid mine, IM mine-onehot (10.4 MB).
- The 10 figure runs total about 1.2 GB.
- `pim/figures/waterfall.py` itself reads no files (arrays in, figure out).

## 3. Probe cache keys and renames (task 3)

Filename = `probes_<blake2b-8 of repr(sorted(provenance))>.pt`. The provenance always contains:
- `model` = 12-hex blake2b over every `state_dict` tensor's bytes, or `"none"` for observation probes;
- `span` = `model.state_span` (39 for Rayworld L, 59 for Othello);
- `v = 2`.

A cache hit is verified against the stored provenance.

| cache | other key fields | instance in key? | path / run name? |
|---|---|---|---|
| dw forward LIN/MLP (run `probes/`) | target, n_seq, split (`probe_120k`/`probe_250k`), family, basis, seed, **data=`discworld/<inst>`**, [epochs], [encoder=`tokens:V<size>:<count sum>`] | yes | no |
| dw inverse map (continuous / categorical) | kind, target, [state], n_seq, split, basis, seed, **data**, hidden, epochs, point, [encoder] | yes | no |
| Othello grid / IM | kind, targets, families, splits, holdout, epochs, batch, lr, seed, n_seq, n_rows, n_points / target mine-onehot, split `sequence`, n_games | no | no |
| `_baselines` dw observation / random-init | as dw forward (observation: model `"none"`, kind, [align]); random-init fp = seeded CPU init (reproduced here for every in-scope (inst, arch)) | yes | no (the dir itself is `_baselines/<inst>`) |
| `_baselines` Othello observation / random-init | kind, target, family, holdout, seed, n_seq, n_rows, vocab, [epochs, align] | no | no |

**Renaming run directories** (e.g. `runs/rayworld/standard`): every cache still hits, since the cache is the run's own `probes/` and keys hold no path. What else must change:
- `SETTINGS["dw_extra_targets"]` keys `"<topic>/<run>"`, and every replicate's `config.json` `replicate.of`. If left unchanged, the parent gets no extra targets while its replicates still resolve the old key. The skip path still does no work, but a fresh rescore loses all categorical blocks.
- Run dirs must stay exactly `runs/<topic>/<run>`, and names must stay unique across topics (`find_run` and the replicate glob match by name only).
- Hard-coded run names in the notebooks (`RUNS_OTH` / `RUNS_DW`), the figure scripts (`VARIANTS`, `RAY128`), and `history_rewrite` / `predictive_quality`.

**Renaming dataset instances:**
- *Discworld*: every run and `_baselines` cache **misses**, and a class rename breaks `layout._check`.
  - The skip path is unaffected only if `scores.json` / `config.json` agree with SETTINGS.
  - `SETTINGS["dw_bases_by_instance"]` (`dw-8ray-obs5`) must be renamed too. Otherwise obs5 is owed a `frustum` block, which is a regression refit.
  - Figures would fail on `require_cached`; a rescore would refit every regression probe and IM and silently lose categorical blocks.
  - Names to update: `config.json` `data.instance`, `scores.json` `instance` (tables join `_baselines` by it), `baselines.json` `instance`, the `_baselines/<inst>` and `datasets/<cls>/<inst>` dirs, `dw_cat_im`, `tables.BASIS_BY_INSTANCE`, `FILTER_INST`.
  - `scripts/migrate_datasets.py` (`rekey` / `prov_fname` / `apply_cache`) is the in-repo precedent for re-keying caches in place without refitting.
- *Othello*: caches still hit, but `corpus.INSTANCES` (the flip / placement rules) and `oc.corpus_dir` reject unknown names, so a rename requires a code change.

**Checkpoint format.** The fingerprint covers weights only. Re-saving `best_model.pt` with metadata stripped keeps every key. Casting to fp16/bf16 or changing the state_dict breaks all of them. The checkpoints already hold no optimizer state (about 101 MB fp32 = weights only).

## 4. `runs/_baselines/<instance>/` (task 4)

| instance | JSONs (MB) | probes | observation | random-init | random-init full (tables 1d/1e) | categorical floors | legacy / out-of-scope |
|---|---|---|---|---|---|---|---|
| oth-uniform / -adjacent-flip / -adjacent / -noflip | 0.10–0.17 each | 41.5 each | 30.6 | 10.9 | – | – | – |
| dw-noiseless | 0.04 | 129.5 | 80.2 | 49.3 | 5.4 | 72.8 | 5.8 (recurrent_l) |
| dw-blink | 0.03 | 78.7 | 46.6 | 32.1 | 5.4 | 50.8 | 0 |
| dw-128ray | 0.03 | 77.6 | 46.1 | 31.5 | 5.4 | 49.8 | 0 |
| dw-16ray | 0.04 | 41.8 | 6.8 | 35.0 | 5.4 | 33.6 | 0 |
| dw-8ray | 0.06 | 52.5 | 5.5 | 47.0 | 10.7 (L + tok) | 22.2 | 16.6 (recurrent + `split=test` legacy) |
| dw-5ray | 0.04 | 30.9 | 2.2 | 28.7 | 5.4 | 24.5 | 0 |
| dw-smooth | 0.01 | 78.7 | 46.6 | 32.1 | 5.4 | 50.8 | 0 |
| dw-8ray-obs5 | 0.01 | 75.3 | 19.3 | 56.0 | 2.7 | 69.1 | 0 |

The in-scope `_baselines` total about 731 MB, almost all probes.

**Read by the tables:**
- `baselines.json` (Table 1 floors; the notebooks glob every `*/baselines.json` and key them by the JSON's `instance` field);
- `bayes_floor.json` (appendix A1 and its diagnostics cell). dw-smooth and dw-8ray-obs5 have none, so their A1 row reads "—";
- the random-init full-target probes, only through `tables_components(above_floor=True)` (Tables 1d/1e). That is called only by `build_full_tables.ipynb`, not by the paper or appendix notebooks.

**Read by scripts:** `reachability.json` (Othello) is a model-free cache for `reachability_table.py` / `two_flip_editability.py` (cheap: at most 9.5 min). `corpus_stats.json` is written by `othello_corpus_stats.py` and read by nothing.

**Droppable:** `bayes_floor.smoke.json`, `bayes_floor.<date>.json` (backups), `INDEX.md`, and all observation / categorical probes. Those are needed only if a user deletes `baselines.json`. Observation and regression floors would then be refit (needs `probe_120k` / `probe_250k.h5`; for Othello the `probe_large` split and its 1.37 GB label cache, which a ~10 min replay rebuilds). Categorical floors are `require_cached` and would be skipped.

## 5. Datasets (task 5)

**What each consumer reads.**
- **Scorer, discworld**: `edits/v1/edits.h5` + `selection.json` (the bench); `probe/probe_120k.{h5,json}` (IM always, plus regression refits); `probe/probe_250k.{h5,json}` (categorical IM on dw-128/16/8/5ray). The token run uses its own `vocab.npz`.
- **Scorer, Othello**: `probe/probe_20000.npz`, `eval/test_10000.npz`, `edits/v1/cases_1000.pkl`. The two npz files are regenerated deterministically by `corpus.build` if missing; `cases_1000.pkl` is not.
- **Baselines cell**: `tokens/vocab.npz` (dw-8ray).
- **Baseline refits**: dw `probe_120k` + `probe_250k`; Othello `probe_large_170000.npz` + labels cache.
- **Bayes floor**: dw `eval/test.h5` + `probe_120k.h5`; Othello test + probe splits.
- **`score_prediction`**: dw `eval/test.h5`.
- **Tables**: `edits/v1/edits.h5` (only its `config_json` attribute, in `table_gridified`) for each instance with gridified blocks.
- **Rayworld qualitative figure**: `edits.h5` attributes (sim config) for all 5 instances; `probe_120k.*` for all 5; `probe_250k.*` for 16/8/5-ray (128-ray too for `qualitative_main`).
- **Othello figure**: `probe_20000.npz`, `cases_1000.pkl`.
- **Reachability / two-flip scripts**: `cases_1000.pkl` (+ `reachability.json`).

**Dead** (no canonical consumer):
- Othello: `edits/edits_10000.npz` (only `make_othello_edits.py`, which regenerates it), `probe/probe_20000_labels_20000.npz` (experiments only), `probe_large_170000_labels_40000.npz` (oth-uniform, experiments).
- Every `_unused/`.
- `tokens/train.i16` (training only), all of `train/`, `instance.json` (never read by code).

**Needed only to refit probes:** nothing for forward probes on a cache hit. **But** both inverse-map paths need their corpus (finding 2).

**Per-instance sizes (MB).**

| instance | total | train (not shipped) | _unused | dead | tables (edits.h5) | rescore (edits/v1 + p120 [+ p250 if cat-IM] [+ vocab]) | p120 | p250 | eval |
|---|---|---|---|---|---|---|---|---|---|
| oth-uniform | 3,471 | 1,220 | 377 | 486 | 0 | 1.9 | – | – | 0.6 |
| oth-adjacent-flip | 2,770 | 1,220 | 0.1 | 162 | 0 | 1.9 | – | – | 0.6 |
| oth-adjacent | 2,770 | 1,220 | 0.1 | 162 | 0 | 1.9 | – | – | 0.6 |
| oth-noflip | 2,776 | 1,220 | 5.6 | 162 | 0 | 1.9 | – | – | 0.6 |
| dw-noiseless | 438,135 | 435,683 | 66 | 0 | 62 | 796 | 734 | 1,529 | 61 |
| dw-blink | 437,884 | 435,682 | 58 | 0 | 109 | 752 | 643 | 1,339 | 54 |
| dw-128ray | 439,713 | 435,682 | 0 | 0 | 104 | 3,927 | 1,240 | 2,583 | 103 |
| dw-16ray | 787 | 0 (not local) | 0 | 0 | 21 | 766 | 242 | 504 | 20 |
| dw-8ray | 53,842 | 51,683 | 27 | 1,600 (train.i16) | 14 | 519 | 164 | 341 | 14 |
| dw-5ray | 42,517 | 42,083 | 0 | 0 | 11 | 424 | 134 | 279 | 11 |
| dw-smooth | 441,847 | 435,682 | 0 | 0 | 160 | 2,057 | 1,897 | 3,949 | 158 |
| dw-8ray-obs5 | 155,862 | 154,083 | 5 | 0 | 46 | 592 | 546 | 1,137 | 46 |

- Othello also has 1,375 MB per instance of `probe_large_170000_labels_170000.npz`, needed only for baseline refits and regenerable.
- Rescore totals: Othello 7.6 MB, Rayworld paper instances 7,184 MB, smooth + obs5 2,649 MB.
- `eval/test.h5` for the prediction block and Bayes floor: 263 + 204 MB.
- `probe_250k` for non-cat-IM instances (noiseless, blink, smooth, obs5; baseline or categorical refits only): 7,954 MB.
- Also ship the tiny `layout.json`, `edits/v1/edits.json`, `probe/*.json`, `eval/test.json` and `tokens/meta.json`. The `probe/*.json` files are read for `sim` on any fit or IM pass.

## 6. The `__seed0_s<step>` members (task 6)

`scripts/layout_checkpoint_replicate.py <topic>/<run> <step | nearest:N>`:
1. loads `runs/<parent>/ckpt/step_<step>.pt`;
2. sets `val_loss` / `val_loss_step` to the nearest validated step in the parent's `metrics.jsonl`, and stamps `arch` from the parent config (the L-oth-20m ckpts carry the legacy `arch: "theirs"`);
3. saves it as `runs/<parent>__seed0_s<step>/best_model.pt`;
4. writes the parent's `config.json` plus `replicate: {of, seed, steps, checkpoint: true, source, val_loss_from_step, note}`, and copies `commit_sha`.

With `nearest:`, an existing member within ±10% of the requested budget is reused.

The member has no `metrics.jsonl` (so `training_complete` is true) and no `ckpt/`. It gets its own `probes/` and `scores.json`, keyed to its own fingerprint.

I verified all 12 members, 10 `s512000` plus the adjacent-flip and noiseless `s421875`: `model_state` is tensor-identical to the parent's `ckpt/step_000<step>.pt`, and the fingerprints match. **If the parent's `ckpt/` is not shipped, this 101 MB file is the only copy.**

The two `s421875` members are not pooled in any ± under the default ±10% budget guard (422k vs 512k → separate group). They appear only as "not pooled" in Table 5 of `build_full_tables.ipynb`. Recommend omitting them.

Trained replicates have `replicate.steps = 512000`, but their `best_model.pt` is the best-validation checkpoint (for example 8-ray seed1 at step 390k).

## 7. Recommended bundle layout

**(a) Rebuild tables and figures (no rescoring)**
- `runs/<topic>/<run>/{scores.json, config.json}` for the 13 main/extra runs and 30 replicates: **~31 MB**. This rebuilds the paper and appendix notebooks, and `build_full_tables` except its `L-dw-20m` and training-curve items.
- `runs/_baselines/<inst>/{baselines.json, bayes_floor.json}` (+ Othello `reachability.json`, `corpus_stats.json`): ~1 MB. Add the random-init full-target probes (~51 MB) only if Tables 1d/1e matter.
- `datasets/discworld/<paper inst>/edits/v1/edits.h5`: 321 MB for the 6 paper instances; +206 MB if smooth/obs5 rows are shown.
- *Qualitative figures on top:*
  - `best_model.pt` + the figure probes for the 10 figure runs: ~1.2 GB (or simply bundle (b) for those runs);
  - `probe_120k.*` for noiseless / blink / 16 / 8 / 5-ray: 1.9 GB;
  - `probe_250k.*` for 16 / 8 / 5-ray: 1.1 GB;
  - dw-128ray `probe_120k` + `probe_250k`: 3.8 GB (main-text figure only);
  - Othello `probe_20000.npz` + `edits/v1/cases_1000.pkl`.

**(b) Rescore from scratch**
- Every run: `best_model.pt`, `config.json`, `scores.json`, the requested probes (+ `vocab.npz` for tok; `metrics.jsonl` / `commit_sha` optional). Main 1.43 GB, extras 0.53 GB, replicates 3.76 GB.
- Datasets: discworld `edits/v1/*`, `probe/probe_120k.*` for all instances, `probe/probe_250k.*` for dw-128/16/8/5ray, `tokens/{vocab.npz, meta.json}` (dw-8ray), `eval/test.*` (prediction block / Bayes floor), and `layout.json`. Othello `probe_20000.npz`, `test_10000.npz`, `edits/v1/cases_1000.*`, `layout.json`. That is **~10.3 GB**.
- Optional: `probe_250k` for noiseless / blink / smooth / obs5 (8.0 GB, only for baseline or categorical refits), full `_baselines/*/probes` (0.73 GB), Othello `probe_large_170000.npz` (41 MB, plus a 10-min label rebuild).

**(c) Optional / drop**
- Replicates beyond `scores.json` + `config.json`: 3.76 GB.
- Token run: 226 MB (if shipped, `datasets/.../dw-8ray/tokens/vocab.npz` is mandatory).
- smooth (139 MB), obs5 (160 MB).
- The `s421875` dirs (255 MB).
- Drop always:
  - `ckpt/` and `latest.pt` (70.1 GB);
  - dead probes (617 MB), including the stale pre-512k probes and the nullspace / probe-seed caches;
  - `scores_backup/` and `scores.*.json` (76 MB);
  - `figures/`, `variance*.json` (an experiment output);
  - every `_unused/`, `tokens/train.i16`, `train/`, and the Othello label caches;
  - `ray_ablation/_R-dw-8ray-20m` (it would be scanned).

**Minor:** `config.json` `data.corpus`, the `scores.json` `probe_dir` field, and the L-oth-20m `train_config` contain absolute `/home/<user>/…` paths. No scoring or table code reads them, so they could be scrubbed without affecting the skip logic.

## Scratchpad artifacts
All under `/tmp/claude-1000/-home-sevan-research-PIM-physically-implicit-modeling/709bf97c-b816-42ca-a426-62ce849c7a7f/scratchpad/`:
- `settings.py`, `inventory_runs.py` → `inventory_runs.json`;
- `tabulate_runs.py` → `run_table.{json,txt}`, `probe_summary.txt`;
- `inventory_baselines.py` → `inventory_baselines.json`, `baselines_summary.txt`;
- `datasets_files.txt`, `tabulate_datasets.py` → `dataset_table.txt`;
- `verify_seed0.py`.
