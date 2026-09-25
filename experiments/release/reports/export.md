# Export worker report

The artifact bundle is built, verified and locked read-only.

- **Script:** `experiments/release/export_artifacts.py`. It is deterministic and re-runnable (`all --force` rebuilds a locked tree).
- **Helpers:** `experiments/release/work/export/`.
- **Output:** `/home/sevan/research/PIM/gms-release-artifacts` (STAGING). It has 1490 files, 15.76 GB, and ends with `MANIFEST.json` and `SHA256SUMS`.
- **PRIVATE:** I never wrote to PRIVATE `runs/`, `datasets/`, `logs/` or `outputs/`.

## Incident: a git pull deleted PRIVATE `runs/initial_othello_comparison/L-oth-20m/scores.json`

- **What happened.** At 22:13–22:14, `git checkout main` and then `git pull origin main` ran. The old main still tracked the file, and commit 2ebc669 untracks it. Git overwrote the ignored working copy, then deleted it. `L-dw-20m/scores.json` was lost the same way; it is out of scope.
- **What I recovered.** I copied the file read-only from the scoring host where `nn_r2_shortlist` wrote it (`wsl-sevan`, 09:33) into `work/export/recovered/L-oth-20m.scores.json`. It is 303,664 bytes, the exact size the artifact audit recorded at 21:36, with sha256 `8e9fb261…1bf6`.
- **What restored PRIVATE.** At 22:42 someone else restored the PRIVATE file (see `runs/MOVES.md`). That was not me. The restored file is byte-identical to my copy (same sha256).
- **What the final export used.** The final export and the reference tables both read the restored PRIVATE file. `SCORES_OVERRIDE` stays in the script only as a fallback, pinned by sha256.
- **Cross-check against the paper.** This file matches the paper's standard row exactly:
  - Table 1: 0.975 / 0.976, and IM 0.884 / 0.827.
  - Table 2: −0.93; PI +0.82 / 0.70; GS +0.83 / 0.72; IM +0.81 / 0.62.
  - The NN latent R² is 0.726. That value exists only in this nn_r2 version; the paper rounds it to 0.73.

## What is shipped

**Runs.** 13 main and extra runs, plus 30 replicates.
- Replicate names: `<new>__seed0` comes from `<old>__seed0_s512000`; `__seed1` and `__seed2` keep their suffix.
- Not shipped: `s421875` and `_R-*`.

**Per-run files.**
- Always: `best_model.pt`, `config.json`, `scores.json`.
- Where present: `metrics.jsonl`, `variance.json` (7 files), `editability_by_reachability.json` (4), `two_flip_editability.json` (3), and `vocab.npz` (token run).
- The in-scope `probes/`.

**Never shipped.** `ckpt/`, `latest.pt`, `scores_backup/`, `scores.pre-*`, `*.s390000.json`, `figures/`, `commit_sha`, `index_ceiling.json`, `INDEX.md`, `_superseded/`. In total that is 70.1 GB left behind, 69.98 GB of it `ckpt/`.

**`config.json`.**
- Renamed: `data.env`, `data.instance`, `replicate.of` and `replicate.source`.
- Made repo-relative: `data.corpus` and `data.vocab` (e.g. `datasets/rayworld/8-ray/train`).
- Dropped: `commit_sha`, `train.run_dir`, and the timestamp `at` inside the trainer's `resumed` records (from/to steps kept).
- Date-stripped: the replicate note.

**`scores.json`.**
- Names follow SPEC; `probe_dir` becomes `runs/<id>/probes`.
- Deleted: `commit_sha`, the `settings` dump, `blocks_added`, `inverse_added` (dated history), and `inverse_cleared` (the "(Sevan)" notes).
- Every ND arm and ND entry is removed (arms, `best`, `best_by_dims`, `alphas`, `probe_sources`).
- Out-of-scope blocks are removed. What remains:
  - Othello: `bases` is `{}` (mine_signed removed).
  - standard, blink, smooth and their replicates: frustum and cartesian.
  - obs5: cartesian.
  - 128-, 16- and 5-ray, and all ray-family replicates: frustum, cartesian, appearance-fac.
  - 8-ray-tokens: frustum, cartesian, appearance-fac.
  - 8-ray main: frustum, cartesian, appearance-fac, appearance, grid-6x5, grid-10x3, grid-16x8, pos@appearance.
- Prose: `bench` and `bench_selection.rule` lose their dates, and `bench_selection.file` becomes `datasets/rayworld/<i>/edits/selection.json`.
- Versions: every date-shaped version becomes "1.0". The map is in `reports/version_map.json` (218 fields rewritten). Probe provenance stores no versions.

**Other per-run JSON** (variance, reachability splits, two-flip). The `written` and `created` timestamps are dropped, ND entries are removed from `variance.json`, and names are renamed.

**Checkpoints.**
- Only `othello/standard` and `othello/standard__seed0` were re-saved, without `train_config.run_dir`.
- The state_dict is tensor-identical and the fingerprint is unchanged.
- The archive's inner name stays `best_model`.
- The other 41 are verbatim copies.

**Probes.** Scope is decided from each file's stored provenance:
- The model fingerprint must equal the run's checkpoint.
- `v` = 2 and seed 0.
- The recipe must match the kept blocks:
  - forward LIN+MLP full per kept basis;
  - IM full at 9 points per kept basis;
  - categorical forward at 200k / probe_250k / 50 epochs on the instance's first basis;
  - categorical IM only where the kept block carries IM arms;
  - snapped pos@appearance on the 8-ray main run;
  - Othello: grid mine LIN+MLP n=20000 plus IM mine-onehot at 9 points.

Every kept block is fully covered (0 coverage gaps). Rayworld files are re-keyed: only `data` changes, from `discworld/dw-X` to `rayworld/<inst>`. The filename is recomputed with the exact `pim/probes/cache.py` hash, and the file is written through the same tmp→replace path as `ProbeCache.store`. Othello files are copied verbatim, since their provenance is unchanged.

| probes | kept | dropped |
|---|---|---|
| 13 main + extra run dirs | 328 files, 234 MB | 214 files, 723 MB |
| 30 replicate dirs | 648 files, 480 MB | 201 files, 464 MB |
| 12 baseline dirs | 222 files, 326 MB | 150 files, 405 MB |
| total | 1198 files, 1040 MB | 565 files, 1592 MB |

The drop counts include `INDEX.md` and `_superseded/`. The per-run table and every drop reason are in `work/export/probe_accounting.json`.

What was dropped:
- probe-seed replicates;
- stale fits of other checkpoints (e.g. adjacent-flip__seed1 30 files, standard seeds 1–2);
- nullspace cascades (18 on 8-ray);
- the retired `pos` target;
- the Othello 10k / state-grid / linear-only grids;
- mine_signed;
- out-of-scope categorical targets (grid-4x2/8x4/32x16, appearance-d2/-d3/-lat, 16/5-ray appearance and grids, noiseless/blink/smooth/obs5 appearance-fac, and on the token model everything except appearance-fac);
- pos@appearance on the 8-ray replicates.

**Baselines (`runs/_baselines/<env>/<inst>/`).**
- `baselines.json` keeps the archs `transformer_l`, plus `transformer_l_tokens` on 8-ray (and `transformer_l_tokens` on Othello). `recurrent_l` is dropped. Kept targets are the same set as the runs, and all four observation kinds plus `random_init` are kept. `commit_sha` is dropped, `baseline_version` is "1.0", and `env` and `instance` are renamed.
- `bayes_floor.json` (canonical only; smooth and obs5 have none, as in PRIVATE), `reachability.json` and `corpus_stats.json` (Othello flip rates) are shipped with `created` dropped and the version map applied.
- `bayes_floor.smoke.json` and `bayes_floor.<timestamp>.json` are not shipped.
- The 12 legacy path-keyed 8-ray baseline probes are dropped.

**Datasets (`datasets/<env>/<inst>/`).**
- Othello: `eval/test_10000.npz`, `edits/cases_1000.{pkl,json}`, `probe/probe_20000.npz`, `probe/probe_large_170000.npz`.
  - Where the npz stored `instance`, the string is rewritten to the new name with arrays byte-identical (adjacent-flip, adjacent-noflip, standard-noflip; standard's files have no such key).
  - Not shipped: the 1.37 GB label caches (`probe_data` regenerates them) and `edits_10000.npz` (`make_othello_edits` regenerates it).
- Rayworld: `eval/test.{h5,json}`, `edits/{edits.h5,edits.json,selection.json}` (flattened from `edits/v1`), `probe/probe_120k.{h5,json}` on all 8 instances, `probe/probe_250k.{h5,json}` on 128/16/8/5-ray, and `tokens/vocab.npz` on 8-ray.
  - JSON manifests have `generated_at` / `created` dropped. In `selection.json`, `instance` is renamed and `vocab` becomes repo-relative.
  - `tokens/meta.json` is not shipped: only `load_tokens` reads it, together with `train.i16`, which is not shipped, and the tokenizer script rewrites both.
- Never shipped: `instance.json`, `layout.json`, `_unused/`, `train/`.

## Bundles (`MANIFEST.json` `bundle` field)

| bundle | files | size | content |
|---|---|---|---|
| core | 780 | 2.891 GB | all 43 runs' config/scores/metrics/JSON; main+extra `best_model.pt` (1.32 GB) + probes (234 MB); `_baselines` incl. probes (327 MB); datasets eval + edits + `tokens/vocab.npz` (0.99 GB) |
| corpora | 32 | 9.352 GB | `datasets/**/probe/**` (Rayworld 9.31 GB, Othello 46 MB) |
| replicates | 678 | 3.521 GB | replicate `best_model.pt` (3.04 GB) + probes (480 MB) |

## Verification (all run on the final tree)

1. **Hash proof on the source.** `cache_fname(provenance)` reproduces the filename of 1707/1707 probe files in the 55 in-scope PRIVATE run and baseline dirs (`work/export/hash_proof.json`).
2. **(a) Re-key.** 1198/1198 shipped probe files: the stored provenance hashes to the filename. The `data` values are only `rayworld/<8 instances>`. Each re-keyed payload was reloaded and deep-compared equal to the original (tensor bytes, stats).
3. **(b) Numbers.** 170 transformed JSON files were re-derived from their sources. 360,484 numeric leaves are bit-identical at identical paths (NaN-aware), with none missing and none extra.
4. **(c) Identity scan.** Zero hits for every token in the brief, across all 1490 files. The scan covers text/JSON, pickle string opcodes of all 1241 `.pt` files and 4 `.pkl`, npz keys and strings, HDF5 attrs and string datasets, and file and dir names. Dates are excepted only in HDF5 `config_json` `generated_at` (28 files, listed in `work/export/verify.json`); nothing else in them hits.
5. **Checkpoints.**
   - The fingerprint through PRIVATE `load_checkpoint` + `fingerprint` matches the source for 43/43.
   - Three-way agreement for all 43: the export's state fingerprint, the audit's `inventory_runs.json`, and the RELEASE loader.
   - The 307 verbatim copies are byte-identical, and there are no symlinks or hardlinks.
6. **RELEASE-side.**
   - All 43 checkpoints load with the RELEASE `pim.models.load_checkpoint`.
   - For 976/976 run probes, the RELEASE `ProbeCache.key(model, **fields)` reproduces the shipped filename and `ProbeCache.load` hits.
   - The RELEASE `layout.probe_key` agrees with every stored `data`/`split`.
   - 222/222 baseline probes reproduce with random-init models rebuilt by RELEASE code (seed 0) and with the model-free observation key.
7. **`SHA256SUMS`.** `sha256sum -c` passes for all 1490 entries.
8. **Table reference** (`reports/reference_tables.json`). PRIVATE `tables.collect` was run exactly as the two notebooks call it, each in a fresh process:
   - paper: cartesian;
   - appendix: its first collect runs under the module default basis (frustum), because the notebook calls `set_basis` only later; then cartesian with `select="fidelity"` and `select="index"`;
   - extra: the appendix list, additional_rw and tokens under cartesian.

   It contains every frame (df rows with `run_id` + `shipped`, `rep_sd`, floors, perdim, prediction rows) and every drawn table cell by cell. Its canonical rows match the paper's Tables 1 and 2.
9. **Self-check before the gate** (`work/export/staging_selfcheck.json`). The same collect calls over the scrubbed STAGING `scores.json` reproduce every shipped row of every reference frame exactly (19/19 paper, 19/19 A2, 21/21 appendix, 3/3 additional_rw, 9/9 tokens). The only differences are the expected ones:
   - the `ND …` columns, which are gone;
   - `rep_sd.dropped_steps` `[421875]` becomes `[]` (noiseless, adjacent-flip; `s421875` not shipped);
   - no ± for 8-ray pos@appearance, because its replicate blocks are deleted per SPEC. The paper's tab:categorical shows no ±.
10. **Lock.** `chmod -R a-w` is applied; a write attempt fails with "Permission denied".

## Requests to other owners

- **scoring** (`notebooks/master_eval.ipynb` SETTINGS, `pim/scoring/*`)
  - **(1) Extra targets.** Set `rw_extra_targets` as follows:
    - `rayworld/8-ray`: appearance-fac, appearance, grid-6x5, grid-10x3, grid-16x8, pos@appearance.
    - `rayworld/128-ray`, `rayworld/16-ray`, `rayworld/5-ray` and `rayworld/8-ray-tokens`: appearance-fac.
    - `rayworld/8-ray__seed0`, `rayworld/8-ray__seed1`, `rayworld/8-ray__seed2`: explicit entries with appearance-fac only.

    Without the explicit replicate entries, the replicates inherit the parent's six targets through `replicate.of`. pos@appearance is a regression target, so the scorer would REFIT it inline (needs `probe_120k` and a GPU), and the dry run would print WOULD-add lines.
  - **(2) Othello extra targets.** `oth_extra_targets = ()`. Othello `bases` is `{}`, so otherwise mine_signed is refit inline.
  - **(3) Bases override.** `rw_bases_by_instance` must key obs5 by whatever the release uses for the instance (the shipped `instance` field is `obs5`, env `rayworld`). Only the cartesian block and probes ship for obs5.
  - **(4) Versions.** `EVAL_VERSION_BY_ENV = {"rayworld": "1.0", "othello": "1.0"}`, `BASELINE_VERSION`, `IM_VERSION`, `PRED_VERSION`, both `FLOOR_VERSION`s, and the reachability / two-flip `VERSION` = "1.0". Every shipped artifact stores "1.0".
  - **(5) 8-ray floors.** `transformer_l` on 8-ray has no grid-6x5 / grid-10x3 / grid-16x8 floors; they were never fitted in PRIVATE either. If SETTINGS asks for them, the baselines dry run prints a WOULD-fit (`require_cached`, then nothing new) exactly as PRIVATE did.
  - **(6) Repo-relative writes.** The scorer must write repo-relative `probe_dir` (`runs/<id>/probes`) and must not re-add `settings` / `commit_sha` / dated `blocks_added` records if the shipped files are to stay anonymous after a re-score.
- **tables** (`pim/figures/tables.py`)
  - Key floors and Bayes floors by `(env, instance)`. The shipped `baselines.json` / `bayes_floor.json` carry a bare `instance` (`standard` in both envs) plus an `env` field (baselines.json only), and live at `runs/_baselines/<env>/<inst>/`.
  - The table-diff gate should exclude the ND columns and expect the two `rep_sd` differences listed above.
- **core** (`pim/probes/base.py`). All 1198 shipped probe pickles reference the global `pim.probes.base.WorldStateProbe`. Keep that import path and its pickled attribute layout. The other globals are `torch.nn.Linear` / `Sequential` / `ReLU`, `torch._utils`, `collections.OrderedDict` and `set`. Checkpoints reference torch and collections only.
- **scripts** (`scripts/make_edit_selection.py`, `scripts/make_rayworld_tokens.py`)
  - `selection.json` `vocab` is now repo-relative (`datasets/rayworld/8-ray/tokens/vocab.npz`); write it that way.
  - `tokens/meta.json` is not in the bundle; the tokenizer regenerates it with `train.i16`.

## Open issues / judgement calls

- **HDF5 timestamps.** The HDF5 `config_json` attrs keep their `generated_at` timestamps (28 files), per the brief. If zero timestamps are wanted anywhere, the attr can be rewritten in the copies with the data untouched; the file sha256 values would then change.
- **8-ray replicates lose pos@appearance.** Their blocks and probes are dropped (SPEC: 8-ray main only). The only effect is the missing ± on that row of Table 2c; see request (1) for the scorer consequence.
- **Othello `bench` prose.** Now reads "…from standard's own edits games" (instance renamed, date removed). Nothing reads it.
- **Leftover helper trees.** The reference helpers leave symlink trees in `work/export/shadow*` (into PRIVATE `runs/` and into STAGING). They are read-only uses and can be deleted.
- **Missing Bayes floors.** smooth and obs5 have no `bayes_floor.json` (same as PRIVATE), so their appendix A1 row stays blank.
