# fix-export

**Status: done.** All 8 artifact findings are applied in `experiments/release/export_artifacts.py`. The export was re-run, re-verified and locked. No shipped number changed. Probe cache keys, probe provenance, checkpoint state dicts and fingerprints are unchanged. The scores.json schema is unchanged: 6 `bench_selection` fields went from null to a record.

- Work dir: `experiments/release/work/fix-export/`
  - `before/`: the old script, SHA256SUMS, MANIFEST.json and verify.json
  - `export_artifacts.diff`: the script diff
  - `logs/`: export, verify and rerun logs
  - `sha_diff.txt`: which files were added, removed or changed
  - `prove_selection.*`: the bench proof
  - `check_changes.*`: the independent numeric check
  - `gate/`: the tables gate, before and after
  - `removed/`: files that are no longer shipped
- The script still keeps its state in `work/export/`: `sha_cache.json`, `written.json`, `verify.json`, and the new `h5_scrubbed.json`.

## Changes (all in `export_artifacts.py`, so a re-run reproduces them)

1. **HDF5 timestamps.**
   - How: `put_h5` writes a fresh file for each of the 28 Rayworld `.h5` files. It copies every dataset with H5Ocopy (raw chunks, filters and fill value move unchanged) and copies the root attributes. `config_json` loses `generated_at` and is written back in its stored form, `json.dumps(indent=2)`. The new file goes to `.partial` and then replaces the old one.
   - Check at write time: before the replace, `h5_compare(full=True)` compares against the PRIVATE source. It checks names, dtype, shape, chunks, filters, fill value and time, and alloc time. It also checks every raw chunk (offset, mask, size and bytes) and every decoded array. For all 28 files, 49.75 GB decoded, everything is identical.
   - Determinism: the rewrite is deterministic (same bytes twice). No object carries HDF5 header times.
   - Re-runs skip the rewrite when the source sha256 and the shipped sha256 both match `h5_scrubbed.json`.
2. **`othello/standard` and `othello/standard__seed0`.**
   - `train` block: it now holds exactly the 10 `TrainConfig` fields, in `TrainConfig` order. The values are the ones the legacy block recorded (asserted): 780000 / 256 / 0.001 / 0.0001 / 1.0 / constant / 2000 / 1000 / 5000 / 0.
   - Dropped from the train block: `run_name`, `rung`, `window`, `arch`, `epochs`, `limit`, `warmup_frac`, `d_model`, `n_layers`, `n_heads`, `mlp_ratio` and `val_fraction`. `val_fraction` goes because it is not a `TrainConfig` field; the split is still recorded in the top-level `train_games` / `val_games`.
   - Dropped at the top level: `rung` and `w16_reference_steps`.
   - Kept at the top level: `model`, `arch`, `n_params`, `state_span`, `data` and `replicate`. `unique_games`, `train_games`, `val_games`, `total_steps`, `warmup_steps` and `epochs_over_pool` also stay: nothing reads them and none carry a name.
   - What reads each key (grep of RELEASE `pim/`, `scripts/`, `notebooks/`):
     - `train.steps`: `scoring/runs.training_complete`
     - `train.seed`: `make_replicate_member`
     - `data.env` / `data.instance`: the scorers, tables and scripts
     - `arch` / `n_params`: `scan_runs`
     - `replicate.steps` / `replicate.seed`: the tables
     - Nothing reads any dropped key.
   - Both `best_model.pt` files were re-saved:
     - `train_config` is now the same 10 fields, equal to config.json.
     - The top-level `rung` key is dropped too. It is the same internal label as config.json's `rung`, and no loader reads it: `_infer_arch` uses `arch` or the state-dict keys.
     - The state dict is tensor-identical, the fingerprints are unchanged (`6ea008f2c111`, `bac0cf35de59`), and every other key is equal.
   - Release-form JSON:
     - `layout` was dropped from the 18 Rayworld manifests.
     - `corpus/` became `train/` in the 6 Othello config.json files (`datasets/othello/<i>/train/train_20000000.npz`, the path `layout.othello_split_file` gives). This is done by a `rename_str` rule.
3. **Unread files.**
   - `UNREAD_RUN_FILES` leaves out `runs/othello/adjacent-flip/variance.json` and `runs/rayworld/standard/variance.json`. Both were moved to `work/fix-export/removed/`.
   - `prune_unplanned` now moves stale files; it never deletes.
   - Still shipped: variance.json for `othello/standard`, `othello/adjacent-flip__seed0/1/2` and `rayworld/8-ray`, exactly what `tables.probe_refit_spread` (tables.py:661) reads. No other reader exists.
   - `corpus_stats.json` ships for all four Othello instances.
4. **`bench_selection`.** 6 of the 7 null blocks are filled from their same-run `cartesian` record (`BENCH_SELECTION_FILL`). **`rayworld/obs5` cartesian stays null**: obs5 has no sibling block that holds a record.
   - Proof (`prove_selection.py`, GPU, on a copy of RELEASE `pim/`): for each block, the unedited scorecard was recomputed with `rwb.load_bench(..., use_selection=True/False)` and compared with the shipped block.

     | block | selected cases: max diff | first 1000 cases: max diff |
     |---|---|---|
     | 5-ray, blink, 8-ray, standard, obs5 | 0 (7/7 keys bitwise) | 0.0015 – 0.10 |
     | 16-ray | 1.65e-9 (only `edit_index`; 6/7 keys bitwise) | 0.047 |
     | smooth | 0 | 0 |

     smooth's `selection.json` is exactly cases 0–999, so both benches hold the same case ids.
   - The record rebuilt from `selection.json` equals the sibling `cartesian` record in all 6 runs.
   - For 8-ray and standard, verify-rescore's full fresh score already matched every arm bitwise.
5. **Othello `cases_1000.json`.** The recipe is set to the fix-scripts string, exactly: `one occupied non-center token recolored; rejected if the legal set is unchanged or empty (bench.synthesize_cases)`. The code asserts that the old string was the British one.
6. **`probes/INDEX.md`: shipped** in all 55 probe dirs (43 runs and 12 baselines).
   - They are rendered by the RELEASE `ProbeCache.write_index`, run in a RELEASE subprocess. A capture object stands in for the directory, so release code writes nothing; the export then writes the text.
   - They contain no dates and no names. The rows cover exactly each dir's `probes_*.pt`.
7. **Generated artifacts.** Every file under `experiments/release/generated/` is copied verbatim to the same path, in the core bundle. Today that is one file, `runs/othello/standard/im_reconstruction.json`. RELEASE `tables.im_reconstruction` renders it: ratios 9.48 … 1.95 / 1.80 … 15.69.
8. **Identity scan.**
   - The `generated_at` exception is removed, so HDF5 attributes are date-checked like everything else.
   - New tokens: `big20m` and `w16_reference`.
   - New: a raw-byte scan of every file, binary included, for the date pattern and 22 exact tokens (names, old run names, `generated_at`, `BIG20M`, …).

## Verification (on the final tree)

**Export's own verify:**
- (a) Re-key: 1202/1202 probe files hash to their name.
- (b) Numbers: 169 JSON files, 363,309 numeric leaves, 0 failures.
- (c) Identity scan: **0 hits, dates and HDF5 attributes included**. The raw-byte scan of 1550 files (15.82 GB) also has 0 hits.
- HDF5: 28/28 match their source (properties, filters, raw chunks, attributes). None contains its old timestamp string or `generated_at`.
- Release form:
  - 0 `layout` keys and 0 `corpus/` paths;
  - all 43 train blocks are `TrainConfig`;
  - no legacy keys remain in any config or checkpoint;
  - the unread files are absent;
  - the generated file is byte-identical to its source;
  - the 6 filled records equal their sibling's.
- INDEX.md: 55/55 equal the release output.
- Checkpoints: 43/43 fingerprints match.
- 280 verbatim copies are byte-identical, and there are no links.

**Independent numeric check (`check_changes.py`, no export code):**
- scores.json: 310,109 PRIVATE numeric leaves over all 43 runs are bit-identical at the same paths, with 0 unexpected extras. The only differences are the 6 null → record replacements: 243 added leaves, each record equal to cartesian's.
- config.json: 43/43 identical after the documented drops.
- Both re-saved checkpoints: state dict identical, `train_config` equal to PRIVATE's field for field, every other key equal.

**RELEASE side:**
- 43/43 checkpoints load, with the same fingerprints as before.
- 980/980 run probes: the release `ProbeCache.key` reproduces the filename, and `load` hits.
- 222/222 baseline probes pass.

**Tables:**
- PRIVATE self-check: paper 21/21, A2 21/21, appendix 23/23 at both bases, additional_rw 4/4, tokens 9/9, with 0 diffs.
- Tables gate: `check_results.json` is **byte-identical** before and after. It checks 1165 / 855 / 29 values; the only mismatches are the 2 known `tab:im_by_point` cells and the known 0.106.

**Checksums and grep:**
- `sha256sum -c SHA256SUMS` gives 1548/1548 OK (exit 0), both before and after the lock.
- MANIFEST paths, bytes and sha256 values equal the disk and SHA256SUMS.
- A shell `grep -a` of all 28 new `.h5` files for the 28 old timestamps, `generated_at` and the date pattern finds nothing. The PRIVATE source used as a control does hit.

**Reproducibility:** a second `export --force` rewrote no HDF5 file, and SHA256SUMS came out byte-identical.

**Lock:** `chmod -R a-w` is applied. 0 writable files or dirs remain, and a write attempt fails with "Permission denied".

**Links:** the 5 RELEASE symlinks (`runs/{othello,rayworld,_baselines}`, `datasets/{othello,rayworld}`) resolve.

## What changed in STAGING

- **Added (56):** 55 `probes/INDEX.md` and `runs/othello/standard/im_reconstruction.json`.
- **Removed (2):** the two variance.json files.
- **Changed (66):**
  - 28 `.h5`;
  - 18 Rayworld manifests (`layout`);
  - 4 `cases_1000.json`;
  - 8 Othello config.json: 2 legacy and 6 `corpus/` paths;
  - 2 `best_model.pt`;
  - 6 scores.json.
- **Unchanged:** `version_map.json` (218 fields).

## Bundle sizes

| bundle | files | bytes | before |
|---|---|---|---|
| core | 808 | 2,944,358,636 | 784 / 2,944,327,433 |
| corpora | 32 | 9,351,735,972 | 32 / 9,351,784,054 |
| replicates | 708 | 3,521,284,037 | 678 / 3,521,165,248 |
| total | 1548 (+ MANIFEST.json, SHA256SUMS) | 15,817,378,645 | 1494 / 15,817,276,735 |

## Requests

- **infra (`README.md`)**
  - The bundle table's file counts and sizes are now the numbers above.
  - Optionally say that each `probes/INDEX.md` lists every hashed probe file's provenance.
  - variance.json now ships for exactly the README refit loop (`rayworld/8-ray othello/standard othello/adjacent-flip__seed{0,1,2}`).

## Open issues

- **`rayworld/obs5` cartesian `bench_selection` stays null.** The brief allows filling only from a sibling block, and obs5 has none. The block was proven to be scored on the selected cases: bitwise on the selected cases, a 0.0084 difference on the first 1000. The scoring worker's full rescore writes the record from `datasets/rayworld/obs5/edits/selection.json` (n 1000, min_rays 2, pool 4000). To ship it, `transform_scores` would take the record from that file instead of from a sibling. That is one entry; say if wanted.
- **Not in this brief, left as shipped:**
  - the `replicate.note` "the canonical run's own checkpoint …" in the 10 `__seed0` config.json files (fix-scripts open issue);
  - the older sidecar layout of the Rayworld manifests (verify-pipeline minor).
- The reference tables (`reports/reference_tables.json`) were not regenerated. Nothing in PRIVATE changed, and the gate is identical.
