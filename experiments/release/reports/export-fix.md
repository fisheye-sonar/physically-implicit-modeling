# Worker report: export-fix

**Task.** Ship the `appearance-fac` block, with its forward probes, for `rayworld/standard` and `rayworld/blink` (main runs only). This is the figures worker's request 1.

**Status.** Done and re-verified. STAGING is locked again (`chmod -R a-w`).
- The only content changes are 4 added probe files and the 2 `scores.json` files that gained the block.
- Everything else in STAGING is byte-identical.

**One knock-on needs the tables owner (request 1 below).** Without it, three tables gain Standard and Blink categorical rows, and the tables gate crashes.

Helpers, logs and before/after snapshots are in `experiments/release/work/export-fix/`.

## Change to `experiments/release/export_artifacts.py`

- New `FIGURE_BLOCKS = {"rayworld/standard": ("appearance-fac",), "rayworld/blink": ("appearance-fac",)}`, keyed by run id, so replicates get nothing.
- New `Run.run_extras` = `extras` + `FIGURE_BLOCKS`.
- `keep_blocks` and `classify_run_probe` now use `run_extras`.
- `baseline_scope` still uses `extras`, so `baselines.json` and the baseline probes are unchanged, as required.

The block goes through the same path as every other block:
- ND arms and ND entries are stripped: 264 arms become 138; `alphas`, `best` and `best_by_dims` lose ND.
- `inverse_cleared` is dropped.
- `probe_recipe.probe.instance` is renamed: `dw-noiseless` → `standard`, `dw-blink` → `blink`.
- `bench_selection.rule` loses its date.
- The block has no version fields, so the "1.0" map had nothing to rewrite. `version_map.json` is unchanged (218 fields).
- There are no IM arms. `best.IM` and `best.IM-NN` are `null`, as in PRIVATE.

Probe scope uses the existing categorical rule: seed 0, `probe_250k`, `n_seq` 200000, 50 epochs, frustum. Each forward file holds all 9 residual points. The rule selects exactly the two files per run named in the figures report; the probe-seed replicates 1–5 of the Standard linear fit stay dropped.

## Added and changed files

| file | from PRIVATE | bytes |
|---|---|---|
| `runs/rayworld/standard/probes/probes_56ff19304c766a12.pt` (linear) | `noise_ablation/L-dw-noiseless-20m/probes/probes_3201354ff714f717.pt` | 19,413,941 |
| `runs/rayworld/standard/probes/probes_c35d55fcb934f87d.pt` (mlp) | `…/probes_474fa1d299d82fb0.pt` | 7,301,333 |
| `runs/rayworld/blink/probes/probes_27c7d687d087f562.pt` (linear) | `blink_ablation/L-dw-blink-20m/probes/probes_d0bf8c88ee6b18f7.pt` | 19,413,941 |
| `runs/rayworld/blink/probes/probes_2015e2ea7a793132.pt` (mlp) | `…/probes_22304240197eca90.pt` | 7,301,333 |
| `runs/rayworld/standard/scores.json` (changed) | + `bases["appearance-fac"]` | 250,389 → 324,395 (+74,006) |
| `runs/rayworld/blink/scores.json` (changed) | + `bases["appearance-fac"]` | 250,403 → 324,349 (+73,946) |

- **Added:** 4 files, 53,430,548 B, all in the `core` bundle.
- **Bundles:**
  - core: 780 → 784 files; 2,890,748,933 → 2,944,327,433 B (+53,578,500).
  - corpora: 32 files, 9,351,784,054 B, unchanged.
  - replicates: 678 files, 3,521,165,248 B, unchanged.
- **Tree:** STAGING holds 1494 manifest files plus `MANIFEST.json` and `SHA256SUMS`, 15.82 GB in total.
- **Block order** in `bases` follows PRIVATE: frustum, appearance-fac, cartesian.

## Verification (all run on the final tree)

1. **Nothing else changed.**
   - The old and new `SHA256SUMS` differ only by the 4 added lines and the 2 `scores.json` hashes.
   - Deleting `bases["appearance-fac"]` and re-serializing each new `scores.json` reproduces the old file's sha256 exactly.
   - Probe, baseline and dataset accounting are identical for every other run, every baseline and every dataset.
   - `othello/standard` and `othello/standard__seed0` `best_model.pt` are re-saved on every export run. Their mtimes changed, but their sha256 did not.
2. **(a) Re-key.** 1202/1202 shipped probe files hash to their filename (was 1198). The `data` counts for rayworld/standard and rayworld/blink are now 110 each.
   - An independent PRIVATE-process check of the 4 new files found the payload and stats deep-equal to the source. Only the `data` provenance key changed (`discworld/dw-noiseless` → `rayworld/standard`, `discworld/dw-blink` → `rayworld/blink`).
3. **(b) Numbers.**
   - The export's own check re-derived 170 JSON files: 364,018 numeric leaves (was 360,484), 0 failures.
   - An independent check (`check_new_blocks.py`, no export code) compared each shipped block with the raw PRIVATE block minus ND arms and `inverse_cleared`.
     - Result: 1767 numeric leaves per block, 0 diffs, 0 missing, 0 extra.
     - The only string changes are the instance rename and the date strip.
     - There are no ND, date, old-name or version strings in either block.
4. **(c) Identity scan.** 0 hits over the full tree: text/JSON, pickle strings of 1245 `.pt` files and 4 `.pkl`, npz keys and strings, HDF5 attrs and string datasets, and file and dir names. The 28 HDF5 `config_json` `generated_at` exceptions are unchanged. Pickle globals are unchanged in kind.
5. **Checkpoints.** 43/43 fingerprints match, 307 verbatim copies are byte-identical, and there are no links.
6. **RELEASE side.**
   - All 43 checkpoints load with the release loader.
   - The release `ProbeCache.key` reproduces the filename, and `load` hits, for 980/980 run probes (was 976).
   - 222/222 baseline probes pass.
   - `layout.probe_key` agrees with every stored `data`/`split`.
7. **A rescore of the block works from the cache.**
   - The scorer's own path was called: release `rwa.probe_recipe("appearance-fac", inst)` + `rwa.fit_probes(..., require_cached=True)`.
   - It returns a cache HIT on the 4 shipped files, with 9 residual points each and 262 classes.
   - Probe Skill recomputed from the cached stats (`probe_skill_from_stats`, `by_point`) is bit-identical to the block's `probe_skill_linear` and `probe_skill_mlp` on both runs.
   - A cache hit returns before the corpus is read, so no Standard/Blink `probe_250k` is needed.
8. **Figures equivalence.**
   - The 4 shipped probes are payload- and provenance-identical to the figures worker's `tree_ext` files. Only the zip's inner archive name differs: mine is written through the same `.pt.partial` tmp path as `ProbeCache.store`.
   - The PI and GS arms equal `tree_ext`'s non-ND arms.
   - Their pixel-identical Rayworld appendix grids therefore carry over to STAGING. I did not re-run the figure scripts myself.
9. **Scorer dry run** (the scoring worker's `dry_run.py`, from RELEASE): 55 skips, 0 WOULD, both to-do lists `[]`. The log is identical to theirs, because `cat_inverse_in_scope` is False on Standard/Blink and floors are driven by SETTINGS.
10. **`MANIFEST.json` + `SHA256SUMS`.**
    - Both were regenerated, and `sha256sum -c` passes for all 1494 entries, before and after the lock.
    - MANIFEST paths, sizes and hashes equal what is on disk and in `SHA256SUMS`.
11. **Reference tables.**
    - `reports/reference_tables.json` was regenerated by the export's `tables` step; the old copy is in `work/export-fix/export_before/`.
    - The diff is exactly 11 fields, all `shipped` false → true on the Standard/Blink `appearance-fac` df rows across the frames. No number changed.
    - The spread flags stay false, because the replicates do not ship the block.
    - The export's `staging_selfcheck.py` matches every shipped reference row: paper 21/21, A2 21/21, appendix 23/23 at both bases, `additional_rw` 4/4, tokens 9/9. It found 0 extra rows and 0 diffs.
12. **Lock and links.**
    - `chmod -R a-w` is applied, 0 writable files remain, and create and append attempts fail with "Permission denied".
    - The 5 RELEASE symlinks (`runs/{othello,rayworld,_baselines}`, `datasets/{othello,rayworld}`) resolve; `runs/rayworld/standard/probes` lists 24 files.

## Requests

**1. tables (`pim/figures/tables.py`) — needed.**

**Problem.** `blocks()` returns `appearance-fac` for every Rayworld run, and `main_rows` puts every such row into the categorical section. With the two new blocks:
- `tab:editability` (Table 2), `tab:seed_spread` and `tab:fidelity_selected` each gain rows "standard (2889 bins)" and "blink (2889 bins)" (`work/export-fix/tables_knockon.txt`).
- The tables gate (`check_tables.py`) crashes with `KeyError: ('rayworld/standard', 'appearance-fac')` at the seed-spread lookup.
- Tables 1, `tab:additional_rw`, `tab:categorical`, `tab:im_vs_nn` and `im_steps` are unaffected.

**Proposed change** (`work/export-fix/tablefix/tables_request.diff`):
- Add after `N_RAY_FAMILY`:
  ```python
  # the runs whose appearance-fac block the tables report; standard and blink carry one for the figures only
  CAT_RUNS = N_RAY_FAMILY + ("rayworld/8-ray-tokens",)
  ```
- Change `def blocks(s: dict)` to `def blocks(s: dict, run: str | None = None)`, with the Rayworld return:
  ```python
  return {k: s["bases"][k] for k in REPORTED_BLOCKS
          if k in s["bases"] and (run is None or k != CAT_BLOCK or run in CAT_RUNS)}
  ```
- In `collect`, call `blocks(s, run)` and `blocks(read_json(rep), run)`, passing the parent id for replicates.
- The other `blocks()` callers read only `cartesian` or the Othello block and need no change.

**Checked.** On a throwaway copy of RELEASE `pim/` over the new STAGING, the gate's `check_results.json` is **byte-identical** to the pre-change run, with both the old and the regenerated reference:
- 1165 reference values with 0 mismatches, 855 paper cells, 29 in-text numbers;
- the same 2 known `tab:im_by_point` rounding items and the 1 known "at most 0.10" number;
- 14/14 spreads.

**2. figures — informational.**
- Request 1 is applied.
- The "categorical cells blank for Standard/Blink" note and the fallback are no longer triggered on STAGING.
- `scripts/figures/qualitative_rayworld.py` should now draw those cells from the shipped files.

## Open issues

- **`export.md` is superseded in part.** Its bundle table, probe counts (1198 files / 1040 MB; run dirs 328 files) and leaf counts are superseded by the numbers above. I did not edit that report.
- **One timestamp changed after the lock.** A `touch` write probe on the read-only `runs/rayworld/standard/scores.json` succeeded in updating its mtime; owners may set timestamps. Its content and sha256 are unchanged.
- **Figures not re-run.** The qualitative figure scripts were not re-run against the real STAGING; item 8 is an equivalence argument. One run of `scripts/figures/qualitative_rayworld.py` (about 75 s on a GPU) would confirm it directly.
