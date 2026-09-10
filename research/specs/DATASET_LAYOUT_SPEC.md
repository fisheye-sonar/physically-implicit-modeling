# Spec — the dataset layout migration (layout v2)

**Status: EXECUTED 2026-09-10 (locked in by Sevan the same day).** `scripts/migrate_datasets.py
--apply` ran at ~15:00 PT after the `probe_targets_5` chain was stopped and the GPU was idle;
`--verify` PASSED on every check (inodes and sizes of all 115 files, bench arrays identical
through both the new `instance=` and the legacy `data_dir=` call forms on all four discworld
instances, Othello benchmarks and `corpus.build` resolutions identical, 372 probe blobs re-keyed
and loadable, 104 canonical run-probe keys + 64 observation-floor keys HIT through the code's
own recipe, token vocab unchanged with 0 UNK on the retired `val.h5`); 240 tests pass. Log:
`research/scratch/2026-09-10-layout-migration-log.json`; snapshot
`research/scratch/2026-09-10-layout-snapshot.json`. Purely cosmetic by contract: no trained
model, probe, score, or bench case changed. Deviations from the plan below: `os.chdir(REPO)` in
`master_eval` was kept (harmless; only its comment changed); the two big notebooks' cells were
edited with `nbformat` because they exceed the notebook reader's cap (GOTCHAS 2026-09-10).

Companion: this layout also reserves the home of the paired-counterfactual edit bench
(`edits/v2/`) that the model-referenced Edit Index will need (separate spec, not written).

---

## 1. Why

The current tree names things by the accident of how they were generated, not by what they
are. Measured 2026-09-10 (`experiments`-wide path map, Explore agent report in the session
transcript; summary in §2):

- `probe/test.h5` is the probe **fit** corpus (the hold-out is an internal 80/20 split by
  sequence inside the same file), while `probe/{train,val,edits}.h5` are 100-sample stubs
  the generator writes unconditionally and nothing reads.
- `eval/` holds the **edit bench** (`edits.h5`), a held-out `test.h5` used only by the
  waterfall notebook, a `val.h5` whose only reader is the token-vocab builder, and on
  `dw-pn04` a 90k `train.h5` (1.6 GB) that nothing reads. Training validation is the last
  tenth of the `train/` memmap, not `eval/val.h5`.
- Othello uses a different vocabulary (`corpus/`), CWD-relative paths (hence the
  `os.chdir(REPO)` in `master_eval`), and keeps Li's 1001 edit cases inside `pim/` while
  every other instance keeps its cases under `datasets/`.
- Probe cache filenames hash a provenance dict whose `data` field is the **absolute
  resolved path** of the probe directory. Moving `probe/` without a migration orphans all
  394 cached probe files under `runs/**/probes/` and forces refits (~24 GB each).

## 2. Target layout

One shape for both environment classes. Split **role** is the directory name; the file
name carries the size where more than one size exists.

```
datasets/<class>/<inst>/
  instance.json               hand-written summary (unchanged; never read by code)
  layout.json                 machine-written: {"version": 2, "migrated": <date>, "moves": [...]}
  train/                      UNCHANGED — the training corpus and its contract
      discworld: obs.f32  meta.h5  corpus.json  _done_*     (val = last 10 % of obs.f32, by contract)
      othello:   train_20000000.npz
  probe/                      probe FIT corpora (hold-out = internal 80/20 by sequence — documented in the manifest)
      discworld: probe_120k.h5  probe_120k.json  probe_250k.h5  probe_250k.json
      othello:   probe_20000.npz  probe_large_170000.npz  + <stem>_labels_<n>.npz label caches
  eval/                       held-out sequences never used to fit anything
      discworld: test.h5  test.json          (waterfalls; future prediction-skill / Bayes-floor estimates)
      othello:   test_10000.npz              (gates; source games for synthesised edit cases)
  edits/
      v1/                     the CURRENT bench — teleports (discworld) / single-tile flips (othello)
          discworld: edits.h5  edits.json  selection.json   (selection only where one exists: dw-8ray)
          othello:   cases_1001.pkl  cases_1001.json
      v2/                     RESERVED — paired counterfactual bench (pairs.h5 + manifest.json with magnitude + filters)
  tokens/                     UNCHANGED (dw-8ray only): train.i16  vocab.npz  meta.json  test.npy  edits.npy
  _unused/                    files no code reads — MOVED, never deleted, original relative path preserved
```

`datasets/archive/` stays exactly as it is (Sevan, 2026-09-10).

### 2a. Per-file move table — discworld (each of dw-pn04, dw-noiseless, dw-8ray, dw-blink)

| from | to | why |
|---|---|---|
| `probe/test.h5` | `probe/probe_120k.h5` | the probe FIT corpus, named for what it is |
| `probe/dataset.json` | `probe/probe_120k.json` | its manifest (`sim` config read by `arms.py`) |
| `probe_250k/test.h5` | `probe/probe_250k.h5` | categorical-probe FIT + large observation floor |
| `probe_250k/dataset.json` | `probe/probe_250k.json` | manifest |
| `probe/{train,val,edits}.h5` | `_unused/probe/…` | 100-sample stubs, no reader |
| `probe_250k/{train,val,edits}.h5` | `_unused/probe_250k/…` | 100-sample stubs, no reader |
| `eval/edits.h5` | `edits/v1/edits.h5` | THE edit bench (first 192 cases; dw-blink 20k) |
| `eval/dataset.json` | `edits/v1/edits.json` **and** `eval/test.json` (copy) | the one manifest describes both surviving splits |
| `edits_selection.json` (dw-8ray) | `edits/v1/selection.json` | filtered case list, next to the cases it filters |
| `eval/test.h5` | `eval/test.h5` | unchanged location, unchanged role |
| `eval/val.h5` | `_unused/eval/val.h5` | only reader was the token-vocab builder (see §4d) |
| `eval/train.h5` | `_unused/eval/train.h5` | dw-pn04: 90k dset-4-era sequences, no reader; others: 100-sample stub |
| `tokens/probe.npy`, `tokens/val.npy` | `_unused/tokens/…` | no reader (token probes re-encode `probe_120k.h5`) |
| `probe/`, `probe_250k/`, `eval/` leftovers | — | `probe_250k/` and `eval/`-only-if-empty directories removed after the moves; `eval/` survives (test.h5) |

### 2b. Per-file move table — othello (each of oth-uniform, oth-noflip, oth-adjacent; oth-adjacent-flip has no data)

| from | to | why |
|---|---|---|
| `corpus/train_20000000.npz` | `train/train_20000000.npz` | the training corpus |
| `corpus/test_10000.npz` | `eval/test_10000.npz` | held-out gates + edit-case source |
| `corpus/probe_20000.npz` | `probe/probe_20000.npz` | probe FIT |
| `corpus/probe_large_170000.npz` | `probe/probe_large_170000.npz` | large observation floor / capacity |
| `corpus/probe_20000_labels_20000.npz`, `corpus/probe_large_170000_labels_170000.npz` | `probe/…` | label caches (derived; regenerable) |
| `corpus/probe_20000_labels_50.npz` (oth-uniform) | `_unused/corpus/…` | smoke artefact |
| `corpus/train_90000.npz`, `train_1000000.npz`, `train_5000000.npz` (uniform), `train_90000.npz` (noflip) | `_unused/corpus/…` | legacy ladder rungs; reachable only via `train.py --limit` — see §4c |
| `corpus/train_90000.npz.regen-dup` | `_unused/corpus/…` | stray |
| `edits/cases_1001.{pkl,json}` | `edits/v1/cases_1001.{pkl,json}` | the current bench |
| `pim/environments/othello/vendor/intervention_benchmark.pkl` | **copied** (cmp-verified) to `oth-uniform/edits/v1/cases_1001.pkl`; original stays in git | one loader rule for every instance; Li's file is kept (Sevan) |

## 3. The path module — one place paths are built

New: `pim/environments/layout.py`. Every path under `datasets/` is built here and nowhere
else. REPO-anchored absolute paths for both classes (ends Othello's CWD-relative regime;
`os.chdir(REPO)` in `master_eval` becomes unnecessary and is removed).

```python
LAYOUT_VERSION = 2
def instance_root(cls: str, inst: str) -> Path          # REPO/datasets/<cls>/<inst>
def train_dir(cls, inst) -> Path                        # …/train
def probe_file(cls, inst, size: str) -> Path            # discworld: size ∈ {"120k","250k"} → probe/probe_<size>.h5
                                                        # othello:   size ∈ {"20000","large_170000"} → probe/probe_<size>.npz
def probe_manifest(cls, inst, size) -> Path             # discworld only: probe/probe_<size>.json
def eval_file(cls, inst) -> Path                        # discworld eval/test.h5 · othello eval/test_10000.npz
def eval_manifest(cls, inst) -> Path                    # discworld eval/test.json
def edits_dir(cls, inst, version: str = "v1") -> Path   # …/edits/<version>
def edits_file(cls, inst, version="v1") -> Path         # discworld edits.h5 · othello cases_1001.pkl
def edits_selection(cls, inst, version="v1") -> Path | None   # selection.json if present
def tokens_dir(inst) -> Path                            # discworld …/tokens (unchanged)
def unused_dir(cls, inst) -> Path
def is_migrated(cls, inst) -> bool                      # layout.json exists with version >= 2
```

**Transition fallback.** Until `layout.json` exists for an instance, every function returns
the v1 path (`probe/test.h5`, `eval/edits.h5`, `corpus/…`). This lets the code land before
the moves and keeps the running chain safe. The fallback is deleted in the commit after
`--verify` passes; it is not a permanent feature.

**Logical split names replace paths in probe cache keys** (§5). `layout.py` also owns the
map from an old resolved `data` path to `(cls, inst, size)`, used once by the migration.

## 4. Code changes (all in the same commit as the module)

### 4a. Discworld canonical (`pim/`)
- `bigcorpus.py` — `instance_dir`/`train_dir` delegate to `layout`; the dead fallback to
  `datasets/20_dwscale_20m` is removed.
- `bench.py` — `DATA` default and `bench_arrays(data_dir=…)` become `bench_arrays(inst=…)`
  resolving through `layout.edits_file(…, "v1")`; `selection_path` → `layout.edits_selection`.
  The dead fallback to `datasets/4_fixed_refl_inview` is removed. `grid_selection` follows.
- `arms.py` — `probe_recipe(target, inst_root, …)` returns `{"probe": "120k" | "250k", …}`
  instead of a `data_dir`; `fit_probes`/`observation_probes` open `layout.probe_file` and
  `layout.probe_manifest`; the cache key gets `data=f"discworld/{inst}"`, `split="probe_120k"`
  or `"probe_250k"` (§5). `GRID_PROBE_RECIPE["split_dir"]` → `"probe": "250k"`.
- `loading.py` — `load_dataset(dir)` gains `load_eval(inst)` / `load_edits_v1(inst)` wrappers;
  the positional form stays for archived callers.
- `token_bench.py` — follows `bench.py`.
- `tokens.py` — `H5_SPLITS` becomes `(("probe", probe_120k), ("test", eval/test), ("edits", edits/v1))`;
  `val` is dropped (§4d). `vocab.npz` is **not rebuilt**.

### 4b. Othello canonical (`pim/`)
- `corpus.py` — `INSTANCES[*]["dir"]` removed; `corpus_dir()` becomes `split_dir(instance, split)`
  through `layout`; `build()` writes each split into its role directory and its prefix-reuse
  glob searches that directory only. `CACHE` alias removed (no callers after the change).
- `bench.py` — `cases_path(instance)` → `layout.edits_file("othello", instance, "v1")` for
  every instance; if the file is missing for `oth-uniform`, fall back to the vendored pkl
  (so a fresh clone without `datasets/` still runs Li's bench). `BENCHMARK_PKL` stays as the
  name of the vendored file.
- `corpus.probe_data` label-cache path follows the probe file.

### 4c. Behaviour notes that are NOT changes to any result
- `train.py --limit 90000/1000000/5000000` used to load the separately generated legacy
  corpora; after the move it takes a prefix of the 20M corpus (different games). No
  canonical run uses `--limit`; the runs trained on those files are archived. Reproducing
  one requires pointing at `_unused/corpus/train_<n>.npz` explicitly. Recorded in GOTCHAS.
- `eval/val.h5` leaving the token-vocab build: the vocabulary is a stored artefact
  (`tokens/vocab.npz`, 421 realisable patterns + UNK, `frames_only_outside_train: 0`) and
  is not rebuilt, so the token model and every token score are untouched. A future rebuild
  over the three remaining splits yields the same vocabulary iff every pattern in `val.h5`
  also occurs elsewhere — assert this once in `--verify` by encoding `val.h5` through the
  stored vocab and checking for zero UNK.

### 4d. What `eval/val.h5` was
Generated because `generate_dataset.py` always writes four splits. Not the training
validation set (that is the memmap tail). Read by exactly one thing: the token-vocab builder,
which also emitted the unused `tokens/val.npy`. It goes to `_unused/` and the builder stops
naming it.

### 4f. Producers — a future instance lands in layout v2 with no hand work

The layout is enforced at the point of writing, not only at the point of reading. A new
instance = one registry row (`bigcorpus.INSTANCES` or `corpus.INSTANCES`) + one driver run,
and every file it produces is already in §2 form with `layout.json` written by the producer.

- `scripts/generate_dataset.py` gains `--role {probe,eval,edits}` and `--size <n>`. It writes
  **only the split the role needs** — `probe`: one file `probe_<size>.h5` (+ `.json`), `eval`:
  `test.h5` (+ `.json`), `edits`: `edits/v1/edits.h5` (+ `.json`) — directly at
  `layout.<role>_file(...)`. The current "always four splits" behaviour goes away, which
  is what ends the stub files at the source. The legacy positional form (an arbitrary output
  directory, four splits) stays available behind `--legacy-dir` for pilots, and pilots keep
  living under `experiments/`.
- `bigcorpus.py` (`train/`) is unchanged; its `__main__` additionally writes `layout.json`
  (version 2) when the corpus is verified, so `is_migrated()` is true for every new instance.
- `scripts/drivers/dw_8ray.sh`, `dw_blink.sh` (the templates for a new discworld instance)
  call the three roles above in place of today's `probe/`, `probe_250k/`, `eval/` steps.
- `corpus.build()` (Othello) writes each split into its role directory (§4b); its
  `__main__` writes `layout.json`. `scripts/make_othello_edits.py` writes to
  `layout.edits_dir(..., "v1")`.
- `scripts/make_discworld_tokens.py` / `tokens.tokenize_instance` read `train/`,
  `probe_120k.h5`, `eval/test.h5`, `edits/v1/edits.h5` through `layout` and write `tokens/`
  exactly as today minus `probe.npy`/`val.npy`. Tokenising a future instance therefore works
  on a v2 tree with no extra steps; the `meta.json` gains `"layout": 2` and the list of
  source files.
- `edits/v2/` producers are out of scope here (the paired-bench spec); the directory name
  and the manifest requirement (magnitude + filters per case) are reserved so that spec has
  a home.

### 4e. Everything outside `pim/`
- **Canonical, updated to `layout`:** `scripts/train.py`, `scripts/fit_probes.py`,
  `scripts/make_discworld_tokens.py`, `scripts/make_othello_edits.py`,
  `scripts/drivers/dw_8ray.sh`, `dw_blink.sh` (existence gates),
  `notebooks/master_eval.ipynb`, `notebooks/make_waterfalls.ipynb`,
  `notebooks/build_full_table.ipynb` (one h5 read), `experiments/interface_ablation/edits_audit/scripts/make_selection.py`
  (writer of the selection file).
- **Tests (5):** `test_grid_target.py`, `test_oracle_editors.py`, `test_othello_instances.py`
  (currently pins the CWD-relative form — rewrite to assert the last three path parts),
  `test_probe_targets.py`, `test_token_bench.py`. Plus new `tests/test_layout.py`:
  fallback-before-migration, v2-after-migration, cache-key map round-trip.
- **Experiment one-offs (25 py + 3 sh under `experiments/`):** mechanical replacement of the
  two patterns (`inst_root/"probe"` → `layout.probe_file(…,"120k")` with the manifest
  alongside; `inst_root/"eval"` → `layout.edits_dir(…,"v1")`), **not re-run**; their outputs
  are historical and stay as they are. Two scripts recompute a cache key by hand
  (`dw_tokens/bridge/scripts/score_bridge.py`, `inlp/scripts/inlp_dw.py`): switch them to
  `arms.probe_recipe` so the key is built in one place.
- **Docs:** `CLAUDE.md` §5 (where things live), `research/REGISTRY.md` instance rows (split
  column), `research/GOTCHAS.md` (probe hold-out is internal; `--limit` note; cache-key
  change), `datasets/MOVES.md` (every move, §6), `experiments/README.md` if it names splits.

## 5. Probe cache migration

Key today: `prov = {model, span, v, target, n_seq, split="test", family, basis, seed,
data="/abs/path/to/<inst>/probe[_250k]", …}`; filename `probes_<blake2b(repr(sorted(prov)))>.pt`;
the blob stores `{"provenance": prov, "probes": …}` and `load()` raises on any mismatch.

New key: `data="discworld/<inst>"`, `split="probe_120k" | "probe_250k"`; everything else
unchanged. (`ProbeCache.VERSION` is **not** bumped — a bump is a refit signal, and nothing
about the fitted probes changes.)

Algorithm (`scripts/migrate_datasets.py --apply`, after the moves):
1. For every `runs/**/probes/probes_*.pt` (394 files on 2026-09-10; also
   `experiments/**/probes/` if any exist — enumerate, do not assume): load the blob on CPU,
   read `provenance`.
2. If `data` starts with `/`: map the resolved path to `(inst, size)` via `layout`
   (`…/<inst>/probe` → `120k`, `…/<inst>/probe_250k` → `250k`; anything unmapped → **abort
   before writing anything**, print the path). Rewrite `data` and `split`.
3. Recompute the filename with `ProbeCache.key`'s hash over the new dict. Assert no
   collision with an existing file. Write the new blob to `<new>.pt.partial`, `replace()` it
   into place, then remove the old file. Rewrite `INDEX.md` via `write_index()`.
4. Record `{old_name → new_name}` per directory in `runs/MOVES.md`.

Verification (§6 step 3) reloads every migrated file through `ProbeCache.load(new_name,
new_prov)` — the provenance check is the test.

## 6. The gate — `scripts/migrate_datasets.py --plan | --snapshot | --apply | --verify`

- `--plan` prints every move and every cache rename; touches nothing.
- `--snapshot` (before apply) writes `research/scratch/2026-09-XX-layout-snapshot.json`:
  for every file under `datasets/{discworld,othello}` (excluding `train/_done_*`): inode,
  size, path; for every instance: `bench_arrays` output hashed (positions, obs, edit metadata),
  Othello `load_benchmark` case hash, `oc.build(only=…)` resolved paths + inodes; every
  run's `scores.json["probe_sources"]` (or equivalent) → the cache filenames it names.
- `--apply` performs §2 moves (`os.rename`, same filesystem — `/dev/nvme0n1p2` holds both
  `datasets/` and `runs/`, verified 2026-09-10), copies the manifests, copies Li's pkl,
  writes `layout.json`, runs §5, appends to `datasets/MOVES.md` and `runs/MOVES.md`.
- `--verify` (CPU only, no GPU, no refits):
  1. every snapshot inode is present at its new path with the same size; nothing under
     `datasets/` was deleted (file count unchanged);
  2. `bench_arrays` / `load_benchmark` / `oc.build` hashes equal the snapshot, per instance;
  3. every migrated probe file loads through `ProbeCache.load` with its new provenance;
     count of `.pt` files per directory unchanged; every `probe_sources` entry resolves;
  4. `arms.probe_recipe` → `store.key` for each (run, target, basis) named in a run's
     `scores.json` produces a filename that EXISTS (the new key hits the migrated cache);
  5. `tokens`: `eval/val.h5` (now under `_unused/`) encodes with zero UNK under the stored
     vocab (§4c);
  6. `poetry run pytest` green; `bash harness/check.sh` clean;
  7. `git status` shows only the intended source changes.
- Rollback: `layout.json` absent ⇒ fallback paths; the move list in `MOVES.md` is exact, so
  `--rollback` can reverse every rename and every cache rename. Implement it; it is cheap.

## 7. Preconditions for `--apply`

- `logs/probe_targets_5/driver.log` contains `STAGE chain complete` **and**
  `systemctl --user is-active probe_targets_5` is `inactive`.
- `nvidia-smi --query-compute-apps=pid --format=csv,noheader` is empty.
- No process holds a file under `datasets/` or `runs/*/probes/` open
  (`lsof +D datasets/discworld/<inst>/probe …` per instance — small trees; never `lsof +D`
  on `train/`).
- The two notebooks show no source diff (`git diff --stat notebooks/` empty after
  nbstripout) — they are rewritten in place by the chain's nbconvert runs.
- `--snapshot` taken **after** the chain finishes (it may write new probe files until then).

## 8. Effort

Code + tests ≈ half a day. Moves + cache migration ≈ minutes. `--verify` ≈ 30 min (loading
394 probe blobs and every bench). Doc updates ≈ 1 h. No GPU time.

## 9. Out of scope (deliberately)

- `edits/v2/` contents — the paired-counterfactual bench is its own spec.
- Any change to `train/` contents, seeds, or `corpus.json`.
- Deleting anything. `_unused/` and `archive/` are kept indefinitely.
- Bumping `EVAL_VERSION` or `ProbeCache.VERSION`. Neither the scores nor the probes change.
