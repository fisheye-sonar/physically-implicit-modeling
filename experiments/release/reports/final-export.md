# final-export

**Status: done.** Both changes are in `experiments/release/export_artifacts.py`, so a re-run reproduces them. The export was re-run, verified and locked again. Exactly 16 STAGING files changed (6 Othello `.npz` and 10 `__seed0/config.json`). No number changed. Probe keys, checkpoints and every other file are byte-identical to before.

Gates finding F1 is fixed at the source. With the core bundle only, the Othello figure script now regenerates all four `probe_20000.npz` files **byte for byte**, and `sha256sum -c --ignore-missing SHA256SUMS` passes both before the figure scripts run and after them.

- Work dir: `experiments/release/work/final-export/`
  - `before/`: the previous script, SHA256SUMS, MANIFEST.json, verify.json and written.json
  - `export_artifacts.diff`: the script diff
  - `sha_diff.txt`: the SHA256SUMS diff (16 files)
  - `core_tree/`: the core-only proof tree
  - `gen_tree/`: a tree with code only, where all 12 Othello splits are regenerated
  - `check_changes.py`: an independent check that uses no export code
  - `dry_splits.py`: the rendering, dry-run before STAGING was touched
  - `verify.final.json` and `logs/`
- Nothing was written to RELEASE, to PRIVATE `runs/` `datasets/` `logs/` `outputs/`, or anywhere in STAGING outside the export run.

## Changes

### 1. Othello splits in the release generator's form (F1)

**Why byte identity is possible.** `np.savez` writes each member through `ZipFile.open(name, "w")`, and that call stamps every entry with the ZIP default date, 1980-01-01 00:00. All 24 PRIVATE and shipped Othello `.npz` files carry only that date. So the zip timestamps carry no information, and the bytes depend only on:
- the key order;
- the dtypes;
- the values.

**How the files are made.** Each of the 12 Othello splits is now the output of the RELEASE generator itself:
- `export_othello_splits` runs `OTHELLO_SPLIT_HELPER` in a RELEASE subprocess (`PYTHONPATH=RELEASE`; it asserts `pim.__file__` is under RELEASE).
- The helper calls `pim.environments.othello.corpus.build(only=(split,), instance=<new>)`, with two stand-ins:
  - `_generate` returns the PRIVATE source file's `tokens` and `lengths`, after asserting that `lo` and `n` match;
  - `corpus_dir` is an in-memory directory, so `np.savez` writes into a buffer and the helper writes nothing to disk.
- The file therefore has the generator's own key set, key order, rule values and dtypes: `tokens, lengths, lo, seed, flip, placement, instance`.

**Checks at write time (`othello_split_check`, in the PRIVATE process).**
- Every source array except `instance` is bit-identical in dtype, shape and bytes: `tokens`, `lengths`, `lo`, `seed`, and also `flip` / `placement` where the source has them.
- `flip` and `placement` equal the PRIVATE rules of the old instance, which are the rules the games were generated under. The rules `build` passed to the generator equal them too.
- `instance` holds the new name, with the generator's dtype.
- The only added keys are `flip`, `placement` and `instance`.

**What was added to each instance:**

| instance | added keys | values |
|---|---|---|
| `standard` (test, probe, probe_large) | `flip`, `placement`, `instance` | True, `"enclosure"` (<U9), `"standard"` (<U8) |
| `standard-noflip` (same three) | `placement` | `"enclosure"` |
| `adjacent-flip`, `adjacent-noflip` | none | byte-identical to the previous shipped files (same sha256) |

**New verify check (`verify_othello_splits`).** For each of the 12 shipped files, the RELEASE rendering is repeated and compared byte for byte. `othello_split_check` is rerun on the shipped bytes. The RELEASE `corpus.verify_splits` also runs on each instance's three shipped files: the index ranges are disjoint, and 66 rows per split regenerate under the recorded rules. Result: 12/12, bad [].

The export's own records:
- `work/export/othello_splits.json` holds each file's key set and its old and new instance names.
- In `written.json` the `how` field is "npz rendered by the release corpus.build".
- The three `standard` files are no longer verbatim copies: 277 copies are checked, down from 280.

### 2. The replicate note

- `transform_config` sets `replicate.note` to `REPLICATE_NOTE = "the main run's own checkpoint at the replicates' step budget"`. It first asserts that the value after the date strip is the old string (or already the new one).
- `verify_release_form` checks three things:
  - the string appears verbatim, as `"note": "<string>"`, in RELEASE `scripts/make_replicate_member.py`;
  - all 10 notes equal it;
  - no shipped `config.json` contains "canonical run".
- Result: 10 notes with the new string, 0 files with "canonical run".

## Proof of the F1 fix

**Core-only tree (`core_tree/`).**
- Setup: the RELEASE code, plus per-file links to the 808 `bundle == "core"` files in the locked STAGING. It has no `datasets/othello/*/probe/`.
- Before any script: `sha256sum -c --ignore-missing SHA256SUMS` gives 808 OK, rc 0.
- `scripts/figures/qualitative_othello.py` (GPU, 84 s, rc 0):
  - it regenerated `datasets/othello/{standard,adjacent-flip,adjacent-noflip,standard-noflip}/probe/probe_20000.npz` in the tree;
  - it wrote nothing else outside `outputs/` (`find -newer`), so every probe and inverse map was a cache hit.
- The regenerated files are **byte-identical** (`cmp`) to STAGING:
  - standard `17e51e31…`
  - standard-noflip `58b0dcd7…`
  - adjacent-flip `afddb0c4…`
  - adjacent-noflip `12eeb433…`
- After the script: `sha256sum -c --ignore-missing SHA256SUMS` gives **812 OK, rc 0**. Before the fix it gave 810 OK and 2 FAILED.
- The 10 figure files (5 seeds, PDF and PNG) are byte-identical to `figures-recheck/treeC`.

**Code-only tree (`gen_tree/`, no downloaded data).**
- `scripts/make_othello_corpus.py --instance <i> --splits test,probe,probe_large` ran for all four instances: rc 0, 31 to 63 s each, and the script's own `verify_splits` passed.
- All 12 regenerated files are byte-identical to STAGING, and `sha256sum -c --ignore-missing` passes on them.
- `scripts/othello_flip_rates.py` on the regenerated test splits rewrites the four shipped `corpus_stats.json` files byte for byte (2.2449 and 0.2687 tokens per move).

**No reader changes behavior.** The only release code that reads `flip` / `placement` from a split file is `verify_splits`. For a file without them it defaults to True / `"enclosure"`, which are exactly the values now stored for `standard` (and the placement stored for `standard-noflip`). Probe-cache keys do not hash data files.

## Verification (final locked tree)

`export_artifacts.py verify` (rc 0, 2.5 min). It was run after the export, and again after the reproducibility re-run:
- (a) Re-key: 1202/1202 probe files hash to their name.
- (b) Numbers: 169 JSON files, 363,309 numeric leaves, 0 failures, all bit-identical to PRIVATE.
- (c) Identity scan: 0 hits, dates and HDF5 attributes included. The raw-byte scan of 1550 files (15.82 GB) also has 0 hits.
- HDF5: 28/28 match their source, with 0 old timestamp strings.
- Othello splits: 12/12 (see above).
- `probes/INDEX.md`: 55/55 equal the release `write_index` output.
- Checkpoint fingerprints: 43/43 match.
- Verbatim copies: 277/277 byte-identical.
- Release form:
  - no `layout` keys and no `corpus/` paths;
  - every train block is `TrainConfig`, with no legacy keys left;
  - the unread files are absent and the generated file is verbatim;
  - the 6 filled `bench_selection` records equal their sibling's.

**Independent check** (`check_changes.py`, no export code):
- The changed files are exactly these 16.
- Each new `__seed0/config.json`, with the old note put back and re-serialized, hashes to its previous shipped sha256, so the note is the only change.
- The 6 changed `.npz` files hold the PRIVATE source arrays bit for bit, with the dtypes above.

**Checksums:**
- MANIFEST.json and SHA256SUMS were regenerated: 1548 files.
- `sha256sum -c SHA256SUMS` in STAGING gives 1548/1548 OK, rc 0, after the final lock.

**Reproducibility:**
- A second `export --force` produced a byte-identical SHA256SUMS and MANIFEST.json.
- The only files it touched were the two re-saved `othello/standard{,__seed0}/best_model.pt`, rewritten with the same bytes (pre-existing behavior).

**Lock:** `chmod -R a-w` is applied. 0 writable files or directories and 0 symlinks remain in STAGING, and a write attempt fails with "Permission denied".

**Links:** the 5 RELEASE symlinks resolve:
- `runs/othello` (16 entries)
- `runs/rayworld` (27)
- `runs/_baselines` (2)
- `datasets/othello` (4)
- `datasets/rayworld` (8)

## What changed in STAGING

| file | bytes before → after | bundle |
|---|---|---|
| `datasets/othello/standard/eval/test_10000.npz` | 611,004 → 611,811 | core |
| `datasets/othello/standard/probe/probe_20000.npz` | 1,221,004 → 1,221,811 | corpora |
| `datasets/othello/standard/probe/probe_large_170000.npz` | 10,371,004 → 10,371,811 | corpora |
| `datasets/othello/standard-noflip/eval/test_10000.npz` | 611,553 → 611,839 | core |
| `datasets/othello/standard-noflip/probe/probe_20000.npz` | 1,221,553 → 1,221,839 | corpora |
| `datasets/othello/standard-noflip/probe/probe_large_170000.npz` | 10,371,553 → 10,371,839 | corpora |
| 10 × `runs/<env>/<variant>__seed0/config.json` | −5 bytes each ("canonical" → "main") | core |

Nothing was added or removed, and nothing was moved to REMOVED.

## Bundle sizes

| bundle | files | bytes | before |
|---|---|---|---|
| core | 808 | 2,944,359,679 | 808 / 2,944,358,636 |
| corpora | 32 | 9,351,738,158 | 32 / 9,351,735,972 |
| replicates | 708 | 3,521,284,037 | unchanged |
| total | 1548 (+ MANIFEST.json, SHA256SUMS) | 15,817,381,874 | 1548 / 15,817,378,645 |

The README's 2.9 / 9.4 / 3.5 GB and its file counts are still correct.

## README (infra)

- **Not needed:** reverify-gates' F1 alternative sentence. Byte identity holds, so the checksum check passes after the figure scripts regenerate the probe splits.
- **Already true:** the current text, "Without them, the scripts regenerate the same games in place within about a minute". It could optionally be tightened to: "Without them, the scripts regenerate the same files in place (byte for byte, so the checksum check still passes) within about a minute."

## Open issues (not in this brief; left as shipped)

- **`rayworld/obs5` cartesian `bench_selection`** is still null, for the reason fix-export gives.
- **Other "canonical" wording in shipped JSON** is untouched: the 3 two-flip `editor` strings and the `variance.json` keys, which the tables read.
- **Pre-existing lint in `export_artifacts.py`:** an unused `defaultdict` import and an unused `files = []` in `export_datasets`. It does not affect the output.
- **`REMOVED`** still points to `work/fix-export/removed/`. Nothing was moved there this run.
