# Verifier report: final-gates

**Verdict: every gate passes. No blocker, no major.** The real scoring paths run end to end on the current RELEASE code: Othello refit, Rayworld frame model (PI zspace, GS, IM, categorical IM), and the token bench (`tkb.pinv_arm`). Every table-reported arm and every printed value is unchanged. There is one new minor finding (F1: a README caveat is too narrow) and three nits.

- The change set under test is the RELEASE state after the last fix. Against `reverify-gates/tree1`, 7 files differ: `pim/scoring/rayworld.py` (the two `space="zspace"` call sites removed), `pim/training/stream.py`, `pim/environments/rayworld/{tokens,bench}.py`, `scripts/{bayes_floor,make_replicate_member}.py` and `notebooks/appendix_tables.ipynb`. STAGING is the locked tree from final-export (1548 files).
- RELEASE did not change while the gates ran: the sha256 snapshots of all 123 non-`.git` files taken at the start and at the end are identical, and no `__pycache__` appeared.
- STAGING did not change either. `find -newer` finds nothing, `sha256sum -c SHA256SUMS` gives 1548/1548 OK, and there are 0 symlinks and 0 writable files. PRIVATE `runs/ datasets/ logs/ outputs/` were not touched (`find -newer`).
- One side effect: the directory mtime of RELEASE `.git/` changed once, at my first read-only `git status`. No file inside `.git/` changed.
- Work dir: `experiments/release/work/final-gates/`
  - trees: `tree_oth`, `tree_5ray`, `tree_tok`, `tree_rep`, `tree_bl2`, `tree_repro`, `tree_quick`, `tree_static`, `vt/tree`;
  - PRIVATE-code trees (copies only): `ptree`, `ptree2`;
  - `results/` (every comparison as JSON/TXT), `logs/`;
  - helpers: `compare_scores.py`, `cmp_bypoint.py`, `cmp_pt.py`, `kwcheck.py`, `anon_*.py`, `oth_gs_spot*.py`, `private_tokens_regression.py`.

## Gate 1: real rescores (fresh throwaway trees, master_eval SETTINGS, nbconvert)

Each tree holds a copy of the RELEASE code and a writable copy of the run and its baselines. The datasets are symlinked read-only into STAGING. Arms are matched by (editor, point, alpha, dims). "EI+fid" means `edit_index*`, `zone_edit_index_expected` and `fidelity_ratio*`.

### 1a. `othello/standard`: `scores.json` and `probes/` absent, so everything is refit

- The run took 12.5 min and rc was 0. The baselines were skipped.
- The scorer refit 10 probe files plus `INDEX.md`, with exactly the shipped filenames.
  - 9 `.pt` files and `INDEX.md` are byte-identical to the shipped ones.
  - `probes_37e891b4279cd312.pt` (the LIN + MLP grid) differs only in pickle memo layout (`data.pkl`, `.data/serialization_id`). All 1582 leaves are equal: state dicts, stats and module attributes.
- The rescore wrote no file into the (read-only) datasets.

| group | leaves | max abs diff |
|---|---|---|
| PI arms / GS arms | 1188 / 300 | **0** |
| probe_skill / probe_stats / gates / unedited | 18 / 126 / 8 / 7 | **0** |
| best PI / best GS | 11 / 10 | **0** |
| IM arms / IM-NN arms | 216 / 225 | 8.34e-7 / 9.23e-8 |
| best IM / best IM-NN | 24 / 25 | 1.05e-7 / 3.7e-8 |
| inverse_map | 27 | 1.52e-10 |

These are the same maxima the scoring worker found.

The key differences are the documented ones:
- the fresh file adds the case-level spread fields to the PI / GS arms, `best` and `unedited`;
- it drops `gates.output_kind`, `out_sum_mean` and `out_neg_mass_mean`, and the `prediction` block.

Tables:
- Table path (`tables.block_row`, both `select="index"` and `"fidelity"`): 22 cells, 0 changed.
- `tab:im_by_point` path (`tables.by_point`): 126 cells, 0 changed; the largest Edit Index difference is 3.8e-7.
- Selected arms: PI pt4 α3, GS pt4 α0.2, IM pt5, IM-NN pt4. Values: LIN 0.975, MLP 0.976; PI +0.82, GS +0.83, IM +0.81.

### 1b. `rayworld/5-ray`: `scores.json` absent, `probes/` kept

- The run took 9.3 min and rc was 0. All 34 probe files are unchanged, so every probe and inverse map was a cache hit.
- All three blocks were scored for real: frustum, cartesian, appearance-fac.
- 7925 numeric leaves were compared.

| block | Probe Skill, unedited, perdim, sanity | PI | GS | IM | IM-NN | inverse_map |
|---|---|---|---|---|---|---|
| frustum | 0 | **0** | **0** | 2.5e-8 | 3.2e-7 | 7.7e-9 |
| cartesian | 0 | **0** | **0** | 1.8e-8 | 3.5e-8 | 8.4e-9 |
| appearance-fac (categorical) | 0 | **0** | **0** | **0** (categorical IM) | – | 0 |

- Categorical GS is exact on this main run. Nothing exceeds 1e-4.
- Key differences: the fresh file adds the per-case spread fields and drops `prediction`. The `bench_selection` records are equal.
- Table path: 44 cells, 0 changed.
  - Cartesian: PI pt4 α175 +0.14, GS pt0 α0.7 −0.10, IM pt0 +0.81, IM-NN pt6 +0.60.
  - appearance-fac: PI pt1 α20 +0.51, GS pt0 α0.35 +0.60, IM pt5 +0.91.

### 1c. `rayworld/8-ray-tokens`: `scores.json` absent, `probes/` kept (the token bench path)

- The run took 3.4 min and rc was 0. All 34 probe files are unchanged. 11,139 numeric leaves were compared.

| block | skill / unedited | PI: EI+fid / all leaves | GS: EI+fid / all leaves | IM | IM-NN |
|---|---|---|---|---|---|
| frustum | 0 / 5.1e-8 | 1.4e-7 / **4.5e-3** (`readout_err_after`, pt0 α175, 86.724 vs 86.728) | **2.2e-4** / 2.2e-4 | 1.2e-7 | 6.8e-8 |
| cartesian | 0 / 5.1e-8 | 1.3e-7 / **0.173** (`readout_err_after`, pt0 α175, 501.43 vs 501.26; `readout_err_before` 1.5e-3) | **3.0e-3** (EI 1.85e-3) / 0.010 (`li_error_vs_post`) | 8.3e-8 | 6.3e-8 |
| appearance-fac | 0 / 5.9e-8 | 3.6e-7 / 0.076 (`readout_landed` α0.5, the known argmax tie) | **1.03e-2** (zone EI; EI 4.0e-3) / 1.26e-2 | 1.2e-7 | – |

- Table path, including the mean-frame rows: 58 cells, 0 changed.
  - The largest shift of a selected arm is 2e-3, on appearance-fac GS pt0 α0.35 (+0.35).
  - Cartesian: PI pt5 α175 +0.00, GS pt0 α0.7 −0.01, IM pt8 +0.65, IM-NN pt8 +0.23.
- **Cause: the shipped file's compute environment, not the release.**
  - I rebuilt a PRIVATE-code tree (`ptree`: a copy of PRIVATE `pim/`, the PRIVATE token run and the `dw-8ray` files) and scored the frustum and cartesian blocks on this GPU.
  - PRIVATE-here equals RELEASE-here **bitwise** on every arm: PI 2592, GS 800, IM 153 and IM-NN 162, per block. Probe Skill and `unedited` are equal too. `inverse_map` differs only in its `version` string.
  - Both differ from the shipped file identically.
- See F1.

### Extra coverage of scoring paths

**Seed replicates** (`tree_rep`: cached probes, `scores.json` absent):
- `rayworld/8-ray__seed1`: 10,620 leaves, **all exact** apart from `minutes`, categorical GS included.
- `othello/standard__seed1`:
  - PI Edit Index and fidelity agree to 3e-6, and IM / IM-NN to 6.5e-7;
  - `gates.top1_acc` differs by 3.4e-6;
  - **GS differs** by up to 0.0165 in Edit Index (pt0 α1.5: +0.0326 vs +0.0491) and 0.093 in fidelity ratio (pt0 α0.7: 3.948 vs 4.041).
- Every selected arm is unchanged: the selected GS is pt4 α0.2, with Edit Index diff 9e-9. Every 2-decimal value is unchanged.
- A PRIVATE-vs-RELEASE spot check on this GPU (`oth_gs_spot*.py`; GS from pt0 / pt2 at α 0.7 / 1.5, plus PI pt4 α3) gives 86/86 arrays bitwise equal. Two RELEASE runs are also bitwise equal to each other. So this is the same environment effect. See F1.

**Other paths:**
- **Prediction block:** `scripts/score_prediction.py --runs <id>` on the three rescored runs adds the block back in 1–3 s. Max differences: Othello 0, 5-ray 4.5e-11, tokens 0.
- **Baselines** (`tree_bl2`: `baselines.json` absent, datasets copied writable): `score_all_baselines` rebuilt both files from cached floor probes.
  - `othello/standard`: 127 of 128 leaves exact; only `minutes` differs.
  - `rayworld/5-ray`: 344/344 exact; only the key order differs.
  - No probe file was written. (With read-only datasets, the Othello floors fail; see F3.)
- **Static call-site check** (`kwcheck.py`): 1203 resolved call sites into `pim` across `pim/`, `scripts/` and the notebooks. Every call was bound against the callee's signature, with **0 problems**.
  - Positive control: on the previous `scoring/rayworld.py` it reports exactly the two `space="zspace"` calls, at lines 100 and 153.

## Gate 2: dry runs and notebooks

- **Dry runs** (`dryrun.py`, the notebook's SETTINGS, under strace), on a fresh tree linked to STAGING:
  - 12/12 baselines and 43/43 runs print skip, and both calls return `[]`;
  - there was no write-mode open, mkdir, rename or unlink.
- **Notebook copies:**
  - `paper_tables` (16 cells) and `appendix_tables` (54): rc 0, with no error or stderr output.
  - `master_eval` under strace -f: rc 0. Its only write is its own output notebook, plus Jupyter's `~/.jupyter/migrated` marker.
- **Against reverify-gates:** all executed outputs are cell-for-cell identical. The only source difference is reverify-gates' F2 header fix in cell `c3f81346`.
- **Shipped notebooks:** 0 outputs, null execution counts, `python3` kernelspec.

## Gate 3: table gates

- `check_tables_after.py` (the fix-tables copy, run from the RELEASE root):
  - 1165 / 855 / 29 checked;
  - reference 0 mismatches; paper 2 (the accepted `tab:im_by_point` +0.25 / +0.38); numbers 1 (the accepted 0.106);
  - the result JSON is **byte-identical** to `reverify-gates/gate/check_results_reverify.json`.
- verify-tables tool (`verify_tables.py`, `verify_numbers.py`, `paper_parse.py`, fresh `vt/tree`):
  - 871 cells; text mismatch 2, raw mismatch 2, text vs raw 0;
  - both result JSONs and both logs are **byte-identical** to reverify-gates'.
- The paper was unchanged during the run (sha256 `494b803c…`).

## Gate 4: figures

All six figure scripts ran against STAGING in a fresh tree, as part of gate 5. Their 31 outputs (15 PNG, 15 PDF, `history_rewrite.json`) are **byte-identical, with 0 differing pixels**, to the outputs of `reverify-gates/readme_all` and `figures-recheck/treeC`. The two prediction caches in `outputs/cache/` are byte-identical to reverify-gates'.

## Gate 5: README commands, verbatim (interpreter adapted, `PYTHONPATH` in place of the editable install)

- **"Reproduce the paper"** (9 commands; fresh tree with all bundles linked read-only):
  - 9/9 rc 0, taking 2, 4, 4, 46, 27, 20, 1, 3 and 8 s;
  - the executed notebooks equal gate 2's;
  - writes stayed in `outputs/` and `notebooks/*.executed.ipynb`.
- **"Quick check"** (13 commands; fresh tree, no download):
  - 13/13 rc 0 in 54 s;
  - **no UserWarning any more**, so reverify-gates' F3 is fixed;
  - `make_rayworld_tokens.py`, which now uses the layout helpers, ran;
  - every dataset file written (25 files, including `tokens/train.i16`, `vocab.npz`, `corpus.json` and all `.h5`) is **byte-identical** to reverify-gates' quick tree, which ran the previous `tokens.py`;
  - written files hold no path, date or name.
- **Checkpoints and `metrics.jsonl`** differ between runs. This is GPU run-to-run nondeterminism: two identical 100-step Othello runs in the same tree end at val loss 3.0771 and 3.0660. See N3.
- Command lists re-extracted from the README equal reverify-gates' lists.

## Gate 6: lint, imports, `--help`

- **ruff 0.15.7 with the repo config:** "All checks passed!" on 97 `.py` files and 3 notebooks.
  - Inside PRIVATE, ruff's gitignore handling silently skips `pim/probes/`. I reran with `--no-respect-gitignore` and on `pim/probes/*.py` explicitly: all pass.
- **Imports:** 72/72 `pim` modules import, and every `__all__` name exists.
- **Python 3.12.11:** 141 code units compile (97 files, 44 notebook cells).
- **`--help`:** 24/24 scripts rc 0 (`figures/style.py` has no main). No `outputs/` was written.

## Gate 7: lexical anonymity sweep

Patterns, case-insensitive:
- the names, institution and geography;
- `wsl`, `tailscale`, `ntfy`, `owner@`, `fisheye`, `physically-implicit` / `_implicit`, `/home/`, `/Users/`, `discworld`, `dw-`, `dw_`, `L-dw`, `L-oth`, `canonical run`;
- emails, IPv4 addresses, `20XX-XX-XX`, compact dates, date-plus-time stamps, and month-name dates.

**RELEASE** (141 files, symlinks not followed):

| group | files | real hits |
|---|---|---|
| would ship (not ignored) | 109 | **0** |
| gitignored local: `.ruff_cache/` | 14 | 12 cache files hold the absolute checkout path, which contains the username (known) |
| `.git/` | 18 | `.git/config` origin URL names the author's personal account (known blocker) |

- The would-ship group has 3 raw matches, all `dw_` / `dW-` inside `assets/teaser.png` IDAT compressed data, i.e. random bytes. The PNG has only IHDR, pHYs, IDAT and IEND chunks: no text or time chunks.
- File names: 0 hits. `.scratch/` is empty.

**STAGING** (1550 files):
- Structured pass over 220,384 strings: 0 hits. It covered:
  - every path and directory name;
  - 259 text files, including MANIFEST.json and SHA256SUMS; the clock-time pattern was included here;
  - pickle strings of 1245 `.pt` files and their non-tensor zip members;
  - 4 `.pkl`;
  - 14 `.npz` (keys, npy headers, string arrays);
  - 28 `.h5` (names, attributes, string datasets).
- Zip timestamps: `.pt` entries carry torch's zero date (1980-00-00) and `.npz` entries 1980-01-01.
- Raw-byte GNU grep of all 15.8 GB for the long patterns: 29 hits, all mixed-case 4-byte matches such as `l-Dw` or `L-DW` in HDF5 float payloads. The count matches the random expectation (about 30), and there are 0 exact-case `L-dw` / `L-oth`.
- Raw-byte regex pass: 0 dates, 0 IPv4, 0 old names. There are 44 email-shaped random byte runs in HDF5 float payloads, such as `OW@zC1.uz`.
- "canonical run": 0. Other "canonical" strings in STAGING are the known open items: the `variance.json` keys and the 3 two-flip `editor` strings.

## Findings

**F1 [minor] The README does not say that GS (and the token model's PI readout diagnostics) shift on a rescore for runs scored on another GPU. The reviewer caveat drafted so far covered only categorical GS, and that is too narrow.**
- Evidence (1c and Extra coverage):
  - `rayworld/8-ray-tokens`: GS moves on the regression blocks too, by up to 3.0e-3 in Edit Index on cartesian and 1.03e-2 in zone Edit Index on appearance-fac. PI `readout_err_after` moves by 0.173 (relative 3.5e-4).
  - `othello/standard__seed1`: GS moves by up to 0.0165 in Edit Index and 0.093 in fidelity ratio.
  - PRIVATE code on this GPU equals RELEASE bitwise in both cases, so the release code is not the cause.
  - `rayworld/8-ray__seed1`, `5-ray`, `othello/standard`, and verify-rescore's `8-ray` / `standard` reproduce exactly.
  - No selected arm changes, and no printed value changes. The largest shift of a selected arm is 2e-3.
- Fix (infra, `README.md`, Scoring paragraph): after "On the shipped runs it changes nothing." insert:
  > "A rescore reproduces most values to about 1e-6. Some runs, the token model and several seed replicates among them, were scored on a different GPU; there the GS arms, and the token model's PI `readout_*` diagnostics, can move by up to about 0.02 in Edit Index and 0.1 in fidelity ratio. The selected settings move by at most 0.002, and every value the tables print stays the same."

**F2 [nit] Stored `nn_r2` key sets differ from what the scorer now writes.** Tables are unaffected; they read NaN either way, and `SEED_COLS` has no `nn_r2`.
- Five shipped categorical blocks carry `inverse_map.nn_r2 = [NaN]*9`, which a fresh score no longer writes:
  - `rayworld/8-ray`: appearance, grid-6x5, grid-10x3, grid-16x8;
  - `rayworld/8-ray-tokens`: appearance-fac.
- The shipped replicates lack `nn_r2`, which a fresh score adds (for example `othello/standard__seed1`, and `rayworld/8-ray__seed1` frustum and cartesian).
- Fix (export, optional): drop `nn_r2` from those five categorical `inverse_map`s, which also removes their non-standard `NaN` JSON tokens. Leave the replicates as they are.

**F3 [nit] Rebuilding an Othello floor writes a 1.4 GB label cache into `datasets/`.**
- `pim/environments/othello/corpus.py:probe_data` writes `datasets/othello/<instance>/probe/probe_large_170000_labels_170000.npz` (1,375,471,236 bytes). It is not in the bundle and not mentioned in the README.
- With read-only datasets, `score_all_baselines` stops with `PermissionError` (seen in `tree_bl`).
- Fix (infra, README Scoring paragraph, after the `generate_dataset.py ... --size 250k` sentence): "Refitting an Othello floor caches the probe labels beside the probe split, about 1.4 GB per instance."

**N3 [nit, optional] GPU training is not bitwise reproducible run to run.**
- Two identical 100-step runs differ at the third decimal of val loss. The README claims nothing either way, and the paper's seed replicates cover the variance.
- Fix (optional, README "From scratch", after "The scripts regenerate every dataset from its recorded seeds."): "Training on a GPU is not bitwise deterministic, so a retrained model matches the shipped one statistically, not bit for bit."

## Still open (rechecked, unchanged)

- **Release repo identity (blocker, a human step):** the `.git/config` origin names the author's personal account, and there are no commits yet.
- **Packaging:** `.ruff_cache/` holds 12 files with the absolute path. Package only via `git archive`, or remove the cache before archiving.
- **README placeholders:** `<ARTIFACTS_URL>` and `<ANON_HF_REPO>`.
- **Paper lines** listed in reverify-gates, including the accepted `tab:im_by_point` +0.26 / +0.39 and "at most 0.10".
- **Artifacts:**
  - `rayworld/obs5` cartesian `bench_selection` is still null;
  - the "canonical" keys in `variance.json` and the two-flip `editor` strings;
  - `generate_dataset.py` prints the absolute output path (stdout only).

## Requests

- **infra (`README.md`):** F1 sentence; F3 sentence; N3 sentence (optional).
- **export (optional):** F2, dropping the NaN `nn_r2` from the five categorical blocks, then regenerating SHA256SUMS and MANIFEST.

## Bundle sizes (MANIFEST, unchanged)

| bundle | files | bytes |
|---|---|---|
| core | 808 | 2,944,359,679 |
| corpora | 32 | 9,351,738,158 |
| replicates | 708 | 3,521,284,037 |
| total | 1548 (+ MANIFEST.json, SHA256SUMS) | 15,817,381,874 |
