# Verifier report: reverify-gates

**Verdict: all seven gates pass.** Two new minor findings and two nits, none of which changes a number.

- RELEASE code and STAGING did not change while the gates ran (sha256 snapshot of every RELEASE file; `find -newer` on STAGING).
- The paper changed while this check ran. Every gate used `paper/paper_draft.tex` at sha256 `494b803c…`, whose reconstruction sentence no longer quotes ratios.
- Work dir: `experiments/release/work/reverify-gates/`. It holds the throwaway trees (`tree1`, `readme_all`, `core_tree`, `quick_tree`, `trace_tree`, `analysis_tree`, `vt/tree`), the gate copies (`gate/`, `vt/`), the executed notebooks (`executed/`), `logs/` (including strace), and `removed/` (shipped outputs moved out of my writable copy before regenerating them).
- Nothing was written to RELEASE, STAGING or PRIVATE `runs/` `datasets/`.
- Note on tooling: in this shell, `grep` is a wrapper around ugrep that adds `-I --ignore-files`, so it skips binary files. Every raw-byte scan below was rerun with `command grep` (GNU grep 3.12) or with Python.

## Gates

| # | gate | result |
|---|---|---|
| 1 | ruff (repo `pyproject.toml`: line length 100, E/W/F, E501 ignored) over `.` (97 .py files and 3 notebooks) | All checks passed |
| 1 | `compileall -f` on a copy of `pim/` and `scripts/` | rc 0, 97 files. Code and notebook cells also compile under Python 3.12.11, matching the README's "3.12 or newer" |
| 1 | import every `pim` module | 72/72, and every `__all__` name exists. `pim.__file__` is in my tree |
| 1 | `--help` on all 24 runnable scripts | 24/24 rc 0, no `outputs/` written. Third-party imports are only numpy, torch, h5py, matplotlib, pandas and tqdm, all in `pyproject.toml` |
| 2 | `score_all_baselines` / `score_all` dry runs with the notebook's SETTINGS cell, under strace | 12/12 baselines and 43/43 runs skip; both return `[]`; no write-mode open, mkdir, rename or unlink |
| 2 | execute copy of `master_eval.ipynb` (strace -f, kernel included) | rc 0, no error or stderr output; 12 + 43 skips; both calls return `[]`; no write in the tree or STAGING |
| 2 | execute copies of `paper_tables` (16 cells) and `appendix_tables` (54 cells) | rc 0, no error or stderr output, no files created. The shipped notebooks have 0 outputs, null execution counts and the `python3` kernelspec |
| 3 | `work/tables/check_tables.py` (copy; only the path assert adapted) | Stops with `KeyError: 'unreachable'` at the removed partners-searched print, as documented |
| 3 | fix-tables' adapted gate (`check_tables_after.py`; the diff against the original is only the two removed checks and the output name) | 1165 / 855 / 29 checked. Reference 0; paper 2 (the accepted `tab:im_by_point` cells, +0.25/+0.38 against +0.26/+0.39); numbers 1 (the accepted 0.106 against "at most 0.10"). The result JSON equals the pre-fix gate's except the two removed-print entries (their kept parts, 0.013 and 0.11, still match), and equals fix-tables' after and final runs |
| 3 | verify-tables tool (`verify_tables.py`, `verify_numbers.py`, `paper_parse.py`, in a fresh `vt/tree`) | 871 cells: text mismatch 2, raw mismatch 2, text vs raw 0. Both result JSONs are byte-identical to the original verifier's and fix-tables' final runs |
| 3 | executed notebooks vs the pre-fix verifier run, cell by cell | Differences are only the removed prints (two-flip reachable/unreachable/undecided columns; "categorical cases kept 951") and the three new cells |
| 3 | new cell: flip rates | 2.244878 (standard) and 0.268657 (adjacent-flip). Independently, n_flipped / n_moves = 1,345,948 / 599,564 and 161,194 / 600,000. The paper says "0.27 flipped tokens per move against 2.2": match |
| 3 | new cell: `pi_landing` | Equals the raw `readout_err_after` of every PI α = 1 cartesian arm (54/54 exact). Point 0 is 1.6 to 2.6; points 1 to 8 are 8.1e-7 to 4.5e-6. The PNG renders correctly |
| 3 | new cell: `im_reconstruction` | Ratios 9.48, 9.42, 7.99, 5.42, 1.95, 1.80, 6.16, 10.13 and 15.69 are bit-identical to PRIVATE `recon_overwrite.pre_fidelity`, and ratio = g_error / model_error exactly. Rerunning `scripts/im_reconstruction.py` (0.7 min, every inverse map a cache hit) rewrites the shipped file byte for byte |
| 4 | all six figure scripts | 31/31 outputs (15 PNG, 15 PDF, `history_rewrite.json`) byte-identical to `figures-recheck/treeC`, with 0 differing pixels. Also identical to `fix-tables/figtree2`, and in a core-bundle-only tree |
| 5 | README "Reproduce the paper", 9 commands verbatim (interpreter adapted, `PYTHONPATH` in place of the editable install), fresh tree with all bundles | 9/9 rc 0: 3, 3, 5, 47, 27, 21, 1, 2 and 9 s. The executed notebooks equal the gate-2 runs |
| 5 | the same on a core-only tree (per-file links to `bundle == "core"`), with the table notebooks CPU-only | 9/9 rc 0 and the same 31 outputs. master_eval returns `[]` (13 runs: replicates without weights are not listed). The Othello figures regenerate `probe_20000.npz` in place, as documented, with identical games (see F1) |
| 5 | README "Quick check", 13 commands verbatim, fresh tree with no download | 13/13 rc 0 in 52 s. The written `config.json`, `corpus.json`, manifests, `INDEX.md`, `.h5` and `.pt` files hold no absolute path, date or name. One UserWarning is printed (F3) |
| 5 | README Demos with `--save` | Both rc 0 (8 s and 14 s) |
| 5 | README download globs vs MANIFEST | The excludes select exactly `bundle == "core"` (808 files). The corpora (32) and replicates (708) includes select exactly their bundles. Sizes are 2.94 / 9.35 / 3.52 GB against the README's 2.9 / 9.4 / 3.5 |
| 6 | residue greps (the brief's list, `-i`, dates) over RELEASE | Only `Nanda et al. (2023)` (a citation) and `grid_target.py`'s `nd` (depth-cell count). No dates. Years appear only in the license and citations. No history wording in comments. Also clean: banners, private-file references, names, hosts, paths, URLs other than the vendored credits, and British spellings. `assets/teaser.png` has no text or time chunks |
| 6 | the same over STAGING | Text files (`json`, `jsonl`, `md`, `SHA256SUMS`): no SPEC-listed residue. The strings embedded in all 1291 binaries (1245 `.pt` pickles, 4 `.pkl`, 14 `.npz` keys and string arrays, 28 `.h5` names and attributes): 0 hits. A raw-byte GNU grep of all files for names, old run names, paths, `generated_at`, dates and clock times: 0 hits. A positive control finds 274 files |
| 7 | `sha256sum -c SHA256SUMS` in STAGING | 1548/1548 OK, rc 0. MANIFEST, SHA256SUMS and disk list the same 1548 files, and bytes, hashes and bundle totals all agree. STAGING holds no symlinks and no writable file |
| 7 | every file the code reads exists (strace of openat/stat over the full reproduce run) | 241 distinct files read under `runs/` and `datasets/`. The only misses are the 10 `__seed0/metrics.jsonl`, which `runs.training_complete` treats as complete when absent (by design) |
| 7 | analysis reruns in a writable copy, outputs moved aside first | Byte-identical: `im_reconstruction.py`, `othello_flip_rates.py` (4 files), `reachability_table.py --runs othello/standard`, `bayes_floor.py --force` (4 Othello). `two_flip_editability.py --run othello/standard-noflip --no-legal` is identical except `minutes` |

## Findings (new)

**F1 [minor] The README's checksum command fails after the figure scripts run on a core-only download.** Two regenerated Othello probe splits differ from the shipped files in metadata only.
- Shipped `datasets/othello/standard/{eval/test_10000,probe/probe_20000,probe/probe_large_170000}.npz` lack the `flip`, `placement` and `instance` keys that the release generator writes. The three `standard-noflip` files lack `placement`. The other two variants carry all three.
- With the core bundle only, the Othello figure scripts regenerate `probe_20000.npz` in place, as the README says. Afterwards, `sha256sum -c --ignore-missing SHA256SUMS` prints `datasets/othello/standard/probe/probe_20000.npz: FAILED` and the same for `standard-noflip` (rc 1; 810 OK).
- `tokens`, `lengths`, `lo` and `seed` are equal, so no number is affected.
- Fix, either:
  - export: ship the six `standard` / `standard-noflip` Othello `.npz` files with the release generator's key set (standard: `flip=True`, `placement="enclosure"`, `instance="standard"`; standard-noflip: `placement="enclosure"`), assert the four data arrays equal, then regenerate SHA256SUMS and MANIFEST;
  - or README, after the checksum line: "Run the check right after the download; without the corpora bundle, the Othello figure scripts write their probe splits into `datasets/`, and two of them then differ from the checksums in metadata only."

**F2 [minor] The appendix reconstruction header quotes ratios the paper no longer quotes.** `notebooks/appendix_tables.ipynb` cell `c3f81346` ends "(within about twice at points 4 and 5; five to sixteen times elsewhere)".
- The paper (line 823) now says the write "keeps the output's error low only at points 4 and 5, and raises it substantially elsewhere".
- The printed values support both wordings: 1.80 and 1.95 at points 4 and 5; 5.42 to 15.69 elsewhere.
- Fix, either: restore the ratios in the paper, which the release now reproduces; or change the header's parenthesis to "(low only at points 4 and 5, higher elsewhere)".

**F3 [nit] Every Rayworld training run prints a UserWarning.** The warning is "The given NumPy array is not writable …", from `pim/training/stream.py:43`. It comes from the unshuffled validation stream, which hands a read-only memmap view to `torch.from_numpy`. It is pre-existing: PRIVATE has the same line. The Quick check prints it twice.
- Fix: in `BlockStream._worker`, use `blk = np.array(self.obs[s: s + self.block])`, a copy. Values and RNG draw order are unchanged.

**F4 [nit] The `bayes_floor.py` docstring understates the Othello runtime.** It says "Othello: exact (…; seconds, CPU)", but `--force` on the four Othello instances took 302 s, about 75 s each.
- Fix: "about a minute per instance".

## Still open from the earlier verification (rechecked, unchanged)

- **Release repo identity (blocker, a human step):** `.git/config` origin still points to the author's personal account, and `user.name` resolves from the global config. The repo has no commits yet.
- **Packaging:** `.ruff_cache/` still holds 12 files containing an absolute home path. Package only via `git archive`.
- **README placeholders:** `<ARTIFACTS_URL>` and `<ANON_HF_REPO>` still need filling.
- **Paper text** (sha `494b803c…`, not a release file). These lines are unedited:
  - `tab:im_by_point` Othello IM at points 3 and 6 (+0.26 / +0.39; the release gives +0.25 / +0.38);
  - "at most $0.10$" at 990 (0.106);
  - "most common class of each cell" at 224 (the code pools over cells);
  - "gradient steps of size η" at 211 (Adam at η × activation scale);
  - "skip cases whose edit leaves the observation unchanged" at 219 (the rule needs 2 or more rays);
  - "both probe-derived editors rise steadily" at 346;
  - "reads the edited state exactly" at 346 (`pi_landing` now shows 1.6 to 2.6 at point 0);
  - "a different setting on each seed" at 465;
  - "at 780,000 steps" at 467;
  - "highest Edit Index instead of the table's fallback" at 1069 (the release figures draw the fallback);
  - "on the same sequences" at 288.
- **Artifacts:**
  - "canonical" wording in shipped JSON: 10 `replicate.note` values, 3 two-flip `editor` strings, and the `variance.json` keys, which the tables read;
  - `rayworld/obs5` cartesian `bench_selection` is still null;
  - `generate_dataset.py` prints the absolute output path (stdout only).

## Requests

- **export:** F1 (the six Othello `.npz` files, then SHA256SUMS and MANIFEST), or infra takes the README alternative.
- **infra (`README.md`):** F1's alternative sentence, if export does not reship the files.
- **tables (`notebooks/appendix_tables.ipynb`, cell `c3f81346`):** F2, unless the paper restores the ratios.
- **core (`pim/training/stream.py`):** F3.
- **scripts (`scripts/bayes_floor.py` docstring):** F4.
- **paper owner:** the paper lines listed above. The line-1069 figure sentence and the line-346 "exactly" now contradict release outputs directly.
