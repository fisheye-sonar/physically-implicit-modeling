# Worker report: infra

**Status: done.** Every owned file is written, and the fresh-environment test passes on Python 3.14 and 3.12. Two small lint requests to other owners are open.

Everything is in `experiments/release/work/infra/`:
- `fresh/` (the 3.14 copy, with its `.venv`, 5.8 GB, kept for reuse) and `fresh312/` (its venv was deleted);
- logs, `compare_nb.py`, `dump_tables.py` and `import_all.py`;
- `ref/`: the same notebooks run with the reference interpreter.

## Files (RELEASE)

- **`LICENSE`**: standard MIT, "Copyright (c) 2026 Anonymous Authors".
- **`.gitignore`**
  - The requested entries: Python basics, `.venv/`, `.ipynb_checkpoints/`, `*.executed.ipynb`, `outputs/`, `.scratch/`, and `runs/*` / `datasets/*` with `!runs/.gitkeep` / `!datasets/.gitkeep`.
  - Added `/MANIFEST.json`, `/SHA256SUMS` and `/.cache/`. The artifact download writes these into the repo root: `hf download --local-dir .` creates `.cache/huggingface/`, which I confirmed in a dry run.
  - Checked in a scratch git repo: `git status --ignored` tracks only code and the two `.gitkeep` files. Downloaded runs and datasets, the manifest files, outputs and executed notebooks are all ignored.
- **`runs/.gitkeep`, `datasets/.gitkeep`**: empty.
- **`pyproject.toml`**
  - PEP 621 with setuptools>=77, `license = "MIT"`, and `license-files` covering both the root LICENSE and the vendored one. No authors and no URLs.
  - Packages come from `find include = ["pim*"]`.
  - Dependencies are unpinned: numpy, torch, h5py, matplotlib, pandas, tqdm, jupyter, nbconvert, ipykernel.
  - A static AST scan of `pim/`, `scripts/` and the notebooks found only those six third-party imports. There is no seaborn, scipy, PIL or torchvision. The GIF writers use matplotlib's own pillow writer, and pillow comes in with matplotlib.
  - Ruff config: line-length 100, select E/W/F, ignore E501, no per-file ignores.
- **`setup.sh`**: `set -euo pipefail`, venv, activate, `pip install --upgrade pip`, `pip install -e .`, and a one-line next step.
- **`assets/teaser.png`**
  - Made with `pdftoppm -png -r 200 -singlefile` from `paper/figs/teaser.pdf`: 11112×2778, 718 KB.
  - I viewed it and extracted its text: no names or affiliations.
  - PNG chunks are only IHDR, pHYs (resolution), IDAT and IEND, so there is no text, time or EXIF metadata.
- **`README.md`**: sections in the order asked:
  - title, summary, teaser with a one-line caption, Highlights (4 bullets taken from the abstract, introduction and discussion);
  - Setup, Download (bundle table plus HF CLI commands), Reproduce (tables, scoring, figures), From scratch (5 ordered steps), Variants (13 rows), Layout, License, Citation (anonymous `@misc`).
  - Scanned for no em or en dashes, no semicolons in prose, and none of the forbidden names, paths, dates or URLs.

## README decisions worth checking

1. **Bundles (from `figures-recheck.md`).** Core is "every table and every figure". The note says the Othello edit figures (and the overview's Othello panel) read the 5 MB Othello probe splits from corpora, and regenerate them in place in a few seconds if they are absent. Sizes come from `MANIFEST.json`: 2.94 / 9.35 / 3.52 GB.
2. **HF CLI syntax.** The current `hf` CLI (huggingface_hub 2.0.0) takes one pattern per flag, so the README repeats `--include` / `--exclude`.
   - A dry run showed that `--include A B` makes `B` a positional filename, and the command then fails.
   - The bundle globs partition MANIFEST exactly (0 mismatches over 1494 files): `datasets/*/*/probe/*` = corpora; `runs/*/*__seed*/best_model.pt` + `runs/*/*__seed*/probes/*` = replicates.
   - I also added `--exclude ".gitattributes"` so the HF repo's LFS file does not land in the code repo.
3. **From scratch.** Every command comes from `stage_b_commands.md` plus the `scoring-settings.md` request (appearance-fac fits on standard and blink). All 147 expanded command lines were run with `--help` appended in the fresh env, and all exit 0.
   - This checks choices and types; a negative control confirms that bad values fail.
   - Flags were also checked against each script's `add_argument` calls.
4. **250k probe corpora for all 8 Rayworld instances.** Stage B listed them for the N-ray family only. Every shipped `baselines.json` stores its `observation_right_large` floor at n_seq 250000 for all 8 instances, and Table 1 reads that floor first. So a rebuild from scratch needs the 250k corpus everywhere.
5. **Notebook commands** use `--output <name>.executed.ipynb`, not `--inplace`, so the tracked notebooks stay clean. nbconvert writes the output next to the input.

## Fresh-environment test (exact results)

**Setup.** The release code was copied without `runs/` or `datasets/`, and `diff -r` against RELEASE was empty. `bash setup.sh` ran under `env -i` with PATH `/usr/local/bin:/usr/bin:/bin`, so it used the system `python3`.

| | Python 3.14.4 (system) | Python 3.12.11 |
|---|---|---|
| `bash setup.sh` | rc 0, **54 s**, no warnings | rc 0, **42 s** |
| versions | torch 2.14.0+cu130 (CUDA available, RTX 5090), numpy 2.5.3, pandas 3.0.6, matplotlib 3.11.2, h5py 3.16.0, tqdm 4.70.1, nbconvert 7.17.1, ipykernel 7.3.0, jupyter 1.1.1, setuptools 84.0.0 | same |
| import every `pim` module (from an outside cwd, no PYTHONPATH) | **72/72**, `pim.__file__` inside the copy | 72/72 |
| `scripts/train.py --help`, `scripts/figures/qualitative_rayworld.py --help` | rc 0 / rc 0 | rc 0 / rc 0 |
| `paper_tables.ipynb` (README command, STAGING symlinked in) | rc 0, 2.4 s, 0 error/stderr outputs | rc 0, 0 errors |

Pip reused 164 cached files and downloaded 52 more (torch and CUDA wheels at about 100 MB/s).

**Numbers against the reference.** I reran the notebook's calls in each env and pickled the Tables 1 and 2 `values`/`text`, the IM vs IM-NN gain and the seed-SD maxima.
- They are **bitwise identical** to the same calls run with the reference interpreter (PRIVATE venv, Python 3.13.5) on RELEASE, on both 3.14 and 3.12.
- Table 2 matches the paper cell for cell, including the four daggers.
- The mean IM gain is 0.53633 (53.6%). Seed-SD maxima: 0.0037 (Probe Skill) and 0.034 (landing EI), with 4 cells above 0.1.

**Text outputs match; images differ by matplotlib version.**
- Text outputs match the reference run exactly in every cell.
- The table images differ in about 6–8% of pixels, because the reference venv has matplotlib 3.10.8 and the fresh one 3.11.2. The 3.12 and 3.14 images are pixel-identical to each other.

**Extra checks, not required.**
- **`appendix_tables.ipynb`:** rc 0, 3 s, 0 errors, and the text of all 30 cells is identical to the reference.
- **`master_eval.ipynb`:** rc 0, 4 s, 0 errors, 55 skips (43 runs + 12 baselines) and 0 WOULD lines, so it really is a no-op.
- **Figure scripts:**
  - `editability_by_point.py` ran in 1 s and `predictions.py` in 8 s, both rc 0, printing the paper's cases.
  - Their PNGs render correctly (viewed), but under matplotlib 3.11 they are 1–4 px narrower or wider than the paper renders (tight bbox).

**Clean up.** RELEASE has no `__pycache__` (none created). No writes to STAGING or PRIVATE `runs/`, `datasets/`, `logs/` or `outputs/`, and no git operations beyond a read-only `check-ignore`.

## Requests

1. **core, `pim/probes/cache.py:78`.** Ruff E702: split `seen.add(k); cols.append(k)` into two lines. This is only the INDEX.md column loop, so hashing and numerics are untouched.
2. **tables (`notebooks/paper_tables.ipynb`, `notebooks/appendix_tables.ipynb`) and scoring (`notebooks/master_eval.ipynb`).** Ruff E402 on the `from pim... import` line that follows `sys.path.insert(0, str(REPO))` in the first code cell. Append `  # noqa: E402` to that import line, using NotebookEdit.

With both applied, `ruff check .` passes under the new config. Right now it reports exactly those 4 findings.

## Open issues

1. **Placeholders to fill:** `<ARTIFACTS_URL>` and `<ANON_HF_REPO>`.
   - GitHub's renderer hides the raw `<ARTIFACTS_URL>` as an HTML tag until it is replaced with the URL.
   - The commands assume `--repo-type dataset`, with the repo root mirroring STAGING (`runs/`, `datasets/`, `MANIFEST.json`, `SHA256SUMS`). Change the flag if the upload is a model repo.
   - The CLI syntax was verified by dry runs on a public repo, not on the real artifact repo.
2. **Dependencies are unpinned.** The tables' numbers are bitwise identical under the latest versions, but figure pixels depend on the matplotlib version. Pin `matplotlib==3.10.8` only if pixel-identical figures matter.
3. **Not tested:**
   - CPU-only runs.
   - The From-scratch steps beyond argument parsing (they take days of GPU time).
   - The full HF download.
4. **Tested versions are not stated in the README.** "Python 3.12 or newer" is backed by the 3.12 and 3.14 tests here and the 3.13 reference.
