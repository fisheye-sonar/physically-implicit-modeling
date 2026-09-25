# Re-verifier report: anon-semantic

**Scope.** Read-only on RELEASE and STAGING.
- I read every shipped text file in RELEASE in full: README, both LICENSEs, .gitignore, pyproject.toml, setup.sh, all 92 `.py` files and every cell of the 3 notebooks. All notebooks have 0 outputs, `execution_count` null and a `python3` kernelspec.
- I viewed `assets/teaser.png`. Its chunks are IHDR, pHYs, IDAT and IEND only, so it has no text or time metadata, and the diagram itself holds no identifying content.
- In STAGING I scanned every JSON and JSONL file (203 files: 534 keys, 3,360 distinct strings) and all 28 HDF5 `config_json` attributes. I also read all npz fields, the 4 case pickles, the 55 `INDEX.md` files, and the headers of MANIFEST.json and SHA256SUMS.
- For all 1,245 `.pt` files I checked the zip entry roots and dates, the pickle globals and the non-tensor checkpoint metadata. For the 1,202 probe files I also read the provenance values.
- I checked the README claims against `paper/paper_draft.tex`.
- Work dir: `experiments/release/work/reverify-anon-semantic/`. It holds `strings_scan.py`, `bin_scan.py`, `pt_scan.py`, `keys.txt`, `strings.txt`, and `tree/` and `tree2/`, which are read-only copies of RELEASE `pim/`.

## Verified fixed since the first pass
- **HDF5 timestamps.** No HDF5 file has `generated_at` any more; the only root attribute is `config_json`.
- **Legacy names.** `BIG20M`, `rung`, `w16_reference_steps` and `window` are gone from the configs and checkpoints.
- **Layout paths.** No shipped manifest has a `layout` key, and all Othello corpus paths read `.../train/...`.
- **Spelling and jargon.**
  - Identifiers now use American spelling: `CENTER`, `synthesize_cases`, `FactorizedTarget`.
  - Prose now says "labeled", "centers" and "non-center token recolored", including in the 4 `cases_1000.json`.
  - `LADDER` is now `N_TRAIN_GAMES`.
- **Hardware.** The 32-core timing docstring is gone.
- **Unread files.** `rayworld/standard/variance.json` and `othello/adjacent-flip/variance.json` are no longer shipped.
- **Clean scans.**
  - Code, notebooks and STAGING contain no personal name, institution, place, username, host, IP or absolute path.
  - No date appears anywhere except the citation year and the LICENSE year.
  - No zip entry date is later than 1980.
  - The only URLs are the public othello_world and minGPT credits.
  - Probe provenance `data` values are only `rayworld/<instance>`.
- **README claims.** They stay within the paper:
  - The intro, the four highlights, the variants table and the reproduction steps match the abstract, the introduction, results line 344 and discussion line 360.
  - The bundle sizes 2.9, 9.4 and 3.5 GB match MANIFEST.json.
  - The claim that the 250k probe corpora ship only for 128-ray, 16-ray, 8-ray and 5-ray matches the STAGING listing.
  - The hardware and training times appear in the paper itself (line 1052), so `metrics.jsonl` `elapsed_s` reveals nothing new.

## Findings (most severe first)

### BLOCKER 1 (still open): the release repo is wired to the author's GitHub account and identity
- **Evidence:**
  - `.git/config` has `url = git@github.com:<author's personal account>/generative-models-as-simulators.git`.
  - The effective `user.name` is the author's real name, and `user.email` is a personal gmail address (masked).
  - The repo has 0 objects, so the first commit would record this identity.
- **Fix (a human step before the first commit):**
  - `git -C RELEASE remote remove origin`.
  - `git -C RELEASE config user.name Anonymous` and `git -C RELEASE config user.email anonymous@example.invalid`.
  - Commit with `TZ=UTC`, then check `git log --format='%an <%ae> %ad | %cn <%ce> %cd'`.
  - Publish from an anonymous account or through anonymous.4open.science, and create `<ANON_HF_REPO>` under an anonymous HF account.

### MAJOR 2 (correctness, found in passing): every Rayworld rescore crashes
- **Evidence:**
  - `pim/scoring/rayworld.py:100` calls `rwa.pinv_arm(model, b, lin, a_pi, space="zspace", dims=dims)`.
  - Line 153 calls `tkb.pinv_arm(..., uns, space="zspace", dims=dims)`.
  - Both callees now have the signature `(model, b, probes, alphas, dims="all")` (token version: `(..., alphas, uns, dims="all")`).
  - `inspect.signature(...).bind(...)` raises `TypeError: got an unexpected keyword argument 'space'` for both.
- **Impact:**
  - A full rescore of any Rayworld run, or adding a missing block to one, fails at the first PI arm.
  - On the shipped runs master_eval skips everything, so its no-op check does not catch this.
- **Fix (owner: scoring):** delete `space="zspace", ` from both calls. The numbers are unchanged, and the arms keep the label `PI[zspace]`.

### MINOR 3 (still open, a human decision): the package name `pim` is the private project's acronym
- **Evidence:** the private repo is `<org>/PhysicallyImplicitModeling`, and all 1,202 probe pickles reference `pim.probes.base.WorldStateProbe`.
- **Fix:**
  - Keep the name if the private repo stays private and "PIM" is never used publicly.
  - Otherwise rename the package (e.g. `gms`) and re-save the probe blobs under the new module path. Probe filenames hash only the provenance, so they do not change.

### MINOR 4: `othello/standard` (and `__seed0`) keeps a second trainer's schema
- **What it reveals:** the reference model was trained by code other than `scripts/train.py`. This is a history tell, not an identity one.
- **Evidence, `config.json`:**
  - The `model` block is in minGPT GPTConfig form: `vocab_size`, `n_layer`, `n_head`, `n_embd`, `embd_pdrop`, `resid_pdrop`, `attn_pdrop`.
  - It has top-level `unique_games`, `train_games`, `val_games`, `total_steps`, `warmup_steps`, `state_span` and `epochs_over_pool`.
  - Its `data` block has no rules and no corpus path.
  - The other 41 configs use the `train.py` schema.
- **Evidence, both `best_model.pt`:**
  - They have no `arch` key, so they load only through the bare-minGPT inference path in `registry._infer_arch`.
  - They carry `best: True` and `vocab: 61`.
  - Their `model_config` is in the same GPTConfig form.
- **Fix (owner: export), in STAGING copies:**
  - Rewrite `config.json` to what `train.py` writes:
    - `{"arch": "transformer_l_tokens", "model": {"vocab": 61, "block_size": 59}, "train": <unchanged>, "data": {"env": "othello", "n_total": 20000000, "n_train": 18000000, "n_val": 2000000, "objective": "ce", "block": 59, "instance": "standard", "flip": true, "placement": "enclosure", "corpus": "datasets/othello/standard/train/train_20000000.npz"}, "n_params": 25312768, "steps_per_epoch": 70312.5, "epochs": 11.093333333333334}`.
    - Keep `replicate` on `__seed0`.
  - Re-save both checkpoints with:
    - `ck["arch"] = "transformer_l_tokens"`
    - `ck["model_config"] = {"vocab": 61, "block_size": 59}`
    - `ck["epoch"] = ck["step"] / 70312.5`
    - `best` and `vocab` removed
    - `model_state` untouched
  - Then regenerate SHA256SUMS and MANIFEST.json.
- **Safety check done:**
  - The model fingerprint `6ea008f2c111` depends only on the state dict.
  - `random_init_model("transformer_l_tokens", cfg)` gives fingerprint `5ae54dd313e0` under both the old and the new `model_config`. This is the key the shipped Othello random-init floor probes carry.
  - So no probe key or floor changes.

### NIT 5: a comment hints at unreleased datasets
- **Evidence:** `pim/environments/rayworld/bigcorpus.py:89` reads "seed ranges of data outside this release; every corpus is still checked against them".
- **Fix (owner: rayworld-env):**
  - Change it to `# reserved seed ranges; every corpus is checked against them`.
  - Keep `RESERVED` as it is, because `verify()` uses it.

### NIT 6: a comment names an unreleased observation mode
- **Evidence:** `pim/environments/rayworld/config.py:72` reads "a top-down raster observation; not implemented here, kept so stored configs rebuild". The `omni2d*` fields also ship in every Rayworld manifest.
- **Fix (owner: rayworld-env):**
  - Change it to `# unused; kept so stored configs rebuild with SimConfig(**d)`.
  - Keep the fields, as invariant 1 requires.

### NIT 7: no-op resume records
- **Evidence:** `rayworld/128-ray`, `rayworld/16-ray` and their `__seed0` configs carry `"resumed": [{"from_step": 780000, "to_steps": 780000, "batch_order_exact": false}]`.
- **Why it matters:** a restart at the final step shows the run's history and carries no information. The other `resumed` records are real and truthful.
  - Note that `rayworld/standard__seed1` and `__seed2` resumed with `batch_order_exact: false`, so `train.py` cannot reproduce those two replicates bit for bit.
- **Fix (owner: export):** drop only the 4 no-op records, then regenerate SHA256SUMS and MANIFEST.json. Nothing reads `resumed`.

### NIT 8 (no change needed): leftover keys in the Othello `gates` block
- **Evidence:** every Othello `scores.json` has `output_kind: "logits"`, `out_sum_mean` and `out_neg_mass_mean` in its `gates` block. The release `gates()` does not write them; they are left over from the removed squared-error head.
- **Decision:** invariant 5 keeps the schema of in-scope blocks, and nothing reads these keys. Leave them.

### NIT 9 (still open): packaging from the working tree would leak the username
- **Evidence:**
  - `RELEASE/.ruff_cache/0.15.7/*` hold `/home/<user>/research/PIM/generative-models-as-simulators/...` (seen with `strings`).
  - The 5 `runs/` and `datasets/` symlinks point to absolute paths.
  - Both are gitignored.
- **Fix:** package only from `git archive` or a clean clone, never a zip of the working directory.

### NIT 10 (outside the release; affects the submission's anonymity)
- **Evidence:**
  - `paper/paper_draft.tex` has 164 comment lines. They include the author's first name with a date (line 243), dated notes (lines 164, 193, 246, 408, 820), and private paths (`research/findings/...`, `datasets/discworld/dw-8ray`, `paper_ci`).
  - `paper/figs/teaser.pdf` carries `/Creator (Keynote)`, `/Producer (macOS Version 26.6.2 ...)` and a CreationDate. pdfTeX copies these into the paper PDF.
- **Fix:**
  - Submit a comment-stripped source (e.g. arxiv_latex_cleaner).
  - Add `\pdfsuppressptexinfo=-1`, `\pdfinfoomitdate=1` and `\pdftrailerid{}` to the preamble.
  - Strip the teaser's metadata (`exiftool -all= teaser.pdf` or `qpdf --empty`).
  - Check the built PDF with `strings paper.pdf | grep -i -E 'keynote|macos|ptex|<first name>'`.

### Accepted, no change
- The citation `year = {2026}` and the LICENSE `Copyright (c) 2026 Anonymous Authors` are the usual form for an anonymous release.
- The name `Transformer-L` implies a smaller sibling. It is stored as `arch` in checkpoints and in `scores.json`, so leave it.
- `n_workers: 16` in the manifests and the `--workers 16` default are too weak a hardware signal to matter.
- The prose vocabulary (lands, arm, floors, residual point) is shared with the paper. The code adds no further stylistic linkage.

## Requests
- **scoring:** `pim/scoring/rayworld.py` lines 100 and 153, delete `space="zspace", ` (MAJOR 2).
- **export:** apply the `othello/standard` and `__seed0` config and checkpoint rewrite (MINOR 4) and drop the 4 no-op `resumed` records (NIT 7), then regenerate SHA256SUMS and MANIFEST.json.
- **rayworld-env:** apply the two comment rewordings (NITs 5 and 6).
