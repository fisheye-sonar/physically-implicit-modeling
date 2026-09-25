# Verifier report: anon-semantic

Scope: I read every shipped text file in RELEASE (README, LICENSE, .gitignore, pyproject.toml,
setup.sh, all 91 `.py` files in `pim/` and `scripts/`, every cell of the 3 notebooks and the vendored
LICENSE). I viewed `assets/teaser.png`. In STAGING I read every JSON and JSONL string and key, the
`config_json` attribute of all 28 HDF5 files, the npz fields, the non-tensor content of all 43
`best_model.pt`, the provenance and pickle globals of all 1202 probe files, the 4 case pickles, and
the MANIFEST.json and SHA256SUMS headers. Work dir: `experiments/release/work/anon-semantic/`
(`strings_scan.py`, `pt_scan.py`, `ckpt_meta.py`, `prov_scan.py`, `strings.txt`, `keys.txt`, `tree/`).

Clean, with evidence:
- The code contains no personal name, institution, place, username, absolute path or hostname.
- The only URLs are the public othello_world and minGPT credits.
- The notebooks have no outputs, `execution_count: null` and a `python3` kernelspec.
- The teaser PNG holds only IHDR/pHYs/IDAT/IEND chunks, so it has no metadata.
- npz and `.pt` zip entry dates are normalized (1980).
- Probe provenance `data` values are only `rayworld/<instance>`.
- The README claims (highlights, variants table, bundle sizes 2.9/9.4/3.5 GB, 410 GB corpus, 5 MB
  Othello probe splits) match the paper text and MANIFEST.json.

## Findings (most severe first)

### BLOCKER 1: the release repo is wired to the author's personal GitHub account and real identity
- Evidence: `generative-models-as-simulators/.git/config` has
  `url = git@github.com:SevanBrodjian/generative-models-as-simulators.git`.
- `git config --get user.name` returns the author's real name, and `user.email` a personal gmail
  address (masked here). The first `git commit` and `git push` would publish the paper's code under a
  named account, with the author's name and email in every commit.
- Fix:
  - Remove the remote: `git remote remove origin`.
  - Publish through an anonymous account or org, or through anonymous.4open.science with redaction
    terms `SevanBrodjian`, `Sevan`, `Brodjian` and `sevanbro`.
  - Commit with a repo-local anonymous identity: `git config user.name Anonymous`,
    `git config user.email anonymous@example.invalid`. Then check
    `git log --format='%an %ae %cn %ce'`.
  - Create the Hugging Face dataset (`<ANON_HF_REPO>`) under an anonymous account and upload with
    that account's token. HF shows the uploader on the repo page and in its commit history.

### MAJOR 2: internal run name and a legacy, contradictory config on `othello/standard` (+ `__seed0`)
- Evidence, `runs/othello/standard/config.json` and `runs/othello/standard__seed0/config.json`:
  - `"run_name": "BIG20M_othello_L"`, `"rung": "D"`, `"arch": "theirs"`, `"window": 16` and
    `"w16_reference_steps": 95100`.
  - `"d_model": 256, "n_layers": 4, "n_heads": 4` in the `train` block. The `model` block of the same
    file says `n_layer 8, n_embd 512`.
  - `"warmup_frac": 0.05`, and top-level `rung`, `unique_games`, `total_steps` and
    `epochs_over_pool`.
- The same `train_config` (with `BIG20M_othello_L`) is pickled inside both `best_model.pt`. Output of
  `ckpt_meta.py`: `"train_config": {"arch": "theirs", ..., "run_name": "BIG20M_othello_L", "rung": "D", "window": 16}`.
- The export's "identity scan: 0 hits" claim missed this: `BIG20M` was not in its token list.
- Beyond the internal name, a reader of `train` would conclude the Othello reference model is 4×256.
- Fix, in the STAGING copies only:
  - Rewrite both `config.json` to the schema every other run uses: `arch`, `model`
    (`{"vocab": 61, "block_size": 59}`), `train` = the `TrainConfig` fields (steps 780000, batch 256,
    lr 1e-3, wd 1e-4, clip 1.0, constant, warmup_steps 2000, ckpt_base 1000, val_every 5000, seed 0),
    `data` (env, instance, objective, rules, `corpus: datasets/othello/standard/train/train_20000000.npz`),
    `n_params`, `steps_per_epoch` and `epochs`. Keep the `replicate` block on `__seed0`.
  - Re-save both checkpoints with `ck["train_config"] = <that train dict>`, and drop `rung` and
    `best`. Leave `model_state` byte-identical: `fingerprint` hashes only the state dict, so probe
    keys are unaffected.
  - Regenerate SHA256SUMS and MANIFEST.json.

### MAJOR 3: dated timestamps in every shipped HDF5 file
- Evidence: the `config_json` attribute of all 28 `datasets/rayworld/*/*/*.h5` files has
  `generated_at`, from `2026-08-31T17:38:17` (standard) to `2026-09-15T20:58:20` (128-ray probe_250k).
  These are local wall-clock times, so they reveal the research timeline.
- The spec says "No dates or timestamps anywhere". The export kept them knowingly (`export.md:169`).
  The code no longer writes or reads the field (`core-ray-fix.md:21-23`).
- Fix, per STAGING file:
  ```
  h5py.File(p, "r+").attrs["config_json"] = json.dumps({k: v for k, v in json.loads(old).items() if k != "generated_at"}, indent=2)
  ```
  This leaves the datasets untouched. Then regenerate SHA256SUMS and MANIFEST.json.

### MINOR 4: the package name `pim` is the private project's acronym
- Evidence: the private repo is `github.com/fisheye-sonar/PhysicallyImplicitModeling`. The release
  installs a package `pim` that the README never expands. All 1202 probe pickles reference
  `pim.probes.base` (`prov_scan.py`).
- If that repo, or any talk or poster, uses "PIM" publicly, the name links the release to its authors.
- Fix, a human decision:
  - Keep the name if the private repo stays private.
  - Otherwise rename the package (e.g. `gms`). This needs a one-time re-save of the 1202 probe blobs
    under the new module path. Probe filenames hash only the provenance dict, so they do not change.

### MINOR 5: internal "data-scale ladder" name
- Evidence: `pim/environments/othello/corpus.py:70` has `LADDER = {"D": 20_000_000}`, used at
  `corpus.py:140,206`, `othello/bayes.py:54,57`, `scoring/othello.py:24,75`, `scripts/train.py:97`,
  `scripts/make_othello_corpus.py:26` and `scripts/figures/qualitative_othello.py:56`. `"rung": "D"`
  in the configs above has the same origin.
- Fix: replace it with `N_TRAIN_GAMES = 20_000_000` and use `oc.N_TRAIN_GAMES` at each call site.
  The numerics are unchanged.

### MINOR 6: description of the authors' hardware
- Evidence: `pim/environments/othello/corpus.py:144` says
  "CPU only, about 4.7k games/s on 32 cores (20M train games in about 70 min)."
- Fix: "CPU only; games are generated in parallel, one process per core."

### MINOR 7: layout-versioning and old-layout paths in shipped JSON
- Evidence:
  - `"layout": 2` appears in 18 Rayworld manifests (e.g. `datasets/rayworld/128-ray/edits/edits.json`).
  - `data.corpus = "datasets/othello/<inst>/corpus/train_20000000.npz"` appears in 6 Othello configs
    (adjacent-noflip, adjacent-noflip__seed0, adjacent-flip, adjacent-flip__seed0, standard-noflip,
    standard-noflip__seed0). The release writes `.../train/...`.
- Fix: drop `layout`, and rewrite `corpus/` to `train/` in those 6 files.

### MINOR 8: British spellings (a locale tic; the spec requires American)
- Evidence:
  - `othello/data.py:19` `CENTRE` (re-exported in `othello/__init__.py`).
  - `othello/bench.py:89` `synthesise_cases`.
  - `rayworld/grid_target.py:298` `FactorisedTarget`, `:1,30` "labelled", `:276` `centres`.
  - `scripts/make_othello_edits.py:51` "one occupied non-centre tile recoloured". This string is
    shipped in all 4 `datasets/othello/*/edits/cases_1000.json`.
- Fix:
  - Prose: "labeled", "centers", "non-center tile recolored" (also in the 4 STAGING json files).
  - Identifiers: `CENTER`, `synthesize_cases`, `FactorizedTarget`. None are pickled; the only pickle
    global is `pim.probes.base`.

### NIT 9: history and unreleased-work hints
- `runs/rayworld/standard/variance.json` is shipped, but no table or paper item reads it. The paper
  refits only standard Othello, adjacent-flip and 8-ray. It holds 20 seeds and a removed dim set
  `"dims": "pos"`. Drop it or document it.
- `resumed` records in 8 configs (e.g. `adjacent-flip__seed1`: from_step 390000) show interrupted
  trainings. This is truthful and harmless; drop it only if you want no history at all.
- `rayworld/bigcorpus.py:86` "seed ranges of data outside this release" and `config.py:72`
  "a top-down raster observation; not implemented here". Reword both to "reserved seed ranges" and
  "unused fields, kept so stored configs rebuild".
- `metrics.jsonl` `elapsed_s` (e.g. 28870 s for 780k steps) and `n_workers: 16` hint at hardware
  class. This is weak; keep it or drop it.

### NIT 10: non-git packaging would leak the username
- `.ruff_cache/0.15.7/*` holds `/home/sevan/research/PIM/generative-models-as-simulators/...`.
- `runs/` and `datasets/` hold symlinks into `/home/sevan/...`.
- Both are gitignored. A zip upload (supplementary material, 4open upload) would still include them.
- Fix: package only from a clean clone or `git archive`.

### NIT 11 (outside the release, but anonymity of the submission)
- `paper/paper_draft.tex` comments contain the author's first name and dates (line 245
  "Sevan 2026-09-22") and private paths.
- `paper/figs/teaser.pdf` carries `/Creator (Keynote)`, `/Producer (macOS Version 26.6.2 ...)` and a
  CreationDate. pdfTeX copies these into the paper PDF as `/PTEX.InfoDict`.
- Fix: add `\pdfsuppressptexinfo=-1` to the preamble, never upload the `.tex` with comments, and check
  the final PDF with `strings | grep -i -E "keynote|macos|sevan|PTEX"`.
- Also: README line 44 `<ARTIFACTS_URL>` is swallowed as an HTML tag by the GitHub renderer. Replace
  it before publishing.

## Checks run
- Full read of the release text files (pass, findings 4-8 and 10).
- git config and remote, read-only (fail, finding 1).
- STAGING JSON strings and keys (fail, findings 2 and 7-9).
- `best_model.pt` metadata (fail, finding 2).
- Probe provenance and pickle globals (pass).
- HDF5 attrs (fail, finding 3).
- npz and zip dates (pass).
- Teaser metadata (pass).
- README against the paper (pass).
