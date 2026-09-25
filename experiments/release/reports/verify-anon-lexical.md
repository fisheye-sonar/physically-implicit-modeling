# verify-anon-lexical: lexical anonymity scan of RELEASE and STAGING

**Verdict: issues.** The shipped code text and the artifact contents are free of names, institutions, emails, IPs, hostnames and absolute paths. There are three real leaks:

1. The release repo's git setup: the remote names the author's GitHub account, and the effective git identity is the author's real name and personal email with a `-0700` offset.
2. `.ruff_cache/` holds absolute `/home/<user>/` paths.
3. 28 HDF5 files keep `generated_at` timestamps.

The first is a blocker because the default commit/push path exposes it. The other two ship only if the directory is archived wholesale, or they break the spec's no-timestamp rule.

Work dir: `experiments/release/work/anon-lexical/`. It holds the scanners (`tokens.py`, `scan_release.py`, `scan_artifacts.py`, `classify.py`, `h5_times.py`, `h5fix_test.py`), the raw hits (`release_hits.jsonl`, `artifact_hits.jsonl`), the full string vocabulary of the bundle (`artifact_vocab.json`) and the per-hit judgement inventory (`inventory.json`).

## What was scanned

- **RELEASE**: every file, 141 in all.
  - Included: hidden files, `.gitignore`, `.ruff_cache/`, `.scratch/`, `.git/{config,HEAD,description,info,hooks}`.
  - Excluded: `.git/objects` (no commits exist yet).
  - UTF-8 files were scanned as text; binaries as printable runs. File and directory names were scanned too.
- **STAGING**: all 1496 files (1245 `.pt`, 171 `.json`, 33 `.jsonl`, 28 `.h5`, 14 `.npz`, 4 `.pkl`, `MANIFEST.json`, `SHA256SUMS`).
  - JSON/JSONL: parsed. Every key and value was scanned, plus epoch-like numbers and a key-name watchlist (time/host/user/path/git/commit/device/...).
  - `.pt`: zip member names, and a check that each inner archive prefix matches its file name (it does in all 1245). Every string opcode of `data.pkl` (640 distinct strings, all read by eye), plus `version`, `byteorder` and `serialization_id`. I also loaded every `best_model.pt` with `weights_only=True` and dumped all non-tensor leaves.
  - `.pkl`: pickle opcode walk.
  - `.npz`: member names, zip timestamps, dtypes and every string array.
  - HDF5: all object names, attribute names and values (with `config_json` parsed), string datasets, the `track_times` flag, and the raw object headers (message 0x12, v2 time fields).
  - PNG: all chunks. `assets/teaser.png` has only IHDR/pHYs/IDAT/IEND, and I viewed it (no names in the image).
  - xattrs on all 1839 paths: none.
- **Belt and braces**: GNU `grep -rai` over the raw bytes of both trees for the identity tokens (not the `ugrep` wrapper, which honors `.gitignore` and skips binaries). Also base64, hex and UTF-16 forms of the name tokens in RELEASE.
- **Tokens** (case-insensitive):
  - Names, places and accounts: sevan, brodj, sbrodjia, sevanbro, caltech, hobley, perona, pietro, pasadena, california, socal, los angeles, gmail, owner@, the machine's hostname, tailscale, ntfy, fisheye, sonar, physically[-_ ]?implicit, pim-master, anonymous.
  - Hosts and hardware: wsl, ubuntu, lab/lab box/the lab, 5090/4090/3090/rtx/a100/h100/14900/psu/whea/bios, host(name), remote, slurm, wandb.
  - Paths: /home/, /Users/, `C:\`, ~/, /mnt/, /tmp/, `.pim/`, `research/PIM`.
  - Regexes: email, IPv4, URL, git@, domain, @handle.
  - Dates and times: ISO/DMY/compact dates, month-name dates, years, weekdays, clock times, am/pm, timezones.
  - First person and attribution: my, I, me, "we decided", "X's pick/rule/choice", "the user".
  - Internal names: discworld, dw-/dw_/L-dw/L-oth/oth-, research/, harness/, findings, GOTCHAS, experiments/, paper_ci, queue, claude/anthropic/co-authored, TODO/FIXME, legacy/nullspace/nanda/oracle, generated_at/timestamp/datetime/gethostname/getuser/environ/commit_sha, ⛔/⚠.
- **Result**: 31 k raw hits, and each was assigned a judgement by `classify.py` (0 unclassified). Leaks are below. Everything else is benign, and the reasons are in `inventory.json`:
  - `pim` package name and the pickled global `pim.probes.base.WorldStateProbe`.
  - `OTH_*` identifiers.
  - The variables `lab`, `todo` and `my`, and decorators.
  - Citations of Li et al. and Nanda et al. (2023), the MIT line "Copyright (c) 2023 Kenneth Li", and the URLs `https://github.com/likenneth/othello_world` and `https://github.com/karpathy/minGPT`.
  - The `Anonymous` LICENSE/bibtex.
  - The `omni2d` SimConfig field and the prose word "noiseless".
  - Digits inside floats, hashes and `\u2014` escapes.
  - The zip placeholder dates `1980-01-01 00:00:00`.
  - Torch member paths `.../data/N`, and printable noise in PNG/IDAT and compressed HDF5 chunks (mixed-case 4-byte matches such as `NtFy` and `L-Dw`, with no exact-case long tokens).
- **URLs, complete list**:
  - The two vendored-code citations above: benign.
  - `https://facebook.github.io/watchman/` in a stock `.git/hooks` sample: benign, not shipped.
  - `git@github.com:SevanBrodjian/generative-models-as-simulators.git`: **leak**.
  - `<ARTIFACTS_URL>` and `<ANON_HF_REPO>` in README.md:44,57-63: placeholders.
- **No hits at all**: emails, IPv4 addresses, hostnames, month-name dates, weekdays, timezones, "the user", "we decided", personal attributions, and the old `discworld`/`dw-`/`L-dw`/`L-oth`/`oth-` names (checked in exact case in raw bytes too).

## Findings

### B1 (blocker): the release repo's git remote and identity name the author

**Evidence:**
- `.git/config:7`: `url = git@github.com:SevanBrodjian/generative-models-as-simulators.git`.
- `git var GIT_AUTHOR_IDENT` in RELEASE gives `Sevan Brodjian <s***@gmail.com> 1790269763 -0700`. It comes from `~/.gitconfig` (user.name, user.email); there is no repo-local override.

**Impact:** The repo has no commits yet, so the first `git commit` will stamp the real name, the personal email and the `-0700` (PDT) offset into history. `git push` then publishes under the named account. A reviewer who searches the paper title on GitHub can find `SevanBrodjian/generative-models-as-simulators` if it is public. Zipping the working dir also ships `.git/config`.

**Fix** (a human step; workers may not touch git config):
```
git -C RELEASE config user.name  "Anonymous"
git -C RELEASE config user.email "anonymous@example.com"
git -C RELEASE remote remove origin      # or set-url to a repo under an anonymous account
TZ=UTC git -C RELEASE commit ...          # avoids the -0700 offset
git -C RELEASE log --format='%an <%ae> %ad | %cn <%ce> %cd'   # check before any push
```
Then either publish from an anonymous account, or keep the named GitHub repo private and serve it through an anonymizing mirror. Ship any archive via `git archive`, never a zip of the working dir.

### M1 (major): `.ruff_cache/` contains absolute paths with the username

**Evidence:** All 12 files in `.ruff_cache/0.15.7/` contain `/home/sevan/research/PIM/generative-models-as-simulators/{pim,scripts,scripts/figures,scripts/demos,notebooks}`. For example, `.ruff_cache/0.15.7/312607013437321849` holds `/home/sevan/research/PIM/generative-models-as-simulators/pim`.

**Impact:** The directory is gitignored (`.gitignore:8`), so a git push is safe. Any archive or upload of the working directory ships it.

**Fix:** Delete `RELEASE/.ruff_cache/`, plus `.scratch/` and any `__pycache__/` (compiled `.pyc` files also embed absolute paths), before packaging. Package only with `git archive --format=zip HEAD`.

### M2 (major): 28 HDF5 files keep `generated_at` timestamps

**Evidence:** The root attribute `config_json` has `"generated_at": "2026-08-31T17:38:17"` … `"2026-09-15T20:58:20"`: 28 distinct date-plus-local-time values. They are plaintext in the file bytes (`grep -rac generated_at --include=*.h5` → 28 files). The files are every Rayworld `eval/test.h5`, `edits/edits.h5`, `probe/probe_120k.h5` and `probe/probe_250k.h5` (all 8 instances, 250k on 128/16/8/5-ray).

The JSON sidecars already dropped the field, so the h5 attribute and the sidecar now disagree. The export report left it in on purpose ("per the brief"), but SPEC says "No dates or timestamps anywhere". The clock times (11:15–23:35) are also a working-hours and timezone fingerprint.

There is no other time metadata:
- HDF5 object headers carry no modification-time messages (checked in `h5_times.py`).
- The zip dates in `.pt` and `.npz` are the fixed placeholders `1980-0-0` and `1980-01-01`.
- The torch `serialization_id`s are random.

**Fix:** Rewrite each file into a fresh file. Do not rewrite it in place: I tested that, and it leaves the old attribute bytes, `generated_at` and `2026-09` included, inside the file.
```python
with h5py.File(src, "r") as s, h5py.File(tmp, "w") as d:
    for k in s: s.copy(s[k], d, name=k)                 # raw chunk copy, data untouched
    c = json.loads(s.attrs["config_json"]); c.pop("generated_at", None)
    d.attrs["config_json"] = json.dumps(c, indent=2)
os.replace(tmp, src)
```
Tested on a copy of `datasets/rayworld/5-ray/eval/test.h5`:
- The sha256 of every dataset array, its chunks, compression and dtype are identical.
- `config_json` is equal apart from the dropped key.
- No `generated_at` or `2026-09` bytes remain.

No code change is needed: every reader uses only `["dataset"]["sim"]`. Afterwards, regenerate `SHA256SUMS` and the MANIFEST `sha256`/`bytes` entries for the 28 files, then re-run `grep -rac 'generated_at\|2026-0' --include=*.h5`.

### m1 (minor): symlinks in RELEASE point to absolute `/home/<user>/` paths

**Evidence:** `runs/othello`, `runs/rayworld`, `runs/_baselines`, `datasets/othello` and `datasets/rayworld` point to `/home/sevan/research/PIM/gms-release-artifacts/...`. They are gitignored (`.gitignore:19,21`), but tar and zip keep link targets.

**Fix:** Remove the five symlinks before packaging, or package with `git archive`.

### m2 (minor): legacy run name and research-ladder fields in `othello/standard` and its `__seed0` replicate

**Evidence:**
- `runs/othello/standard/config.json` and `runs/othello/standard__seed0/config.json` contain:
  - line 3 and line 37: `"rung": "D"`
  - line 8: `"run_name": "BIG20M_othello_L"`
  - line 45: `"w16_reference_steps": 95100`
- The same config.json files also have `"arch": "theirs"` and a stale `d_model: 256 / n_layers: 4`, which contradict `model.n_embd: 512 / n_layer: 8`.
- Both `best_model.pt` files carry the same `train_config` (`run_name`, `rung`, `arch: theirs`) and a top-level `rung`.

This is not identifying, but it is internal history under an old name, and it misleads readers.

**Fix:** In both config.json files, drop `train.run_name`, `train.rung`, the top-level `rung` and `w16_reference_steps`. No shipped code reads them (grep over `pim/`, `scripts/` and `notebooks/`: 0 hits). Optionally re-save the two checkpoints with the same keys dropped from `train_config` and the top level. `fingerprint` hashes only `state_dict`, so probe keys do not change. Regenerate SHA256SUMS and MANIFEST afterwards.

### n1 (nit): British spellings in prose

This is a weak geography signal, and SPEC asks for American spelling.

**Evidence:**
- `pim/environments/rayworld/grid_target.py:1` and `:30`: "labelled".
- `pim/environments/rayworld/grid_target.py:276`, `:278`: `centres`, a local variable.
- `scripts/make_othello_edits.py:51`: "non-centre tile recoloured". This string also ships in the 4 `datasets/othello/*/edits/cases_1000.json` `recipe` values.
- The identifiers `synthesise_cases` and `CENTRE` are in-scope public API.

**Fix:**
- Change the prose to "labeled", "centers" and "non-center tile recolored".
- Optionally update the 4 JSON recipe strings.
- Keep `synthesise_cases` and `CENTRE`, or rename them to `synthesize_cases` and `CENTER` with the old names kept as aliases.

### n2 (nit, advisory): publishing channel metadata

- If the bundle is ever shipped as a tarball, the tar headers store owner and group names (`sevan`) and mtimes. Use `tar --owner=0 --group=0 --numeric-owner --mtime=1980-01-01`.
- HF commits carry the uploader's account name, so fill `<ANON_HF_REPO>` and `<ARTIFACTS_URL>` (README.md:44,57,59,62,63) from an anonymous HF account.
- The paper still has `% TODO: add the anonymous code and artifact links.` (paper_draft.tex:379).

### n3 (nit): the package name `pim` is the private project's acronym

The private remote is `fisheye-sonar/PhysicallyImplicitModeling`. This only links the release to that project if the private repo becomes public during review. Renaming would break the pickled global `pim.probes.base.WorldStateProbe` in the 1198 probe files, so keep the name and keep the private repo private.

## Claims of the export reports that I checked

- export.md and export-fix.md say the identity scan found 0 hits across the full tree. **Confirmed for STAGING contents.**
  - This scan does not cover RELEASE `.git/config`, `.ruff_cache/` or the symlink targets (all three leak, above).
  - Nor does it cover the in-place-rewrite residue risk in HDF5 files: there is none today (no old names in the raw bytes), but it would appear if the M2 fix were done in place.
- "The `inverse_cleared` '(Sevan)' notes are deleted": **confirmed.** There are 0 `sevan` bytes anywhere in STAGING.
- The `commit_sha`, `settings` and dated records are dropped from the JSON: **confirmed.** There are no ISO dates in any JSON or JSONL file, no `commit_sha` keys, and no absolute paths. `probe_dir` is repo-relative.
- infra.md says the README has no forbidden names, paths, dates or URLs: **confirmed.**
