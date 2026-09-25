# reverify-anon-lexical: lexical anonymity re-scan of RELEASE and STAGING

**Verdict: STAGING is clean. RELEASE has one blocker and two leaks, all outside the git-tracked set.**

- **Blocker:** the git remote and the effective git identity name the author.
- **Leaks:** gitignored local files contain absolute `/home/<user>/` paths. They would ship if the working tree were zipped or tarred instead of exported with `git archive`.
- **Shipped content is clean:** the 109 files `git add -A` would commit, and all 1550 files of the artifact bundle, contain no names, institutions, places, emails, IPs, hostnames, usernames, absolute paths, dates, clock times or timezones.

The scanner was written from scratch and checked with planted canaries before use. Everything is in `experiments/release/work/reverify-anon-lexical/`.

## Method

**Scope**
- RELEASE: every entry, including hidden files, `.gitignore`, `.ruff_cache/`, `.scratch/`, the symlinks and `.git/{config,HEAD,description,info,hooks}`. `.git/objects` is excluded; it is empty (no commits, no index).
- STAGING: all 1550 files: 1245 `.pt`, 170 `.json`, 55 `INDEX.md`, 33 `.jsonl`, 28 `.h5`, 14 `.npz`, 4 `.pkl`, plus `MANIFEST.json` and `SHA256SUMS`.

**What was read, per file type (`lexscan.py`)**
- Text: line by line. JSON/JSONL/ipynb were also parsed, and every numeric leaf was checked against epoch ranges (s and ms).
- `.pt`: every zip member name, zip comments, extras, dates and creator fields. Every string and number opcode of each `data.pkl` (pickletools, no unpickling). The small members (`version`, `byteorder`, `.format_version`, `.storage_alignment`, `.data/serialization_id`). A check that the archive prefix equals the file name: 1245/1245 match.
- `.pkl`: the pickle opcode walk.
- `.npz`: member names, zip metadata, and every `.npy` header. Every string-typed or object array was read (none are object).
- `.h5`:
  - every object name and every attribute name and value, including root `config_json` parsed as JSON;
  - every string-typed dataset;
  - per-object header times and object comments (`h5times.py`);
  - the object-header message types present (`h5msgs.py`).
- PNG: every chunk. `assets/teaser.png` has only IHDR/pHYs/IDAT/IEND. I also viewed it: a diagram with no names or marks.
- Also covered: file and directory names, symlink targets, xattrs (none anywhere), owners, modes and mtimes.

**Raw bytes**
- Every binary (1306 files, 15.8 GB) was run through `strings -a -n 8`, then a case-insensitive grep for 47 patterns (`rawpat.txt`): identity tokens, paths, `generated_at`, old run names, emails, URLs, ISO dates, clock times with seconds, and compact date-times.

**Vocabulary review**
- `vocab.py` lists the 548 distinct keys and 3361 distinct string values across all JSON/JSONL, HDF5 attributes and npz string arrays. I read by eye the 204 values that are not hashes, run ids or numbers.
- I also read all 488 non-hash pickle strings. Nothing identifying: module paths, parameter names, target and block names, `cuda:0`/`cpu`, and 12-hex model fingerprints.

**Tokens and patterns (case-insensitive)**
- Identity and places: the author's first and last name and email prefix, `caltech`, `hobley`, `perona`, `pietro`, `pasadena`, `california`, `los angeles`, `socal`, `gmail`, `owner@`, `tailscale`/`tailnet`, `ntfy`, `fisheye`, `sonar`, `physically[-_ ]implicit`.
- Machine: `wsl`, `ubuntu`, the machine's hostname, `lab box`/`the lab`/`lab`, GPU and CPU model numbers, `keynote`/`macos`/`darwin`, `anthropic`/`claude`/`co-authored`.
- Paths: `/home/`, `/Users/`, `~/`, `C:\`, `/mnt/`, `/tmp/`, `research/PIM`, `.pim/`, `gms-release`.
- Runtime identity calls and time fields: `gethostname`/`getuser`/`getlogin`/`platform.node`/`environ`, `generated_at`/`timestamp`/`datetime`/`strftime`/`commit_sha`.
- Old names: `discworld`/`dw-`/`dw_`/`L-dw`/`L-oth`/`oth-`, `BIG20M`, `w16_reference`, `rung`.
- Regexes: email, IPv4, URL, `git@`, domain, GitHub/HF user, @handle, ISO/US/compact dates, month-name dates, weekdays, years, clock times, am/pm, timezone abbreviations (exact case), UTC offsets, epoch numbers.
- First person and attribution: `I`, `my`/`me`/`mine`, `we`/`our`/`us`, "X's pick/choice/…", "the user". Also TODO/FIXME/XXX.

**Canary test**
- A planted tree held leaks in JSON (value and epoch), an h5 root attribute, an h5 group attribute, an h5 string dataset, an npz string array, a `.pt` pickle (string and int), a `.pkl`, a PNG tEXt chunk and prose.
- All 25 planted leaks were reported (`canary/hits.jsonl`).

**Result**
- 3802 structured hits and 523 raw-byte lines.
- `classify.py` gave every structured hit a judgement: 0 unclassified (`inventory.json`). The raw-byte lines were judged by hand; they are below.

## Leaks and exact fixes

### L1 (blocker): the git remote and identity name the author

**`.git/config`, verbatim except the account name:**
```
[core]
	repositoryformatversion = 0
	filemode = true
	bare = false
	logallrefupdates = true
[remote "origin"]
	url = git@github.com:<author-account>/generative-models-as-simulators.git
	fetch = +refs/heads/*:refs/remotes/origin/*
[branch "main"]
	remote = origin
	merge = refs/heads/main
```

**Evidence:**
- `<author-account>` is the author's real first and last name.
- The effective identity comes from the global `~/.gitconfig`, with no repo-local override. `git var GIT_AUTHOR_IDENT` gives the author's real name, a personal gmail address and a `-0700` offset.
- There are no commits yet. The first `git commit` would write all of this into history, and `git push` would publish under the named account.
- `HEAD`, `description` and `info/exclude` are the stock git files. `hooks/*.sample` are stock samples, which are never pushed or archived.

**Fix** (a human step: workers may not touch git config):
```
cd RELEASE
git config user.name  "Anonymous"
git config user.email "anonymous@example.invalid"
git remote remove origin     # or set-url to a repo under an anonymous account, pushed with that account's key
TZ=UTC git commit ...        # a +0000 offset; optionally also set GIT_AUTHOR_DATE / GIT_COMMITTER_DATE to a neutral value
git log --format='%an <%ae> %ad | %cn <%ce> %cd'   # check before any push
```
Alternatively, keep the named repo private and publish through an anonymizing mirror. Never ship `.git/`.

### L2 (major if the tree is archived wholesale): `.ruff_cache/` has absolute home paths

**Evidence:**
- All 12 files in `.ruff_cache/0.15.7/` contain `/home/<user>/research/PIM/generative-models-as-simulators/{pim,scripts,scripts/figures,scripts/demos,notebooks}`.
- Their mode is 600, and they are the only binaries in RELEASE besides the teaser.
- `.gitignore:8` ignores the directory, so `git archive` and `git push` are safe.

**Fix:**
- Before any non-git packaging, move `RELEASE/.ruff_cache/` and the empty `RELEASE/.scratch/` out of the tree (or run `ruff clean`).
- Package the code only with `git archive --format=tar.gz -o release.tgz HEAD`.

### L3 (minor if the tree is archived wholesale): the five test symlinks point into the home directory

**Evidence:**
- `runs/{othello,rayworld,_baselines}` and `datasets/{othello,rayworld}` point to `/home/<user>/research/PIM/gms-release-artifacts/...`.
- They are ignored by `runs/*` and `datasets/*`, but tar and zip keep link targets.

**Fix:** move the five links out before packaging, or use `git archive` (L2).

### L4 (advisory): file-system metadata in any tar or zip

**Evidence:**
- All 1870 entries in both trees are owned by `<user>:<user>`.
- RELEASE mtimes span the build days, and `pim/environments/othello/vendor/LICENSE` keeps an older mtime copied from PRIVATE.
- STAGING mtimes span two days.
- Modes: 664 / 600 / 775 (setup.sh).

**Fix:**
- Code: `git archive` (L2). It writes commit-time mtimes and no owner names.
- Bundle, either:
  - upload the files to an anonymous HF account or org (HF records only the uploader account and commit time, so use an anonymous account and token); or
  - use `tar --sort=name --owner=0 --group=0 --numeric-owner --mtime=@0 --pax-option=delete=atime,delete=ctime`.
- Avoid zip of the raw tree.

### Placeholders to fill before publishing (not leaks)

- `README.md:45`: `<ARTIFACTS_URL>`.
- `README.md:58,60,63,64`: `<ANON_HF_REPO>`.

Both must name an anonymous host or account.

## No leak found (checked explicitly)

- **HDF5, all 28 files and 382 objects:**
  - No `generated_at`, no date and no clock time in any attribute. None in the raw bytes either (strings pass).
  - Header times are all 0.
  - No object carries a comment (0x0D) or modification-time (0x0E/0x12) message. Datasets hold only dataspace, datatype, fill, layout and filter messages. Root groups hold attribute, continuation and symbol-table messages.
  - `get_obj_track_times()` reports 1 only because that is the library default for v1 headers.
  - h5py's `get_comment` returns uninitialized buffer bytes when no comment exists. That is not stored data.
- **`.pt` / `.npz` zip dates:** only the fixed placeholders. torch writes `1980-00-00` (36,501 entries) and numpy writes `1980-01-01` (80 entries). `serialization_id`s are random 40-digit strings.
- **Old names and legacy fields:** 0 hits anywhere for `discworld`, `dw-`/`dw_`, `L-dw`, `L-oth`, `oth-`, `BIG20M`, `w16_reference`, `rung`, `generated_at` and `commit_sha`.
- **Runtime writers:** no code calls hostname, user, platform, datetime, strftime, getcwd or git. `time.time` is used only for durations (`minutes`, `elapsed_s`). Every path written to JSON goes through `relative_to(REPO)`.
- **British spellings:** 0 in RELEASE and in STAGING strings.
- **Personal names:** none beyond the third-party credits (Andrej Karpathy; Kenneth Li in the vendored MIT license).

## Benign hit inventory (from `inventory.json`)

| hits | judgement |
|---|---|
| 2080 | Digits inside a hash, probe key, serialization id, float, count or `\u2014` escape. This covers every STAGING `year`, epoch-like and 3090/4090/5090/14900 hit (MANIFEST sha256 values, probe file names, `metrics.jsonl` floats, `bytes: 1896973875`). |
| 847 | `mine`: the Othello mine/theirs target |
| 268 | "environment"; `os.environ` passes PYTHONPATH in `bigcorpus.py:184` |
| 104 | Python decorators; `@misc` bibtex |
| 65 | `OTH_*` constants and `oth_*` settings keys |
| 54 | Stock git hook samples (perl `my`, "Junio C Hamano 2006, 2008", `/usr/bin/perl`, the watchman URL, `update-server-info`, TODO, `XXXXXXX`) |
| 42 | Variable `lab`; "the label" |
| 41 | Variable `todo` |
| 39 | Duration timers |
| 23 | Attribute access matched as a domain (`self.dev`, `probe.net`, `rwb.DEV`, `oa.DEV`) |
| 19 | `#!/usr/bin/env` shebangs |
| 17 | README shell loop variable `$I` |
| 9 | Citation URLs |
| 8 | The count 2000 |
| 8 | Placeholders and the `Anonymous` LICENSE/bibtex |
| 5 | Citation year 2023 |
| 4 | `viz.py` margin variable `my` |
| 3 | Authorial "we"/"our" in README prose |
| 3 | Publication year 2026 in LICENSE and bibtex (year only) |
| 2 | Vendored minGPT comments |
| 1 | Demo key binding `I/J/K/L` |
| 1 | README macOS checksum hint |

Leak-attributed hits: L1 = 7, L2 = 127, L3 = 25.

**Raw-byte pass (523 lines):**
- The only real matches are the 12 `.ruff_cache` files (L2).
- Everything in STAGING is noise:
  - mixed-case 4-letter runs inside float or compressed bytes (`pdt`/`pst` variants, `nTFy`, `wsL-`, `l-Dw`, `>PDT>`, `=UTC=`);
  - `@xx.yy` fragments;
  - two digit runs inside a torch `serialization_id` (`…0302032061610048…`, `…5643194101172606…`).
- No ISO date, clock time, URL, `git@` or email appears in any binary.

**URLs, complete list:**
- `https://github.com/likenneth/othello_world` (vendored `mingpt_model.py:1`, `othello.py:3`): benign citation.
- `https://github.com/karpathy/minGPT` (`mingpt_model.py:3`): benign citation.
- `https://facebook.github.io/watchman/` (stock hook sample): benign, not shipped.
- `git@github.com:<author-account>/generative-models-as-simulators.git` (`.git/config:7`): **leak L1**.

**GitHub/HF usernames:** `likenneth` and `karpathy` (third party, benign); `<author-account>` (L1). No HF id appears; only the `<ANON_HF_REPO>` placeholder.

## Gitignored-but-present local files (from `git ls-files --others --ignored --exclude-standard --directory`)

| path | leaks if the tree is zipped |
|---|---|
| `.ruff_cache/` (12 cache files + `CACHEDIR.TAG`, `.gitignore`) | yes: absolute home paths (L2) |
| `.scratch/` (empty) | no content; move it out anyway |
| `runs/othello`, `runs/rayworld`, `runs/_baselines`, `datasets/othello`, `datasets/rayworld` (symlinks) | yes: absolute home-path targets (L3); following them would also pull in the 15.8 GB bundle |

There are no `__pycache__/`, `.ipynb_checkpoints/`, `outputs/`, `MANIFEST.json`, `SHA256SUMS` or `.cache/` in RELEASE.

## Nit (optional)

`pim/environments/rayworld/bench.py:187` falls back to `str(_sp)`, an absolute path, when the selection file is outside REPO. That branch cannot be reached today, because `layout.edits_selection` always builds the path under `REPO`. It could be reduced to `str(_sp.relative_to(_REPO))`; this changes no shipped value.
