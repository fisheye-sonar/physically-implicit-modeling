# Anonymous-release audit: identity strings (A) and the discworld → rayworld rename (B)

2026-09-23 · read-only audit · nothing in the repo, runs/ or datasets/ was modified.
Helper outputs in the scratchpad: `scan_full.tsv` (every code hit: file, line, category, snippet),
`scan_summary.txt`, `rename_counts.tsv` (per-file rename-token counts), `artifact_scan.json`,
`pickle_scan.txt`; scanners `scan_ids.py`, `scan_artifacts.py`, `scan_pickles.py`, `literals.py`.

## Headline answers

- **Do best_model.pt pickles embed paths or usernames?** Yes, but only 2 of 45:
  `initial_othello_comparison/L-oth-20m` and `L-oth-20m__seed0_s512000` both have
  `train_config.run_dir = "/home/sevan/research/PIM/physically-implicit-modeling/runs/scaling"`.
  The other 43 contain no strings at all beyond `arch` and `lr_schedule`. The pickle globals are
  torch/collections only, with no `pim.*` classes. So renaming the package does not break loading,
  and rewriting `run_dir` leaves the model fingerprint unchanged, because the fingerprint hashes
  parameter bytes only.
- **Probe caches?** None of the 1402 run probe files contain a path or username. The stored
  provenance is a logical key such as `data: "discworld/dw-8ray"`. There are two exceptions, and
  neither is in a place the current code reads:
  - **12 legacy files in `runs/_baselines/dw-8ray/probes/`** have
    `data: "/home/sevan/research/physically-implicit-modeling/datasets/discworld/dw-8ray/probe"`
    with `split: "test"`. These are the old path-keyed form. The current code never produces this
    key, so these files are never hit. `INDEX.md` does not list them (it is stale). Drop them.
    Files: `probes_{fd623f4dd5730b3e, f0683d9fc630dda8, 686f803d380c1a75, f4697975c6cd2a9f,
    fe853930cc201b32, 74a296358b4f4bb9, 2569885c9b4ad2f0, a38153a2be6a26b6, ce3ee535152915ae,
    1285c4719ea3ee03, 768b62f307b4160e, 041f6687f824d058}.pt`.
  - **`runs/noise_ablation/L-dw-noiseless-20m/probes/_superseded/`** holds one file keyed by the
    relative path `datasets/discworld/dw-noiseless/probe`. Exclude it.
- **HDF5 attrs?** No. Each eval/edits/probe `.h5` has a single string attribute, `config_json`,
  holding simulator parameters, `base_seed`, `n_workers`, the schema, and a `generated_at`
  timestamp. It contains no path, username or environment name. `train/meta.h5` has no attributes.
  The Othello `.npz` files store only `instance`/`placement` strings (`oth-*`). They are clean.
- **Does a cache key or fingerprint contain an absolute path, so that scrubbing would break
  cache hits?** No live key does. The filename is `blake2b(repr(sorted(provenance.items())))`.
  I confirmed this on 40 sampled files (40/40 match). The model fingerprint is a hash of parameter
  bytes only. The absolute paths that do exist sit in fields no code reads back:
  - `scores.json → probe_dir`
  - `config.json → data.corpus`, `data.vocab`, `train.run_dir`
  - `selection.json → vocab`

  Scrubbing them breaks no lookup.

  **The catch is the rename, not the scrub.** Every rayworld probe key contains the string
  `discworld/dw-<inst>`: 1068 run files plus 276 live baseline files. Renaming therefore changes
  every hash, so the files must be re-keyed (see B).

---

## Task A: identity strings

### A1. Code (pim/, scripts/, tests/, the three notebooks, paper/figs qualitative code, pyproject.toml, poetry.lock)

**Counts per category**, true positives only:

| category | code | where |
|---|---|---|
| **Name "Sevan"** (incl. "Sevan's pick/call/spec/ask", "(Sevan)") | **84 lines in 45 files** | pim 31 lines / 19 files · paper/figs 31 / 12 (3 of them JSON, 3 README) · scripts/drivers 15 / 12 · notebooks source 7 / 2 · scripts/ outside drivers 0 · tests 0 |
| brodjian / caltech / hobley / perona / pasadena / california / email / `owner@` / IP / tailscale / wsl | **0** | the 54 "IPs" and the `github`/`ssh` hits in poetry.lock are version numbers and extras names |
| **Geography: "PT" timezone** | 10 | 9 in drivers, plus `paper/figs/paper_style.py:7` ("until 11:50 PT") |
| **Machine / host** | 4 | `pim/environments/discworld/arms.py:226` "tmpfs (RAM) on the lab box" · `pim/environments/othello/corpus.py:110` "OOM a 59 GB box" · `scripts/drivers/score_pending.sh:29` "on the lab box … on the remote" · `scripts/drivers/dw_128ray.sh:19` "(~8 h on the 5090)" |
| **Secret: personal notification URL** `https://ntfy.sh/swirl…` (masked) | **27 files** | every `scripts/drivers/*.sh` that pings. Treat it as a credential and **exclude `scripts/drivers/` entirely** |
| Absolute paths in source | 0 | none, including stored notebook outputs |
| **Absolute paths in `__pycache__/*.pyc`** | **140 files** | `co_filename = /home/sevan/research/PIM/…`. **Never copy `__pycache__`** |
| Dates in comments/docstrings | 569 (+19 clock times) | pim 314 · tests 60 · notebooks source 42 / outputs 56 · paper/figs 36 · drivers 37 · scripts 23 · pyproject 1 |
| Load-bearing date-shaped version tokens | 7 constants | `EVAL_VERSION "2026-09-01.4"`, `EVAL_VERSION_BY_ENV {"discworld": "2026-09-12.2", "othello": "2026-09-12.1"}` (master_eval cell 2), `BASELINE_VERSION "2026-09-06.b4"` (scoring/baselines.py:74), `IM_VERSION "2026-09-15.1"` (scoring/blocks.py:93), `PRED_VERSION` / both `FLOOR_VERSION "2026-09-19.1"`. They are compared against the same strings stored in scores.json, baselines.json and bayes_floor.json. **Rewrite them in code and artifacts together, or leave them.** |
| Notebook metadata | 3 notebooks | every code cell has nbconvert `metadata.execution` timestamps ("2026-09-23T22:07:33Z"); kernelspec `".pim (3.13.5)"` |
| First person / "the user" / "my" | 0 real | one "I am" in vendored minGPT (third-party); every "my" is the variable `mx, my` |
| Third-party people | fine to keep | Kenneth Li (vendored `othello_world` URL + MIT LICENSE), "Li et al.", `nanda.py` (cited method) |
| Repo name | 1 | `pyproject.toml name = "physically-implicit-modeling"`. This de-anonymizes if the GitHub repo of that name is public; `authors = []` is clean |
| PDFs / PNGs in paper/figs | 228 files | PDF `/CreationDate D:20260922`, Creator/Producer Matplotlib; PNG `Software: Matplotlib`. No name or path. `pdftotext` finds no dw/discworld/Sevan text in any figure |
| Stored notebook images | 7 PNGs | table row labels are `r['run']`, so the rendered tables show `L-dw-…` names. Regenerate them after the rename, or strip the outputs |
| Stored master_eval text output | 105 outputs | lists every run and baseline in runs/, including unreleased ones (dw-pn04, dw-pair, oth-uniform baselines, …). Re-execute it on the release tree or strip it |

**Every "Sevan" line, by file and line number** (all are comments/docstrings or JSON prose; none is
load-bearing):
- pim/editors/inverse.py:2
- pim/environments/discworld/arms.py:766
- pim/environments/discworld/bench.py:161
- pim/environments/discworld/bigcorpus.py:89, 96, 107, 116
- pim/environments/discworld/frustum.py:59
- pim/environments/discworld/grid_target.py:578
- pim/environments/discworld/observers.py:1, 7
- pim/environments/discworld/token_bench.py:151
- pim/figures/tables.py:22, 46, 53, 69, 82, 91, 801
- pim/metrics/edit_index.py:65
- pim/metrics/prediction.py:104
- pim/metrics/replicates.py:59
- pim/metrics/selection.py:16
- pim/models/recurrent.py:19, 53
- pim/models/transformer_s.py:402
- pim/probes/inverse.py:2, 49
- pim/scoring/discworld.py:45
- pim/scoring/othello.py:64
- pim/scoring/runs.py:54
- notebooks/master_eval.ipynb cell 2 (lines 14, 53, 58) and cell 3 (lines 1, 15)
- notebooks/build_paper_tables_and_figs.ipynb cell 1 (lines 14, 21)
- paper/figs/paper_style.py:7
- qualitative_edits/make_figure.py:138, 187, 234, 370; README.md:29, 48, 50, 99
- qualitative_edits_othello/make_figure.py:60, 149, 151, 175; README.md:26
- qualitative_main/composite_final.py:1, 37, 39, 50
- qualitative_main/rayworld_panel.py:7, 47, 213, 347
- qualitative_main/othello_panel.py:8
- qualitative_main/README.md:4, 67, 78, 97, 164
- qualitative_main/composite_final.json:774, composite_final_sidebyside.json:781, composite_final_sidebyside_single.json:539 ("(Sevan, round 5)")
- drivers: dw_128ray.sh:13, 33; dw_5ray.sh:9; dw_8ray_obs5.sh:2; dw_smooth_gen.sh:5; oth_adjflip_seeds.sh:4, 17, 29; probe_targets_2.sh:3; probe_targets_4.sh:3; probe_targets_5.sh:7; rescore_2026-09-12.sh:2; score_cartesian.sh:2; score_im.sh:2; seed_variance.sh:2

Typical forms: "(2026-09-20, Sevan)", "Sevan's call, 2026-08-21", "(Sevan: \"only show examples which change …\")".
Also note `pim/scoring/runs.py:54` is an inline comment on live code: `# … (2026-09-15, Sevan:`.

Low-severity items to tidy (not identity): 59 references to `logs/`, 44 to `experiments/`, 20 to
`research/`, 3 to `harness/`, and 2 each to `GOTCHAS` and `findings/`. These are dangling pointers
to files that will not ship.

### A2. Artifacts

Scope: 45 run dirs (13 parents plus `__seed*` siblings), `runs/_baselines/<12 in-scope instances>`,
and `datasets/{discworld: 8, othello: 4}`.

| artifact | finding | category |
|---|---|---|
| `config.json` (45) | **38/45 carry absolute paths**: `data.corpus` (36), `data.vocab` (1: L-dw-8ray-tok-20m), `train.run_dir` (2: the L-oth-20m family). Two roots appear: `/home/sevan/research/PIM/physically-implicit-modeling` (62 occurrences) and `/home/sevan/research/physically-implicit-modeling` (23; the second machine's clone). The 7 clean ones are the Othello adjacent-flip / adjacency / noflip parents and their `seed0_s*` members. Also `run_name "BIG20M_othello_L"` (legacy, harmless). No code reads these fields | abs path / username |
| `scores.json` (45) | **45/45: `probe_dir` = absolute run path**. **24 files / 44 occurrences** of `"… removed 2026-09-20 (Sevan)"` in `bases.<cat-block>.inverse_cleared.why` (written by `experiments/categorical_inverse/scripts/clear_continuous_im.py`). Also `commit_sha`, `blocks_added.*.date`, `inverse_added.*.date`, and a `settings` snapshot that names unreleased runs (`ray_ablation/R-dw-8ray-20m`, `training_curve/L-dw-8ray-20m_s032000`). **The live scorer re-creates the absolute path**: `pim/scoring/discworld.py:98,152` and `othello.py:96` write `str(run_dir/"probes")`. Fix the code to write a repo-relative path | abs path / name / date |
| `commit_sha` (45) | 43 are bare private-repo SHAs. The L-oth-20m family's 2 files contain prose ("trained_with_commit: 785e32b … recoverable_code_tag: pre-cleanup-2026-08 … housekeeping phase 0"). SHAs identify only if the original repo is public | low (drop or neutralise) |
| `metrics.jsonl` (33; missing in the 12 `seed0_s*` members) | clean: keys step, train_loss, val_loss, lr, elapsed_s | none |
| `variance.json` (8) | clean (instance names only) | none |
| `best_model.pt` (45) | top-level keys are `arch, step, model_state, model_config, train_config, val_loss, epoch[, val_loss_step]` (43 runs) or `step, model_state, model_config, train_config, val_loss, rung, vocab, best / arch, val_loss_step` (the L-oth-20m family). **Only the 2 `run_dir` strings above.** No env or instance names inside, so the rename needs no checkpoint rewrite | abs path (2) |
| `ckpt/*.pt`, `latest.pt` | not in the ship list. The 11 L-oth-20m ckpts and `latest.pt` carry the same `run_dir`. Exclude them | abs path |
| `probes/probes_*.pt` (1402) + `INDEX.md` (45) | provenance only; see the headline. Othello keys have no instance, env or path component | clean |
| `runs/_baselines/<inst>/*.json` | clean (no path, no name). `bayes_floor.2026-09-23_15xx.json` and `bayes_floor.smoke.json` have timestamped or scratch names; exclude them | none |
| `runs/_baselines/dw-8ray/probes` | the 12 legacy absolute-path-keyed files (headline) | abs path / username |
| `datasets/*/instance.json` (hand-written; code never reads it) | "(Sevan, 2026-09-1x)" in 5 dw instances; `dw-16ray`: "built on the WSL remote", "unit dw_16ray on wsl-sevan"; `dw-128ray`: "on the lab box"; `oth-adjacent-flip`: `"machine": "sevan-ubuntu-lab (local); the WSL remote's first attempt … Windows WHEA log"`. **Do not ship. Rewrite from scratch if one is wanted** | name / host |
| `datasets/othello/oth-adjacent-flip/layout.json` | `"migrated": "2026-09-11 05:05 (pulled from wsl-sevan in v1 layout …)"` | host / name |
| `datasets/discworld/dw-8ray/tokens/meta.json` | `"files_revised": "2026-09-19 by hand (Sevan): …"` | name |
| `datasets/discworld/dw-8ray/edits/v1/selection.json` | `"vocab": "/home/sevan/…/tokens/vocab.npz"` (code reads only `select`) | abs path |
| `probe/*.json`, `eval/test.json`, `edits/v1/edits.json`, `train/corpus.json`, `layout.json` (the others) | clean (dates plus instance labels) | none |
| HDF5 attrs, Othello `.npz`, `cases_1000.pkl` | clean (see headline) | none |

**Summary, code vs artifacts.** Names: 84 lines in code; 44 occurrences in 24 scores.json;
8 dataset files (5 instance.json on the dw side, oth-adjacent-flip instance.json and layout.json,
dw-8ray tokens/meta.json). Absolute paths: 0 in source (140 in `.pyc`); in artifacts, 38 config.json,
45 scores.json, 2 best_model.pt, 12 legacy probe files, 1 selection.json (plus ckpts). Hosts:
4 lines in code; 4 dataset files (dw-128ray and dw-16ray instance.json, oth-adjacent-flip
instance.json and layout.json). Secret: 27 driver scripts.

**Exclude outright:** `__pycache__/`, `scripts/drivers/`, `ckpt/`, `latest.pt`, `scores_backup/`,
`scores.pre-*.json`, `scores.s390000.json`, `variance.s390000.json`, `figures/`, `probes/_superseded/`,
the 12 legacy baseline probe files, `instance.json`, `bayes_floor.<timestamp>.json` /
`bayes_floor.smoke.json`, `.git`, and the extra JSONs in L-oth-20m (`editability_by_reachability.json`,
`two_flip_editability.json`, `index_ceiling.json`).

---

## Task B: the discworld → rayworld rename

### B1. Surface (full per-file table in `rename_counts.tsv`)

Note: `dw-` counts include the `dw-` inside `L-dw-`.

| group | files hit | discworld | Discworld | DISCWORLD | dw- | dw_ | L-dw | other dw idents (dwa, dwb, DW_*, runs_dw, …) | total |
|---|---|---|---|---|---|---|---|---|---|
| pim/environments/discworld/ | 16 | 76 | 1 | 0 | 257 | 0 | 1 | 9 | 344 |
| pim/scoring/ | 7 | 68 | 3 | 0 | 10 | 70 | 0 | 32 | 183 |
| pim/ (other) | 37 | 146 | 5 | 1 | 12 | 1 | 0 | 17 | 182 |
| scripts/ (non-driver) | 13 | 92 | 2 | 0 | 40 | 2 | 11 | 19 | 166 |
| scripts/drivers/ | 19 | 41 | 1 | 0 | 124 | 45 | 39 | 10 | 260 |
| notebooks (source) | 3 | 26 | 2 | 1 | 40 | 17 | 30 | 3 | 119 |
| notebooks (stored outputs) | 2 | 64 | 0 | 0 | 187 | 0 | 104 | 0 | 355 |
| paper/figs qualitative code/JSON/README | 16 | 14 | 0 | 0 | 131 | 0 | 23 | 22 | 190 |
| tests/ | 24 | 163 | 1 | 0 | 122 | 13 | 20 | 56 | 375 |

The heaviest files are:
- `bigcorpus.py` (237: the INSTANCES registry and seed-range labels)
- `tests/test_layout.py` (82)
- `tests/test_scoring_package.py` (66)
- `pim/scoring/discworld.py` (60), `pim/scoring/baselines.py` (54)
- `tables.py` (47), `make_figure.py` (45)

Filenames that must be renamed:
- `pim/environments/discworld/` (24 modules)
- `pim/scoring/discworld.py`
- `scripts/make_discworld_tokens.py`
- `tests/test_discworld_tokens.py`
- `scripts/drivers/dw_*.sh` (9)
- 72 piece files `paper/figs/qualitative_main/pieces/composite_final/col*_dw-*_*.{pdf,png}`

Prose "Discworld" occurs 15 times, all in comments. Figure code already partly uses
Rayworld/`RW_RUNS`/`rw_appendix`. No `rw-`/`rw_` token collides with anything existing.

### B2. Where the name is load-bearing

| # | location | what reads it | artifact rewrite needed to stay consistent? |
|---|---|---|---|
| 1 | **Package path** `pim.environments.discworld`: 94 import lines in 31 files (+ tests) | Python imports | **No.** No pickle references it (best_model.pt: torch/collections only; probes: `pim.probes.base.WorldStateProbe` / `pim.probes.nullspace.NullspaceCascade` only) |
| 2 | **Env string** `"discworld"` compared at runtime. Also the env-dispatch comparisons: `tables.py:183, 248, 298–307, 377, 460, 649, 707, 926`; `summary.py:21`; `baselines.py:159, 218, 293, 333`; `training/sources.py:62` (meta env); `train.py:62` (`--env` choices), `:56`, `:199`, `:208` | `scan_runs` takes `r["env"]` from **config.json `data.env`** | **Yes:** config.json `data.env` (28), scores.json `env` (28), baselines.json `env` (8). See the table below this one for the `score_all` "already scored" gates |
| 3 | **Class dir / layout** `CLASSES = ("discworld","othello")`, `layout.<fn>("discworld", inst)` (~30 call sites in arms, bench, bayes, tokens, prediction, tables, baselines) → `datasets/<cls>/<inst>/` | every dataset read | **Yes:** move `datasets/discworld` to `datasets/rayworld`; `layout.json "class"` (cosmetic) |
| 4 | **Probe-cache key** `layout.probe_key → data = f"{cls}/{inst}"`, hashed into the filename | `ProbeCache.load` (hit by filename, then an exact provenance-equality check) | **Yes. 1068 run + 276 baseline files must be re-keyed** (B4). Othello keys are unaffected |
| 5 | **Instance names** `dw-*`: config.json `data.instance` (28); scores.json `instance`, `bases.*.probe_recipe.probe.instance`, `bases.*.bench_selection.file`, `prediction.instance`; baselines.json / bayes_floor.json `instance`; variance.json | `layout` paths, probe keys, `dw_bases_for`, `cat_inverse_in_scope`, baselines dir lookup, tables floor join (`base_json[b["instance"]]` vs scores `instance`) | **Yes**, all of these. Also move `runs/_baselines/dw-*` to `rw-*` |
| 6 | **Instance-keyed dicts**: SETTINGS `dw_bases_by_instance {"dw-8ray-obs5": ("cartesian",)}`, `tables.BASIS_BY_INSTANCE` (same), SETTINGS `dw_cat_im.instances`, `layout.DEFAULT_INSTANCE`, `train.py CANONICAL_INSTANCE`, `bigcorpus.INSTANCES` keys and range labels | scorer and tables | code-only, but must match the renamed instance values. **If the obs5 key is missed**, obs5 is owed a `frustum` block, and the scorer fits regression probes inline to add it |
| 7 | **SETTINGS keys `dw_*`** (17 keys, read as `s["dw_…"]` in blocks.py, discworld.py, baselines.py, driver.py) | scorer | code-only. scores.json `settings` is a snapshot the driver never compares, so rewrite it (or drop it) only for anonymity |
| 8 | **Run names `L-dw-*`** (13 parents, 28 dirs with seeds): SETTINGS `dw_extra_targets` keys `"topic/L-dw-…"`; config.json `replicate.of` / `replicate.source` (16 replicates); `tables.find_run` globs `runs/*/<name>/scores.json`, replicates by `<name>__seed*`; notebook run lists; figure scripts (`"ray_ablation/L-dw-…"`); scores.json `run` | `extra_targets_of`, `missing_blocks`, tables, figures | **Yes**, if run dirs are renamed: config.json `replicate.of/source` and scores.json `run` must follow. A mismatch in `dw_extra_targets` silently drops extra-target blocks from the "required" set (no refit, but a later fresh score would lack them). Row labels in the rendered tables are the dir names |
| 9 | HDF5 `config_json`, Othello npz, best_model.pt | nothing name-bearing | **No** |
| 10 | dataset JSON: `corpus.json instance/disjoint_from`, `selection.json instance`, `tokens/meta.json instance`, `probe_*.json holdout` prose | `tokens.tokenize_instance` reads corpus.json (training-time only) | cosmetic, but rewrite for anonymity |

The `score_all` gates behind row 2 (all in `pim/scoring/driver.py` unless noted):
- `eval_version(r) = EVAL_VERSION_BY_ENV.get(r["env"], EVAL_VERSION)` (master_eval cell 2)
- `scorer_for`, `SCORED_ENVS` (`driver.py:43–49`)
- `missing_inverse` (`r["env"] != "discworld"`, `driver.py:77`)
- `missing_blocks` / `add_inverse` (`r["env"] == "othello"`)

### B3. How the driver decides a run is already scored

`scan_runs` (pim/scoring/runs.py) lists every `runs/<topic>/<run>/config.json` that meets all of:
- the topic is not `archive` and does not start with `_`;
- `best_model.pt` exists;
- training is complete: the last `metrics.jsonl` step is at least `train.steps`, and a missing
  metrics.jsonl counts as complete.

For each listed run, `env` and `instance` come from `config.json data.*`.

`score_all` then **skips** a run only if all of these hold:
1. `scores.json` exists and `prev["eval_version"] == EVAL_VERSION_BY_ENV.get(env, EVAL_VERSION)`,
   and the run is not in `PIM_FORCE_RESCORE`.
2. `missing_blocks` is empty. For rayworld runs this means every basis in
   `dw_bases_for(instance)` (default `("frustum","cartesian")`, or the per-instance override) plus
   every extra target of `SETTINGS["dw_extra_targets"][topic/run]` (or of the parent named in
   `replicate.of`) is a key of `prev["bases"]`. For Othello it means `"mine_signed"`.
3. `missing_inverse` is empty. Every regression block and the Othello top level must have an
   `"IM"` arm. Categorical rayworld blocks are exempt unless `PIM_ADD_CAT_IM=1`, and nothing
   `nn_r2`-related triggers unless `PIM_ADD_NN_R2=1`.

Baselines (`score_all_baselines`) skip when `runs/_baselines/<instance>/baselines.json` has
`baseline_version == BASELINE_VERSION`, every arch among the runs, and every extra target and
basis present.

**On the skip path no probe is ever loaded.** Probes are loaded only when a block or IM arm is
added, or on a (re)score. Regression probes are then fitted inline on a cache miss. Categorical
probes use `require_cached`: on a miss the block is silently skipped. The qualitative figure code
(`qualitative_edits/make_figure.py:244–274`, `require_cached=True`) always loads probes, so it
raises on a miss.

### B4. Recommended scheme

- **Class**: `discworld` → `rayworld` everywhere. That covers the package dir, `pim/scoring/rayworld.py`,
  `layout.CLASSES`, `datasets/rayworld/`, env strings in code, the `EVAL_VERSION_BY_ENV` /
  `SCORED_ENVS` / `CANONICAL` keys, config.json `data.env`, scores.json / baselines.json `env`,
  layout.json `class`, and the probe-key prefix.
- **Instances**: `dw-<x>` → `rw-<x>` (rw-noiseless, rw-blink, rw-128ray, rw-16ray, rw-8ray,
  rw-5ray, rw-smooth, rw-8ray-obs5). This parallels `oth-*`, can be applied mechanically, and
  collides with nothing.
- **Runs**: `L-dw-<x>-20m` → `L-rw-<x>-20m` (keep `__seed*` / `_s<steps>` suffixes). **Rename the
  directories.** A display map alone would leave `L-dw` in the dir names, in `replicate.of`, and in
  the rendered row labels.
- **Identifiers**:
  - SETTINGS keys `dw_*` → `rw_*`
  - `dwa`/`dwb` → `rwa`/`rwb`, `DW_PROBE_SIZES` → `RW_PROBE_SIZES`, `runs_dw` → `runs_rw`
  - `score_discworld*` / `inverse_discworld` / `discworld_blocks` / `discworld_source` → `rayworld_*`
- **Artifact migration, in one script, run against a copy of the release tree**:
  1. `config.json`: `data.env`, `data.instance`, `replicate.of` / `replicate.source`. Scrub
     `data.corpus`, `data.vocab`, `train.run_dir` (set them to repo-relative paths or drop them).
  2. `scores.json`: `env`, `instance`, `run`, `bases.*.probe_recipe.probe.instance`,
     `bases.*.bench_selection.file`, `prediction.instance`. Make `probe_dir` relative. Remove the
     `(Sevan)` in `inverse_cleared.why`. Rewrite the `settings` snapshot keys and values (or drop
     it). Leave `eval_version` unchanged.
  3. **Probe re-key.** For every `probes_*.pt` whose `provenance["data"]` starts with `discworld/`:
     - replace it with `rayworld/rw-…` and leave every other field untouched (`model`, `span`,
       `v=2`, and all value types);
     - set `h = blake2b(repr(sorted(prov.items())).encode(), digest_size=8)`;
     - `torch.save({"provenance": prov, "probes": probes}, f"probes_{h}.pt")`;
     - delete the old name in the copy, then regenerate `INDEX.md` with `ProbeCache.write_index()`.

     No model is needed, because the fingerprint is already stored in `prov["model"]`. Apply this
     to `runs/<topic>/L-rw-*/probes` and `runs/_baselines/rw-*/probes`, after dropping the 12
     legacy files.
  4. Rename `runs/_baselines/dw-*` to `rw-*`; set `instance`/`env` in baselines.json and
     `instance` in bayes_floor.json.
  5. Rename `datasets/discworld/dw-*` to `datasets/rayworld/rw-*`; rewrite layout.json,
     selection.json (plus scrub `vocab`), tokens/meta.json, corpus.json and probe/eval/edits
     manifests. HDF5 needs nothing.
  6. Scrub the 2 `best_model.pt` `train_config.run_dir` values. Scrubbing is fingerprint-safe.
     Load with `weights_only=False`, edit the string, and re-save.
- **Verification gate**, on the release tree, before packaging:
  - (a) `score_all(RUNS, SETTINGS, eval_version, dry_run=True)` and
    `score_all_baselines(RUNS, SETTINGS, dry_run=True)` must print only `skip` lines, with no
    `WOULD` and no `stale`.
  - (b) For every probe file, the filename must equal the hash of its provenance.
  - (c) Run `paper/figs/qualitative_edits/make_figure.py` (`require_cached=True`) and the
    Othello panel end to end.
  - (d) Re-execute the paper and appendix notebooks and diff the numeric output against the
    current outputs.
  - (e) `grep -ri 'discworld\|dw-\|dw_\|sevan\|/home/'` over the tree, including `.json`, `.md`,
    and `strings` of every `.pt` / `.pkl`.

### B5. Riskiest spots

1. **`EVAL_VERSION_BY_ENV` keyed by env** (master_eval cell 2). This is the worst silent failure.
   If configs say `rayworld` and the dict still says `discworld` (or the reverse), every rayworld
   run falls back to `EVAL_VERSION "2026-09-01.4"` and is judged **stale**. It is then either
   **rescored from scratch** (about 30–45 min each for 28 runs, overwriting scores.json) or,
   with a partial rename of `SCORED_ENVS`, skipped as "env has no scorer". Either way the
   printed table silently changes.
2. **Probe-cache hashes** embed `discworld/dw-<inst>`. Without the re-key:
   - regression probes are refit inline on any add or rescore, with slightly different numbers;
   - categorical blocks are silently skipped;
   - the qualitative figures crash.

   The re-key must reproduce the new code's provenance dict exactly. Any drift in a value or its
   type changes the repr and therefore the hash.
3. **`dw-8ray-obs5` basis override** (SETTINGS `dw_bases_by_instance` and `tables.BASIS_BY_INSTANCE`).
   If this key is missed, obs5 gets owed a `frustum` block, which is fitted inline. The table
   then shows frustum for obs5 when frustum is requested.
4. **`dw_extra_targets` / `replicate.of` run keys.** A mismatch drops extra targets from the
   required set without error, and tables built on those blocks lose rows only if someone
   rescores later.
5. **Baselines lookup by instance**, both the directory and the `instance` field. A mismatch
   causes a full baseline refit, and the table floors go missing (`base_json[instance]`).
6. **Scorer writes absolute `probe_dir`** (and train.py writes absolute `corpus` / `vocab` / `run_dir`,
   and make_edit_selection.py writes `vocab`). Any re-run in the release tree reintroduces a
   `/home/<reviewer>` path. Change the code to write repo-relative paths.
7. **Date-version tokens.** If the date scrub rewrites `EVAL_VERSION*`, `BASELINE_VERSION`,
   `PRED_VERSION` or `FLOOR_VERSION` in code but not in the artifacts, every run is stale
   (same effect as spot 1). Safest is to leave them as opaque tokens.
7a. **A naive global `sed dw-→rw-` over artifacts** also rewrites the `settings` snapshot's
   unreleased run names. That is harmless. But a find-replace applied to pickles in place would
   corrupt them, so the probe `.pt` files must go through `torch.load` / `torch.save`, never
   byte substitution.
