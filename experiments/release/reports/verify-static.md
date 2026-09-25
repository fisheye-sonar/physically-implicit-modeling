# verify-static: static quality of the release tree

Verifier: static. I did not modify the release tree or the artifacts. Everything I made is in
`experiments/release/work/static/`: a copy of the tree, the `tools/` scripts (`reach.py` builds the
reachability graph from the entry points, `dead.py` does a flat reference scan, `prose.py` extracts
comments, docstrings and markdown), their outputs (`reach.json`, `reach_paths.json`, `dead.json`,
`prose.json`), `help.txt` (the `--help` of every script) and `majority_check.py`.

## Verdict

**Issues: 1 blocker, 3 major, 10 minor, and some nits.**

- The blocker is in the release repository's git setup, not in any shipped file. If the release is
  committed and pushed as it is set up now, the authors are identified.
- Most major and minor items are dead code or out-of-scope code that the builders' reports say
  they removed, or say is still used.
- A paper/code mismatch in the Probe Skill definition is not a release bug in itself. It turned
  up because the per-cell function the paper describes is dead code.

## Checks that pass

| check | evidence |
|---|---|
| Ruff with the repo config (E, W, F; ignore E501) | `ruff check --no-cache --config pyproject.toml .` gives "All checks passed!". `--show-files` lists 101 files: all 97 `.py`, the 3 notebooks and `pyproject.toml`. |
| compileall | `python -m compileall -q -f` exits 0. Compiling every file with `-W error` gives 0 failures (no SyntaxWarnings). |
| Every `pim` module imports | `pkgutil.walk_packages`: 72 modules, 0 failures. `pim.__file__` is under the copy. |
| Every script's `--help` | 24 of 24 exit 0. See the argparse item below for `editability_by_point.py`. |
| README flags exist | Every `--flag` in a README `python scripts/...` line appears in that script's `--help`: 0 missing. |
| Out-of-scope identifiers | No hits for nanda, nullspace, oracle, freeze_interp, transformer_s, TransformerS, recurrent, mine_signed, oth_reg, mse_onehot, pn04, rw-pair, L-rw, grid-4x2/8x4/32x16, appearance-d/-lat, index_ceiling, probe_capacity, probe_targets, intervention_benchmark, legacy, discworld, dw-/dw_, L-oth/L-dw, `PIM_*` flags, or any `os.environ` read other than thread-count defaults. The "Nanda et al. (2023)" in the README is a citation. `lambert` and `omni2d` appear only as kept `SimConfig` fields, which spec invariant 1 requires, and `check_supported` rejects them. |
| No dates, banners, TODOs or private references | No `20XX-XX-XX`, ⛔, ⚠, TODO, FIXME or XXX anywhere. No research/, harness, findings, GOTCHAS, REGISTRY, PROGRESS, experiments/, paper_ci, queue, host or GPU-box references. No personal name, institution, `/home/` or `physically-implicit` in any shipped text. |
| Artifacts carry no identity strings | `grep -rlaE "/home/\|[Ss]evan\|brodj\|physically-implicit\|discworld"` over all 15 GB of the artifacts exits 1 (no match). Only uncompressed bytes are searched. `assets/teaser.png` has no tEXt, iTXt or eXIf chunks. |
| Notebook structure | All 3 notebooks: 0 outputs, no execution counts, no cell metadata, kernelspec `python3`. Every code cell is one call into `pim.*` or a list of run ids; there is no metric math. |
| Paper tables covered | All 13 `\label{tab:...}` in the paper are produced by name in `pim/figures/tables.py` and the notebooks. |
| Every module reachable | Reachability from every script and notebook reaches all 97 `.py` files, and every script-level function. |
| Unmatched dependencies | Third-party imports are h5py, matplotlib, numpy, pandas, torch and tqdm, all listed in `pyproject.toml`. |

## Findings

### BLOCKER

**B1. Committing or pushing from this repository de-anonymizes it.** (`RELEASE/.git/config`; not a tracked file)
- **Evidence:**
  - `cat .git/config` shows `[remote "origin"] url = git@github.com:<the author's personal GitHub account>/generative-models-as-simulators.git`.
  - `git config --show-origin --get user.name` resolves to `file:/home/sevan/.gitconfig` with the author's real name, and the global email is personal. The repo is at "No commits yet", so the first commit would record that identity.
- **Fix, before the first commit:**
  - `git -C RELEASE config user.name "Anonymous"` and `git -C RELEASE config user.email "anonymous@example.invalid"`.
  - Point `origin` at an anonymous host: an anonymous account or org, or push only a squashed export to anonymous.4open.science.
  - Check with `git log --format='%an %ae %cn %ce'` before any push.

### MAJOR

**M1. The paper's classification Probe Skill does not match the code that produced the numbers, and the pim function that follows the paper is dead.**
- **Where:**
  - Paper: `paper/paper_draft.tex:226` says "the most common class of each cell as the trivial predictor".
  - Reported numbers: `pim/probes/base.py:192-194` and `pim/probes/baselines.py:198`.
  - Dead per-cell functions: `pim/metrics/decodability.py:44` (`trivial_error_rate`) and `:63` (`probe_skill_classification`).
- **Evidence:**
  - `majority_check.py` runs in the copy on each instance's 20,000-game probe split and the scorer's held-out split.
  - The code's denominator is the majority class pooled over all cells, `1 - bincount(y_tr.reshape(-1)).max()/y_tr.size`. It reproduces the stored `majority_class_error_rate` exactly: 53.119% on `standard`, 53.125% on the other three variants.
  - The per-cell majority error the paper describes is 46.104% on `standard` (adjacent-flip 48.063, adjacent-noflip 47.467, standard-noflip 46.358).
  - The MLP skill `standard` reports is 0.9756; the paper's definition gives 0.9718. At a weak point the gap is large: point 0 linear, error 34.96%, gives 0.342 as reported and 0.242 under the paper's definition. So the observation and random-init floors in Table 1 are most affected.
  - `pim.metrics.probe_skill_classification` is exported but nothing calls it. It computes the per-cell version, so a reader who uses it gets different numbers from the table.
- **Fix (numbers must not change, per invariant 1):**
  - Change the paper sentence to "with the single most common class over all cells (from the training split) as the trivial predictor".
  - Delete `probe_skill_classification` and `trivial_error_rate` and their `pim/metrics/__init__.py` exports, or make them pool over cells. If you keep them, have `base.fit_probe` and `baselines.fit_probe_stream` call them.
  - Reword the `probe_skill_from_stats` docstring to "majority class pooled over every cell, from the train split".

**M2. `pim/figures/waterfall.py::waterfall_grid` (line 30) is dead code, and the tables report's reason for keeping it is stale.**
- **Evidence:**
  - Reachability from every entry point never reaches it. The only references are its own module and the re-export in `pim/figures/__init__.py:6`.
  - `reports/tables.md:50-52` says it was "Kept: `waterfall_grid` with the arguments the history-rewriting script passes". But `scripts/figures/history_rewrite.py` now draws with `style.waterfall` (line 117) and never imports `waterfall_grid`.
  - The editor-comparison grid is not a paper item. The only live parts of the module are `DARK_BG`, `DIFF_CMAP` and `EDIT_LINE`, which `scripts/figures/style.py:26` imports. `GHOST_C` and `TARGET_C` are live only through `DIFF_CMAP`.
- **Fix:**
  - Move the palette constants (`EDIT_LINE`, `TARGET_C`, `GHOST_C`, `DIFF_CMAP`, with `DARK_BG = viz.BG_HEX`) into `scripts/figures/style.py`.
  - Delete `pim/figures/waterfall.py`.
  - Make `pim/figures/__init__.py` docstring-only (drop the "editor-comparison waterfall" text and the `waterfall_grid` export).
  - Drop the comment at `pim/environments/rayworld/viz.py:18` ("pim.figures.waterfall imports it").
  - Side effect: importing `pim.figures.tables` for the CPU-only table notebooks would no longer load the Rayworld demo module `viz.py`.

**M3. The README download commands contain unresolved placeholders.** (`README.md:48`, `:56-64`)
- **Evidence:** `<ARTIFACTS_URL>` and `<ANON_HF_REPO>` appear literally, so `hf download <ANON_HF_REPO> ...` fails as written.
- **Fix:** Fill in the anonymous artifact URL and repo id before publishing. If the artifacts are not on Hugging Face, replace the commands.

### MINOR

**m1. Unreferenced top-level definitions** (reachability graph plus grep; each one has no caller in pim, scripts, notebooks or README).

| definition | judgement | fix |
|---|---|---|
| `pim/editors/pinv.py:41` `PinvMap`, `:50` `pinv_maps` | Dead. Left over from the multi-space PI API; it returns only `{"zspace": ...}`. | Delete both and their exports in `pim/editors/__init__.py:7,10,17,18`. |
| `pim/metrics/prediction.py:58` `floor_bracket` | Dead. The tables use `floor_estimate`. | Delete it, and drop "(``floor_bracket``)" from the module docstring at line 4. |
| `pim/environments/layout.py:52` `probe_dir`, `:74` `eval_dir`, `:121` `othello_split_file`, `:16` `INSTANCES` | Dead. The error messages at lines 63 and 83 point callers to the dead `othello_split_file`. | Delete them and their `__all__` entries. Change the messages to name `othello_split_dir`. |
| `pim/environments/rayworld/bigcorpus.py:106` `instance_dir` | Dead. | Delete it. |
| `pim/environments/othello/corpus.py:26` `BLOCK` | Dead; only `MAXLEN` is used. | `MAXLEN = od.MAXLEN`. |
| `pim/environments/rayworld/bayes.py:57` `World.n_rays`, `pim/environments/rayworld/tokens.py:54` `FrameVocab.obs_dim` | Dead properties: no attribute use anywhere. | Delete them. |
| `pim/figures/tables.py:237` `Table._repr_png_` | Used dynamically by Jupyter's rich display. | Keep. |

**m2. Parameters that now accept only one value, left over from removed options.**
- `pim/editors/pinv.py:58`: `pinv_step(..., space="zspace")` raises for any other value. It is threaded through `rayworld/arms.py:348,378`, `token_bench.py:173` and the scorer (`pim/scoring/rayworld.py:100,153` passes `space="zspace"`).
- `pim/environments/rayworld/token_bench.py:78`: `frame_probs(kind="logits")` together with `TransformerL.output_kind` (`pim/models/transformer_l.py:40`). This is what is left of the removed "raw" output kind.
- `pim/environments/othello/arms.py:243,252`: `linear_arm(mode="pinv")`.
- `pim/editors/grad_steer.py:69,93`: `optimizer="adam"`.
- `TransformerL._seq_mask` (line 75) returns None, and `_run(attn_mask=...)` (line 86) ignores it. This was plumbing for the removed architectures.
- **Fix:** Drop these parameters and the attribute; `frame_probs` becomes a softmax. Keep the literal `"PI[zspace]"` editor label, which is part of the `scores.json` schema.
- **Keep, because they are in the probe-cache key:** `fit_probe_grid(targets=..., splits=...)` and `_split(how)`.
- **Signature note:** `pinv_step` is public API. Dropping `space` is an in-scope signature change under invariant 4. It is numerically a no-op.

**m3. The add-IM-only scoring path is catch-up code.** (`pim/scoring/driver.py:56` `missing_inverse`, `:69` `add_inverse`)
- A full score already computes IM: `score_othello` at `pim/scoring/othello.py:86` and `score_rayworld` through `inverse_rayworld`. This path only backfills IM onto a `scores.json` that was written without it.
- Its Othello branch (lines 71-91) re-implements the IM record assembly of `score_othello:86-92` with different handling: it filters each record with `np.isscalar` and picks `best` with `top_arm`, where `score_othello` uses an unfiltered list and `max`. So the two write paths can produce differently shaped records.
- **Fix:** Delete `missing_inverse`, `add_inverse` and the `missing_im` branch of `_complete`. Keep `missing_blocks`, which the documented `fit_probes` then `master_eval` flow needs. Or, if the path stays, make it call the same assembly helper as the full scorer.

**m4. Scorer settings are copied as literals into six scripts.**
- `PROBE_GAMES, GS_STEPS, GS_BETA = 20_000, 100, 0.2` in `scripts/reachability_table.py:35`, `scripts/two_flip_editability.py:36` and `scripts/figures/qualitative_othello.py:38`.
- `PROBE_SEQS = 30_000` in `scripts/figures/qualitative_rayworld.py:50` and `scripts/figures/history_rewrite.py:37`.
- `RW_PROBE_SEQS, RW_BENCH_N, OTH_PROBE_GAMES` in `scripts/probe_refit_variance.py:28`.
- `scripts/train.py:32` `DEFAULT_INSTANCE` duplicates `layout.DEFAULT_INSTANCE`.
- **Consequence:** If someone edits `SETTINGS` in `master_eval.ipynb`, the figures miss the probe cache. `reachability_table.py` catches the drift; the figure scripts refit silently.
- **Fix:** Define the defaults once, for example `pim/scoring/settings.py: DEFAULTS = {...}`. Import it from the notebook (`SETTINGS = DEFAULTS | {...}`) and from the scripts. In `train.py`, use `layout.DEFAULT_INSTANCE`.

**m5. Paths are spelled outside `layout`, although `layout.py` line 1 says no other module spells one.**
- `runs/_baselines/...` is built by hand in `scripts/fit_probes.py:73`, `scripts/reachability_table.py:48`, `scripts/othello_flip_rates.py:37` and `scripts/bayes_floor.py:58`.
- `pim/environments/rayworld/tokens.py:86` has its own `tokens_dir(instance_dir)`, a namesake of `layout.tokens_dir(inst)` with a different signature. Lines 108 and 110 spell `train/corpus.json` and `train/obs.f32`.
- **Fix:** Use `layout.baselines_dir(env, inst)`, `layout.tokens_dir(inst)` and `layout.train_dir("rayworld", inst)`, and delete `tokens.tokens_dir`.

**m6. A second CLI and a leftover constant name in `pim/environments/othello/corpus.py`.**
- A second, undocumented CLI: the `if __name__ == "__main__":` block (line 205) and "Usage: python -m ..." (line 11) duplicate `scripts/make_othello_corpus.py`.
- `LADDER = {"D": 20_000_000}` (line 70) is left over from the data-size ladder. The othello-env report kept it only because callers index `LADDER["D"]`.
- **Fix:** Delete the main block and the Usage line. Rename to `N_TRAIN = 20_000_000` at all 8 call sites: corpus.py 140 and 206, `scoring/othello.py` 24 and 75, `bayes.py` 54 and 57, `scripts/train.py:97`, `scripts/make_othello_corpus.py:26`, `scripts/figures/qualitative_othello.py:56`. This does not change numerics.

**m7. Docstrings are longer than the spec allows** (spec: module docstrings 1–4 lines, function docstrings 1–3 lines).
- Module docstrings over 8 lines:

  | module | lines |
  |---|---|
  | `pim/environments/othello/reachability.py` | 24 |
  | `scripts/figures/qualitative_overview.py` | 14 |
  | `scripts/figures/qualitative_rayworld.py` | 13 |
  | `scripts/figures/history_rewrite.py` | 12 |
  | `scripts/figures/qualitative_othello.py` | 11 |
  | `pim/environments/othello/corpus.py` | 11 |
  | `scripts/figures/predictions.py` | 10 |
  | `pim/environments/othello/__init__.py` | 10 |
  | `pim/scoring/baselines.py` | 9 |

- 38 modules exceed 4 lines.
- 19 function docstrings exceed 3 lines; the worst are `othello/arms.py:332` `inverse_arms` (11), `scoring/rayworld.py:18` (7), `othello/arms.py:104,243,289` (6).
- **Fix:** In the figure scripts, keep the one-line purpose plus the usage line in the docstring (it is also the `--help` text) and move the figure description into the paper caption. Trim `reachability.py` to its first paragraph plus the verdict definitions.

**m8. Inconsistent CLI flags across scripts** (from `help.txt`).

| issue | where | suggested change |
|---|---|---|
| `--instance` means different things | `bayes_floor.py` takes `ENV/INSTANCE` (nargs `+`); `othello_flip_rates.py` takes bare names (nargs `+`); every other script takes one bare name | Rename the multi-valued, env-qualified one `--instances` |
| Worker-count flag differs | `--n-workers` in `generate_dataset.py`; `--workers` in `build_rayworld_corpus.py`, `reachability_table.py` and `two_flip_editability.py` | Use `--workers` everywhere |
| Observation-noise flag differs | `--obs-noise` in `demos/play.py`; `--obs-noise-std` in `demos/demo.py` and `generate_dataset.py` | Use `--obs-noise-std` |
| `--seeds` means two things | Seed values in the figure scripts; seed counts per target in `probe_refit_variance.py` ("default 10 6") | `--n-seeds` for the counts |
| Same meaning, two names | `--force` vs `--recompute` | Use one name |
| Instance validation differs | Rayworld `--instance` has `choices` in `build_rayworld_corpus.py` only; `generate_dataset.py`, `make_edit_selection.py` and `make_rayworld_tokens.py` accept any string | Add `choices=sorted(bigcorpus.INSTANCES)` |

**m9. `scripts/figures/editability_by_point.py` has no argparse, so `--help` runs the whole figure.** Evidence: `--help` wrote `outputs/figures/appendix/editability_over_res_point.{pdf,png}` in my copy. Fix: add `argparse.ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter).parse_args()` in `main()`.

**m10. British spellings in shipped prose and artifacts.**
- Prose: "labelled" (`pim/environments/rayworld/grid_target.py:1`, `:30`).
- A string written into shipped files: "one occupied non-centre tile recoloured" (`scripts/make_othello_edits.py:51`). It sits in all four staged `datasets/othello/*/edits/cases_1000.json`. `pim/environments/othello/bench.py:138` already says "non-center", and no code reads the `recipe` field.
- Identifiers: `CENTRE` (`othello/data.py:19`, exported), `synthesise_cases` (`othello/bench.py:89`), `FactorisedTarget` (`grid_target.py:298`), local `centres` (`grid_target.py:276`).
- **Fix:** Use "labeled" and "non-center tile recolored" in the script. Have the export worker patch the four manifests' `recipe` string. Rename the identifiers to `CENTER`, `synthesize_cases` and `FactorizedTarget` if public-API changes are acceptable; otherwise leave them and note them.

### NIT
- `pim/environments/rayworld/bigcorpus.py:86` comment: "seed ranges of data outside this release" points at unreleased datasets. Reword to "seed ranges no split may use" (keep the ranges).
- Lowercase "rayworld"/"othello" in prose comments: `pim/metrics/__init__.py:51,61` and `pim/environments/layout.py:22`.
- Title-case figure labels vs the paper's `texttt{adjacent-noflip}`: "Adjacent NoFlip" / "Standard NoFlip" in `scripts/figures/qualitative_othello.py:33-34`, `qualitative_overview.py:31`. They are acceptable if they match the published figures.
- The paper never says "Transformer-L" (it says "the minGPT model"), and the suffix implies a removed S variant. The spec endorses the name, but prose such as `scripts/train.py:2` and README:232 could say "the transformer (minGPT)"; keep the `transformer_l` arch key.
- "canonical" is private jargon in shipped strings: `make_replicate_member.py:49` writes "the canonical run's own checkpoint" into every `__seed0/config.json`, and `probe_refit_variance.py` has `--im-points canonical`. The README says "main run".
- Box-drawing section banners (`# ── name ───`) in 7 files, 31 lines. They are not the banned ⛔/⚠, but they are decorative.
- `pim/probes/baselines.py:229` `fit_baseline_probe = fit_probe_stream`: two names for one function, both used. Keep one.
- Unused parameters: `rayworld/arms.py:278` `score(model, ...)`, `pim/figures/tables.py:399` `_decodability_values(F, rows)`.
- `pim/scoring/baselines.py:165` `oc.probe_data(...)` writes a label cache beside the corpus in `datasets/othello/<inst>/probe/`. With read-only artifacts that write fails, and it is a derived file inside `datasets/`. Cache it under `outputs/` instead.
- The staged Othello splits carry different key sets: `standard` has no `flip`/`placement`/`instance`, and `standard-noflip` has no `placement`. The fallback at `othello/corpus.py:116` is needed and correct as written, but regenerating `standard` produces a file with more keys than the shipped one.
- The release tree's root holds an empty `.scratch/` and a `.ruff_cache/`. Both are gitignored; exclude them if the release ships as an archive rather than through git.
- Extended ruff (not the repo config): the B023 closures at `othello/arms.py:273,412`, `tables.py:507`, `rayworld/bayes.py:285` and `scripts/bayes_floor.py:47` are all called inside their loop iteration, so they are harmless.

## Builder claims I checked and found overstated
- `reports/tables.md:50-52`: "`waterfall_grid` kept with the arguments the history-rewriting script passes". False now; see M2.
- `reports/core.md:32`: `probe_skill_classification` / `trivial_error_rate` kept "code-identical" as in-scope metrics. They are unused, and they compute a different number from the one reported; see M1.
- `reports/core.md:42` and `reports/othello-env.md:69`: `space`/`optimizer` kept "whether or not core drops those parameters". Nobody dropped them; see m2.
