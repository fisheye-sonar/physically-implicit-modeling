# othello-env: report

Scope: RELEASE `pim/environments/othello/**`, including `vendor/`. The package went from 2,250 to 1,622 lines. Every module imports, ruff is clean, and every check against PRIVATE (a–e below) came out **bitwise identical**.

## Deleted

- **`arms.py`**
  - ND (`mode="add"` and `"add_sub"`) and the `pim.editors.nanda` import.
  - Every regression-probe (`mine_signed`) branch, in `linear_arm`, `grad_steer_arm`, `observation_probes` and `fit_probe_grid`.
  - The `"frame"` split in `_split`.
  - The squared-error-head gate statistics (`out_sum_mean`, `out_neg_mass_mean`) and the `output_kind` field from `gates`.
  - `getattr(model, "output_kind")` plumbing, and the unused `_REPO`.
- **`data.py`**
  - `REGRESSION_TARGETS`, `signed_mine` and the `mine_signed` branch of `flatten_rows`.
  - `OUTPUT_KINDS` and the `raw` / `clipnorm` kinds: `move_probs` and `board_probs` now always use the logits softmax.
  - `synthetic_games`, which nothing called, and the `multiprocessing` import.
- **`bench.py`**
  - `BENCHMARK_PKL`, `load_li_benchmark`, `shipped_length_distribution`.
  - The `intervention_benchmark.pkl` fallback in `cases_path`, and the unused `_REPO`.
- **`counterfactual.py`**: `mine_board` and `search_cf`. Only the dropped `index_ceiling.py` used them. I checked `two_flip_editability`, `reachability_table`, `othello_corpus_stats` (current version) and every qualitative figure script.
- **`corpus.py`**
  - The `M` / `L1` / `L2` data-scale ladder rungs and `rung()`.
  - The `ensure_marker` / `layout.json` stamping in `build` (layout versioning is out of scope).
  - The rest of the "v1 / v2" layout language.
- **`vendor/othello.py`**
  - Module level: `mask`, `permit`, `start_hands`, `wanna_use`, and the unused `copy` / `deepcopy` imports.
  - Methods: `get_occupied`, `get_state`, `get_age`, `get_next_hand_color`, `plot_hm`, `get_gt`.
  - The empty `__main__`.
- **`vendor/mingpt_model.py`**: `GPTforProbing`, `GPTforIntervention`, `GPTforProbeIA`, `GPT.get_block_size`, `GPT.configure_optimizers`. Plus one whitespace-only line (W293).
- **`__init__.py` exports**: `load_li_benchmark` and `synthetic_games`.

## Kept, and why

- **`corpus.py`**
  - The four instances under their new names, with the same rules: standard (flip, enclosure), standard-noflip (no flip, enclosure), adjacent-noflip (no flip, adjacent), adjacent-flip (flip, adjacent).
  - Every index range unchanged: train 0, test 90M+10k, probe 91M+20k, probe_large 92M+170k, edits 93M+10k.
  - `probe_large` stays because the large observation floor reads it.
  - `LADDER = {"D": 20_000_000}` stays because consumers call `oc.LADDER["D"]`.
  - `verify_splits` keeps its fallback for files without a `flip` / `placement` field. The shipped `standard` npz files and three `standard-noflip` files lack them.
- **`vendor/othello.py`**
  - `OthelloBoardState` logic is untouched: `update`, `umpire`, `_captures`, `_legal`, `tentative_move`, `get_valid_moves`.
  - `__print__` stays because `update(prt=True)` calls it.
  - `get_ood_game` and its use of the global `random` state are unchanged.
  - `rows`, `columns`, `permit_reverse` and `eights` stay. `permit_reverse` is imported by the environments-overview figure, which `predictive_quality/othello.py` loads.
- **`vendor/mingpt_model.py`**: `GPTConfig`, `CausalSelfAttention`, `Block`, and `GPT` with `_init_weights` and `forward`. The retained lines are byte-identical to upstream `mingpt/model.py`, checked against a local clone.
- **`arms.py`**
  - The probe-cache keys are exactly as before: `othello_grid`, `othello_observation` (which carries `align` only when it is not "left"), and `inverse_map`.
  - `DEV`, `BLOCK`, `ProbeGrid`, `legal_sets`, and `_split` (still with its `how` argument).
- **`reachability.py`, `bayes.py`**: logic unchanged.

## Docstrings, headers, versions

- Every docstring and comment is rewritten to the SPEC writing rules: no dates, no history, no names, no research-file references, American spelling. The "59 GB box" line is gone. A scan for names, dates, banners, `dw-`/`oth-`/`L-oth`, paths and history words is clean.
- **`vendor/othello.py` header** now says the file is modified from Li et al.'s `data/othello.py` (MIT, Copyright (c) 2023 Kenneth Li). It lists the changes: the `flip` / `placement` rules with the `_captures` / `_legal` split, and the removed loader, plotting and helpers.
- **`vendor/mingpt_model.py` header** credits Li et al. and adds one line saying the file derives from Andrej Karpathy's minGPT (github.com/karpathy/minGPT, MIT).
- **`vendor/LICENSE`** (Li's MIT) is unchanged.
- **Versions**: `reachability.VERSION` changed from `"2026-09-23.1"` to `"1.0"`, and `bayes.FLOOR_VERSION` from `"2026-09-19.1"` to `"1.0"`. Both match `version_map.json`.

## API changes (in-scope symbols)

- `linear_arm(..., mode="pinv", ...)`: `mode` now defaults to `"pinv"`. Any other value raises `ValueError`, and so do regression probes.
- `grad_steer_arm`: the `optimizer` parameter is gone. No caller passed it, and the editor's default is Adam. Regression probes now raise `ValueError`.
- `data.move_probs(outputs)` and `data.board_probs(outputs)`: the `kind` parameter is gone. Only the package's own code passed it.
- `data.flatten_rows` and `arms.observation_probes` now raise `ValueError` for any target other than `"state"` or `"mine"`. Before, `"mine_signed"` would silently have fitted the wrong labels.
- `_split` raises for anything other than `"sequence"`.
- `gates()` returns `legal_mass, top1_legal, top1_acc, ce, bayes_ce, bayes_top1, n_positions, n_games`. `output_kind`, `out_sum_mean` and `out_neg_mass_mean` are gone. `tables.py` reads only `legal_mass`, and `summary.py` prints only kept fields.
- Removed symbols are listed under Deleted. The only one an in-scope consumer still uses is `bench.shipped_length_distribution` (see request 1).
- I extracted all 46 symbols that the in-scope consumers take from the Othello package. All of them exist in RELEASE except that one.
- `pinv_step` is now called without `space=`, and `make_intervention_hook` without `optimizer=`. Both rely on the defaults (`zspace`, `adam`), so they keep working whether or not core drops those parameters.

## Requests to other owners

1. **scripts — `scripts/make_othello_edits.py`**
   - Drop the `shipped_length_distribution` import, and the `--length 0` option ("Li's 5–30 mix").
   - Make `--length` required, or default it to 20 with no 0 path. Use `lengths = {a.length: a.n}`.
   - Drop `layout.ensure_marker(...)`.
   - Write the cases to `pim.environments.othello.bench.cases_path(a.instance)` (and the `.json` beside it) instead of `layout.edits_dir(..., "v1")`.
   - Replace the "shipped set" wording in the docstring.
2. **scoring — `pim/scoring/othello.py`**
   - In `othello_arms`, remove the `("add_sub", "ND", a_nd)` loop entry. `oa.linear_arm` now raises `ValueError` for any mode other than `"pinv"`.
   - Remove the `oth_extra_targets` (`mine_signed`) block: `fit_probe_grid` / `flatten_rows` now reject that target.
   - Drop `ND` from `probe_sources`.
3. **scoring — `pim/environments/layout.py`**
   - Keep `othello_split_dir(inst, name)`, with `OTH_ROLE` including `"edits" -> "edits"`.
   - Keep `othello_cases_file(inst, n_cases=1000)`. It must resolve to `datasets/othello/<inst>/edits/cases_<n>.pkl` (flattened; `bench.cases_path` calls it with the instance only).
   - `corpus.build` no longer calls `ensure_marker`, so it can go.
4. **core**
   - Keep `pinv_step`'s z-space default and `make_intervention_hook`'s Adam default (or remove those parameters).
   - Keep `WorldStateProbe.n_classes`, `model.decode(idx, edit=hook)`, `model.logits`, `model.residual_stack` and `model.n_layers`.
   - Keep `load_checkpoint` rule 4: the standard checkpoint has no `arch` key.
5. **scripts — `scripts/bayes_floor.py`** (for information): the docstring of `bayes.py` names the output as `runs/_baselines/othello/<instance>/bayes_floor.json`, per SPEC. `bayes_floor()` writes `"instance": <bare instance>` and `"version": "1.0"`.

## Verification

All helper scripts are in `experiments/release/work/othello-env/`: `check_env.py`, `check_arms.py`, `check_fit.py` and `compare.py`. `compare.py` checks exact structural equality, compares arrays by dtype, shape and bytes, and ignores only the three dropped gate fields. Each check ran once under PRIVATE and once under RELEASE, in separate processes. `pim.__file__` was confirmed under the right tree each time.

The PRIVATE runs were deterministic run to run: the arm probe (`out_private` vs `out_private2`) and the fit probe (`fit_private` vs `fit_private2`) were each identical across two runs. `compare.py` does catch small changes: a planted one-ulp change to a float32 probability, a 1e-15 change to a float, one int8 label and one node count were all reported.

- **(a) Generator — IDENTICAL.**
  - Coverage: all 4 variants, the first 200 games of each of the 5 index ranges (train, test, probe, probe_large, edits).
  - Compared: token rows and lengths (serial `_regen_row`, checked equal to the pooled `_generate`), legal sets (`legal_sets`), `tokens_and_labels` output (tokens, labels, mine, mask, lengths), `flatten_rows("mine")`, the constants (index ranges, `SEED`, `BLOCK` / `MAXLEN`, `LADDER["D"]`, vocab), the `exact_ce_floor` values on 200 test games, and `verify_splits` on the stored PRIVATE test / probe / probe_large / edits files (read-only).
  - Plus: the fit probe regenerated the test (10k), probe (20k) and edits (10k) splits for all 4 variants through `oc.build`, into scratch. The files are byte-identical to PRIVATE's except for the `instance` string, and pass `verify_splits`. `bayes_floor(inst, 2000)` and `flips_per_move` (500 games) are also identical: 2.252 flips per move for standard, 0.261 for adjacent-flip, 0 for both no-flip variants. `probe_data` (fresh and cached) is identical too.
- **(b) Bench synthesis — IDENTICAL.**
  - Recipe: 10,000 edits games regenerated, then `synthesise_cases(hist, 1000, {20: 1000}, seed=0)`.
  - Cases and quota/stats equal PRIVATE's for all 4 variants, as does `benchmark_from_cases` (tokens, case_ids, pos_int, new_class, legal_pre, legal_post, cur_lab, tgt_lab).
  - In both trees the regenerated edits games equal the stored `edits_10000.npz`, and the synthesized cases equal the stored `datasets/othello/<old>/edits/v1/cases_1000.pkl` (`==` True for all 4).
- **(c) Reachability — IDENTICAL.** 20 cases per variant, budget 200k nodes. Verdicts, node counts and witnesses are all equal:

  | variant | reachable | unreachable | undecided |
  |---|---|---|---|
  | standard | 8 | 9 | 3 |
  | adjacent-flip | 2 | 3 | 15 |
  | adjacent-noflip | 0 | 20 | 0 |
  | standard-noflip | 0 | 20 | 0 |
- **(d) Editor arms — IDENTICAL.**
  - Setup: PRIVATE `L-oth-20m/best_model.pt` (read-only), a copy of its `probes/` in the work dir, the first 20 stored standard cases.
  - Arms and outputs compared:
    - `unsteered_probs` and `unsteered`;
    - the probe grid (cache HIT, same filename `probes_37e891b4279cd312.pt`);
    - PI at point 4, α = 3;
    - GS from start point 4, α = 0.2, 100 steps, β = 0.2;
    - IM and IM-NN at point 5 (inverse-map cache HIT), records, stats and distributions;
    - the two-disc variants: PI with `second=`, GS with `second=`, IM with `post_boards=`;
    - `gates` on 100 test games;
    - `observation_probes`, linear, left and right (baseline cache HITs).
  - Both trees wrote zero new cache files.
  - Plus (the fit probe): cache-miss paths on 400 games give identical results. That covers `fit_probe_grid` (2 epochs; stats and every probe weight), `observation_probes` (linear and MLP × left and right, 2 epochs), a fresh `inverse_arms` fit at point 3 (which writes the same cache filename, `probes_b9da42d89e0ca3b3.pt`), and PI through the fresh probes.
- **(e) Imports and ruff — pass.**
  - Imports: all 11 modules import in RELEASE, with `pim.__file__` under RELEASE.
  - Ruff: `ruff check` passes with the defaults, with `--isolated --select F821,F822,F823,F401,F811,F841`, and with PRIVATE's `pyproject` (E, W, F).
- **PRIVATE writes: none.** No write went into PRIVATE `runs/` or `datasets/`, and the `layout.json` mtimes are unchanged. All outputs are in the work dir.

## Open issues

- The shipped `standard` npz files have no `flip` / `placement` fields, so `verify_splits` keeps its standard-rule fallback. It is documented in one comment.
- RELEASE `layout.py` still has the v1 edits level and `ensure_marker`. Until scoring flattens `edits/`, `bench.cases_path` resolves to `edits/v1/cases_1000.pkl`.
- The PRIVATE `runs/initial_othello_comparison/L-oth-20m/scores.json` is missing. `export.md` already records this: a git pull deleted it and the export worker recovered a copy. I did not need it; my picks of PI / GS / IM arms came from `scores_backup/`.
- `__pycache__/` directories exist under my package, left by imports. They should be gitignored by infra.
