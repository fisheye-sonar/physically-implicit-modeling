# fix-lib

Scope: `pim/environments/{rayworld,othello}/**`, `pim/{editors,probes,metrics,training,models}/**` in RELEASE.
Work files: `experiments/release/work/fix-lib/` (`baseline/` is the pre-edit copy; `harness_editors.py`,
`cmp_npz.py`, `cmp_corpus.py`, `othello_games.py`, `fp.py` are the checks; `fix-lib.diff` is the full diff).

## Changes (all 13 items)

1. `editors/grad_steer.py`: module, `_descend` and `make_intervention_hook` docstrings say `n_steps` Adam steps
   at learning rate `alpha * probe.act_scale` (median per-dimension SD), toward the target, hold weight `beta`.
2. "latent state" meaning the simulator's state is now "simulator state": `rayworld/sim.py` (2), `dataset.py`,
   the `viz.py` panel title. `interactive.py` had none.
3. Dead code removed: `PinvMap`, `pinv_maps` (+ exports); `metrics.prediction.floor_bracket` (+ docstring
   mention); `bigcorpus.instance_dir`; `othello/corpus.BLOCK`; `bayes.World.n_rays`; `FrameVocab.obs_dim`.
   Single-value parameters removed: `pinv_step(space=)`, `rayworld.arms.pinv_rollout/pinv_arm(space=)`,
   `token_bench.pinv_arm(space=)` (label stays the literal `"PI[zspace]"`), `frame_probs(kind=)` (now a softmax),
   `othello.arms.linear_arm(mode=)`, `_descend`/`make_intervention_hook(optimizer=)`, `TransformerL.output_kind`,
   `TransformerL._seq_mask`, `_run(attn_mask=)` (now `_run(tokens, edit=None, want_resid=False)`).
   Kept: `fit_probe_grid(targets, splits)`, `_split(how)`, every cache-key field.
4. `metrics/decodability.py`: `probe_skill_classification`, `trivial_error_rate` and their exports deleted.
   The module and `probe_skill_from_stats` docstrings, and a one-line comment at `majority_class_error_rate`
   in `probes/base.py` and `probes/baselines.py`, say the trivial predictor is the single most common class
   pooled over all cells of the train split.
5. `training/sources.py`: `rayworld_source` raises `ValueError` when fewer than one block is left for training
   (minimum `3 * block` = 6,144). `training/stream.py`: `BlockStream` raises if its range holds no block.
6. `rayworld/arms.py`: `_read_corpus` raises `ValueError` ("<file> holds N sequences; this fit needs M");
   `probe_recipe(target, instance, n_seq=30_000, *, cat_n_seq=200_000, cat_epochs=50)`.
7. `rayworld/bigcorpus.py`: `use_instance(inst, *, n_shards=None, shard_n=None)` (None = 40 x 500,000; range
   checked) rebinds `SHARD_N`, `N_SHARDS`, `N_TOTAL`, `VAL_N = N_TOTAL // 10`, `TRAIN_N`; new `corpus_n()` reads
   `n` from `train/corpus.json` when it exists; `open_obs` and `verify` use it. Shard commands and seeds unchanged.
8. `othello/corpus.py`: `N_TRAIN_GAMES = 20_000_000` replaces `LADDER`; the `__main__` CLI and the usage line are
   gone; `build` says "CPU only; games are generated in parallel, one process per core."
9. `synthesise_cases` -> `synthesize_cases`, `CENTRE` -> `CENTER`, `FactorisedTarget` -> `FactorizedTarget`,
   `centres` -> `centers`, "labelled" -> "labeled". No other British spelling left in my files.
10. `grid_target.py` glosses `appearance-fac` as each disc's appearance bin as a center and a length label
    (15 and 5 classes on 8-ray; checked with `factor_sizes`).
11. `viz.py`: `plt.Circle(facecolor=..., edgecolor="white")`; the demos run without the matplotlib warning, and
    the disc outline is white, as `edgecolor` specifies.
12. Module docstrings trimmed: `othello/reachability.py` (the prune rules became one-line comments at the code
    and the `_flippable` docstring), `othello/corpus.py`, `othello/__init__.py`. No other module in my packages
    is over 6 lines.
13. `blink.py`: the module docstring says one disc is hidden at a time, and a disc cannot start a new blackout
    on the frame it reappears. The code enforces both.

Signature changes (invariant 4, all numerically no-ops): the removed `space`/`mode`/`optimizer`/`kind`/`attn_mask`
parameters, and the added keyword-only `probe_recipe(cat_n_seq, cat_epochs)` and `use_instance(n_shards, shard_n)`.

## Verification

| check | result |
|---|---|
| import every `pim` module | 71 modules, 0 failures (`pim.__file__` under RELEASE) |
| ruff (repo config) on my files and on the whole repo | All checks passed |
| PI (zspace) and GS, 20 bench cases: 8-ray frame model (full/cartesian and appearance-fac), 8-ray token model, othello/standard, vs pre-edit copy | 576 arrays (rollouts, probabilities, every arm record field, editor labels, recipe dict) bitwise equal; the baseline repeated bitwise, so the check is deterministic; every probe load was a cache hit with `require_cached=True`, so the default recipe keys are unchanged |
| random-init fingerprints, `state_span`, parameter order | identical before and after |
| Rayworld shard 0 (8-ray, blink), `--shards 1 --shard-n 2000` | first 3, and all 2,000, sequences of `obs.f32` and every `meta.h5` field bitwise equal to the PRIVATE 20M corpora |
| small corpus reads its size | `corpus.json` records `n`, `shard_n`, `n_shards`; after `use_instance(inst)` with full defaults, `open_obs` is (2000, 40, R) and `verify` reports 2,000. A 2 x 3,072 corpus matches PRIVATE shards 0 and 1, and `train.py` runs 20 steps on it |
| too-small `--limit` | `train.py --limit 5000` exits at once with the message above; 6,143 raises, 6,144 trains |
| Othello games | first 200 games at every split's `lo` (train, test, probe, probe_large, edits) equal PRIVATE for all four variants, and equal the staged splits |
| `synthesize_cases` | `make_othello_edits.py` regenerates all four `cases_1000.pkl` byte-identical to STAGING; the edits split equals PRIVATE's; only the manifest `recipe` string differs |
| demos | `animate_scene` and `demo.py` run with `-W error::UserWarning` |

## Requests

- fix-tables, `pim/scoring/rayworld.py:100` and `:153`: drop `space="zspace"` from the `rwa.pinv_arm` and
  `tkb.pinv_arm` calls. Until then, scoring any Rayworld run raises `TypeError`.
- fix-tables: `rwa.probe_recipe(target, instance, n_seq=30_000, *, cat_n_seq=200_000, cat_epochs=50)` is in
  place; `scoring/blocks.py:rw_probe_recipe` already calls it this way.
- fix-scripts, `scripts/demos/play.py:115`: `"2D world  (latent state)"` -> `"2D world  (simulator state)"`.
- fix-scripts, bigcorpus interface (already used by `build_rayworld_corpus.py` and `train.py`):
  `use_instance(inst, *, n_shards=None, shard_n=None)`, `corpus_n()`, `open_obs()` shaped `(corpus_n(), FRAMES, OBS_RES)`.
- fix-scripts, old -> new names for any remaining call site: `oc.LADDER["D"]` -> `oc.N_TRAIN_GAMES`;
  `synthesise_cases` -> `synthesize_cases`; `CENTRE` -> `CENTER`; `FactorisedTarget` -> `FactorizedTarget`;
  `pinv_step/pinv_rollout/pinv_arm(space=)`, `linear_arm(mode=)`, `make_intervention_hook(optimizer=)` and
  `frame_probs(kind)` are gone. None remain in `scripts/` or `notebooks/` now.
- export: set the `recipe` of the four shipped `cases_1000.json` to exactly what `scripts/make_othello_edits.py`
  writes now: `"one occupied non-center token recolored; rejected if the legal set is unchanged or empty (bench.synthesize_cases)"`.

## Not applied (outside the 13 items)

`bigcorpus.RESERVED` comment wording, `rayworld.arms.score(model)` unused parameter, the `fit_baseline_probe`
alias, `tokens.tokens_dir` vs `layout.tokens_dir`, box-drawing banners.
