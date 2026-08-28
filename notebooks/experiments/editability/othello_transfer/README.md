# othello_transfer — our probe and our editor, on Li et al.'s Othello-GPT

**The direction matters.** This thread is the *mirror* of [`../othello_gpt/`](../othello_gpt/):

| thread | model | probe & editor | asks |
|---|---|---|---|
| [`../othello_gpt/`](../othello_gpt/) | **ours** (`runs/transformers/W16`) | **theirs**, ported exactly | does Li et al.'s method work in our world? |
| **`othello_transfer/`** (here) | **theirs** (`gpt_synthetic`) | **ours**, unmodified | does our implementation work in a world where the answer is published? |

`../othello_gpt/` answered its question on 2026-08-18: the probing half replicates, the
intervention half does not. That leaves one alternative explanation standing — that **our editor
code is simply wrong** — and no experiment in the thread could rule it out, because every
editability number in the repo comes from that same code. This thread is the positive control
that removes it.

**Read the two together.** Neither is complete alone, and Table 4 of
[`probe_transfer.ipynb`](probe_transfer.ipynb) puts all three columns side by side: Li et al.'s
published numbers, ours on their model, and ours on our model.

## What is fixed, and what moves

The rule throughout: **their world, their model, their design decisions; our probe and our editor,
byte-identical to what runs on discworld.**

- The probe, the edit objective, the descent and the multi-layer schedule are imported from
  [`../othello_gpt/othello_probe.py`](../othello_gpt/othello_probe.py) — *the same module*, not a
  copy. It gained a 3-way classification head on 2026-08-20 because their state is 64 ternary tiles
  rather than continuous scalars; the regression path is untouched and the repo's 178 tests pass.
- [`othello_shim.py`](othello_shim.py) is the entire bridge: ~135 lines supplying the seven names
  our editing code calls (`embed`, `_run`, `_seq_mask`/`_win_mask`, `norm_out`, `decoder`,
  `residual_stack`, `decode`) over their unmodified minGPT `GPT`. **It contains no editing logic.**
  Cell [2] of the notebook gates it as bit-identical to their `GPT.forward` and to
  `GPTforProbing` at all nine residual points.
- Data and hyperparameters mimic theirs: synthetic games from their own generator,
  `beta = 0.2` (their `reg_strg`), board state as absolute colour, their 1001-case benchmark.

## Files

| file | holds |
|---|---|
| [`probe_transfer.ipynb`](probe_transfer.ipynb) | the pipeline, top to bottom |
| [`controls.ipynb`](controls.ipynb) | the four controls that sit *under* the pipeline: probe-data scaling, the raw-observation baseline, the random-init baseline (all on **our** model), and Nanda's linear-direction write vs our pseudoinverse (on **theirs**) |
| [`controls_lib.py`](controls_lib.py) | the three probe controls on our model |
| [`linear_intervention.py`](linear_intervention.py) | Nanda's addition and our `inject_state`, on their model |
| [`nanda_on_discworld.py`](nanda_on_discworld.py) | Nanda's addition applied to `W16` |
| [`single_layer.py`](single_layer.py) | the single-residual-point sweep, for both write mechanisms |

`probe_scaling.py`, `obs_window_probe.py`, `random_init_control.py` and `occlusion_probe.py` are
the **original one-off runners** that produced the numbers the 2026-08-21 scratch notes cite, and
are kept as that provenance. `controls_lib.py` is the implementation `controls.ipynb` uses; where
the two overlap, `controls_lib.py` is canonical.
| [`othello_shim.py`](othello_shim.py) | the bridge to their minGPT model |
| [`othello_data.py`](othello_data.py) | games, board labels, activation harvest, the benchmark, the metrics |
| [`transfer_pipeline.py`](transfer_pipeline.py) | probe-grid and intervention-arm orchestration |
| [`board_grid.py`](board_grid.py) | the qualitative panel — boards, not observation waterfalls |
| [`OTHELLO_TRANSFER_RUNS.md`](OTHELLO_TRANSFER_RUNS.md) | the run registry, including checkpoint provenance |

Outputs land in `runs/othello_transfer/` — `figures/`, `results.json` (from `probe_transfer`),
`results_controls.json` (from `controls`), `probe_cache/`, and the run logs; the checkpoint lives
there too and is gitignored.

**Two notebooks, two questions.** `probe_transfer.ipynb` asks *does our implementation work in a
world where the answer is published?* (yes). `controls.ipynb` asks the two questions that answer
sat on top of: *is our probe reading a learned state or the observation?* and *which write
mechanisms actually move their model?*

## The metric situation

Their metric and ours are **both** reported, never merged:

- **Li error** — top-*N* predicted moves against the legal set, false positives + false negatives.
  This is the replication target (their null baseline 2.68 → 0.12 intervened).
- **Li error vs the pre-flip world** — the guard they do not have. Their metric alone cannot
  separate "the edit worked" from "the model was destroyed"; scoring against both worlds does.
- **Edit Index (union support)** — this repo's own formula, translated, so the result lands on the
  same axis as every other editor in the editability thread. The ground-truth reference is uniform
  over legal moves, which is *exact* here rather than approximate, because their generator draws
  moves uniformly from the legal set.

Formulas, units, and the calibration floors are in the definitions table at the top of the notebook.

## Dependencies

Their repo is vendored read-only at `/home/sevan/research/PIM/othello_world` and is added to
`sys.path`; nothing in it is modified. Its `data/othello.py` imports `seaborn`, `psutil` and `pgn`
at module level, so those are now in the `.pim` venv (`pandas` came along with `seaborn` — see
`../../../../research/GOTCHAS.md`, 2026-08-20).
