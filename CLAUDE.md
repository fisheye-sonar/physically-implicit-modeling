# CLAUDE.md

How to work in this repo. *What* we're trying to learn lives in `RESEARCH.md`;
the scientific knowledge base lives in `research/`. This file is mechanics,
conventions, and gates — not science. When you're tempted to explain a finding
or motivate a research question here, it belongs in `research/` or `RESEARCH.md`.

## Research workflow

Two reading paths, by role:
- **Driving the project (orchestrator):** read `research/README.md`, `research/PROGRESS.md`,
  and `research/ORCHESTRATION.md`; `RESEARCH.md` for the north star.
- **A spawned worker subagent:** read ONLY `research/WORKER.md` and your assigned
  `research/directions/<X>.md` — not README / PROGRESS / ORCHESTRATION.

**IMPORTANT — the findings gate.** Draft freely in `research/scratch/`. Never write to
`research/findings/` or `RESEARCH.md` yourself — promotion is the human's call. [...keep the rest...]


## Commands

```bash
# tests
poetry run pytest

# demo animation (interactive)
python scripts/demo.py
python scripts/demo.py --seed 7 --n-objects 4 --waterfall-mode human
python scripts/demo.py --fixed-reflectivities --always-in-frustum

# dataset (splits: train/val/test/edits)
python scripts/generate_dataset.py data/my_run
python scripts/generate_dataset.py data/my_run --n-train 100000 --n-workers 8 --fixed-reflectivities

# train
python scripts/train_gru.py
python scripts/train_rssm.py

# eval (auto-detects model type; writes figures + metrics.json + eval_config.json to a timestamped dir)
python scripts/run_eval.py <checkpoint>

# lint / format
poetry run ruff check pim tests
poetry run black pim tests scripts
```

## Environment

- Python 3.12 venv in `.pim/`; [direnv](https://direnv.net/) auto-activates via
  `.envrc` (else `source .pim/bin/activate`).
- Dependencies via Poetry (`pyproject.toml`).

## Where things live

<!-- FILL IN: stable paths for datasets, checkpoints, eval outputs, scratch.
     One line each. This is the kind of thing worth stating because it's stable
     and saves Claude guessing. e.g. -->
- Datasets: `data/<run_name>/` (train/val/test/edits HDF5 + `dataset.json`)
- Checkpoints: `<...>`
- Eval outputs: timestamped dirs under `<...>`

## Architecture — the contracts that matter

Five strictly separated layers: `simulator/` (ground-truth world, rendering,
datasets) → `world_models/` (architectures behind a shared Protocol) →
`extractors/` (probes) → `editors/` (interventions) → `eval/` (metrics) and
`figures/` (drawing). The per-module map is deliberately not duplicated here —
it goes stale; use `Grep`/`Read` to navigate, or see `pim/CLAUDE.md` if present.

Four invariants you cannot infer from any single file:

1. **eval/figures purity.** `pim/eval/*` returns arrays/scalars and never imports
   matplotlib. `pim/figures/*` takes pre-computed arrays and returns
   `matplotlib.Figure`, and never calls models or computes metrics. The
   notebook/script is the orchestrator that wires them. Change *how a metric is
   computed* → `eval/`; *how it's drawn* → `figures/`. Do not collapse these.
2. **Model-agnostic eval via Protocol.** All eval flows through the
   `HiddenStateModel` protocol (`flat_state` / `state_from_flat` / `decode` /
   `observe_sequence` / `predict_step`). Eval never special-cases GRU vs RSSM. A
   new architecture implements the protocol and the whole suite works unchanged —
   do not add `if isinstance(model, ...)` branches in `eval/`.
3. **Probe uniformity.** Any number of probes flow as `list[ProbeSpec]` through
   `fit_probes` → `eval_recovery_multi` → figure builders. Add a probe by
   instantiating a `ProbeSpec`. Probe hyperparameters live in the extractor
   constructor (notebook-controlled), never in `EvalConfig`.
4. **Notebook = orchestrator.** Eval notebooks are explicit top-to-bottom
   pipelines: one operation per cell, passing named artifacts (`states_tf`,
   `probes`, `decoded_roll`, `steered`, `metrics_*`) to the next. Keep them
   linear and readable; don't hide pipeline logic in helpers.

## Conventions

- **Notebooks:** prefer to edit `.ipynb` with the `NotebookEdit` tool; inspect with
  `Read`/`Grep`. Less preferred to manipulate notebook JSON via Bash. Live under
  topical subdirs: `notebooks/experiments/<topic>/<name>.ipynb`.
- **Number every cell and every figure.** Each experiment notebook is referenced later
  in discussion, so make every artifact addressable: prefix each code cell with a
  `# [N]` tag (sequential), and give every figure a number in its title/suptitle
  (e.g. `Fig 3 — convergence`) with sub-panels lettered `(a)/(b)/(c)`. A claim should be
  citable as "cell [7] / Fig 3a" without hunting. *(Figure-heavy notebooks can exceed the
  `Read` token cap — keep outputs lean, or expect to edit setup cells before outputs accrue.)*
- **Visual aesthetic:** results / metrics / analysis → light academic theme
  (white bg, Okabe-Ito, `style_ax(ax)`). Simulator artifacts (2D scene,
  waterfalls, "model running in the sim") → dark theme (`#0a0a14`, `dark=True`).
  When in doubt: metrics → light, simulator output → dark.

## Visualization for analysis

Experiment notebooks serve two readers with different strengths: **Sevan** does the
scientific judgment and reads best from *plots*; the **agent** self-verifies from
*numbers*. Always produce **both** — rich visualizations and printed metric tables —
never tables alone. When you claim an effect, visualize it in the space where it
actually occurs: e.g. plot the **1D observation scans / waterfalls** under the perturbation,
not only decoded-scalar positions. Observation-space plots can reveal effects (e.g. one
object moving while another stays) that decoded-scalar tables hide entirely.

## Notebook legibility (hard standard — every experiment notebook)

A notebook is read later by Sevan (from plots) and re-derived by agents (from numbers); it
must **stand alone and be followable top-to-bottom** without the reader hunting for what a
label means. Beyond cell/figure numbering:

- **Definitions table up front.** Right after setup, include a **table defining every
  non-obvious term and — critically — every metric with its explicit formula**, units, and
  better-direction (↑/↓). A metric's definition lives in this table, *not* buried in a print
  sidenote or a code comment. If a figure shows `d_gt` or `ghost ratio`, the reader must find
  its formula in that table. When a term first appears, it must already be defined.
- **Consistent metrics + units across everything you compare.** If two things are compared
  (editors, models, variants, sections), report the **same metric set in the same units**.
  Use **RMSE, not MSE** (matches the rest of the repo); never plot MSE in one panel and tabulate
  RMSE elsewhere, and never compare method A on metric-set-X against method B on metric-set-Y.
- **Tables for dense values.** When a step emits many named scalars, put them in a **table**,
  not free-floating prints. Targeted use — do NOT duplicate every plot as a table; tabulate
  where there are many terms/values to scan.
- **Data-source provenance.** Each section states the exact model/checkpoint/dataset/split it
  uses. When a number is pulled in as a **comparison from another notebook / experiment /
  finding, cite the source** inline (e.g. "GRU fiber resid 0.337 — from `diagnostic_corrections`")
  rather than dropping a bare constant.
- **Always include the reference/GT column** in any comparison figure (waterfalls, editor
  head-to-heads). A comparison with no ground-truth/target column is uninterpretable.
