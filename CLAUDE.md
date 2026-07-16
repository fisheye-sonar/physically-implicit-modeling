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
- **Waterfalls — fixed spec (do NOT invent a colormap).** Observation waterfalls use
  **`cmap="gray"` on the dark background** — the `notebooks/world_model_eval.ipynb` /
  `notebooks/experiments/editability/00_master_editability.ipynb` **Fig 5a** style, which is the
  canonical reference. **NEVER magma / viridis / the pink-purple scheme.** Every waterfall comparison
  also has: a **GT (sim clean-obs) reference column**; ~6 **pre-edit context frames** — the **actual
  (noisy) observations the model was teacher-forced on** (`edits.obs` / `test.obs`), NOT the clean render
  (only the GT column is clean) — above a marked **edit-frame line**; vertical **green = target** /
  **red-dashed = ghost** locators; a **figure-top
  legend** (not inside a panel); a **single what-is-shown title** (no results); and it is sized **wide
  enough that every column stays full-size** (add columns by widening, never by shrinking).

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
- **Tables for dense values — clearly demarcated.** When a step emits many named scalars, render a
  **clearly demarcated table** (a `display(Markdown(...))` table — visible row/column structure; **pandas
  is not in the `.pim` venv**, so don't reach for a DataFrame), **not** an aligned-monospace `print()`
  block. Targeted use — do NOT duplicate every plot as a table; tabulate where there are many terms/values
  to scan.
- **Plain language, not shorthand.** Figure/panel/section titles, print headers, and prose are for a human
  reader — spell things out. No internal shorthand (`~=`, `=>`, `<<`, `!=`, ALL-CAPS jargon like
  "nonlinear-INSTANTANEOUS"): write "≈", "→", "much less than", "≠", or plain words. A **title states what
  is shown, not the current result** — results belong in the dated `Current results` block, never in a
  figure/section title.
- **Define every implementation detail where it's used.** Any threshold, subset, or cutoff a reader would
  ask about (e.g. "late-t = frames t ≥ 15") must appear in the definitions table or a clearly identifiable
  note — never left as an unexplained label on an axis or in a print.
- **Name methods by their mechanism.** A method's name must say what it actually does: "decoder gradient"
  (gradient descent on `h` through the decoder against a target observation), "MLP-probe gradient" (steer
  `h` until a frozen MLP probe reads the target). Never name a method after an incidental implementation
  detail, and never reuse a name that already means something else in this repo.
- **Every reported magnitude needs a reference scale and units.** A residual/distance alone is
  uninterpretable — always show the matched reference next to it (e.g. the same metric on real states) and
  state the normalization (raw ‖·‖ vs fraction). Name the estimator in the metric name: "global-PCA hull
  residual", "leave-out local-PCA residual" — never a bare "manifold residual" or a pet adjective
  ("honest").
- **Precise failure-mode language.** "Reverts" = returns toward the unsteered/pre-edit trajectory.
  "Collapses" = output degenerates off-distribution. "Drifts" = diverges without returning. Look at the
  actual rollout before choosing the word; they are different dynamics outcomes.
- **Data-source provenance.** Each section states the exact model/checkpoint/dataset/split it
  uses. When a number is pulled in as a **comparison from another notebook / experiment /
  finding, cite the source** inline (e.g. "GRU fiber resid 0.337 — from `diagnostic_corrections`")
  rather than dropping a bare constant.
- **Always include the reference/GT column** in any comparison figure (waterfalls, editor
  head-to-heads). A comparison with no ground-truth/target column is uninterpretable.

## Synthesis notebooks (source-of-truth tier)

Some notebooks sit a tier above one-off experiments: a **single source of truth** that consolidates a
research thread across architectures and proposes the language/metrics we may later fold into `pim`
(e.g. `notebooks/experiments/editability/00_master_editability.ipynb`). They are **provisional
proposals**, not the codebase — don't over-index on idiosyncrasies. Extra standards for this tier:

- **Separate the invariant spine from dated results.** Definitions, metric formulas, and the pipeline
  are stable. Every *result* (a number that moves as models/experiments evolve) lives in a clearly
  marked **`Current results (updated YYYY-MM-DD)`** block — never woven into a section header, a
  definition, or a figure title. The reader must tell "what this measures" apart from "what it reads now."
- **Build every figure and table to hold N world models.** No two-model hardcoding: metric categories
  on one axis, **one color-coded bar/series per world model** side-by-side, shared legend; never put a
  model's result in a panel title. Adding a 3rd architecture is a data change, not a re-layout. Report
  the *same* estimator for every model — **compute it, don't write "~same."**
- **Lightweight:** recompute only the cheap things; cite the rest with source-notebook provenance
  (keep the header's source list current as the thread evolves).
- **Comparison sets grow — plan for it.** Editor line-ups and world-model line-ups will gain members;
  lay out comparison figures/tables so a new method or model is an added column/row (wider figure,
  single legend at the top of the figure), not a redesign.
- **Calibrated claims; interpretation lives in the Summary.** Body sections state quantities ("~34% of
  ‖h‖ is not explained by (pos,vel)") without verdict adjectives — don't binarize graded quantities
  ("non-canonical") in body prose. Forward interpretations are confined to the final **Summary** section,
  clearly marked as interpretation, and still quantified.
