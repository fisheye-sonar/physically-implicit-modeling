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
- **Every experiment goes inside the RESEARCH THREAD it serves — never beside it.** The topic dir is the
  *findings thread* the work contributes to, not the kind of experiment it is. While the editability findings are
  the active thread, **everything** — main experiments, controls, ablations, side quests — lives under
  `notebooks/experiments/editability/`, in a subdirectory when it is a coherent group
  (`editability/actions/`, `editability/multistep/`, `editability/controls/`,
  `editability/rssm_structure/`). Creating a sibling like `notebooks/experiments/controls/` orphans the work from
  the thread whose findings it supports and breaks every relative path back to the thread's registries.
  *(Recurring failure: this happened on 2026-07-30 with the michael_controls thread and had to be migrated.)*
  Run artifacts follow the same logic — a thread's checkpoints and eval outputs stay together under one
  `runs/<thread>/` prefix.
- **Number every cell, figure, AND animation.** Each experiment notebook is referenced later
  in discussion, so make every artifact addressable: prefix each code cell with a
  `# [N]` tag (sequential), and give every figure a number in its title/suptitle
  (e.g. `Fig 3 — convergence`) with sub-panels lettered `(a)/(b)/(c)`. A claim should be
  citable as "cell [7] / Fig 3a" without hunting. *(Figure-heavy notebooks can exceed the
  `Read` token cap — keep outputs lean, or expect to edit setup cells before outputs accrue.)*
  **Animations/GIFs are numbered the same way** — a *persistent* figure-level title carrying the
  number (e.g. `Anim 4 — …`, or continue the surrounding letter series like `Anim E3`), separate
  from any per-frame caption, and the saved file named to match (e.g. `animE3_….gif`). GIF playback
  must be legible: slow enough to read (default ~3 fps, not the matplotlib default), and **hold/pause
  on the key frames** (e.g. edit/action frames) by repeating them so the viewer can register the effect.
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
  (only the GT column is clean) — above a marked **edit-frame line**; then, below that line, **every column
  shows its OWN free-run starting at step 0**, and the GT column shows `clean_obs[ef:ef+K]`.

  > ### ⛔ NEVER paint a shared teacher-forced `ef` row across all columns (corrected 2026-07-30)
  > An earlier version of this spec mandated one shared row = `clean_obs[ef]` in *every* column. **That is
  > wrong and is now banned.** It makes every column look as though it were teacher-forced on the post-edit
  > frame when only the **Oracle observation** reference actually was, and it **hides the exact frame the §4
  > scorecard scores** (step 0). It also displayed the *clean* render while the model that legitimately sees
  > that frame is fed the **noisy** `edits.obs[ef]` — a second inconsistency. Seeing the post-edit frame is a
  > **property of one editor**, never a display convention. (Same error was caught and fixed in
  > `eval_editability_endogenous.py` v2; it then leaked back in via this file into the `controls/` notebooks.)

  **Alignment rule (get this exactly right):** `warm_up_to_edit` teacher-forces `obs[0..ef-1]`, so the
  predict-next GRU's rollout **step-0 is `ef`** (`decode(h_edit) ≈ obs[ef]`, i.e. `ROLL[:,0] ↔ clean_obs[ef]`).
  So plot `ROLL[:, 0:K]` against `clean_obs[ef:ef+K]` — no slicing, no dropped step. The one exception is the
  **Oracle observation** column, which was fed `obs[ef]` and therefore **leads by one frame**; label it as
  such rather than re-aligning the other columns to it. Vertical **green = target** / **red-dashed = ghost**
  locators; a **figure-top
  legend** (not inside a panel); a **single what-is-shown title** (no results); and it is sized **wide
  enough that every column stays full-size** (add columns by widening, never by shrinking).
  **This is a hard, recurring-violation spec** — `magma`/`viridis` and "just the K-step rollout with no
  context frames" keep sneaking back in (e.g. `multistep_steering` shipped `cmap="magma"` with no context
  frames until 2026-07-27). **Do not re-implement the panel per notebook** — that is where the drift
  happens; define **one `waterfall_grid(...)` helper** in the notebook that bakes in the whole spec (gray
  cmap, `N_CTX≈6` noisy context frames concatenated above a dashed edit-frame line, each column's own
  free-run from step 0 below it, green/red locators, figure-top legend) and route **every** waterfall
  through it. Reference implementations to copy: `scripts/eval_editability_endogenous.py` `waterfall()` and
  the `waterfall_grid` in `notebooks/experiments/editability/controls/`. Before committing any waterfall,
  eyeball it: gray? noisy context frames above the edit line? **each column its own free-run from step 0**
  (no shared teacher-forced row)? top legend? GT column? If not, it is not done.

- **2D observations — the sanctioned form (approved 2026-08-12).** Everything above assumes a **1D**
  observation, where a frame is one row of pixels and the whole rollout fits in one image with time on the
  vertical axis. When the observation is a **2D raster** (`pim/simulator/render2d.py`), a frame already uses
  both image axes and a literal waterfall **cannot be drawn**. Do not improvise a substitute and do not fall
  back to scalar figures — use the approved pair, implemented once in
  `notebooks/experiments/editability/omniscient_2d/frame_grid.py` and specified in full in
  `notebooks/experiments/editability/omniscient_2d/WATERFALL_SPEC_2D.md`:
  **`frame_grid`** (arms as rows × time as columns, raw model output — catches degradation) **+
  `frame_trails`** (every rollout step composited per arm — shows where the object went). They ship
  **together**; `frame_trails` is what guarantees `frame_grid`'s time subsample hides nothing.
  Every *content* rule above is unchanged and still binding — gray on dark, GT arm first, **noisy** context
  frames, marked edit boundary, **each arm its own free-run from step 0** (the shared-`ef`-row ban is fully
  intact), `ROLL[:, 0:K] ↔ clean_obs[ef:ef+K]` alignment, `leads_by_one` labelling, figure-top legend,
  what-is-shown title, metric in each arm's label. Only three things change, because the extra spatial
  dimension forces them: **arms become rows** (both grid axes are free once a cell is a whole frame);
  **locators become circles** at true world coordinates, which makes `aspect="equal"` mandatory (with
  `aspect="auto"` they render as ellipses and apparent object shape is a lie); and **time is subsampled**
  (3 context + 5 steps by default) because a cell per frame cannot show 21 of them without either shrinking
  cells below legibility or making the figure metres wide. Plus one addition: **fixed `vmin=0, vmax=1`** on
  every cell — per-cell autoscaling makes a collapsed arm look normal, which is the failure these panels
  exist to catch. Same eyeball check before committing, plus: circles round, not elliptical?
  A third, **optional** view, `frame_animation`, applies the animation rules above (numbered persistent
  title, ~3 fps, holds on the edit frame) to the same arms — use it to show *motion*, which stills cannot,
  but it is an **addition, never a substitute**: a GIF cannot be read in a committed notebook diff or a
  paper, so the claim still ships with the grid + trails pair.

## Visualization for analysis

Experiment notebooks serve two readers with different strengths: **Sevan** does the
scientific judgment and reads best from *plots*; the **agent** self-verifies from
*numbers*. Always produce **both** — rich visualizations and printed metric tables —
never tables alone. When you claim an effect, visualize it in the space where it
actually occurs: e.g. plot the **1D observation scans / waterfalls** under the perturbation,
not only decoded-scalar positions. Observation-space plots can reveal effects (e.g. one
object moving while another stays) that decoded-scalar tables hide entirely.

**MANDATORY — any claim about an effect on the generations ships with a waterfall.** If a notebook
compares editors, interventions, models, or scales *by what the model outputs*, it must include an
observation-space waterfall of those same arms, built through the one `waterfall_grid(...)` helper and the
fixed spec below — or, for a **2D** observation, through the sanctioned `frame_grid` + `frame_trails` pair
(same section). A scorecard compresses a rollout to one number and routinely hides the difference between
"the edit landed" and "the output degraded" — the two look identical in an Edit Index that moved.
*(Recurring failure: 2026-08-05, tangent-constrained injection was analysed through four scalar figures
before a waterfall was added; only the waterfall showed the "successful" arms were generating vertical-stripe
garbage.)* Include the arm's headline metric in each column title so the picture and the number are read
together, and add the degenerate/extreme settings as their own columns — that is where collapse shows up.

**Axis labels must be legible — check the rendered PNG, not the code.** Long series names in vertical bar
charts overlap into an unreadable smear at 4+ categories. Use **horizontal bars** (one label per row) for
anything with long names, and never shrink a label below ~7pt to make it fit. Every figure must be eyeballed
after rendering; "it ran without error" is not the check. *(Flagged by Sevan 2026-08-05 across several
figures at once.)*

## Notebook legibility (hard standard — every experiment notebook)

A notebook is read later by Sevan (from plots) and re-derived by agents (from numbers); it
must **stand alone and be followable top-to-bottom** without the reader hunting for what a
label means. Beyond cell/figure numbering:

- **Definitions table up front.** Right after setup, include a **table defining every
  non-obvious term and — critically — every metric with its explicit formula**, units, and
  better-direction (↑/↓). For editability-thread notebooks, **copy the exact name/formula/units from the
  canonical registry `notebooks/experiments/editability/METRICS_AND_EDITORS.md`** (the single source of truth
  for metric + editor definitions) rather than re-inventing terms; a notebook uses the subset it needs and may
  introduce new metrics/editors when that is its point (fold recurring ones back into the registry). A metric's definition lives in this table, *not* buried in a print
  sidenote or a code comment. If a figure shows `d_gt` or `Edit Index`, the reader must find
  its formula in that table. When a term first appears, it must already be defined.
  **§4 editability metrics are also implemented once, in `scripts/editability_metrics.py`** — import
  `build_edit_zones` / `edit_scorecard` / `fidelity_ratio` rather than re-deriving the formulas per notebook
  (that drift is what produced five incompatible versions of "reach"). Prose in the registry, code in that
  module; they must agree. The canonical set is **Edit Index** (−1…+1, which of the two ground-truth worlds the
  output is closer to) plus **Target / Ghost / Collateral / Edit-frame / GT-traj RMSE** and the **fidelity ratio**.
  The old `reach % of swap` / `collateral % of swap` / `selectivity` / `ghost ratio` were **retired 2026-07-30**
  (they scored *change*, not *correctness*, and normalised by a model-dependent soft reference) — do not
  reintroduce them, and treat pre-2026-07-30 numbers on that scale as not comparable.
- **Probes: use the standard, don't hand-roll one.** Every reported position/velocity R² comes from
  `pim.extractors.fit_readability_probes` (linear lstsq + a 2×256 ReLU MLP, both fit on the same 80% of
  **sequences** and scored on the same held-out 20%). An **in-sample R² is not a readability claim**, and a
  by-row split leaks near-duplicate neighbouring frames. The **MLP Grad Steering** editor's frozen 1×128
  `MLPExtractor` is a *different object* and must not be changed — never quote one as the other. Full rationale:
  `notebooks/experiments/editability/METRICS_AND_EDITORS.md` §2.
- **Consistent metrics + units across everything you compare.** If two things are compared
  (editors, models, variants, sections), report the **same metric set in the same units**.
  Use **RMSE, not MSE** (matches the rest of the repo); never plot MSE in one panel and tabulate
  RMSE elsewhere, and never compare method A on metric-set-X against method B on metric-set-Y.
- **One quantity per axis — a shared axis is a claim that the bars mean the same thing.** If the "same"
  metric is computed from **structurally different constructions**, it does not belong on one axis, however
  similar the column header looks. *(2026-08-03: a compositionality figure put `sequential` and `superposition`
  on one axis, where "composed" meant a **two-stage endpoint** in one and a **literal vector sum** in the other —
  so a cosine in one bar was not the same object as a cosine in the next. Only the panel asking "does the
  resulting state work" (Edit Index) was legitimately shared.)* Test before merging panels: *could a reader
  subtract two bars and get something meaningful?* If not, split the figure. Where a downstream metric **is**
  common to both (an outcome metric like Edit Index usually is), that panel may stay shared — say so explicitly.
- **No derived duplicates: never report a number that is an algebraic function of two you already show.**
  Before adding a panel or column, check whether it is recoverable from ones already present. If it is, show it
  *instead of*, not *alongside* — a redundant metric adds no information, feeds the metric zoo, and reads as a
  **contradiction** when the reader cannot see the identity linking them. *(2026-08-03: `relative residual`
  was reported next to `cosine` and `magnitude ratio`, but `residual² = r² + 1 − 2·r·cos θ` — it was fully
  determined by the other two, and looked like it disagreed with them.)*
- **Tables for dense values — clearly demarcated.** When a step emits many named scalars, render a
  **clearly demarcated table** (a `display(Markdown(...))` table — visible row/column structure; **pandas
  is not in the `.pim` venv**, so don't reach for a DataFrame), **not** an aligned-monospace `print()`
  block. Targeted use — do NOT duplicate every plot as a table; tabulate where there are many terms/values
  to scan.
- **Every run/model/variant name MUST be defined where it is used — no bare short codes.** This is a
  **recurring, high-friction failure**: notebooks ship labels like `L3` vs `L3b`, `L3s0`, "weak" vs "strong" with no
  expansion, and the reader cannot tell what was compared. Rules: (1) each experiment thread keeps a **canonical run
  registry** — one markdown table listing *every* checkpoint/run name with its full config (level/objective, dataset or
  world settings, architecture, training length, seed, what it is a control *for*); e.g.
  `notebooks/experiments/editability/actions/ENDOGENOUS_RUNS.md`. (2) Every notebook that mentions a run **copies the
  rows it uses** into its own definitions table — the notebook still stands alone. (3) **Figures and tables use
  self-describing labels**, not raw codes: "L3 force+goal · 512h · seed 0", never `L3s0`. (4) A suffix that encodes a
  variable (`b`, `s0`, `s1`, "strong") must state the variable it encodes (`b` = second seed, `s` = strong config).
  Adding a new run means adding its registry row **in the same commit**. If you catch yourself writing a name whose
  meaning is not on the page, stop and define it.
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
- **When a comparison varies along more than one dimension, the label must carry BOTH — and must not let one
  dimension hide inside the other's naming slot.** A line-up crossing *what is being tested* × *which mechanism
  ran it* needs both visible and visually separable. *(2026-08-03: three arms were named `sequential (freeze-time)`,
  `superposition (counterfactual)`, `superposition (freeze-time)`. Two of the three carry "(freeze-time)", so
  arms testing **different things** looked like one family and Sevan read the wrong bar. The parenthetical held the
  mechanism while the leading word held the test type, and at a glance the parenthetical dominated.)* Prefer labels
  that make the tested dimension unmistakable — `same-object A→B→C · freeze-time` vs `two-object A+B · freeze-time` —
  and if one dimension has only one level (e.g. only freeze-time can run a test), say **why** rather than letting
  the asymmetry look like an omission.
- **Every reported magnitude needs a reference scale and units.** A residual/distance alone is
  uninterpretable — always show the matched reference next to it (e.g. the same metric on real states) and
  state the normalization (raw ‖·‖ vs fraction). Name the estimator in the metric name: "global-PCA hull
  residual", "leave-out local-PCA residual" — never a bare "manifold residual" or a pet adjective
  ("honest").
- **High-dimensional quantities need their intuition stated, because the everyday one is wrong.** Three that
  have bitten this repo, all of which look like one thing and mean another:
  (1) **Cosine is not correlation-like** — cos 0.9 is a **26° angle**, and two equal-length vectors 26° apart
  differ by `2·sin(θ/2)` ≈ 0.45 of their length. Report the **angle** alongside any cosine, or an explicit
  "what would count as aligned here".
  (2) **The mean cosine between random vectors is 0**, not `1/√H` — `1/√H` is the *per-pair standard deviation*.
  Never quote it as a floor the mean should sit at; for a mean, use an **empirical shuffled-pair control**.
  (3) **A random vector already has `√(d/H)` of its norm in any d-dimensional subspace** — so a "small"
  projection fraction can be at or *below* chance. Always report the chance level, and when `H` varies across
  the comparison, plot the **enrichment** `value / chance`, never the raw fraction (the raw version manufactures
  a trend that is entirely the moving chance level).
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
