# directions/ — Candidate Experiments Backlog (index)

Open questions and candidate experiments. Each direction is a self-contained brief a fresh
session can be pointed at: "read `directions/<x>.md` and execute."

Brief conventions and the full lifecycle: `../../harness/WORKFLOW.md`.
Template for a new brief: `../../harness/templates/direction.md`.

## Tagging

Every direction is tagged to keep the backlog honest about novelty:

- `[in-frame]` — a variation or extension within the current approach.
- `[reframe]` — changes the question or a premise (rarer, higher value). If the backlog fills
  with `[in-frame]` items, that is a frame-lock signal: deliberately go find a `[reframe]`.

## Status lifecycle

`proposed` → `active` → `in progress` → `done` / `dropped`

**Only `active` is Sevan's call** — choosing what to work on next is steering. *(Changed
2026-08-17: the agent may now set `done` once the work is finished **and** written into
`findings/`. Previously `done` required human confirmation, which — like the old promotion
gate — became a queue rather than a check.)*

On `done`, move the file into `directions/done/` to keep the active backlog clean.

**Every brief must be cold-start runnable** — it runs from a fresh session with no live state
from another notebook. Include a **Bootstrap** section naming exactly what to load or compute.
**Define every metric and threshold**, state the decision rule for any binary question, and
mandate a control.

## Backlog

_Statuses normalized 2026-08-17 against the artifacts on disk (scratch notes, notebooks, run
directories); several briefs carried no status line at all._

| Direction | Tag | Sub-Q | Status |
|---|---|---|---|
| [`editability-scaling-sweep.md`](editability-scaling-sweep.md) | `[reframe]` | 1, 3 | **planning**, 2026-08-23 — the programme: when does an editable world model emerge, and does the environment move the gate? |
| [`discworld-at-scale.md`](discworld-at-scale.md) | `[in-frame]` | 3 | **superseded**, 2026-08-22 — trigger did NOT fire; editability does not emerge with data. 10× rungs ran anyway (near-free) |
| [`our-architecture-on-othello.md`](our-architecture-on-othello.md) | `[reframe]` | 3 | **in progress**, 2026-08-22 — ladder complete; **environment control shows the world is the variable** (`othello_arch/`) |
| [`othello-architecture-on-discworld.md`](othello-architecture-on-discworld.md) | `[reframe]` | 3 | **proposed — fully specified**, 2026-08-21; awaiting **one** call (pilot `max_epochs`) |
| [`edit-direction-causality.md`](edit-direction-causality.md) | `[in-frame]` | 3 | proposed — follow-up to `latent_linearity`, 2026-08-19 |
| [`curvature-metric-normalization.md`](curvature-metric-normalization.md) | `[in-frame]` | 1 | proposed — **owed fix**, see `../GOTCHAS.md` |
| [`orthogonal-edits.md`](orthogonal-edits.md) | `[reframe]` | 3 | proposed |
| [`endogenous-action-rssm.md`](endogenous-action-rssm.md) | `[reframe]` | 2, 3 | proposed |
| [`multistep-objective-rssm-pure-overshoot.md`](multistep-objective-rssm-pure-overshoot.md) | `[in-frame]` | 1, 2, 3 | held — validity re-run |
| [`learn-to-edit.md`](learn-to-edit.md) | `[reframe]` | 3 | in progress — superseded in practice by the trained-editor work |
| [`history-editing.md`](history-editing.md) | `[in-frame]` | 3 | executed 2026-08-13 |
| [`hidden-size-sweep.md`](hidden-size-sweep.md) | `[in-frame]` | 1, 3 | executed 2026-07-30 · extended 2026-08-13 |
| [`transformer-world-state.md`](transformer-world-state.md) | `[reframe]` | 1, 2, 3 | executed 2026-08-04 |
| [`delta-h-analysis.md`](delta-h-analysis.md) | `[in-frame]` | 3 | executed 2026-08-03 |
| [`encoder-space-editing.md`](encoder-space-editing.md) | `[in-frame]` | 3 | executed 2026-07-30 |
| [`noise-ablation.md`](noise-ablation.md) | `[in-frame]` | 1, 2 | executed 2026-07-30 |
| [`trained-editability.md`](trained-editability.md) | `[in-frame]` | 3 | executed 2026-07-30 |
| [`endogenous-action-interactive-world.md`](endogenous-action-interactive-world.md) | `[reframe]` | 2, 3 | executed 2026-07-28 |
| [`action-space-object-individuation.md`](action-space-object-individuation.md) | `[reframe]` | 2, 3 | executed 2026-07-17 |
| [`action-conditioned-structure.md`](action-conditioned-structure.md) | `[reframe]` | 2, 3 | executed 2026-07-16 |
| [`counterfactual-history-state.md`](counterfactual-history-state.md) | `[in-frame]` | 3 | executed 2026-07-16 |
| [`multistep-steering.md`](multistep-steering.md) | `[in-frame]` | 3 | executed 2026-07-16 |
| [`multistep-objective-rssm.md`](multistep-objective-rssm.md) | `[in-frame]` | 1, 2, 3 | executed 2026-07-16 |
| [`multistep-prediction-objective.md`](multistep-prediction-objective.md) | `[reframe]` | 1, 2, 3 | executed 2026-07-16 |
| [`latent-dit-vae.md`](latent-dit-vae.md) | `[reframe]` | 1, 3 | executed 2026-08-11 |
| [`master-editability-notebook.md`](master-editability-notebook.md) | `[in-frame]` | — | executed 2026-07-08 (living notebook) |
| [`diagnostic-corrections.md`](diagnostic-corrections.md) | `[in-frame]` | 1, 2, 3 | executed 2026-07-08 |
| [`canonical-state-editing.md`](canonical-state-editing.md) | `[reframe]` | 2, 3 | executed 2026-06-24 |
| [`manifold-geometry-diagnostic.md`](manifold-geometry-diagnostic.md) | `[in-frame]` | 1 | executed 2026-06-24 |
| [`geodesic-walk.md`](geodesic-walk.md) | `[in-frame]` | 3 | executed 2026-06-24 |
| [`pca-component-position-analysis.md`](pca-component-position-analysis.md) | `[in-frame]` | 1, 3 | executed 2026-06-23 |

**"executed" ≠ `done`.** These have results in `scratch/` and (as of 2026-08-17) entries in
`findings/`. Moving them to `done/` is a tidy-up pass worth doing once Sevan has read the
backfilled findings; several may also deserve follow-up briefs rather than closure.

## Parked ideas (not yet written up)

- Density-regularized gradient edits.
- VAE / normalizing-flow explicit manifold model.
- "Edit via the dynamics" — feed a post-edit observation (or an interpolated observation
  prefix) and see if even one observed frame shifts the rollout. Leans on the renderer and
  risks violating the pure-latent-intervention framing; do carefully, and only after the
  cheaper latent edits are exhausted.
- Probe-direction calibration: *why* does the position-probe direction have ~10× smaller
  data-σ than top PCA directions? Capacity, dataset range, or geometry?
- Trained editors at larger scale, and whether the learned `(h, start, target) → Δh` map
  transfers across world models.
