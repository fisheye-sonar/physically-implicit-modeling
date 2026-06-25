# directions/ — Candidate Experiments Backlog (index)

Open questions and candidate experiments. **Agent proposes, human disposes.** An
agent may add or refine a direction file; only Sevan marks one **active**. Each
direction is a self-contained brief a fresh session can be pointed at:
"read `directions/<x>.md` and execute."

## Tagging

Every direction is tagged to keep the backlog honest about novelty:

- `[in-frame]` — a variation/extension within the current approach.
- `[reframe]` — changes the question or a premise (rarer, higher value). If the
  backlog fills with `[in-frame]` items, that's a frame-lock signal: deliberately
  seek a `[reframe]`.

Status lifecycle:
- `proposed` — drafted (agent or human).
- `active` — **human** marks it the current focus.
- `in progress` — an agent has executed it; results sit in `scratch/` awaiting review.
  *(An agent may set this; it may NOT skip to `done`.)*
- `done` — **human**-confirmed complete (result promoted to a finding, or intentionally
  closed). On marking done, move the file into `directions/done/` to keep the active
  backlog clean.
- `dropped` — abandoned.

Marking `done` is a commitment, so it's a human power — same line as promotion to
`findings/`. Agents stop at `in progress` and surface for review.

## Brief conventions (every direction file must satisfy)

- **Cold-start runnable.** A brief must run from a *fresh kernel* — never assume
  variables computed in another notebook's live session. Include a **Bootstrap**
  section naming exactly what to load/compute (checkpoint, data, probe, subspace,
  warm-up, helper fns) from the paths in its Context section. *(Surfaced by the
  2026-06-23 PCA cold-start test: the brief said "reuse `states_tf` …" but the
  test-hygiene rule forbids opening the notebook that defines it — a direct
  contradiction.)* Until the shared setup helpers are factored into `pim/`, spell
  the bootstrap out; `notebooks/experiments/manifold_editing/pca_component_position.ipynb` is a
  working reference for the standard cold-start setup.
- **Define every metric and threshold.** If the brief asks a binary question
  ("is it selective?"), state the decision rule. If a magnitude is interpreted,
  define its units (e.g. σ = data-std vs PCA explained-variance) and mandate a
  **control/baseline** (e.g. the α=0 no-edit control the PCA renderer bonus needed).
- **Visualize, don't just tabulate.** Notebooks serve two readers: Sevan does the
  scientific judgment and reads best from *plots*; the agent self-verifies from
  *numbers*. Produce **both** — never tables alone. Crucially, visualize an effect in
  the space where it actually occurs: plot the **1D observations / waterfalls** under
  the perturbation, not only decoded-scalar positions. *(Surfaced 2026-06-23: the PCA
  decoded-position table said "both objects move," but the observation waterfall shows
  PC0 moving only the dim object — an effect invisible in the scalar table.)*

## Backlog

| Direction | Tag | Sub-Q | Status |
|---|---|---|---|
| [`canonical-state-editing.md`](canonical-state-editing.md) | `[reframe]` | 2, 3 | in progress |
| [`manifold-geometry-diagnostic.md`](manifold-geometry-diagnostic.md) | `[in-frame]` | 1 | in progress |
| [`pca-component-position-analysis.md`](pca-component-position-analysis.md) | `[in-frame]` | 1, 3 | in progress |
| [`geodesic-walk.md`](geodesic-walk.md) | `[in-frame]` | 3 | in progress |

## Parked ideas (not yet written up)

- Density-regularized gradient edits.
- VAE / normalizing-flow explicit manifold model.
- "Edit via the dynamics" — feed a post-edit observation (or an interpolated
  observation prefix) and see if even one observed frame shifts the rollout. Note:
  this leans on the renderer and risks violating the pure-latent-intervention
  framing — do carefully, and only after the cheaper latent edits are exhausted.
- Probe-direction calibration: *why* does the position-probe direction have ~10×
  smaller data-σ than top PCA directions? Capacity, dataset range, or geometry?
- RSSM comparison: rerun the editability battery on the RSSM stochastic+deterministic
  state and compare manifold/edit behavior to the GRU `[reframe]`-adjacent.
