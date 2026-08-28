# OURS_ON_OTHELLO_RUNS.md — canonical RUN REGISTRY

**The single source of truth for what every run code in
`notebooks/experiments/editability/ours_on_othello/` means.** Per `harness/STYLE.md` §5: no
deliverable may use a run code without copying its row into its own definitions table, and
**figures use the descriptive label, never the raw code**. Adding a run means adding its row
here in the same commit.

Origin: Sevan, 2026-08-21 — *"take our transformer which we trained on discworld, adapt it to
match the input and output scheme of the OthelloGPT setup, train it similar to how we trained on
discworld, and see if it is still editable."* The mirror of `../othello_transfer/`, and the
cheap half of the 2×2 that `../../../../research/directions/othello-architecture-on-discworld.md`
(run A) completes.

| | their world (Othello) | our world (discworld) |
|---|---|---|
| **their architecture** (8 blocks, `d_model` 512, full causal) | editable — `../othello_transfer/`, 2026-08-20/21 | run A, **not run** |
| **our architecture** (4 blocks, `d_model` 256, RoPE, banded) | **this thread** | not editable — 2026-08-04 … 2026-08-21 |

---

## The model

| code | descriptive label | what changed from `W16` | params |
|---|---|---|---|
| `OthelloTransformer` | **our transformer · Othello I/O** | `Linear(128, 256)+ReLU` → `nn.Embedding(61, 256)`; `Linear(256, 128)` → `Linear(256, 61)`; MSE → cross-entropy. **Nothing else.** | 3,190,845 |

Held fixed from [`../transformers/TRANSFORMER_RUNS.md`](../transformers/TRANSFORMER_RUNS.md)'s
`W16`: `d_model` 256, 4 layers, 4 heads, `mlp_ratio` 4.0, pre-norm blocks, RoPE, banded-causal
attention. **5 residual points** (ℓ = the stream after ℓ blocks, `n_layers + 1`), against
Othello-GPT's 9.

> **Sevan's call, 2026-08-21:** replace the *whole* encoder, `Linear` **and** `ReLU` — a ReLU on
> an embedding lookup would zero half the dimensions for no reason. This mirrors the "no ReLU
> after the input projection" decision already pinned for run A.

**The band width is not a limitation, and this is the fact the whole thread rests on.**
`state_span = n_layers·(window−1)+1`, so `window` 16 gives **61** — a full 60-move game, exactly.
(`window` 4 would give 13 and the test would be void.) What the band costs is *directness*: their
model attends from the last move to any earlier move in one hop at all 8 layers; ours routes
through up to 4 banded hops. `window` 40 (span 157) is run alongside to remove that difference at
**zero** extra compute — measured 25,018 vs 25,001 games/s, because the band mask is applied to
the full T×T attention.

| code | window | `state_span` | one hop from move 59 reaches |
|---|---|---|---|
| `w16` | 16 | 61 | moves 44–59 (whole game in 4 hops) |
| `w40` | 40 | 157 | moves 20–59 (whole game in 2 hops) |

---

## The scale ladder — the reason this thread exists in this shape

Sevan, 2026-08-21: *"I don't really like how we are comparing very different data sizes … I don't
really know if success came from scale."*

`runs/transformers/W16` trained on **90,000 episodes × 40 frames = 3.6M unique frames**, batch
256, lr 1e-3, 5% warmup + cosine over 300 epochs = **95,100 optimiser steps**, with the
checkpoint taken at best val. Training at 20M Othello games would be **222× the unique sequences
and 333× the unique tokens** — a success there would say nothing about our setting.

So **every rung runs the same 95,100 steps at batch 256 with the identical schedule**, and only
the pool it samples from changes. Anything that moves across M → D is data diversity, not compute.

| code | unique games | epochs over its pool | unique tokens | vs `W16`'s 3.6M frames | steps |
|---|---|---|---|---|---|
| `M` | 90,000 | 300 | 5.4M | **1.5×** — matched | 95,100 |
| `L1` | 1,000,000 | 27 | 60M | 17× | 95,100 |
| `L2` | 5,000,000 | 5.4 | 300M | 83× | 95,100 |
| `D` | 20,000,000 | 1.35 | 1.2B | 333× | 95,100 |
| `F` | 20,000,000 | 8 passes | 1.2B | 333× | **8× the steps** |

⚠ **`F` is not a scale datapoint.** It is the only arm with more compute and exists solely to
answer "can this architecture do Othello at all". Never quote it in the same row as M–D.

⚠ **M matches on sequences, not tokens** (90k ↔ 90k), so its token count is 1.5× discworld's
because Othello games are 60 long and discworld episodes 40. Matching tokens instead would mean
60k games — a 1.5× discrepancy on an axis spanning 333×, so sequences is the honest match.
Sevan's call, 2026-08-21.

Run codes are `<rung>_w<window>`: `M_w16`, `M_w40`, `L1_w16`, … Checkpoints at
`runs/ours_on_othello/<code>/{best_model.pt, latest.pt, metrics.jsonl, config.json}`.

---

## Data

All games from **their** generator, `data.othello.get_ood_game`, reached through
`../othello_transfer/othello_data._one_game`. It picks `random.choice(possible_next_steps)` —
uniform over the legal set — which is what makes the uniform-over-legal reference in `edit_index`
the *true* conditional distribution of this data rather than an approximation.

A game is a pure function of its **index** (`random.seed(seed·1_000_003 + i)`), so splits are
disjoint index ranges of one seed and the ladder rungs are strict prefixes — `L1 ⊂ L2 ⊂ D`, not
four independent draws.

| split | index range | n | role |
|---|---|---|---|
| train pool | `[0, 20M)` | 20,000,000 | every rung takes a prefix; 10% held out for val by `--val-fraction 0.1`, matching `train_transformer.py` |
| test | `[90M, 90.01M)` | 10,000 | **the OOD generalisation gate** — never seen in training |
| probe harvest | `[91M, 91.02M)` | 20,000 | the same 20k scale `../othello_transfer/` used |
| intervention benchmark | — | 1,001 | Li et al.'s `intervention_benchmark.pkl`, checked into their repo |

⛔ `corpus.assert_disjoint` **hashes the actual token rows** and fails hard rather than trusting
the arithmetic. The analogous mistake is already pinned in run A's brief: generating an eval
split from an index range the training corpus also covers silently turns held-out data into
training data.

---

## Held-out gates — and why raw accuracy is uninterpretable here

Computed at `best_model.pt` on the **test** split, never on training games. Arm M sees 90k games
300 times, so training loss reaches wherever memorisation can take it; the question is whether
what it learns *generalises*, exactly as `W16` overfits discworld's training loss (best val at
epoch ~40 of 300) and is still a good predictor on the held-out split.

| gate | formula | units | better |
|---|---|---|---|
| **legal-move mass** | mean over positions of Σ_{m ∈ legal(t)} p(m \| history) | 0…1 | ↑ |
| **top-1 legal rate** | fraction of positions where `argmax p` is legal | 0…1 | ↑ |
| **top-1 accuracy** | fraction where `argmax p` is the move actually played | 0…1 | ↑ |
| **held-out CE** | cross-entropy in nats over non-pad positions | nats | ↓ |

⚠ **Top-1 accuracy and CE have a hard ceiling that is not 1 and not 0.** The generator draws
uniformly from the legal set, so the Bayes-optimal predictor scores `mean(1/|legal|)` and
`mean(log|legal|)`. Both are computed on the same games and reported as `bayes_top1` / `bayes_ce`;
**the number that means something is the excess CE over `bayes_ce`**, not the CE.

Reference, cited not recomputed: **Li et al. report 99.98% legal-move mass** for their
25.3M-parameter model (arXiv:2210.13382).

---

## Probes and editors — imported, never re-derived

Everything comes from `../othello_transfer/` unchanged, which is the same code that produced the
numbers on Li et al.'s own checkpoint. `evaluate.attach()` only tells those modules our model has
5 residual points instead of 9 and repoints the probe cache so a 5-point grid can never be served
from the 9-point one.

| axis | levels |
|---|---|
| target | `state` = absolute colour (white/blank/black — **theirs**) · `mine` = blank/mine/theirs relative to the player to move |
| family | `MLP 512 hidden` (**ours**) · `MLP 128 hidden` (**theirs**) · `linear` |
| split | `frame` = pooled rows (**their** convention) · `sequence` = whole games (**our** convention, `harness/ANALYSIS.md` §2) |
| residual point | 0…4 |

Fitting: Adam, 200 epochs, batch 4096, lr 1e-3, 20% held out, inputs standardised inside the
probe. 2 × 3 × 2 × 5 = **60 probes** per checkpoint, never shared across points.

| editor | write | source |
|---|---|---|
| **Layerwise MLP Grad Steering @L_s** | `x ← x − α ∂L/∂x` at the last position, at `L_s` and every point after | `othello_probe.make_intervention_hook`, unmodified |
| **Nanda Direction Addition** | `x ← x + α·p_d` | `linear_intervention.run(mode="add")` |
| **Nanda, target − current** | `x ← x + α·(p_tgt − p_cur)` | same, `subtract=True` |
| **Pseudoinverse Injection** | `Δ = A⁺(target − (Ax+b))` | `pim.editors.probe_steering.inject_state`, unmodified |

Each is run **at every point at once** and **at each single point**, because 2026-08-21 showed
those differ by 28× on their model.

Metrics: `../METRICS_AND_EDITORS.md` §6 — Li error vs post-flip, Li error vs pre-flip (the
guard), Edit Index union and symdiff, legal mass, ‖Δx‖/‖x‖. Definitions are restated in the
notebook, per `harness/STYLE.md` §5.

---

## Cited, not recomputed

| number | source |
|---|---|
| Li et al.: legal-move mass 99.98%; null intervention 2.68 → best 0.12; nonlinear probe 1.7% error, linear 20.4% | arXiv:2210.13382 |
| Nanda et al.: linear-direction addition 0.10; their null 2.723 | arXiv:2309.00941 §4.1, Table 2 |
| On **their** model with **our** code: null 2.723 → gradient editor 0.016 / EI +0.656; Nanda addition 0.062 / +0.603; pseudoinverse 0.052 / +0.697 at point 5 | `../othello_transfer/probe_transfer.ipynb`, `controls.ipynb` |
| discworld `W16`: 90,000 train episodes, 95,100 steps, best val at epoch ~40, position probe R² 0.798 linear / 0.9349 MLP | `../transformers/TRANSFORMER_RUNS.md`, `../othello_gpt/othello_gpt_probing.ipynb` |
| discworld `W16` editability: unsteered Edit Index −0.684; best probe-derived write −0.194; single-point pseudoinverse inert at every point and every α | `../transformers/transformer_world_state.ipynb`, `../othello_transfer/pinv_alpha_discworld.py` |

## No pretrained weights are reused

Sevan's call, 2026-08-21: **fresh init**. The encoder and head must be replaced anyway, so only
the blocks could transfer, and whether discworld-trained blocks transfer to Othello is a
different question that would confound this one.
