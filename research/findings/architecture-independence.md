# Finding: Architecture-independence of the non-canonical / readable≠controllable pattern

*Cross-cutting (sub-Q1+2+3) — does the editability failure generalize beyond the GRU?*
Models: GRU `3_dset3_gru_persistentids_inview_400epochs` (256) and refined RSSM
`4_dset4_refined_best` (det 256 + stoch 64 = 320, `sample=False`), dataset `4_fixed_refl_inview`.
Notebooks: `notebooks/experiments/editability/00_master_editability.ipynb`,
`notebooks/experiments/editability/rssm_structure/rssm_state_geometry.ipynb`,
`notebooks/experiments/editability/diagnostic_corrections.ipynb`.

> **Scope (preliminary, 2026-07-09).** A comparison of **two specific trained checkpoints** — a GRU and
> a refined RSSM, both trained purely to predict observations (next-step / ELBO, no state supervision)
> on `dataset 4`. It says the *failure mode replicates across these two architectures as trained*; it is
> **not** a proof that no implicit architecture can be canonical, nor a claim about RSSMs in general
> (different KL weight / latent size / objective could differ). Two points is a line, not a law.

## Current understanding

> **Updated 2026-08-17.** Two additions change the shape of this concept. (1) **A causal
> transformer has two state objects** — the carried observation buffer and the recomputed
> residual stream — and they come apart, so "editability" was never a single property and every
> cross-architecture claim must name *which* state it concerns (2026-08-04). (2) The pattern now
> replicates on **four** architectures: GRU, RSSM, transformer, and a VAE + latent DiT, the last
> of which refutes the compression hypothesis (2026-08-11). It also survives a change to the
> *world* rather than the model — full 2D observability (see `editability.md`, 2026-08-11).
> The summary below predates all four and is otherwise unchanged.

### Previous synthesis (mutable summary)

The non-canonical / **readable ≠ controllable** editability failure found in the GRU is **not specific
to the GRU**: a refined, KL-regularized RSSM with an explicit stochastic latent reproduces every part
of it, and the structured prior delivers **no** gain in canonicity or controllability.

- **World-state lives in the deterministic recurrent core, not the stochastic latent.** Linear
  position R²: RSSM det-only 0.84 ≈ full 0.85 ≫ stoch-only 0.59; the stochastic `s` is a low-rank
  (6/64 dims @90%) uncertainty code. This **refutes** the "the stochastic latent captures the compact
  world state" expectation — the RSSM behaves like a GRU with a small stochastic block bolted on.
- **Canonicity is on par, not worse.** Apples-to-apples fiber residual `‖block − g(pos,vel)‖/‖block‖`
  (MLP g): GRU `h` **0.337** ≈ RSSM **det-only 0.368**; the stochastic `s` (residual 0.891, legitimately
  not a function of (pos,vel)) inflated the naive full-320 number to 0.602. So the KL structure buys
  **no** extra canonicity — the deterministic core carries the same ~35%-non-canonical code as the GRU.
- **Velocity is nonlinear-instantaneous in both** (single-frame MLP ≈ 2-frame MLP; not temporal).
- **Editing fails identically, and readable≠controllable is if anything sharper.** A perfect-readout
  pseudoinverse edit hits the probe target exactly (readout RMSE 0.000) yet moves the observation
  **0.0%** of a swap and reverts in one step; the global-manifold edit moves obs 36.5% of a swap
  (≈ GRU 37%) but in the wrong/scrambled direction. (Mechanism differs: the RSSM position-probe
  direction is decoder-inert, σ 0.017; the GRU's was matched-magnitude generative — same outcome,
  different route.)

**Why it matters.** Lifts the editability story from "a fact about one GRU" toward "a property of
implicit predictive world models as currently trained," and shows a popular structured-latent remedy
does not fix it — motivating the RESEARCH.md thesis that canonicality likely needs *explicit* physical
scaffolding, not stochastic latents. (Preliminary — two checkpoints.)

## Log

### 2026-08-11 — The compression hypothesis is refuted (VAE + latent DiT) · `observed`

**Evidence:** `scratch/2026-08-11-latent-dit.md` ·
`notebooks/experiments/editability/latent_DiT/` (+ `LATENT_DIT_RUNS.md`) ·
`input_grad_steering/input_grad_steering_latent_dit.ipynb` · `directions/latent-dit-vae.md` ·
new code `pim/world_models/vae.py`, `scripts/train_vae.py`, `pim/world_models/latent_dit/`,
`scripts/train_latent_dit.py`.

Built at Sevan's direction ("treat the latent DiT as a wholly separate architecture"): a
per-frame MLP VAE with a continuous vector latent and LDM-style tiny KL, then a DiT core over
that latent implementing `HiddenStateModel` in observation space.

**The hypothesis was that a compressed latent — where nearly every direction should be
semantic — would be grabbable. It is not.** The failure survives compression, on a model that
is *more* readable and whose decoder is an extra unconditional projector to valid observations.

**What remains is belief dynamics:** the ghost is carried by the clean context frames, and only
evidence consistent *across the window* removes it. Practical corollary recorded at the time:
stop searching for a better *space* to take the gradient in; search for objectives that
synthesise multi-frame, velocity-consistent evidence.

**Owed / not done:** no latent-GRU control (the brief's optional arm, which would separate
bottleneck effects from diffusion effects); z=8 and window-2 latent runs exist and passed the
quality gate but were never put through the editors.

---

### 2026-08-04 — Transformers: the readable state is not the carried state · `replicated` ★-candidate

**Evidence:** `scratch/2026-08-04-transformer-world-state.md` ·
`notebooks/experiments/editability/transformers/transformer_world_state.ipynb`
(+ `TRANSFORMER_RUNS.md`) · new code `pim/world_models/transformer/model.py`,
`scripts/train_transformer.py`, `tests/test_transformer.py` · runs `W2`/`W4`/`W16` (window
2/4/16), all `d_model=256` (**matched to the GRU's hidden size** so geometry chance levels are
comparable), 4 layers, 3.23M params each — window is the only variable.

**This retires a premise every prior result silently assumed:** that the model has exactly one
state, a vector `h` that is both what it carries between steps and what a probe reads. "Edit the
world state" is only well-posed because those are the same object.

A causal transformer has **two** state objects and they come apart:

| | what it is | carried? | history-dependent? |
|---|---|---|---|
| **carried state** — the observation buffer | the frames you must supply to reproduce the model's own next prediction | **yes** | no — each slot is one frame |
| **readable state** — residual stream at (layer ℓ, current position) | what attention has mixed at this position | **no** — recomputed every step | **yes** |

So "does a latent write stick?" splits into two questions with different expected answers, and a
**decayed activation write here is architecture, not the GRU's reversion failure.**

**Why it matters:** it means "editability" was never a single property. Any cross-architecture
claim must say *which* state it is about. It also gives a second architecture where the write
surface is explicit — later exploited by the input-gradient work, which found the transformer's
buffer is a stronger write channel than one GRU recurrent update (+0.27 vs −0.01 on the render
oracle).

---

### 2026-07-08 — Det-only fiber refit: RSSM det core ≈ GRU (KL buys no canonicity) · `established`
`diagnostic_corrections.ipynb`. RSSM det-only fiber residual 0.368 ≈ GRU 0.337; full-320 0.602 was the
stochastic `s` (0.891) inflating it. Retires the earlier "RSSM ~2× less canonical (0.605 vs 0.347)"
reading — that was a measurement artifact of scoring the full 320-d state. On the deterministic core the
two architectures are on par.

### 2026-07-02 — RSSM replicates the GRU editability failure · `established`
`rssm_state_geometry.ipynb`. Geometry replicates (34/320 @90%, tangent 65° vs GRU 56°); position in det
`h` not stochastic `s`; velocity temporal-signature replicates; pseudoinverse edit readout-exact but
obs-0.0% and reverts in one step; global-manifold 36.5% of swap. KL-structured prior did not produce a
more canonical or controllable representation.
