# 2026-08-21 — Is latent object-composition learned? A random-init baseline

Notebook `notebooks/experiments/editability/latent_linearity/random_baseline.ipynb`, library
`composition_lib.py` beside it. N = 256 episodes from `datasets/4_fixed_refl_inview/edits.h5`,
edit frame 20, displacement scales {1.0, 0.5, 0.25, 0.125}, two random seeds. Runs in ~10 s after
a ~2 s render. Results in `runs/latent_linearity/random_baseline_results.json`, figure
`runs/latent_linearity/figures/fig1_composition_random_baseline.png`. **No models trained.**

Prompted by Sevan (2026-08-21): *"see if it is also true for a randomly initialized untrained
model… both when we have a linear encoder/decoder, and a nonlinear encoder/decoder. Be clear about
which model you test this on."*

## The claim under test

`delta_h_analysis` §7 (2026-08-03): in latent space `[move obj0] + [move obj1]` recovers most of
`[move both]` — composed cosine **+0.873**, relative residual 0.39–0.69, `‖composed‖/‖direct‖`
1.13 — recorded at the time as *"the strongest positive in the editability thread so far"*. The
2026-08-05 note found the **decode-space** version to be an affine-decoder artifact and explicitly
recorded the **state-space** readout as unaffected. This note re-examines the state-space readout.

## Why a random baseline is the right control, not a formality

Additivity is a **first-order Taylor property of any smooth map**:
`f(x + δ_A + δ_B) ≈ f(x) + Jδ_A + Jδ_B = f(x+δ_A) + f(x+δ_B) − f(x)`, error second-order in ‖δ‖.
Training has nothing to do with it. Random weights give a smooth *deterministic* function, not a
random one, and at standard init a small perturbation rarely flips ReLU signs, so the network acts
as a fixed linear operator in that neighbourhood. Any additivity metric therefore has a large
architectural floor that must be measured, not assumed away.

## Models tested (all discworld, dataset 4, 2 objects)

| tag | what |
|---|---|
| `TRAINED linear enc+dec` | the thread's trained GRU with the affine encoder/decoder |
| `TRAINED nonlinear enc+dec` | the trained GRU with the `Linear+ReLU` encoder and nonlinear decoder (the `nonlinear_gru` family) |
| `RANDOM s0/s1 …` | **identical config, untrained weights**, seeds 0 and 1, both families |

## Three flaws found and fixed before the numbers below were trusted

1. **Uniform displacement direction.** A first pass nudged both objects the same way, which
   inflates the null. Replaced with the **real teleport edit set** from `edits.h5`.
2. **Shuffled floor permuted only one delta.** That leaves the un-permuted delta's overlap with
   `Δ_AB` intact and reads +0.38. Permuting **both** gives the true floor, **+0.026**.
3. **Displacement measured from `positions[ef]`.** On `edits.h5` the teleport is *already in the
   data*, so that measurement is zero for each episode's own edit object — silently nulling half
   the single-object edits. Fixed by reconstructing the **un-teleported** ballistic world and
   measuring from there. An assertion now guards it.

## The observation ceiling — the floor no latent can beat

The two objects share rays, so the **render itself** is not additive. Ceiling (relative residual of
`obs_A + obs_B − obs_base` against `obs_AB`), by scale: **0.406 / 0.373 / 0.285 / 0.207** at
s = 1.0 / 0.5 / 0.25 / 0.125. Every latent number below is read against the ceiling at its own scale.

## Result 1 — by cosine, composition is essentially architectural

Real teleport scale, N = 256:

| model | composed cos ↑ | *triviality* `cos(Δ_A,Δ_AB)` | relative residual ↓ | ‖comp‖/‖direct‖ | shuffled floor |
|---|---|---|---|---|---|
| *observation ceiling* | — | — | **0.406** | — | — |
| TRAINED linear | +0.904 | *+0.576* | 0.451 | 1.121 | +0.026 |
| RANDOM s0 linear | +0.890 | *+0.418* | 0.505 | 1.152 | +0.027 |
| RANDOM s1 linear | +0.890 | *+0.437* | 0.515 | 1.164 | +0.033 |
| TRAINED nonlinear | +0.835 | *+0.490* | 0.646 | 1.203 | +0.018 |
| RANDOM s0 nonlinear | +0.853 | *+0.393* | 0.605 | 1.189 | +0.032 |
| RANDOM s1 nonlinear | +0.852 | *+0.418* | 0.621 | 1.208 | +0.007 |

Training moves the composed cosine by **+0.014** on the linear family and **−0.018** on the
nonlinear one — where the random net is *higher* than the trained one. The metric is anchored:
trained linear reproduces §7's +0.873 and its 1.13 norm ratio.

**§7's headline cosine does not distinguish a trained world model from an untrained one.**

## Result 2 — the metric is not vacuous, though

The triviality baseline `cos(Δ_A, Δ_AB)` sits at **+0.39 … +0.58**, far below the composed cosine,
and the one-delta residual is 0.80–0.92 against a composed residual of 0.45–0.65. Adding the second
delta genuinely improves both direction and magnitude. *"Composition explains the double edit"* is
true. *"Training is why"* is what fails.

## Result 3 — against the renderer's own limit, training does separate

Excess relative residual over the observation ceiling — the non-additivity the **network** adds,
the only part attributable to the model rather than the renderer:

| scale | 1.0 | 0.5 | 0.25 | 0.125 |
|---|---|---|---|---|
| *observation ceiling (absolute)* | *0.406* | *0.373* | *0.285* | *0.207* |
| TRAINED linear | **+0.046** | **−0.004** | **−0.019** | **+0.004** |
| RANDOM linear (mean of 2 seeds) | +0.104 | +0.064 | +0.037 | +0.028 |
| TRAINED nonlinear | +0.241 | +0.138 | +0.077 | +0.073 |
| RANDOM nonlinear (mean of 2 seeds) | +0.208 | +0.150 | +0.121 | +0.109 |

**The trained linear model tracks the renderer's own additivity ceiling — excess ≈ 0 at every
scale, twice going slightly negative.** Random init of the same architecture cannot: it adds
0.03–0.10, and the gap widens as the displacement shrinks, i.e. exactly where the Taylor argument
predicts everything should converge. On the nonlinear family the separation is present but weak
and only appears at small scales (0.073 vs 0.109 at s = 0.125); at s = 1.0 the trained model is
*worse* than random.

## Where this leaves the claim

- **Dead in the strong form.** "The latent superposes object edits, and that is a learned property"
  is not supported by the composed cosine, which is what §7 led with. A randomly initialised
  network of the same architecture matches it.
- **Alive in a restated form,** for the **linear** family only, and only against the right
  baseline: *the trained model's latent is as additive as the observation it is trained to
  predict, and no less* — a ceiling-tracking claim, not a superposition claim. Random init misses
  that ceiling by 0.03–0.10.
- The nonlinear family does not support even the weak form at the real edit scale.

## Loose ends

- Only one trained checkpoint per family. The separation in Result 3 is small enough that
  checkpoint-to-checkpoint variance could matter; two random seeds bracket the *random* side only.
- The ceiling is computed in observation space and the residual in latent space. They are
  comparable as *ratios*, which is the only sense in which they are compared here, but a
  latent-space ceiling would be a stronger control if one could be defined.
