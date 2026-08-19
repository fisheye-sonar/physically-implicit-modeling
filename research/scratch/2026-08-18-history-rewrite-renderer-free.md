# 2026-08-18 — The Othello write applied to the WHOLE history, renderer-free: still fails. Frame validity is the barrier, not consistency

**Supersedes the reading in** `2026-08-18-history-rewrite.md` (same day; its numbers stand, its
interpretation does not). **Thread:** `notebooks/experiments/editability/othello_gpt/history_rewrite.ipynb`
(+ `history_edit.py`). **Model:** `runs/transformers/W16`. **Data:** `datasets/4_fixed_refl_inview`,
edits split, `ef=20`, `K=15`, **N=256** — same episodes as the Othello port.
**Findings updated:** `findings/editability.md` (2026-08-18, `replicated`).

## What Sevan asked for, and what I built the first time

The ask: apply the Othello MLP edit **over the entire history** so the previous observations change,
using the model's own decoded positions, **no renderer, no ground truth, no oracle**. My first pass
substituted a different design — decode positions, then **re-render each frame with the simulator** —
which smuggled the observation function back in. That arm is retained as a labelled reference; it is
not the answer.

## The two renderer-free variants

1. **Activation history edit** — the paper's Figure 2C schedule widened from one timestep to **every
   window position**: write the residual stream at position `t` so the probe there reads
   `decoded(t) + δ`, at residual point `L_s` and every point after it.
2. **Observation history edit** — the same MLP write pushed to the frames themselves. Residual point
   0 is exactly `relu(Linear(obs))`, so `probe₀ ∘ embed` is differentiable from observation to
   position; gradient descent on the observation changes the frames directly.

Both use only the model, the probe, and `δ` from the decoded track.

## Results

| arm | renderer-free | read-out after | EI step 0 | step 14 | fidelity |
|---|---|---|---|---|---|
| Unsteered | — | — | −0.684 | −0.439 | 1.000 |
| Latent write · single frame | ✅ | 0.018 | −0.538 | −0.428 | 0.994 |
| **Activation history edit @0** | ✅ | **0.008** | **−0.544** | −0.428 | 0.995 |
| **Observation history edit** | ✅ | 0.079 | **−0.459** | −0.302 | 1.001 |
| *Rebuilt history* | ❌ render | — | *+0.626* | *+0.351* | *0.674* |
| *Reconstruction control δ=0* | ❌ render | — | *−0.569* | *−0.375* | *1.039* |

**1. Widening the write to the whole history buys 0.006 index points.** Read-out converges *better*
than the single-frame write (3.64 → **0.008**) and the outcome is −0.544 vs −0.538. **The hypothesis
is cleanly refuted.**

**2. Applied-layer pattern unchanged**: −0.544 (point 0) → −0.637 (point 4), same monotone ordering,
same structural reason.

**3. Writing the observations themselves helps but not qualitatively**: −0.459, and it is the only
renderer-free arm whose effect survives the rollout at all (−0.302 vs unsteered −0.439).

**4. The entire difference is frame validity.** The observation edit and the render reference use the
**same δ**, the **same per-frame targets**, and the **same channel** — and differ by **1.085 index
points**. The only difference is how the frames were produced.

**5. Off-manifold, and not from timidity.** The MLP edit changes the frames **less** than the render
does (relative change 0.539 vs 0.881) while scoring far worse. Real observations here are strongly
saturated — **39.3%** of pixels sit at the intensity rails; after the MLP write only **2.6%** do. Fig 2
shows it directly: broadband vertical striping with the plateau structure destroyed, exactly the
"adversarial fuzz" `input_grad_steering` found for single-frame input gradients, now across 20 frames.

## Reading (not established)

The corrected account: the barrier is **not** coherence across frames (refuted by 1) and **not**
precision of the write (read-out converges to 0.008). It is that a probe-derived write **cannot
synthesise a valid observation**. The probe pins 4 read-out dimensions and leaves the other 124 free;
essentially every member of that solution set is not a picture of this world.

Third independent route to the same place: `2026-08-05-observation-space-geometry` (the required
change is near-orthogonal to what injection applies, **in observation space, with no model**),
`2026-08-11-input-grad-steering` (broadband fuzz on one frame), and now a whole history.

**Standing negative, sharpened:** probe-derived writes fail on **every** surface tried — `h`, encoder
port, VAE latent, residual stream at one frame, residual stream at all frames, and raw observations.
What works needs a **generative model of observations**, which the render arm supplies from outside.

## Owed / next

- The obvious follow-up: replace the simulator's renderer with a **learned** `positions → observation`
  map fit on the training split. If that recovers most of the +0.626, the missing ingredient is
  "any valid observation model", not "the true one" — and that is a mechanism a model could own.
- The model's own decoder is such a map but cannot be used directly: to render "object k at p" you
  would first need a state that decodes to it, which is the latent-editing problem again.
- Untested: an observation edit optimised through a deeper residual point back-propagated to the input.
