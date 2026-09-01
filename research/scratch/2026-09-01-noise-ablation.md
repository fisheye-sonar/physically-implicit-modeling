# 2026-09-01 — Removing ALL noise does not make discworld editable

**The run.** `noise_ablation/L-dw-noiseless-20m`: dw-pn04 with `obs_noise_std = 0.0` AND
`position_noise_std = 0.0`, everything else byte-identical — Transformer-L (25,371,776
params), 20M sequences, 780k steps, the matched canonical recipe (AdamW 1e-3, wd 1e-4,
clip 1.0, batch 256, 2k warmup then constant, seed 0). Fresh seed block (base 30e9);
seeds cannot be paired with dw-pn04 (`always_in_frustum` consumes noise draws inside its
IC acceptance loop — 0/5 seeds matched, ~5-unit divergence).

## Prediction got 21.5x better. Editability did not move.

| | dw-pn04 | dw-noiseless |
|---|---|---|
| best val MSE | 0.022873 @ 660k | **0.001063** @ 770k |
| train-val gap | ~0 | +0.000085 |
| wall clock | 8.04 h | 8.03 h |
| unedited EI | −0.6998 | **−0.9237** |
| PI (best) | +0.175 · fid 1.11 | **+0.218** · fid 1.16 |
| ND (best) | −0.038 | −0.037 |
| GS (best) | −0.195 | −0.114 |

Every editor lands in the same place it did with noise: **PI marginally positive but
destructive** (fidelity 1.16 > 1 — the edited rollout ends FURTHER from the true
post-edit world than doing nothing; collateral 0.513 vs 0.118 unedited, a 4.3x
degradation), **ND and GS negative**. Against Othello's +0.61…+0.65 with clean guards,
discworld is not editable, and observation/position noise was not the reason.

## What the noise WAS responsible for

* **The prediction floor.** pn04's 0.0229 was ~82% irreducible observation noise
  (floor 0.0189). Remove it and the floor is 0; the residual 0.00106 is model error
  alone — inferring the initial state from a 128-ray quantised render, plus rollout
  drift. Prediction quality is therefore *not* what gates editability: a model 21.5x
  better at its own objective is no more editable.
* **The Edit Index scale.** The unedited floor moves −0.700 → **−0.924**, close to the
  theoretical −1. This confirms the ±0.82 compression documented on 2026-08-25 was an
  artefact of scoring a noise-trained model against clean renders. On this instance the
  index is nearly uncompressed, so the editors have MORE headroom to show a win — and
  still do not.
* **Linear decodability of position.** LIN skill drops 0.944 → 0.872 while MLP *rises*
  0.977 → 0.979. Same information, less linearly accessible. Velocity per-component is
  the sharpest case (o1·vy LIN 0.526 → 0.173, MLP 0.651 → 0.542). Worth a thought: with
  noise the model must average over jitter, which may push it toward a smoother,
  more linear encoding; without it, exactness is available through a nonlinear code.

## Reading

Three of the four hypotheses that "discworld's negative is an artefact" are now closed:
not data scale (20M), not architecture (Transformer-L, Li et al.'s own), not noise
(both channels off). The remaining live differences vs Othello are **continuous vs
discrete state**, **regression vs classification objective**, and **the observation
being a projection** (occlusion + perspective) rather than a full-information board.

Open follow-up: what IS the floor for a noiseless world? Not zero — the 128-ray
quantised render limits how precisely the initial state can be inferred. Computing it
would say how much of the 0.00106 is irreducible. Cheap CPU work, not yet run.
