# grid-target control — runs/noise_ablation/L-dw-noiseless-20m (dw-noiseless)

grid 16x8 = 128 cells x 3 classes; probes on 200,000 sequences, 50 epochs; bench 192 cases (20 same-cell teleports dropped); GS 100 steps, beta 0.2

| target | probe skill LIN / MLP | unedited | PI | ND | GS |
|---|---|---|---|---|---|
| grid 3-way (this) | +0.434 / +0.707 | -0.932 | +0.126 / fid 1.58 (pt 1, α 60) | +0.373 / fid 0.91 (pt 3, α 8) | +0.289 / fid 0.87 (pt 1, α 0.75) |
| grid, fidelity ≤ 1.1 | | | +0.083 / fid 1.10 (pt 1, α 20) | +0.373 / fid 0.91 (pt 3, α 8) | +0.289 / fid 0.87 (pt 1, α 0.75) |
| regression, frustum (canonical scores.json) | +0.959 / +0.996 | -0.924 | +0.233 / fid 1.95 (pt 1, α 175) | -0.028 / fid 3.51 (pt 8, α 2) (not reported for discworld) | -0.099 / fid 0.99 (pt 1, α 0.5) |

Probe skill for the grid target is 1 − err / err(majority) (classification, majority = empty); for the regression target it is R² — same axis, different formula. ND on the canonical row is computed but not reported (a fixed direction is incoherent for continuous teleports); on the grid target it is a categorical edit and IS reported.
