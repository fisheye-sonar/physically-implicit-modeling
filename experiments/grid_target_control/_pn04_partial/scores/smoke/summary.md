# grid-target control — runs/initial_othello_comparison/L-dw-20m (dw-pn04)

grid 16x8 = 128 cells x 3 classes; probes on 3,000 sequences, 2 epochs; bench 32 cases (6 same-cell teleports dropped); GS 10 steps, beta 0.2

| target | probe skill LIN / MLP | unedited | PI | ND | GS |
|---|---|---|---|---|---|
| grid 3-way (this) | -20.546 / -0.000 | -0.720 | -0.581 / fid 0.95 (pt 4, α 1) | -0.316 / fid 0.96 (pt 4, α 0.5) | -0.698 / fid 0.99 (pt 4, α 0.05) |
| grid, fidelity ≤ 1.1 | | | -0.581 / fid 0.95 (pt 4, α 1) | -0.316 / fid 0.96 (pt 4, α 0.5) | -0.698 / fid 0.99 (pt 4, α 0.05) |
| regression, frustum (canonical scores.json) | +0.984 / +0.996 | -0.700 | +0.199 / fid 1.98 (pt 2, α 60) | -0.017 / fid 2.97 (pt 4, α 1.5) (not reported for discworld) | -0.221 / fid 0.94 (pt 0, α 0.5) |

Probe skill for the grid target is 1 − err / err(majority) (classification, majority = empty); for the regression target it is R² — same axis, different formula. ND on the canonical row is computed but not reported (a fixed direction is incoherent for continuous teleports); on the grid target it is a categorical edit and IS reported.
