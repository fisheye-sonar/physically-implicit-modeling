# dw-blink subset editability — runs/noise_ablation/L-dw-noiseless-20m

cases available: {'reappearance': 190, 'reappearance_k3': 136, 'mid_blackout': 1135, 'visible': 4675}; used up to 192 per subset; staleness hist {1: 187, 2: 178, 3: 136, 4: 131, 5: 101, 6: 73, 7: 86, 8: 68, 9: 47, 10: 48, 11: 46, 12: 26}

| basis | subset | n | unedited | PI best | fid | GS best | fid |
|---|---|---|---|---|---|---|---|
| cartesian | reappearance | 24 | -0.235 | -0.219 (pt 4, α 1.0, all) | 0.99 | -0.214 (pt 4, α 0.05, all) | 0.98 |
| cartesian | reappearance_k3 | 24 | -0.145 | -0.136 (pt 4, α 1.0, all) | 0.99 | -0.130 (pt 4, α 0.05, all) | 0.98 |
| cartesian | mid_blackout | 24 | -0.367 | -0.366 (pt 4, α 0.75, all) | 0.99 | -0.371 (pt 4, α 0.02, pos) | 0.99 |
| cartesian | visible | 24 | -0.774 | -0.762 (pt 4, α 1.0, pos) | 1.00 | -0.730 (pt 4, α 0.05, all) | 0.98 |

mid_blackout is scored at each case's reappearance step (NaN at step 0 by construction).

## decodability by visibility (held-out probe sequences, best point)

| basis | probe | obj | visible | hidden | since 1 | 3 | 6 | 10 |
|---|---|---|---|---|---|---|---|---|
| cartesian | linear | 0 | +0.698 | +0.357 | +0.435 | +0.397 | +0.363 | — |
| cartesian | linear | 1 | +0.893 | +0.678 | +0.724 | +0.734 | +0.675 | — |
| cartesian | mlp | 0 | +0.857 | +0.616 | +0.762 | +0.637 | +0.577 | — |
| cartesian | mlp | 1 | +0.947 | +0.805 | +0.857 | +0.838 | +0.785 | — |
