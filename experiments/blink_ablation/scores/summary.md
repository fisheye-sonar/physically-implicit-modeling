# dw-blink subset editability — runs/blink_ablation/L-dw-blink-20m

cases available: {'reappearance': 642, 'reappearance_k3': 453, 'mid_blackout': 3697, 'visible': 15661}; used up to 192 per subset; staleness hist {1: 588, 2: 553, 3: 480, 4: 430, 5: 335, 6: 263, 7: 255, 8: 210, 9: 182, 10: 153, 11: 131, 12: 112}

| basis | subset | n | unedited | PI best | fid | GS best | fid |
|---|---|---|---|---|---|---|---|
| cartesian | reappearance | 192 | -0.846 | +0.211 (pt 2, α 100.0, all) | 1.41 | -0.052 (pt 0, α 0.5, pos) | 0.91 |
| cartesian | reappearance_k3 | 192 | -0.826 | +0.203 (pt 2, α 100.0, all) | 1.39 | -0.043 (pt 0, α 0.5, pos) | 0.92 |
| cartesian | mid_blackout | 192 | -0.385 | -0.196 (pt 7, α 175.0, pos) | 38.07 | -0.383 (pt 5, α 0.05, pos) | 1.35 |
| cartesian | visible | 192 | -0.909 | +0.181 (pt 2, α 175.0, pos) | 1.75 | -0.143 (pt 0, α 0.5, pos) | 0.92 |
| frustum | reappearance | 192 | -0.846 | +0.269 (pt 1, α 175.0, all) | 1.37 | -0.035 (pt 0, α 0.5, all) | 0.93 |
| frustum | reappearance_k3 | 192 | -0.826 | +0.241 (pt 1, α 175.0, all) | 1.39 | -0.033 (pt 0, α 0.5, pos) | 0.94 |
| frustum | mid_blackout | 192 | -0.385 | -0.148 (pt 7, α 175.0, pos) | 44.55 | -0.382 (pt 6, α 0.1, pos) | 1.84 |
| frustum | visible | 192 | -0.909 | +0.223 (pt 3, α 175.0, pos) | 1.81 | -0.117 (pt 0, α 0.5, pos) | 0.92 |

mid_blackout is scored at each case's reappearance step (NaN at step 0 by construction).

## decodability by visibility (held-out probe sequences, best point)

| basis | probe | obj | visible | hidden | since 1 | 3 | 6 | 10 |
|---|---|---|---|---|---|---|---|---|
| cartesian | linear | 0 | +0.900 | +0.745 | +0.774 | +0.760 | +0.727 | +0.681 |
| cartesian | linear | 1 | +0.955 | +0.883 | +0.902 | +0.892 | +0.870 | +0.842 |
| cartesian | mlp | 0 | +0.980 | +0.953 | +0.961 | +0.960 | +0.949 | +0.927 |
| cartesian | mlp | 1 | +0.987 | +0.969 | +0.973 | +0.973 | +0.968 | +0.954 |
| frustum | linear | 0 | +0.911 | +0.630 | +0.632 | +0.642 | +0.624 | +0.579 |
| frustum | linear | 1 | +0.958 | +0.889 | +0.909 | +0.899 | +0.875 | +0.841 |
| frustum | mlp | 0 | +0.994 | +0.973 | +0.977 | +0.978 | +0.971 | +0.955 |
| frustum | mlp | 1 | +0.996 | +0.985 | +0.987 | +0.988 | +0.984 | +0.975 |
