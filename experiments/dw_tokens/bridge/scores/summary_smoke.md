# Bridge: dw-tok-smoke scored with the discworld analysis (expected-frame rendering, argmax feedback)

| basis | reading | unedited EI | PI best · EI / fid | GS best · EI / fid |
|---|---|---|---|---|
| frustum | token model · ray-zone via bridge (expected frame) | -0.839 | all·pt4·α175 · -0.065 / 0.80 | pos·pt1·α0.5 · +0.000 / 0.83 |
| frustum | token model · ray-zone via bridge (argmax frame, same arms) | -0.881 | -0.327 / 0.98 | -0.373 / 0.95 |
| frustum | token model · frame-set (canonical scores.json) | -0.631 | all·pt6·α3 · -0.628 / 1.00 | pos·pt0·α0.2 · -0.386 / 0.91 |
| frustum | regression L-dw-8ray-20m · ray-zone (canonical) | -0.888 | pos·pt3·α175 · +0.297 / 1.11 | pos·pt0·α0.5 · -0.097 / 0.95 |

UNK inputs seen by the adapter: 0 · 0.1 min
