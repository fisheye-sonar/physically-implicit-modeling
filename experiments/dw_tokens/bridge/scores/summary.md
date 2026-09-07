# Bridge: L-dw-8ray-tok-20m scored with the discworld analysis (expected-frame rendering, argmax feedback)

| basis | reading | unedited EI | PI best · EI / fid | GS best · EI / fid |
|---|---|---|---|---|
| cartesian | token model · ray-zone via bridge (expected frame) | -0.894 | all·pt6·α175 · +0.137 / 0.89 | all·pt0·α0.5 · -0.138 / 0.90 |
| cartesian | token model · ray-zone via bridge (argmax frame, same arms) | -0.932 | +0.204 / 1.14 | -0.227 / 1.07 |
| cartesian | token model · frame-set (canonical scores.json) | -0.755 | all·pt6·α175 · +0.003 / 0.76 | pos·pt0·α0.5 · -0.027 / 0.76 |
| cartesian | regression L-dw-8ray-20m · ray-zone (canonical) | -0.888 | pos·pt3·α175 · +0.242 / 0.96 | all·pt0·α0.5 · -0.156 / 0.92 |
| frustum | token model · ray-zone via bridge (expected frame) | -0.894 | pos·pt6·α175 · +0.145 / 0.83 | pos·pt0·α0.5 · -0.271 / 0.87 |
| frustum | token model · ray-zone via bridge (argmax frame, same arms) | -0.932 | +0.222 / 1.02 | -0.386 / 1.04 |
| frustum | token model · frame-set (canonical scores.json) | -0.755 | pos·pt6·α175 · +0.004 / 0.75 | all·pt0·α0.5 · -0.097 / 0.78 |
| frustum | regression L-dw-8ray-20m · ray-zone (canonical) | -0.888 | pos·pt3·α175 · +0.297 / 1.11 | pos·pt0·α0.5 · -0.097 / 0.95 |

UNK inputs seen by the adapter: 0 · 3.3 min
