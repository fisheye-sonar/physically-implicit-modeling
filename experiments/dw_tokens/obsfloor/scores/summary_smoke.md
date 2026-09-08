# Observation floors for tokenised dw-8ray (span 39, full state, held out by sequence)

| basis | features | corpus | LIN skill (gap) | MLP-128 skill (gap) | d_in | MLP params | rows/param (MLP) |
|---|---|---|---|---|---|---|---|
| cartesian | frame_onehot · right-aligned | 30k | -10.141 (+11.110) | +0.290 (+0.060) | 16,458 | 2,107,784 | 0.0 |
| cartesian | frame_onehot · right-aligned | 250k | +0.604 (+0.344) | +0.388 (+0.024) | 16,458 | 2,107,784 | 0.0 |
| cartesian | frame_onehot · left-aligned | 30k | +0.204 (+0.779) | +0.173 (+0.103) | 16,458 | 2,107,784 | 0.0 |
| cartesian | frame_onehot · left-aligned | 250k | +0.344 (+0.631) | +0.258 (+0.052) | 16,458 | 2,107,784 | 0.0 |
| cartesian | ray_onehot · right-aligned | 30k | +0.743 (+0.022) | +0.544 (+0.013) | 936 | 120,968 | 0.1 |
| cartesian | ray_onehot · right-aligned | 250k | +0.728 (+0.022) | +0.583 (+0.017) | 936 | 120,968 | 0.2 |
| cartesian | ray_onehot · left-aligned | 30k | +0.416 (+0.359) | +0.505 (+0.013) | 936 | 120,968 | 0.1 |
| cartesian | ray_onehot · left-aligned | 250k | +0.491 (+0.199) | +0.524 (+0.016) | 936 | 120,968 | 0.2 |
| cartesian | float frames (canonical) | 30k / 250k | +0.257 / +0.252 | +0.837 / +0.847 | 312 | 41,096 | 28.5 / 237 |
| cartesian | random-init token model (best point) | 30k | +0.876 | +0.885 | 512 | | |
| cartesian | **trained token model** (best point) | 30k | **+0.899** | **+0.935** | 512 | | |
| frustum | frame_onehot · right-aligned | 30k | -1.968 (+2.959) | +0.309 (+0.080) | 16,458 | 2,107,784 | 0.0 |
| frustum | frame_onehot · right-aligned | 250k | +0.897 (+0.087) | +0.417 (+0.035) | 16,458 | 2,107,784 | 0.0 |
| frustum | frame_onehot · left-aligned | 30k | +0.089 (+0.903) | +0.182 (+0.129) | 16,458 | 2,107,784 | 0.0 |
| frustum | frame_onehot · left-aligned | 250k | +0.379 (+0.608) | +0.280 (+0.065) | 16,458 | 2,107,784 | 0.0 |
| frustum | ray_onehot · right-aligned | 30k | +0.846 (+0.036) | +0.675 (+0.025) | 936 | 120,968 | 0.1 |
| frustum | ray_onehot · right-aligned | 250k | +0.862 (+0.016) | +0.710 (+0.012) | 936 | 120,968 | 0.2 |
| frustum | ray_onehot · left-aligned | 30k | +0.397 (+0.418) | +0.577 (+0.028) | 936 | 120,968 | 0.1 |
| frustum | ray_onehot · left-aligned | 250k | +0.498 (+0.232) | +0.593 (+0.013) | 936 | 120,968 | 0.2 |
| frustum | float frames (canonical) | 30k / 250k | +0.281 / +0.279 | +0.913 / +0.925 | 312 | 41,096 | 28.5 / 237 |
| frustum | random-init token model (best point) | 30k | +0.968 | +0.968 | 512 | | |
| frustum | **trained token model** (best point) | 30k | **+0.968** | **+0.980** | 512 | | |
