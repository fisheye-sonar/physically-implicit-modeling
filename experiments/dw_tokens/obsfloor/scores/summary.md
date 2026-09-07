# Observation floors for tokenised dw-8ray (span 39, full state, held out by sequence)

| basis | features | corpus | LIN skill (gap) | MLP-128 skill (gap) | d_in | MLP params | rows/param (MLP) |
|---|---|---|---|---|---|---|---|
| cartesian | frame_onehot · right-aligned | 30k | +0.907 (+0.004) | +0.921 (+0.021) | 16,458 | 2,107,784 | 0.4 |
| cartesian | frame_onehot · right-aligned | 250k | +0.909 (+0.000) | +0.933 (+0.002) | 16,458 | 2,107,784 | 3.7 |
| cartesian | frame_onehot · left-aligned | 30k | +0.653 (+0.134) | +0.883 (+0.079) | 16,458 | 2,107,784 | 0.4 |
| cartesian | frame_onehot · left-aligned | 250k | +0.728 (+0.014) | +0.924 (+0.015) | 16,458 | 2,107,784 | 3.7 |
| cartesian | ray_onehot · right-aligned | 30k | +0.744 (-0.002) | +0.922 (+0.001) | 936 | 120,968 | 7.7 |
| cartesian | ray_onehot · right-aligned | 250k | +0.743 (+0.001) | +0.924 (+0.000) | 936 | 120,968 | 64.5 |
| cartesian | ray_onehot · left-aligned | 30k | +0.604 (+0.003) | +0.916 (+0.008) | 936 | 120,968 | 7.7 |
| cartesian | ray_onehot · left-aligned | 250k | +0.604 (+0.001) | +0.919 (+0.001) | 936 | 120,968 | 64.5 |
| cartesian | float frames (canonical) | 30k / 250k | +0.257 / +0.252 | +0.837 / +0.847 | 312 | 41,096 | 28.5 / 237 |
| cartesian | random-init token model (best point) | 30k | +0.876 | +0.885 | 512 | | |
| cartesian | **trained token model** (best point) | 30k | **+0.899** | **+0.935** | 512 | | |
| frustum | frame_onehot · right-aligned | 30k | +0.972 (+0.001) | +0.978 (+0.004) | 16,458 | 2,107,784 | 0.4 |
| frustum | frame_onehot · right-aligned | 250k | +0.973 (+0.000) | +0.983 (+0.000) | 16,458 | 2,107,784 | 3.7 |
| frustum | frame_onehot · left-aligned | 30k | +0.638 (+0.154) | +0.954 (+0.031) | 16,458 | 2,107,784 | 0.4 |
| frustum | frame_onehot · left-aligned | 250k | +0.726 (+0.017) | +0.978 (+0.003) | 16,458 | 2,107,784 | 3.7 |
| frustum | ray_onehot · right-aligned | 30k | +0.869 (+0.000) | +0.978 (+0.001) | 936 | 120,968 | 7.7 |
| frustum | ray_onehot · right-aligned | 250k | +0.871 (+0.001) | +0.980 (+0.000) | 936 | 120,968 | 64.5 |
| frustum | ray_onehot · left-aligned | 30k | +0.622 (+0.009) | +0.974 (+0.003) | 936 | 120,968 | 7.7 |
| frustum | ray_onehot · left-aligned | 250k | +0.627 (+0.003) | +0.977 (+0.000) | 936 | 120,968 | 64.5 |
| frustum | float frames (canonical) | 30k / 250k | +0.281 / +0.279 | +0.913 / +0.925 | 312 | 41,096 | 28.5 / 237 |
| frustum | random-init token model (best point) | 30k | +0.968 | +0.968 | 512 | | |
| frustum | **trained token model** (best point) | 30k | **+0.968** | **+0.980** | 512 | | |
