# 2026-09-13 — Everything rescored under the unified edit protocol (overnight chain, stage 2–6)

**What changed (Sevan's decisions 2026-09-12):** n = 1000 cases per instance in BOTH environments, cut from
each instance's own edits split at a fixed edit position (Othello: single-tile flips at a 20-move prefix,
index range disjoint from train/val/test/probe; discworld: EF = 20, the first 1000 cases whose two clean
renders differ on ≥ 2 rays); full-state writes only; shared alpha grids and GS layers; Othello headline =
**symmetric difference** (union kept in the JSON); discworld target = the pre-dynamics state (unchanged
since the 13:15 rescore). `EVAL_VERSION_BY_ENV` discworld `2026-09-12.2`, othello `2026-09-12.1`; unit
`rescore_protocol`, master_eval 22:04 → 00:55 PT (2 h 51 min for 33 runs), then Table 3 alignment refresh,
Haufe editability, test loss, both table notebooks (00:58). Parked predecessors: discworld
`scores.pre-alignment-2026-09-12.json` beside each run (the post-alignment intermediate differed from it by
≤ 0.06, `scratch/2026-09-12-alignment-rescore.md`); Othello had no parked file — the old numbers below are
the REGISTRY run rows (Li's 1001 cases, union support, extended alphas).

## Discworld — before (pre-alignment parked, first-192 bench) → after (1000-case selected bench, pre-dynamics target)

Best arm per editor, Edit Index (guard). Skill = max over points, LIN / MLP.

| run | block | unedited | PI | ND | GS | skill |
|---|---|---|---|---|---|---|
| blink_ablation/L-dw-blink-20m | frustum | -0.909→-0.920 | +0.215→+0.210 (1.78→1.54) | -0.030→-0.089 (3.50→1.72) | -0.088→-0.090 (1.00→0.97) | 0.90 / 0.99 |
| blink_ablation/L-dw-blink-20m | appearance-fac | -0.909→-0.919 | +0.023→+0.006 (2.44→2.26) | +0.531→+0.515 (0.94→0.92) | +0.332→+0.260 (0.95→1.15) | 0.46 / 0.68 |
| initial_othello_comparison/L-dw-20m | frustum | -0.700→-0.700 | +0.199→+0.183 (1.98→1.94) | -0.017→-0.051 (2.97→1.55) | -0.221→-0.161 (0.94→0.94) | 0.98 / 1.00 |
| interface_ablation/L-dw-8ray-tok-20m | frustum | -0.779→-0.745 | +0.006→+0.004 (0.74→0.75) | — | -0.102→-0.154 (0.77→0.80) | 0.97 / 0.98 |
| interface_ablation/L-dw-8ray-tok-20m | appearance | -0.763→-0.734 | +0.260→+0.247 (0.53→0.55) | +0.439→+0.440 (0.41→0.43) | +0.575→+0.486 (0.31→0.39) | 0.90 / 0.91 |
| interface_ablation/L-dw-8ray-tok-20m | appearance-d2 | -0.765→-0.736 | +0.299→+0.260 (0.50→0.54) | +0.299→+0.308 (0.51→0.51) | +0.585→+0.522 (0.29→0.36) | 0.67 / 0.69 |
| interface_ablation/L-dw-8ray-tok-20m | appearance-d3 | -0.763→-0.737 | +0.112→+0.123 (0.65→0.64) | +0.239→+0.262 (0.54→0.54) | +0.535→+0.470 (0.32→0.39) | 0.48 / 0.52 |
| interface_ablation/L-dw-8ray-tok-20m | grid-16x8 | -0.754→-0.737 | +0.011→+0.006 (0.75→0.77) | +0.107→+0.108 (0.66→0.65) | +0.233→+0.168 (0.54→0.61) | 0.17 / 0.23 |
| interface_ablation/L-dw-8ray-tok-20m | appearance-lat | -0.766→-0.736 | +0.061→+0.062 (0.71→0.70) | +0.163→+0.172 (0.61→0.61) | +0.294→+0.282 (0.52→0.55) | 0.91 / 0.92 |
| interface_ablation/L-dw-8ray-tok-20m | grid-8x4 | -0.751→-0.735 | +0.056→+0.034 (0.69→0.73) | +0.096→+0.091 (0.66→0.67) | +0.195→+0.152 (0.59→0.64) | 0.47 / 0.53 |
| interface_ablation/L-dw-8ray-tok-20m | grid-32x16 | -0.753→-0.732 | +0.016→+0.009 (0.75→0.77) | +0.099→+0.114 (0.66→0.64) | +0.182→+0.106 (0.58→0.68) | 0.01 / 0.05 |
| interface_ablation/L-dw-8ray-tok-20m | grid-6x5 | -0.766→-0.732 | +0.028→+0.029 (0.72→0.73) | +0.071→+0.078 (0.69→0.67) | +0.191→+0.182 (0.60→0.63) | 0.49 / 0.55 |
| interface_ablation/L-dw-8ray-tok-20m | grid-10x3 | -0.753→-0.734 | +0.046→+0.049 (0.70→0.69) | +0.108→+0.125 (0.65→0.64) | +0.228→+0.199 (0.57→0.60) | 0.45 / 0.51 |
| interface_ablation/L-dw-8ray-tok-20m | grid-4x2 | -0.759→-0.737 | +0.033→+0.020 (0.74→0.75) | +0.053→+0.051 (0.72→0.72) | +0.091→+0.075 (0.71→0.72) | 0.75 / 0.79 |
| interface_ablation/L-dw-8ray-tok-20m | appearance-fac | -0.763→-0.734 | +0.235→+0.228 (0.54→0.56) | +0.298→+0.283 (0.50→0.53) | +0.403→+0.351 (0.43→0.47) | 0.94 / 0.94 |
| noise_ablation/L-dw-noiseless-20m | frustum | -0.924→-0.930 | +0.233→+0.231 (1.95→1.54) | -0.028→-0.135 (3.51→1.23) | -0.099→-0.082 (0.99→1.07) | 0.96 / 1.00 |
| noise_ablation/L-dw-noiseless-20m | grid-16x8 | -0.932→-0.930 | +0.126→+0.130 (1.58→1.63) | +0.373→+0.370 (0.91→0.95) | +0.289→+0.318 (0.87→0.77) | 0.43 / 0.71 |
| noise_ablation/L-dw-noiseless-20m | grid-8x4 | -0.935→-0.934 | +0.265→+0.255 (1.05→1.04) | +0.343→+0.345 (0.92→0.88) | +0.266→+0.281 (1.11→0.83) | 0.71 / 0.89 |
| noise_ablation/L-dw-noiseless-20m | appearance-lat | -0.924→-0.930 | +0.043→+0.029 (1.68→1.81) | +0.506→+0.484 (0.79→0.81) | +0.295→+0.291 (1.08→0.89) | 0.04 / 0.63 |
| noise_ablation/L-dw-noiseless-20m | grid-32x16 | -0.927→-0.929 | +0.171→+0.166 (1.13→1.17) | +0.342→+0.321 (0.92→0.94) | +0.292→+0.314 (0.90→0.88) | 0.20 / 0.44 |
| noise_ablation/L-dw-noiseless-20m | appearance-fac | -0.924→-0.928 | +0.014→+0.007 (1.95→1.93) | +0.628→+0.612 (0.78→0.80) | +0.352→+0.326 (0.68→0.71) | 0.43 / 0.79 |
| noise_ablation/L-dw-noiseless-20m__seed0_s421875 | frustum | -0.917→-0.926 | +0.217→+0.211 (2.23→1.77) | -0.050→-0.164 (2.85→1.26) | -0.104→-0.086 (1.00→1.04) | 0.96 / 1.00 |
| noise_ablation/L-dw-noiseless-20m__seed0_s421875 | appearance-fac | -0.917→-0.924 | +0.007→-0.001 (1.81→1.78) | +0.614→+0.597 (0.79→0.82) | +0.330→+0.298 (0.85→0.74) | 0.43 / 0.77 |
| noise_ablation/L-dw-noiseless-20m__seed1 | frustum | -0.919→-0.925 | +0.227→+0.215 (1.80→1.96) | -0.087→-0.194 (1.72→1.40) | -0.109→-0.090 (0.99→1.05) | 0.96 / 1.00 |
| noise_ablation/L-dw-noiseless-20m__seed1 | appearance-fac | -0.920→-0.923 | +0.016→+0.003 (1.96→1.74) | +0.644→+0.626 (0.72→0.83) | +0.306→+0.291 (1.09→0.80) | 0.42 / 0.78 |
| noise_ablation/L-dw-noiseless-20m__seed2 | frustum | -0.919→-0.926 | +0.230→+0.223 (2.01→2.13) | -0.050→-0.139 (2.93→1.15) | -0.121→-0.092 (1.00→1.03) | 0.95 / 1.00 |
| noise_ablation/L-dw-noiseless-20m__seed2 | appearance-fac | -0.919→-0.924 | +0.007→-0.003 (1.77→1.78) | +0.616→+0.600 (0.80→0.82) | +0.331→+0.305 (0.83→0.75) | 0.43 / 0.79 |
| ray_ablation/L-dw-5ray-20m | frustum | -0.864→-0.891 | +0.295→+0.245 (1.12→0.94) | -0.178→-0.271 (0.92→0.82) | -0.112→-0.063 (1.02→0.86) | 0.94 / 0.97 |
| ray_ablation/L-dw-5ray-20m | appearance-fac | -0.908→-0.907 | +0.486→+0.513 (0.68→0.70) | +0.555→+0.553 (0.90→0.58) | +0.580→+0.598 (0.47→0.46) | 0.93 / 0.93 |
| ray_ablation/L-dw-5ray-20m | appearance | -0.908→-0.907 | +0.536→+0.556 (0.78→0.78) | +0.474→+0.437 (0.85→0.90) | +0.638→+0.640 (0.40→0.42) | 0.88 / 0.89 |
| ray_ablation/L-dw-8ray-20m | frustum | -0.911→-0.897 | +0.282→+0.256 (0.90→0.99) | -0.148→-0.203 (1.15→0.92) | -0.064→-0.034 (0.82→0.90) | 0.95 / 0.98 |
| ray_ablation/L-dw-8ray-20m | appearance | -0.909→-0.899 | +0.429→+0.437 (0.76→0.89) | +0.429→+0.418 (0.91→0.97) | +0.614→+0.591 (0.39→0.44) | 0.89 / 0.90 |
| ray_ablation/L-dw-8ray-20m | appearance-d2 | -0.910→-0.897 | +0.478→+0.462 (0.67→0.76) | +0.389→+0.386 (0.89→1.03) | +0.573→+0.562 (0.41→0.44) | 0.66 / 0.68 |
| ray_ablation/L-dw-8ray-20m | appearance-d3 | -0.902→-0.897 | +0.406→+0.352 (0.85→1.02) | +0.380→+0.370 (0.93→1.04) | +0.510→+0.493 (0.53→0.50) | 0.46 / 0.51 |
| ray_ablation/L-dw-8ray-20m | grid-16x8 | -0.889→-0.893 | +0.104→+0.116 (1.43→1.65) | +0.309→+0.304 (0.99→1.07) | +0.371→+0.388 (0.66→0.66) | 0.16 / 0.21 |
| ray_ablation/L-dw-8ray-20m | appearance-lat | -0.909→-0.899 | +0.328→+0.325 (0.81→0.93) | +0.417→+0.377 (0.89→1.00) | +0.418→+0.393 (0.61→0.62) | 0.89 / 0.92 |
| ray_ablation/L-dw-8ray-20m | grid-8x4 | -0.898→-0.896 | +0.314→+0.306 (0.82→0.91) | +0.306→+0.280 (0.94→1.01) | +0.350→+0.314 (0.80→0.73) | 0.46 / 0.51 |
| ray_ablation/L-dw-8ray-20m | grid-32x16 | -0.889→-0.888 | +0.140→+0.109 (1.22→1.36) | +0.305→+0.303 (0.97→1.06) | +0.422→+0.452 (0.68→0.70) | 0.01 / 0.04 |
| ray_ablation/L-dw-8ray-20m | grid-6x5 | -0.909→-0.896 | +0.315→+0.305 (0.87→0.98) | +0.287→+0.262 (0.92→1.06) | +0.343→+0.349 (0.66→0.67) | 0.47 / 0.54 |
| ray_ablation/L-dw-8ray-20m | grid-10x3 | -0.900→-0.893 | +0.359→+0.366 (0.85→0.91) | +0.337→+0.335 (0.95→1.01) | +0.340→+0.348 (0.63→0.65) | 0.44 / 0.50 |
| ray_ablation/L-dw-8ray-20m | grid-4x2 | -0.910→-0.898 | +0.414→+0.369 (0.73→0.77) | +0.245→+0.224 (0.92→1.00) | +0.309→+0.340 (0.85→0.79) | 0.75 / 0.78 |
| ray_ablation/L-dw-8ray-20m | pos@appearance | -0.909→-0.899 | +0.336→+0.290 (1.15→1.00) | -0.199→-0.177 (1.07→1.17) | -0.126→-0.046 (0.84→0.93) | 0.96 / 0.99 |
| ray_ablation/L-dw-8ray-20m | appearance-fac | -0.909→-0.899 | +0.414→+0.380 (0.71→0.95) | +0.504→+0.488 (0.86→0.98) | +0.464→+0.458 (0.53→0.54) | 0.94 / 0.94 |
| ray_ablation/_R-dw-8ray-20m | frustum | -0.906→-0.890 | +0.168→+0.121 (1.10→1.25) | -0.278→-0.300 (0.93→0.97) | -0.605→-0.530 (0.91→0.91) | 0.97 / 0.98 |

## Othello — the new 1000-case per-instance benches (symmetric difference = headline; union beside it)

| run | unedited symdiff (union) | PI symdiff (union) / fid | ND | GS | skill LIN / MLP | old REGISTRY: uned · PI · ND · GS (union, Li's 1001) |
|---|---|---|---|---|---|---|
| adjacency_ablation/L-oth-adjacent-20m | -0.960 (-0.655) | +0.176 (+0.014) / 2.14 | +0.488 (+0.164) / 1.98 | -0.157 (+0.002) / 6.68 | 0.988 / 0.990 | — |
| adjacent_flip_ablation/L-oth-adjacent-flip-20m | -0.964 (-0.647) | +0.399 (+0.091) / 2.09 | +0.576 (+0.167) / 1.16 | +0.138 (+0.039) / 1.75 | 0.947 / 0.970 | — |
| dropout_ablation/L-oth-adjacent-nodrop-390k | -0.957 (-0.633) | -0.054 (+0.001) / 6.44 | +0.448 (+0.042) / 3.46 | -0.108 (-0.001) / 4.69 | 0.979 / 0.981 | — |
| flip_ablation/L-oth-noflip-20m | -0.976 (-0.809) | -0.308 (-0.003) / 1.97 | +0.222 (+0.094) / 2.54 | -0.522 (+0.039) / 4.78 | 1.000 / 1.000 | — |
| initial_othello_comparison/L-oth-20m | -0.933 (-0.659) | +0.818 (+0.552) / 0.30 | +0.749 (+0.512) / 0.33 | +0.828 (+0.571) / 0.28 | 0.975 / 0.976 | — |
| objective_ablation/L-oth-20m-mse | -0.936 (-0.780) | +0.848 (+0.682) / 0.26 | +0.807 (+0.678) / 0.23 | +0.858 (+0.706) / 0.26 | 0.961 / 0.960 | — |

## Table 3 — PI at its best point vs PI along the Haufe-corrected direction (same point, own alpha sweep)

| run | target | PI EI / fid (canonical best) | PI-Haufe EI / fid (α) | Δ EI |
|---|---|---|---|---|
| initial_othello_comparison/L-oth-20m | mine/theirs | +0.818 / 0.30 | +0.806 / 0.39 (α 3) | -0.011 |
| objective_ablation/L-oth-20m-mse | mine/theirs | +0.848 / 0.26 | +0.832 / 0.28 (α 2) | -0.017 |
| flip_ablation/L-oth-noflip-20m | mine/theirs | -0.308 / 1.97 | -0.367 / 3.45 (α 10) | -0.059 |
| adjacency_ablation/L-oth-adjacent-20m | mine/theirs | +0.176 / 2.14 | +0.465 / 1.86 (α 10) | +0.288 |
| adjacent_flip_ablation/L-oth-adjacent-flip-20m | mine/theirs | +0.399 / 2.09 | +0.388 / 0.83 (α 3) | -0.011 |
| initial_othello_comparison/L-dw-20m | frustum | +0.183 / 1.94 | +0.232 / 1.55 (α 35) | +0.049 |
| noise_ablation/L-dw-noiseless-20m | frustum | +0.231 / 1.54 | +0.272 / 1.49 (α 60) | +0.041 |
| noise_ablation/L-dw-noiseless-20m | appearance-fac | +0.007 / 1.93 | +0.007 / 1.93 (α 100) | +0.000 |
| ray_ablation/L-dw-8ray-20m | frustum | +0.256 / 0.99 | +0.304 / 1.06 (α 20) | +0.049 |
| ray_ablation/L-dw-8ray-20m | appearance-fac | +0.380 / 0.95 | +0.195 / 1.29 (α 20) | -0.185 |
| interface_ablation/L-dw-8ray-tok-20m | frustum | +0.004 / 0.75 | +0.032 / 0.83 (α 100) | +0.027 |
| interface_ablation/L-dw-8ray-tok-20m | appearance-fac | +0.228 / 0.56 | +0.115 / 0.69 (α 5) | -0.113 |
| ray_ablation/L-dw-5ray-20m | frustum | +0.245 / 0.94 | +0.279 / 0.79 (α 8) | +0.034 |
| ray_ablation/L-dw-5ray-20m | appearance-fac | +0.513 / 0.70 | +0.240 / 1.02 (α 10) | -0.272 |
| blink_ablation/L-dw-blink-20m | frustum | +0.210 / 1.54 | +0.256 / 1.35 (α 60) | +0.045 |
| blink_ablation/L-dw-blink-20m | appearance-fac | +0.006 / 2.26 | +0.006 / 2.26 (α 100) | +0.000 |

## Reading (draft, 01:05 PT — to be checked against the tables in the morning)

- **Discworld: the protocol change does not move any conclusion.** Unedited floors shift by ≤ 0.03 (the
  bench changed: 1000 selected cases vs the first 192); best PI and GS on the regression blocks stay within
  ~0.05 of the parked numbers; the categorical blocks move by ≤ 0.09, the largest drops on the token model's
  GS (appearance +0.58 → +0.49, appearance-d2 +0.59 → +0.52) and on dw-blink's appearance-fac GS (+0.33 → +0.26);
  ND on the regression target — ill-posed there (GOTCHAS 2026-09-01) — moves by up to −0.11 as its guard
  collapses from ~3 to ~1.3 under the shared alpha grid. PI ≈ +0.2…+0.3 on the regression target with guards
  > 1 (except the 8-ray family, ≈ 0.9–1.0); GS ≤ 0 everywhere on the regression target; ND remains the best
  categorical editor; dw-5ray's appearance blocks are still the most editable discworld blocks (GS +0.60 / +0.64).
  The 1000-case selected bench and the 192-case first-block bench agree — the old bench was not the reason
  for any number.
- **Othello: the symmetric-difference headline lowers every unedited floor to ≈ −0.93…−0.98 (union −0.66…−0.82)
  and raises the standard-Othello best arms to +0.72…+0.83 (union +0.55).** The ORDERING is unchanged:
  standard ≈ MSE-objective ≫ adjacent-flip > adjacent ≈ nodrop-390k > noflip. oth-noflip stays non-editable
  (PI −0.31, GS −0.52, ND +0.22 / 2.54); oth-adjacent's only positive arm is ND (+0.49 / 1.98, guard failing);
  oth-adjacent-flip is the partially editable one (PI +0.35 / 0.83, ND +0.32 / 0.61).
- **Haufe correction (Table 3):** no gain where PI is already aligned (standard Othello +0.82 → +0.81, MSE
  +0.85 → +0.83, noflip stays negative); a real gain on oth-adjacent (+0.18 → +0.47 but guard 1.86) and small
  gains on the discworld regression blocks (+0.03…+0.05, guards ~0.8–1.5); on the categorical (appearance-fac)
  blocks the Haufe direction HURTS PI — dw-5ray +0.51 → +0.24, dw-8ray +0.38 → +0.20, 8ray-tok +0.23 → +0.12 —
  and leaves the two near-zero blocks (noiseless, blink) at zero. To be read against Table 3's alignment rows.

_Written by the orchestrator at 01:05 PT while `L-dw-smooth-20m` trains (stage 7); dw-smooth's rows land when the chain completes._
