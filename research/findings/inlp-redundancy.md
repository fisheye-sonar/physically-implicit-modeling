# The whole linear code, written at once, still does not edit discworld

**Status:** measured 2026-09-01 (overnight), `L-dw-20m` best checkpoint on `dw-pn04`, frustum
basis, full target, n_seq 30 000 (the canonical probe rows and the identical seeded 80/20
sequence split). Source: `experiments/inlp/scripts/inlp_dw.py` → `experiments/inlp/scores/inlp_L-dw-20m_frustum.json`.
Objects: `pim.probes.nullspace.fit_nullspace_cascade` (INLP — the deflation cascade of
orthogonal linear probes) and `pim.editors.nullspace.multiprobe_delta` (write the first K
probes at once). Fitted in the canonical probe's standardised space, so probe 1 of the
cascade **is** canonical PI[zspace]: the K=1 write reproduced the canonical step to a
relative difference of 0.0 at every residual point 1–8 (0.19 at the rank-deficient
embedding layer, where min-norm solutions legitimately differ).

## The question

The row-space objection to the discworld negative: *a linear probe reads only d_out = 8 of
d_model = 512 dimensions; if the state is written redundantly, PI moves one copy and the
rest of the network still says "the object is where it was".* INLP measures how large the
linearly-readable code is; the multi-probe editor writes all of it.

## What the cascade found — the code IS large

| point | probes to R²<0.02 | rank | held-out R² per successive orthogonal probe |
|---|---|---|---|
| 0 (embedding) | 40 (cap) | 320 | 0.41 0.38 0.38 0.38 0.38 … (flat) |
| 1 | 20 | 160 | 0.96 0.94 0.90 0.83 0.75 0.64 0.54 0.44 … |
| 2 | 15 | 120 | 0.98 0.96 0.92 0.84 0.70 0.55 0.42 0.29 … |
| 3–8 | 15–16 | 120–128 | 0.98 0.95 0.89 0.79 0.66 0.50 0.39 0.28 … |

At every trained point, position is decodable above R² 0.4 from **eight mutually orthogonal
8-dimensional subspaces**, and the code is not linearly exhausted until 120–160 of the 512
dimensions have been removed. The objection's premise is true: the state is written many
times over. (Point 0's flat 0.38 profile is redundancy inherited from the *input* — a smooth
function of 128 rays, each carrying a little — not something the network built.)

## What the editor found — writing all of it changes nothing that matters

Best arm per point (Edit Index / fidelity ratio; fidelity > 1 = the edit left the model
*worse* at predicting the post-edit world than doing nothing):

| point | K = 1 (= canonical PI, all dims) | best multi-probe (K, mode) |
|---|---|---|
| 1 | +0.149 / 1.96 | +0.221 / 1.90 (K2) |
| 2 | +0.186 / 2.00 | **+0.250 / 1.66** (K8) |
| 3 | +0.171 / 3.31 | +0.220 / 1.61 (K15, shrunk) |
| 4 | +0.167 / 3.03 | +0.166 / 2.69 (K8) |
| 5–8 | +0.12 – +0.15 / 3.6 – 6.0 | no gain over K = 1 |

Writing 2–15 orthogonal copies buys at most **+0.06 Edit Index** over a single probe at
points 1–3 and nothing at all from point 4 on; fidelity never crosses below 1. The uniform
write at large K blows up (fid 3–7), exactly as `multiprobe_delta`'s docstring predicts; the
R²-shrunk write damps it back to the K = 1 numbers, not beyond.

**The sharpest form of the result:** across every residual point, every K from 1 to the full
cascade, both write modes and the whole α grid, **no arm is both positive on the Edit Index
and non-destructive.** The best fidelity-≤1 arm anywhere is point 2, K8, α8: EI **+0.012**
at fid 0.97 — indistinguishable from doing nothing. Every positive Edit Index in the table
above was bought by degrading the prediction.

## What this rules out, and what it leaves

Ruled out: **"you only moved one copy."** Moving every linearly-readable copy the cascade can
find — 120–160 dimensions, orthogonal, exact for every probe simultaneously — lands where one
probe lands. Redundancy of the linear code is real and is not the gate.

Left standing (the remaining candidates for the α=1 puzzle — the exact write lands in probe
space and the next frame does not move):
- **the decoder does not run on the linear code at all** — the readable directions are a
  consequence of the representation, not its substrate (a *nonlinear* consumer; note the
  MLP probe beats the linear one by ~0.015 at every point);
- **recomputation** — later layers rebuild position from earlier *positions* via attention,
  overwriting a write made at the last frame (testable: write at ℓ, read at ℓ+1…8);
- **the prior fights the edit** — a teleport is a physical discontinuity the model has never
  seen; a tile flip is a legal board (testable: scale the teleport, or edit velocity).

Together with `decodability-baselines.md` (an untrained network already supports 0.96 of the
0.996 MLP decodability): the discworld probe reads a code that is large, redundant, mostly
present before training, and causally inert under every linear write tried.

Related: `editability.md`, `decodability-baselines.md`, `state-geometry.md`.
