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


## 2026-09-12 — Othello: copy counts order the models by editability, and writing all copies rescues oth-adjacent

`experiments/adjacent_flip_ablation/scripts/inlp_othello.py` (per-tile colour cascades, ±1 on occupied
rows, closed-form; K-probe writes with the R²-shrink of `multiprobe_delta`). Copies of one tile's colour
(iterations to exhaustion, pts 1–8): standard Othello 48 / 41 / 36 / 32 / 29 / 27 / 24 / 28; oth-adjacent-flip 75 / 52 / 47 / 47 / 49 / 49 / 51 / 54;
oth-adjacent 232 / 138 / 99 / 88 / 90 / 85 / 83 / 83. Edits: standard already moves at K=2 and peaks at K=16 (+0.646 / fid 0.21,
pt 4); oth-adjacent's first 8 copies are inert everywhere and 64 copies at pt 2 give +0.472 / 0.47
(canonical ND +0.12); the flip model saturates at K=8 (+0.244). Unlike discworld (this file, top), where
writing the whole linear code changed nothing that mattered, on Othello's adjacency instance it is the whole
code that carries the edit — the fused-code prediction of the materialisation theory. Full tables and reading:
`adjacent-flip-ablation.md` §INLP.


## 2026-09-12 — Dropout is not the source of the redundancy: oth-adjacent without dropout has MORE copies

`dropout_ablation/L-oth-adjacent-nodrop-390k` (dropout 0, 390k steps, same optimum). Copies per tile pts 1–8:
283 / 227 / 187 / 161 / 150 / 148 / 146 / 162 vs 232 / 138 / 99 / 88 / 90 / 85 / 83 / 83 with dropout 0.1; K-copy edits unchanged in shape (K ≤ 16 inert; K=64 +0.28, K=128 +0.44 at
pt 1 vs +0.28 / +0.39). The residual-stream regulariser was compressing the colour code, not inflating it.
`adjacent-flip-ablation.md` §Dropout ablation.

**2026-09-13 — dropout 0.3 arm and a correction of wording.** `dropout_ablation/L-oth-adjacent-drop03-390k` (dropout 0.3,
390k, same optimum): copies per tile pts 1–8 **185 / 137 / 115 / 101 / 97 / 90 / 87 / 86** — monotone at point 1 across the
rates (283 → 232 → 185 for dropout 0 → 0.1 → 0.3), level with 0.1 from point 4; cascades at 0.3 are plateau-then-cliff at
every point. On the same run the canonical PI (+0.172 / fid 1.04) and ND (+0.159 / 0.73) editors land inside the guard for
the first time on oth-adjacent. The 2026-09-12 phrase above, "compressing the colour code", is too strong: the participation
ratio of the standardised residual covariance at point 1 RISES with dropout (44 / 81 / 84 for 0 / 0.1 / 0.3;
`experiments/dropout_ablation/scripts/covariance_dim.py`), so dropout decorrelates the stream rather than compressing it,
and the first colour directions cost LESS R² to remove with dropout (10 % / 3 % / 6 % for the first 8), i.e. they are
redundant sufficient copies, not additive correlates. What dropout removes is the long tail of weak, additive colour
correlates; what it leaves is a block of individually sufficient copies — the code the editors can move. Extension of the
no-dropout run to 780k (`L-oth-adjacent-nodrop-20m`): 272 / 210 / 158 / 132 / 122 / 119 / 118 / 137 — the tail prunes 10–20 %
with doubled training and stays 1.2–1.6× the dropout run's. `adjacent-flip-ablation.md` §Dropout ablation → Dropout 0.3.

**2026-09-14 — dropout 0.7.** `L-oth-adjacent-drop07-390k`: copies per tile pts 1–8 169 / 126 / 112 / 106 / 98 / 93 / 94 / 93. Point 1
continues the monotone fall (283 / 232 / 185 / 169 for 0 / 0.1 / 0.3 / 0.7) but its initial R² collapses to 0.72 (0.86–0.91 elsewhere), and
points 4–8 turn back up with longer plateaus (half-R² iteration 32–44 vs 15–19 at 0.1): the block of sufficient copies has moved deeper.
Canonical editors fall back to inert/marginal (PI −0.168 guarded, ND +0.033) and the K-copy window sits at points 4–5 — copy count at a fixed
point is not the whole story; WHERE the sufficient block lives relative to the editors' points matters too. `adjacent-flip-ablation.md`
§Dropout ablation → Dropout 0.7.
