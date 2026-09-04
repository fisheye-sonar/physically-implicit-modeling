# Probe capacity: the random reservoir plateaus below the trained model in BOTH worlds — at 0.975 on discworld, at 0.60 on Othello

**Status:** measured 2026-09-02. `experiments/probe_capacity/` — one-hidden-layer probes of
width h ∈ {LIN, 16, 64, 128, 512, 1024, 2048}, one residual point per environment
(discworld point 3, Othello point 7 — each the canonical LIN argmax), three sources
(trained model, random-init model, observation history), on **~5× the canonical probe rows**
so that width, not memorisation, is the binding constraint: discworld 250k sequences
(`dw-pn04/probe_250k`, 9.75M rows), Othello 170k games (`oth-uniform/probe_large`,
10.0M rows). 50 epochs (≈2× the canonical step count), batch 4096, 80/20 by sequence,
seed 0. Every fit persisted (`experiments/probe_capacity/probes/`, 42 files); results
`scores/probe_capacity_{discworld,othello}.json`; figure **Fig 2** in `build_full_table.ipynb`.

## The question

Table 3 showed an untrained network already supports 0.96 of the trained model's 0.996
MLP-128 decodability on discworld, but only 0.58 of 0.98 on Othello. Sevan's hypothesis: the
probe class is *overpowered* for discworld and *right-sized* for Othello — widen the probe and
Othello's random reservoir should catch up too, just later. Alternatively the two worlds
differ in what random features *contain*, and no width closes the gap.

## Result — held-out Probe Skill vs width (in-sample gap in brackets where it exceeds 0.01)

| | LIN | 16 | 64 | 128 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| **discworld** trained | 0.983 | 0.986 | 0.996 | 0.997 | 0.998 | 0.998 | 0.998 |
| discworld random-init | 0.671 | 0.886 | 0.942 | 0.957 | 0.970 | 0.975 | **0.975** |
| discworld observation | 0.251 (.02) | 0.690 (.02) | 0.844 (.02) | 0.875 (.02) | 0.913 (.03) | 0.917 (.03) | 0.918 (.03) |
| **Othello** trained | 0.975 | 0.578† | 0.941† | 0.976 | 0.978 | 0.980 | 0.980 |
| Othello random-init | 0.569 | 0.429† | 0.551† | 0.577 | 0.589 | 0.596 | **0.599** |
| Othello observation | 0.530 | 0.447† | 0.628† | 0.737 | 0.790 (.01) | 0.801 (.02) | 0.805 (.02) |

† h below the EFFECTIVE output dimension, 64 tiles × (3 − 1) free logits = 128 (the third logit
per tile is softmax gauge): a rank-h bottleneck on the read-out, not a weaker function class —
width 128 reproduces the linear probe to three decimals. Discworld's d_out is 8, so no width
bottlenecks it.

**Neither reservoir catches up.** On discworld the random-init curve rises 0.67 → 0.975 and
stops, 0.023 short of the trained model at every width ≥ 1024, with in-sample gaps ≤ 0.001 —
so it is not data-limited and not width-limited: that is what random features of the
observation history contain, and training adds a real 0.02 on top. On Othello the random
curve moves 0.577 → **0.599** across a 16× widening on 8.5× the data (gap ≤ 0.005), against a
trained plateau of 0.980. The hypothesis of a *later* saturation is refuted; the board is not
in random features at any probe capacity we can reach, whereas position on discworld nearly
is. **The difference between the worlds is in what the random network computes, not in the
probe class.**

Two further facts the sweep settles:

- **Plateaus are ordered and never cross.** Discworld: observation 0.918 < reservoir 0.975 <
  trained 0.998 — a random transformer over the ray history is a better feature map for
  position than the raw history, by 0.06 at every width. Othello inverts the first two:
  observation 0.805 > reservoir 0.599 — raw moves beat random features of moves, and a wide
  probe on raw moves keeps climbing (it is learning fragments of the flip rules; its gap grows
  with width accordingly), while the reservoir does not. Random mixing *helps* a smooth
  target and *hurts* a recursive one.
- **The canonical corpus was fine for the model probes and not for the observation floor.**
  Trained and random-init values at LIN and 128 reproduce the canonical 30k/20k numbers to
  ±0.005 (hollow markers on Fig 2); the observation MLP-128 moves from 0.696 (gap 0.25) to
  0.875 (gap 0.02) on discworld and 0.724 → 0.737 on Othello. Table 3's observation rows are
  under-estimates at the canonical size, as flagged there; the 5× values are the honest floor.

## What it means for the programme

Together with `decodability-baselines.md`, `training-curve.md`, `inlp-redundancy.md` and
`recurrent-l.md`: decodability provenance is a property of the *task*, robust to probe class,
probe data, training length, architecture and write strategy. Discworld's state is nearly a
free function of its input; Othello's is manufactured by training. The editing results line
up with that split, and the remaining candidate gates are about the task's dependence on
state (target sharpness — the Othello-as-regression flip) and the edit's manifold.
