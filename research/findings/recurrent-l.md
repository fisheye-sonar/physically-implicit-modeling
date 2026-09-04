# Recurrent-L: carried writes into a GRU's only state edit no better — recomputation is not the gate

**Status:** measured 2026-09-02. `runs/architecture_gate/R-dw-20m` — a stacked GRU, 4 × 1024,
25.46M params (parameter-matched to Transformer-L's 25.37M), the matched recipe (780k steps,
batch 256, lr 1e-3 constant, wd 1e-4, clip 1, seed 0) on the canonical noisy instance
`dw-pn04`, scored by the unchanged `master_eval` with its own random-init and observation
floors. Model: `pim/models/recurrent.py`. Driver: `experiments/recurrent/drivers/recurrent_dw.sh`.
The noiseless twin (`R-dw-noiseless-20m`) is reported below.

## Why this run

One candidate gate for the discworld negative was **recomputation**: a transformer's readable
state at the last position is rebuilt every step by attention over the observation window, so
a write there can be overwritten by later layers re-deriving position from earlier positions.
A GRU has no such route — its hidden state is the only summary of the past it carries — and
`RecurrentL.rollout_with_edit` writes point ℓ, decodes, and then **carries the edited hiddens
forward** (layer ℓ and, through recomputation at that step, everything above it). If a
carried write into the only state there is still cannot edit, recomputation is not the gate.

## Result

| model · basis (dw-pn04) | val | LIN | MLP | unedited | PI EI / fid (arm) | GS EI / fid |
|---|---|---|---|---|---|---|
| **Recurrent-L** · frustum | 0.02302 | 0.962 | 0.993 | −0.688 | **+0.217 / 1.81** (pos, pt4, α175) | −0.379 / 0.98 |
| Transformer-L · frustum | 0.02287 | 0.984 | 0.996 | −0.700 | +0.199 / 1.98 (pos, pt2, α60) | −0.221 / 0.94 |
| **Recurrent-L** · cartesian | | 0.880 | 0.959 | −0.688 | +0.124 / 1.67 | −0.416 / 0.99 |
| Transformer-L · cartesian | | 0.944 | 0.973 | −0.700 | +0.175 / 1.63 | −0.212 / 0.97 |

Same prediction quality (val within 0.7%), same decodability class, and **the same
editability signature**: PI reaches ~+0.2 only in the destructive regime (fidelity 1.7–1.8,
at the largest α in the grid, at the last layer), GS is negative, and — the sharpest form —
**no non-destructive arm is positive**: the best fidelity-≤1 write on the frustum basis is PI
at −0.229 / 0.95, i.e. nothing. Carrying the write in the state, which a transformer cannot
do, bought no persistence that the Edit Index can see.

**Recomputation is ruled out.** Together with `inlp-redundancy.md` (writing all 120–160
linearly readable dimensions at once does not help) and `training-curve.md` (nothing changes
after 64k steps), the remaining candidates are about the *task and the representation it
induces*: the sharpness of the target's dependence on state (legality vs a smooth next frame;
the Othello-as-regression flip tests it), a nonlinear consumer of a linearly readable code,
and the edit being off the data manifold (scale the teleport; edit velocity).

## The noiseless twin (`R-dw-noiseless-20m`, scored 2026-09-02 10:48)

| model · frustum (dw-noiseless) | val | LIN | MLP | unedited | PI EI / fid (arm) | GS EI / fid |
|---|---|---|---|---|---|---|
| **Recurrent-L** | 0.001112 | 0.968 | 0.995 | −0.917 | **+0.093 / 2.27** (pos, pt4, α175) | −0.537 / 0.98 |
| Transformer-L | 0.001063 | 0.959 | 0.996 | −0.924 | +0.233 / 1.95 (pos, pt1, α175) | −0.099 / 0.99 |

Same picture, worse: the carried write buys less than the transformer's transient one, at a
higher price (fidelity 2.27), and the best non-destructive arm is −0.326 / 0.98. The random-init
GRU floor on the noiseless instance is **LIN 0.937 / MLP 0.992** against the trained model's
0.968 / 0.995 — training added 0.03 and 0.003. Best val at step 205k, with the same instability
afterwards (spikes at ~250k and ~450k).

## Two things to read alongside it

*The random-init floor is even higher for the GRU.* Untrained 4 × 1024 GRU on dw-pn04, frustum:
LIN **0.850** / MLP **0.976** (transformer: 0.690 / 0.960), read at its first layer. Training
lifts the GRU's decodability by +0.11 (LIN) and +0.017 (MLP) — the reservoir story of
`decodability-baselines.md`, stronger.

*The recipe is a poor fit for the GRU.* Val bottomed at **0.02302 at step 50k**, drifted upward
for 450k steps, spiked at 525k (0.0234 → 0.0278) and ended at 0.0237; the scored model is the
50k best-val checkpoint, by the same rule every canonical run uses. The transformer's constant
lr 1e-3 was kept deliberately ("exactly the same"); a recurrent-tuned recipe would be a
different experiment, and the plateau at 50k is itself consistent with the training curve's
lesson that more training moves nothing here.

Related: `training-curve.md`, `inlp-redundancy.md`, `decodability-baselines.md`, `editability.md`.
