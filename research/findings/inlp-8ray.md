# INLP on the 8-ray transformer: a deeper, more redundant linear code; writing more of it moves the ceiling from +0.30 to +0.36

**Date** 2026-09-04 · **Run** `ray_ablation/L-dw-8ray-20m` (dw-8ray: radius 1.0, 8 usable rays)
· **Code** `experiments/inlp/8ray/` (driver) + `experiments/inlp/scripts/inlp_dw.py` (the
canonical INLP experiment script, extended with a per-component R² profile) · **Data**
`experiments/inlp/8ray/scores/inlp_L-dw-8ray-20m_{frustum,cartesian}.json`, `summary.md` ·
**Figure** `experiments/inlp/8ray/outputs/perdim_profile.png` · **Cascades** persisted in
`runs/ray_ablation/L-dw-8ray-20m/probes/` (18 `nullspace_cascade` entries, 62 MB)

## Question

Same two questions as `inlp-redundancy.md`, on the 8-ray instance: how large is the
linearly readable code (per residual point and, new, per state component), and does
writing all of it edit any better than the canonical single-probe PI (+0.297 frustum /
+0.242 cartesian)?

## Method

Nullspace cascade at every residual point in the canonical probe's standardised space
(probe 1 ≡ PI[zspace]), 20,000 probe-corpus sequences, same 80/20 split by sequence, deflate
until held-out R² < 0.02 or 40 probes. Per probe: aggregate and **per-component**
held-out R² (train-mean baseline). Edits: write the first K probes at once
(K ∈ {1, 2, 4, 8, 16, 32, all}, uniform and R²-shrunk targets), α ∈ canonical grid, scored
with the canonical Edit Index and fidelity on the 192-case bench. Wiring check: the K=1
uniform write agrees with the canonical PI step within 6–10 % at every point ≥ 1 (the
embedding point is degenerate, as before). Probe 1's per-component R² reproduces Table 1b's
row to the third decimal. ~1 h per basis with the GPU shared with a training run.

## Results

**Redundancy is much larger than on the 128-ray transformer.** Points 1–8 give 29–40
orthogonal probes (total rank 232–320) against 15–20 probes / rank 120–160 on L-dw-20m, and
the aggregate profile barely decays over the first probes: frustum point 4 reads
0.948 / 0.942 / 0.921 / 0.833 at probes 1 / 2 / 4 / 8 (cartesian 0.883 / 0.876 / 0.847 /
0.745). The state is readable from dozens of independent directions of the stream.

**By component, the redundancy is a position phenomenon.** In the per-component profile
(`perdim_profile.png`, point 4) the four position read-outs stay above R² 0.8 for the first
~8 probes and are still above 0.5 at probe 15, while the four velocity read-outs start at
0.47–0.64 (frustum), halve by probe ~10 and fall below 0.2 by probe ~14: velocity lives in
fewer directions than position, though neither is confined to a handful. Probe 1's per-component R² (frustum, point 4:
0.94 / 0.89 / 0.96 / 0.90 · 0.59 / 0.48 / 0.64 / 0.49) is Table 1b's row.

**Writing several slices at once helps a little; writing all of them does not.** Best arms
(Edit Index / fidelity; canonical single-probe in brackets):

| basis | best multi-probe arm | K=1 at that point | all probes at that point |
|---|---|---|---|
| frustum | pt2 K8 **+0.358** / 1.11; pt1 K16 +0.356 / 0.95 | +0.267 / 1.07; −0.057 / 0.72 | K34 +0.349 / 1.13; K40 +0.295 / 1.27 |
| cartesian | pt2 K34 **+0.336** / 1.19; pt1 K8 +0.299 / 1.03 | +0.193 / 0.99; −0.259 / 0.77 | K34 +0.336 / 1.19; K40 +0.255 / 1.30 |

(canonical PI: frustum +0.297 / 1.11, cartesian +0.242 / 0.96). Gains of +0.06 (frustum)
and +0.09 (cartesian) over the single probe, with fidelity at or just above 1; at the late
points (7–8) every K degrades fidelity to 1.4–4.8 and the index falls, as on L-dw-20m.
The ceiling moves from ~0.30 to ~0.36 and stays a factor of two below Othello's +0.6.

## Reading

Two things the 128-ray INLP could not tell apart are now separated by the component
profile: the position code is highly redundant (many orthogonal slices, each reading it
almost fully) while the velocity code is thinner — a few directions carry most of it, and
it is gone from the stream before position is. Writing the redundant position copies
together does recover a little of what a single write leaves behind — the consistent
+0.06–0.09 across bases — but the remaining two thirds of the teleport are not hidden in
further copies: writing every copy is no better than writing eight. The block on
discworld editability is not redundancy of the linear code, on this instance as on the
128-ray one.

## Saved

Both result JSONs (every arm, every profile), `summary.md`, the figure, and all 18 fitted
cascades in the run's probe cache (a re-run is a cache hit). No canonical code changed;
the INLP experiment script gained a per-component profile, a repo-root import and an
optional scratch cache directory for smokes.
