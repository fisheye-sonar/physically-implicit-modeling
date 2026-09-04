# GS lands in probe space and the generation ignores it (discworld)

**Date** 2026-09-02 · **Run** L-dw-20m (dw-pn04), cached MLP-128 probes, both bases ·
**Code** `experiments/gs_readout_pilot/` · **Data**
`experiments/gs_readout_pilot/scores/gs_readout_pilot_L-dw-20m.json` (+ `summary_L-dw-20m.md`)

## Question

Gradient steering (GS) scores a negative Edit Index on every discworld run (Table 2:
−0.07 to −0.54). Is that because the descent never reaches its target — a tuning problem
that more steps would fix — or because the read-out lands and the model's generation does
not follow?

## Method

The canonical GS hook (`pim.editors.grad_steer.make_intervention_hook`, Li's sequential
schedule: descend at every residual point from the start layer up, letting the network
recompute in between) wrapped with a measurement after each layer's write: the MLP-128
probe's read-out on the changed dims (the teleported object's position, or position and
velocity for `all`) against the requested target, as RMSE in basis units and as the
fraction of the requested teleport covered per edit, `1 − |after| / |before|`; the drift of
the held dims; the hook's own loss record; and the canonical scorecard (Edit Index,
fidelity ratio) of the rollout the very same write produces. Swept descent length
{100 (canonical), 500, 2000} × start layer {0, 4, 8} × α {0.05, 0.2, 0.5, 1, 2} ×
dims {pos, all} × basis {frustum, cartesian}: 180 configurations, 192 edits each, 5 min
on the local GPU. The 18 configurations that coincide with the canonical sweep reproduce
the scorer's Edit Index to four decimals.

## Result

**The descent lands.** At the canonical 100 steps, after the write the probe at every
intervened layer reads the target to within a few percent of the teleport distance:

| basis · dims (100 steps) | last-layer coverage, min / median over configs | Edit Index range |
|---|---|---|
| frustum · pos | 0.975 / 0.990 | −0.640 … −0.079 |
| frustum · all | 0.958 / 0.975 | −0.640 … −0.090 |
| cartesian · pos | 0.953 / 0.993 | −0.643 … −0.031 |
| cartesian · all | 0.960 / 0.991 | −0.643 … −0.040 |

For the arm Table 2 reports (frustum · all · L0 · α 0.5) the per-layer picture is: layer 0
covers 95.5 % of the teleport (median 98.2 %; RMSE 0.298 → 0.007 basis units), every
later layer 97–99 %, the held dims drift by ≤ 0.015, the edit loss falls by three to four
orders of magnitude — and the rollout's Edit Index is −0.221 at fidelity 0.94, i.e. the
predicted frame is the *unedited* world, slightly degraded.

**More steps change nothing.** From 100 to 2000 steps the Edit Index moves by
+0.007 on average (range −0.015 … +0.042) across all 60 configuration pairs; the
landing at the first layer improves from ~96 % to ~99 % where it was under-converged
(small α, `all` dims) with no effect on the index. The best index in the whole sweep is
+0.002 (cartesian · all · L0 · α 2 · 2000 steps) at fidelity 1.13 — and the index rises
with α while the coverage *falls* (α 2: 0.80–0.98), so what little movement there is
comes from a larger, more disruptive write, not from a better landing.

## Reading

GS is not failing for lack of optimisation. A write that satisfies the MLP-128 probe
essentially exactly — at every layer, on the driven dims, while holding the rest — leaves
the next-frame prediction at the unedited world. The gradient finds the cheapest direction
that moves the probe's read-out (write ratio 0.05–0.7 of the activation norm), and that
direction is not the one the decoder and the later layers use. This is the same
conclusion the PI and INLP threads reached from the other side (`inlp-redundancy.md`:
writing every orthogonal probe lands where one lands): on discworld the state the probes
read and the state the generation runs on are not the same subspace. On Othello the
identical editor, probe and schedule move the generation (Table 2: +0.65), so this is a
property of the model–environment pair, not of GS.

## What this does not settle

* Whether a write that also matched the *decoder's* view (e.g. a probe fitted on the
  output-side features, or a supervised state decoder) would move the generation — the
  parked decoder-ceiling experiment.
* Whether the landing persists past step 0 of the rollout: the transformer recomputes the
  stream from observations, so any persistence must travel through the predicted frame,
  which here stays unedited.
