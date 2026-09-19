# 2026-09-17 — clipping the predicted frame to [0, 1]: does editability change?

**Why.** The discworld observation is a reflectivity scan in [0, 1] by construction (0 on a miss,
0.4 / 0.8 on a disc) but the regression head is unconstrained, so a large PI write is scored on
values the environment can never render. Sevan asked whether clipping the output changes the
editability scores. Both Edit Index references are clean renders in [0, 1], so clipping can only
move a prediction toward the feasible set.

**Script.** `experiments/clip_output_pilot/scripts/clip_pilot.py --run <run> [--basis --target
--block --no-im]` — the CANONICAL best arm of each editor, at the residual point and α already
selected in `scores.json`, scored twice off ONE rollout (raw, and `np.clip(roll, 0, 1)`). No probe
refitting, no sweep. The unsteered rollout is clipped too, so the Fidelity Ratio denominator gets
the same post-processing (the raw-denominator variant is also stored). **Validation: every raw
number reproduces `scores.json` exactly on all four blocks.** Scores in
`experiments/clip_output_pilot/scores/clip_*.json`. GPU ~3 min per block.

## Result (Edit Index raw → clipped · Fidelity Ratio raw → clipped)

**Continuous state (`full`, cartesian basis — the paper's main table)**

| editor | dw-noiseless | dw-8ray |
|---|---|---|
| unedited | −0.930 → −0.930 | −0.897 → −0.898 |
| PI | +0.197 → **+0.243** · 1.71 → **1.41** | +0.212 → **+0.238** · 1.00 → **0.91** |
| ND | −0.150 → −0.180 · 1.57 → **1.17** | −0.226 → −0.226 · 0.87 → 0.87 |
| GS | −0.056 → −0.056 · 1.07 → 1.03 | −0.070 → −0.070 · 0.88 → 0.88 |
| IM | +0.590 → +0.590 · 0.34 → 0.34 | +0.711 → +0.712 · 0.27 → 0.27 |
| IM-NN | +0.327 → +0.327 · 0.76 → 0.76 | +0.535 → +0.535 · 0.53 → 0.53 |

**Categorical state (`appearance-fac`, frustum basis; IM is target-independent, so it is the
frustum IM above)**

| editor | dw-noiseless | dw-8ray |
|---|---|---|
| PI | +0.007 → +0.011 · 1.93 → **1.52** | +0.380 → **+0.415** · 0.95 → 0.90 |
| ND | +0.612 → +0.618 · 0.80 → 0.80 | +0.488 → +0.504 · 0.98 → 0.98 |
| GS | +0.326 → +0.335 · 0.71 → **0.63** | +0.458 → +0.462 · 0.54 → 0.54 |

## Reading

1. **No conclusion changes.** The largest index move is +0.046 (PI, dw-noiseless, continuous).
   PI still never lands, IM still lands on both instances, GS is still negative on the continuous
   target and mid-range on the categorical one. The paper's sentence "PI reaches its best index
   only at a Fidelity Ratio above 1.3" survives clipping (1.41 at dw-noiseless).
2. **The whole effect is PI, and it lands in the fidelity column, not the index.** PI is the only
   editor whose write leaves the feasible range by a lot: at α=100 its frame spans
   [−2.16, +2.61] against the unedited model's [−0.02, +0.82]. Clipping cuts its Fidelity Ratio
   (1.93 → 1.52, 1.71 → 1.41) far more than it lifts its index. So clipping makes PI **less
   visibly destructive without making it edit** — which is the paper's point about PI sharpened,
   not weakened.
3. **IM and IM-NN are untouched (≤ 0.001).** Their outputs are already in range, which is itself a
   small piece of evidence that the inverse-map write leaves the model on-distribution while the
   probe-derived write does not.
4. ⚠ **Do not read the out-of-range FRACTION as a defect.** 20–41% of rays fall outside [0, 1] even
   with no edit at all, but only by ~0.02 (background rays predicted slightly negative), and
   clipping the unedited rollout moves its index by ≤ 0.001. The magnitude is what matters, not
   the count.
5. ND on the continuous target moves *away* from the edited world when clipped (−0.150 → −0.180)
   while its fidelity improves (1.57 → 1.17): clipping pulls a scrambled frame back toward
   background, which is the unedited world on most differing rays. A reminder that the index and
   the guard move independently, which is why both are reported.

**Decision: keep the tables unclipped.** Clipping is a post-processing choice that flatters the
one editor we disqualify, changes nothing else, and would need explaining in the Metrics section.
If a referee asks, this pilot is the answer.
