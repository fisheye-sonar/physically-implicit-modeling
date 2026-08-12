# Direction: is `readable ≠ grabbable` a property of the *rendering*, before any learning?

**Tag:** `[reframe]` · **Sub-question:** 3 (editability) · **Status:** in progress (2026-08-05) ·
**Complexity:** low (no training, no models loaded) · **Models:** none — that is the point.
Branch `orthogonal_edit_analysis`.

Notebook: `notebooks/experiments/editability/orthogonal_edits/observation_space_geometry.ipynb`.
Origin: Sevan, 2026-08-05, from a conversation about whether the renderer is a function and whether it is
linear.

## The gap this closes

Every `readable ≠ grabbable` result so far is stated about a **trained model**: a linear probe reads position
out of `h` accurately, yet writing to that probe's row space does not move the object. The transformer
notebook's §6 sharpened it to geometry — the decoder's preferred direction sits **87–90°** from the
pseudoinverse direction, with a row-space fraction **at or below chance** — but it was still measured
*inside a network*, which leaves two readings open:

| | claim | consequence |
|---|---|---|
| **inherited** | the misalignment is in the rendering geometry | every model trained on this world has it; no architecture, probe, or training change escapes it |
| **chosen** | the network picked an internal layout where these directions disagree | a fact about our models, potentially fixable |

These have very different scope. The first makes `readable ≠ grabbable` a statement about occlusion-based
sensing; the second makes it a statement about GRUs, RSSMs and transformers as we trained them.

## The hypothesis, and why it is plausible a priori

Write the render of one object as a profile placed at a location, `f(p) = g(· − p)`. Then
`df/dp = −g'(· − p)`, and `∫ g g' = ½∫(g²)' = 0` for compactly supported `g` — **a profile and its own
derivative are exactly orthogonal in L²**. Concretely in this world: an object covers `n ≈ 21–33` rays at
constant intensity; moving it one ray changes **only the two edge rays** and leaves the interior untouched.
By hand, `cos ≈ −√(k / 2n)`, giving ≈ −0.15 (99°) for `k = 1`.

So "where the object is" (a wide plateau) and "how to move it" (thin spikes at the edges) are near-orthogonal
directions in observation space — with no learning involved anywhere.

## What to run

The exact structural analogue of transformer §6 with `h ∈ R^256` replaced by the raw clean observation
`o ∈ R^128` and the world model deleted:

| §6 (inside the network) | here (no network) |
|---|---|
| probe `h → position` | probe `clean_obs → position` |
| `Δh_pinv = (target − (A h + b)) A⁺` | `Δo_pinv = (target − (A o + b)) A⁺` |
| decoder's descent direction in `h` | the **required** change `Δo_true = gt_edited − gt_unedited` |
| chance `√(4/256) = 0.125` | chance `√(4/128) = 0.177` |

1. Fit the linear probe on the `test` split, evaluate on `edits` — never fit and score the same rows.
2. Per sample then averaged: `cos(Δo_true, Δo_pinv)`, its angle, and a **shuffled-pair control**.
3. Row-space fraction `‖Qᵀ Δo_true‖ / ‖Δo_true‖` against chance; report the **ratio to chance**.
4. Both a 1-ray **nudge** (the pedagogical case, matching the hand formula) and the full **teleport** (the
   case that matches §6).
5. The demonstration the in-network experiment cannot make: apply the pseudoinverse edit **to the
   observation itself** and render it. The probe then reads the target exactly; look at whether the picture
   moved.

**Probe-quality control is mandatory.** A weak linear probe invites "your probe is just bad". Fit a nonlinear
(MLP) probe on the same inputs and report both. Note the repo's `_fit_mlp` is tuned for `h`-vectors and
**will not converge on position targets without standardising them** (it returns R² ≈ −0.5, a failed fit
that must not be reported as a result).

## Result (2026-08-05) — inherited

- Linear probe R² **0.259**, MLP probe R² **0.754** on the same inputs: the render→position map is strongly
  nonlinear, so a linear readout is weak *by nature*, not by under-fitting.
- `cos(Δo_true, Δo_pinv)` = **+0.073 (86°)** for the teleport, **+0.011 (89°)** for the nudge, against
  shuffled controls of +0.001 and −0.000. Indistinguishable from the null.
- Row-space fraction **0.135 (0.77× chance)** for the teleport, **0.097 (0.55× chance)** for the nudge —
  *below* chance in both cases.
- Applying the injection directly to the observation drives the probe readout to the target to **1.25e-06
  sim units** and closes **−0.1%** of the RMSE gap to the target world. It renders as a low-amplitude ripple
  spread over all 128 rays; the plateaus do not move.

**Verdict: inherited.** The misalignment exists in raw observation space before any learning. §6 was
measuring geometry, not something the networks chose.

## What this does and does not license

It **does** say: no world model trained on this renderer can be edited by writing to a linear position
probe's row space, regardless of architecture. That retires a whole family of proposed fixes (better probes,
better training, different architectures) for *this world*.

It **does not** say editing is impossible — history overwrite and the freeze-time editor still work, because
they act through the observation sequence rather than through a probe subspace. Nor does it generalise
beyond occlusion-style renderers without further work.

## Follow-ons

- **Does it hold for a smooth renderer?** Replace the hard silhouette with a soft/antialiased profile and
  re-measure. The `∫gg' = 0` argument predicts near-orthogonality *survives* smoothing, which would be a
  much stronger claim — the current result could otherwise be dismissed as an artifact of hard edges.
- **Is there a coordinate system where they are aligned?** If position were encoded as a scalar coordinate
  rather than a rendered plateau, the geometry would be benign. What representation would a model need for
  probe-directed editing to work at all? That question is now well-posed.
- **Does the same geometry explain the GRU/RSSM row-space ceilings** measured in `delta-h-analysis`
  (0.096 vs 0.125 chance; 0.005 vs 0.112)? Those numbers are suspiciously close to this one.
