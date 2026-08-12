# `readable ≠ grabbable` is inherited from the renderer, not learned by the model

**Date:** 2026-08-05 · **Branch:** `orthogonal_edit_analysis` · **Direction:** `orthogonal-edits`
(`[reframe]`, sub-Q 3) · **Status:** → **FLAG FOR PROMOTION** (it relocates the thread's central negative
from the models to the world) · **Author:** orchestrator, at Sevan's request.

## The question

Every `readable ≠ grabbable` result so far was measured inside a trained model. The transformer notebook's §6
sharpened it to geometry — the decoder's preferred direction sits 87–90° from the pseudoinverse direction,
row-space fraction at or below chance — but *still inside a network*. So: did the network **choose** an
awkward internal layout, or did it **inherit** one from the rendering?

This notebook deletes the network and asks the same question of raw observations.

## Setup / provenance

Notebook `notebooks/experiments/editability/orthogonal_edits/observation_space_geometry.ipynb`
(11 cells, 0 errors, 3 figures), PNGs `/tmp/orthogonal_edits/`. `datasets/4_fixed_refl_inview`, `edits`
split, `ef = 20`, 2 objects, `obs_res = 128`, fixed reflectivities (so the clean render is a function of
positions alone). **N = 2000.** Probe fit on `test`, evaluated on `edits`. **No model is loaded anywhere.**

Structural analogue of transformer §6 with `h ∈ R^256` replaced by `o ∈ R^128`:
probe `clean_obs → position`; `Δo_pinv = (target − (A o + b)) A⁺`; required change
`Δo_true = gt_edited − gt_unedited`; chance `√(4/128) = 0.177`.

## Why one would predict it — the hand calculation

An object covers `n` rays at constant intensity. Nudge it one ray: the change is `−0.4` on the ray that goes
dark, `+0.4` on the ray that lights up, and **exactly zero on the `n − 1` rays in between**. So
`cos ≈ −√(k / 2n)`. For `n = 21`, `k = 1`: **−0.154 (99°)**.

Measured over 1996 samples with per-sample ray counts (mean `n` = 32.6, range 15–54): predicted **−0.125**,
measured **−0.151**. The formula is the mechanism, not a fit.

Underlying it: for `f(p) = g(· − p)`, `df/dp = −g'(· − p)`, and `∫ g g' = ½∫(g²)' = 0` for compactly
supported `g`. **A profile and its own derivative are exactly orthogonal in L².** Position information lives
across the whole plateau; moving the object changes only the edges.

## Results

**1. The linear probe is weak, and that is a property of the map, not the fit.** Linear R² **0.259**;
MLP probe on the same inputs R² **0.754**. Per coordinate the linear probe gets obj0 x 0.04, obj0 y −0.06,
obj1 x 0.50, obj1 y 0.57 — a linear map keys on brightness (reflectivities 0.4 vs 0.8) to tell the two
plateaus apart, so the dimmer object is nearly unreadable.

> **Trap for anyone re-running this:** the repo's `_fit_mlp` is tuned for `h`-vectors and does **not**
> converge on position targets without standardising them (returns R² ≈ −0.5, worse than the mean). That is
> a failed fit, not a result. Standardise `y`, fit, un-standardise.

**2. The required change is orthogonal to what injection can apply** (per sample, then averaged, N = 2000):

| intervention | cos(required, pseudoinverse) | angle | shuffled control | row-space fraction | ÷ chance |
|---|---|---|---|---|---|
| teleport (matches §6) | **+0.073 ± 0.19** | **86°** | +0.001 | 0.135 ± 0.069 | **0.77×** |
| 1-ray nudge | **+0.011 ± 0.09** | **89°** | −0.000 | 0.097 ± 0.050 | **0.55×** |
| *(ref)* transformer W16, last residual point, **in-network** | +0.014 ± 0.042 | 89.2° | +0.001 | 0.071 | 0.57× |
| *(ref)* GRU H=256, **in-network** | +0.034 ± 0.100 | 88.1° | −0.017 | 0.189 | 1.51× |

Both cosines sit inside their shuffled-pair controls, and both row-space fractions are **below** chance.
The observation-space numbers land on top of the in-network ones — 86–89° here versus 87–90° there, and
0.55–0.77× chance here versus 0.57–1.51× there. *(Fractions are not directly comparable across the line —
`R = 128` vs `256`, chance 0.177 vs 0.125 — which is why the ÷ chance column exists.)*

**3. The direct demonstration the in-network experiment cannot make.** Apply the pseudoinverse edit to the
observation itself. The probe then reads the target position to **1.25e-06 sim units** — a perfect write by
its own objective. RMSE to the target world: doing nothing **0.2852**, after injection **0.2856**. It closes
**−0.1%** of the gap; it renders as a low-amplitude ripple spread across all 128 rays while the plateaus
stay exactly where they were (Fig 3).

## What this means

> **`readable ≠ grabbable` is a property of the rendering, not of the models.** The misalignment between
> "the subspace a linear position probe writes in" and "the direction that moves an object" is present in
> raw observation space, before any learning. §6 was measuring geometry the networks inherited, not a layout
> they chose.

The mechanism is elementary and worth stating in exactly these terms: **a linear probe reads the plateau;
moving the object changes the edges; a plateau is nearly perpendicular to the spikes at its own edges.**

This retires a family of proposed fixes for this world — better probes, longer training, different
architectures — since none of them changes the geometry. It also explains why every *working* editor in the
thread (counterfactual overwrite, freeze-time teacher forcing, history overwrite) operates through the
**observation sequence** rather than a probe subspace: that is the only channel that can produce an
edge-shaped change.

## Caveats

- **This world's renderer has hard silhouettes.** `obs_intensity` is the first-hit object's reflectivity,
  flat, no depth shading, no antialiasing — so the clean render is *piecewise constant* in position
  (verified: only the values {0, 0.4, 0.8} appear, and 14 of 25 small position steps changed nothing at
  all). The `∫gg' = 0` argument predicts the orthogonality survives smoothing, but **that is untested** —
  and it is the first thing a referee will ask. See the follow-on.
- Scope is one renderer and one probe family (linear, position-targeted). No claim beyond occlusion-style
  sensing.
- The linear probe's low R² (0.259) is a real weakness of the analogue to §6, where in-network probes reach
  0.76–0.83. It does not undermine the conclusion — injection hits *its own* readout target exactly and the
  render still does not move — but the two setups are not perfectly matched.

## Follow-on #1 RESOLVED same day — it survives a soft, differentiable renderer

`orthogonal_edits/soft_render_geometry.ipynb`, new module `pim/simulator/soft_render.py` (optional
extension; `renderer.py` untouched, all knobs default off), new dataset `datasets/5_soft_render`, new GRU
`runs/soft_render/H256_soft` trained with the **identical** protocol — rendering is the only variable.

**The manipulation was large.** Participation ratio `N_eff` of the change under a nudge (threshold-free:
≈1 = a single-ray spike, ≈n = spread over n rays): hard **1.00** → soft edge **10.40** → + lambert **9.17**
→ + psf blur **15.43**. A **15× spread** of the derivative off the silhouette.

**It changed essentially nothing.**

| | cos(required, pseudoinverse) | angle | row-space fraction | ÷ chance | injection closes |
|---|---|---|---|---|---|
| hard renderer | +0.073 | 85.8° | 0.135 | 0.77× | −0.1% of the gap |
| soft renderer | +0.053 | 87.0° | 0.131 | 0.74× | −2.0% of the gap |

With the **exact Jacobian** `∂o/∂x` from the differentiable backend (unavailable on the hard renderer,
whose Jacobian is 0 almost everywhere): **0.066 = 0.37× chance**, i.e. *further* below chance than the
finite-difference version. Hard-occlusion and soft-occlusion backends agree to 3 decimals, so the
"smoothed but not differentiable" control — an ordinary antialiased simulator — behaves identically.

**And inside the retrained GRU** (quality gate: 0.73× its own noise floor, vs 0.68× for the hard model;
position R² 0.824 vs 0.833): cos(decoder descent, pseudoinverse) **+0.054 (86.9°)**, row-space fraction
**1.11× chance**, readout injection **−0.585 → −0.567** — inert, exactly as in the hard world.

**A prediction I got wrong, recorded.** I told Sevan shading would be the structural knob and that
antialiasing/blur would be inert. It is the reverse: **softening the silhouette does nearly all the
spreading** (1.0 → 10.4) and Lambertian shading adds nothing (10.4 → 9.2), because a dome's slope is
steepest at its rim and zero at its apex — still edge-dominated. Sevan's original instinct ("start with
antialiasing and smoothing") was right.

**Bug found and fixed in the process** (would have invalidated the soft-world numbers): `clean_obs` is
*reconstructed* by the loader as `reflectivities[obs_id]`, which is exact only for a flat renderer. Soft
datasets now store `obs_clean` explicitly and the loader prefers it; `build_edit_zones` now inherits the
dataset's rendering settings, without which the §4 reference worlds would have been rendered hard while the
model was trained soft. Also fixed: the soft renderer's relaxed visibility gate (needed for continuous
coverage) reported a "hit" on nearly every ray, corrupting `obs_id`/`obs_depth`.

## Follow-ons

1. ~~**Soft renderer.**~~ Done, see above — the result survives.
2. **What representation would work?** If position were carried as a scalar coordinate rather than a
   rendered plateau, probe-directed editing would be benign. That question is now well-posed and is the
   constructive counterpart to this negative.
3. **Re-read the GRU/RSSM ceilings** from `2026-08-03-delta-h-analysis` (0.096 vs 0.125 chance; 0.005 vs
   0.112) in this light — they may be the same phenomenon measured through a network.
