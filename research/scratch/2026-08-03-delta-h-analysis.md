# Δh analysis: a successful edit is large, edit-specific, and invisible to the probe

**Date:** 2026-08-03 · **Branch:** `delta_h_analysis` · **Direction:** `delta-h-analysis` (`[in-frame]`, sub-Q 3) ·
**Status:** → **FLAG FOR PROMOTION** (turns "readable ≠ controllable" from an observation into a measured
reachability ceiling; plus a sharp new memorisation diagnostic) · **Author:** orchestrator, at Sevan's request.

## The question
Exactly two edit mechanisms in this repo reliably work, and **both need oracle access**: **counterfactual state
overwrite** and **freeze-time teacher forcing**. Note what they share — *neither writes to `h` directly*; both work
by making the model **consume observations over time**. (Sevan's observation, and it is the thread's through-line:
we have never found an edit that works without dynamics.) Because they succeed, they give us ground truth for what
a successful edit **is** as a latent displacement, `Δh = h_post − h_pre`. This characterises it.

## Setup / provenance
Notebook `notebooks/experiments/editability/delta_h_analysis.ipynb` (26 cells, 0 errors, 9 figures), PNGs
`/tmp/delta_h_analysis/`. GRU `runs/controls/H256`, RSSM `runs/rssm/4_dset4_refined_best`, both on
`datasets/4_fixed_refl_inview`, `ef=20`, `K=15`. **N = 256 held-out edits** for all statistics (the prior version
of this measurement was N=64, GRU only, one construction). Metrics = the canonical §4 set
(`scripts/editability_metrics.py`). Δh reported **raw** and **edit-only** (with dynamics/rendering controls
subtracted). All cosines and fractions computed **per instance, then averaged**.

**Alignment.** GRU decoder predicts-next, RSSM reconstructs-current, so the RSSM takes one extra prior step to
reach the same "rollout step 0 ↔ frame `ef`" convention, and each probe is fit on the *same state type it is
applied to*. Verified on ordinary (non-edit) sequences: GRU passes cleanly; **RSSM is ambiguous at this precision**
(k=−1 0.1059 vs k=0 0.1067, a 0.8% gap) because its prior decode is blurry — a caveat on the RSSM Δh numbers.

## Headline
**A successful edit is roughly as large as the state itself, points in a direction that is essentially orthogonal
to everything the position probe can see or move, is different for every edit — even for edits that make the *same
positional change* — and does not generalise when learned.** The two independent oracles nevertheless agree strongly on *which* direction it is — so the edit
direction is well-defined; it is simply not reachable by any probe-directed write. **And — against prediction — it
composes:** edits to the two objects add (83–87% of the direct edit's effect), and the latent is largely
path-independent (94%). The structure is there; the probe simply cannot address it.

## Results (GRU; RSSM in brackets where it differs)

**§1 Both oracles succeed, and Sevan's prediction was right.** Counterfactual overwrite **+0.68** [RSSM +0.61],
freeze-time **+0.54** [+0.09], against unsteered −0.67 [−0.65] and readout injection −0.66 [−0.65, i.e. inert).
Counterfactual is the stronger of the two **and holds better** over the rollout (+0.68 → **+0.44** at step 14, vs
freeze-time +0.54 → +0.26) — consistent with Sevan's reasoning that a full overwrite has no pre-edit remnant to
revert to. Contrast the decoder-gradient oracle, which is a *single-frame* success (+0.94 → −0.12).
*Freeze-time is much weaker on the RSSM (+0.09) than on the GRU (+0.54) — worth its own look.*

**§2 The reachability ceiling — the core measurement.** Row-space fraction `‖P_row·Δh‖/‖Δh‖` (also the largest
cosine any injection-style edit could achieve with the truth):
| | GRU | RSSM |
|---|---|---|
| counterfactual Δh | **0.096** | **0.005** |
| freeze-time Δh | 0.073 | 0.003 |
| **chance for a random vector** `√(d/H)` | **0.125** | **0.112** |
**Both are at or BELOW chance.** A successful edit is *less* aligned with the position probe's row space than a
random direction would be. So readout injection could match at best ~10% of the true edit direction on the GRU and
~0% on the RSSM — not because the editor is weak, but because it is confined to a 4-dimensional subspace that the
edit provably avoids.

**Adding velocity to the probe does not help.** GRU 0.096 → 0.110 while chance rises 0.125 → 0.177, so relative
alignment *falls*. The content a successful edit moves is **not physical state** — consistent with the fiber
residual (~0.87 of ‖h‖ is not a function of (pos, vel)); the edit lives in that 87%.

**§3 The two oracles agree — strongly.** cos(counterfactual, freeze-time) = **+0.799** raw, **+0.816** edit-only
[RSSM +0.569 / +0.596], against a shuffled-pair control of +0.023 and a random baseline of +0.062 — a ~13× signal.
Two completely different oracle constructions land on nearly the same displacement, so **"the edit direction" is a
well-defined object**. Meanwhile cos(oracle, readout injection) = **+0.078** [+0.004]: the failing editor is
almost exactly orthogonal to the thing that works.

**§4 Magnitude.** `‖Δh‖/‖h0‖` = **0.97** — the edit is as large as the entire state. It is **14×** larger than the
injection it replaces [RSSM **275×**], and **3.6×** one ordinary dynamics step [5.2×]. So a successful edit is a
large excursion, several times bigger than anything the dynamics normally does in a step.

**§5 No shared edit direction.** Mean pairwise cosine across different edits: **+0.011** [+0.008] — for unrelated
directions the expected mean is **0** (the often-quoted `1/√H` = 0.062 is the *per-pair standard deviation*, not a
floor for the mean), so this is **indistinguishable from zero**. There is no generic "an object moved" direction;
each edit has its own. Magnitude is far more stable than direction (CV 0.28 [0.34]).

**§6 Same displacement, different starting states — Sevan's question.** §5 compares edits that are *different
moves*, so orthogonality there is unsurprising. The sharper test holds the object's positional change fixed
(δ ∈ {(+1.5,+1.5), (+3,0), (0,+3), (−1.5,+1.5), (−3,0)}, n=64 in-frustum samples each) and varies everything
else — starting state, both objects' absolute positions, history.

| | GRU | RSSM |
|---|---|---|
| mean pairwise cosine **within** a fixed displacement | **+0.071** | **+0.084** |
| mean pairwise cosine **across** displacements (§5 control) | +0.011 | +0.008 |
| expected value for unrelated directions | 0 | 0 |

**Holding the displacement fixed multiplies alignment by 6.6× [10.3×] — but the absolute level stays at ≈0.08.**
So the positional change carries *real* information about Δh (the effect is unambiguous against the across-
displacement control), yet is **nowhere near determining it**: the same "move object 1 by (+1.5, +1.5)" produces
essentially unrelated latent displacements depending on where it starts.

This kills the most attractive remaining hypothesis — that the edit map is a lookup table over displacements and
readout injection was merely using the wrong basis. There is no displacement→Δh function to look up. It also
explains §7 directly: a predictor given `(h0, target)` memorises because there is no low-dimensional regularity to
generalise.

**The perspective prediction: half right, and not in the way I predicted.** I predicted the residual
start-dependence would track **depth**, since the same world-space move changes far more rays at y=3 than at y=11.
Fig 6b tests that directly (alignment vs |depth mismatch| between the two starting positions) and finds it **flat**
— r ≈ +0.02 (GRU) / +0.03 (RSSM). *That specific prediction is not supported.*
But Fig 6a shows a clear asymmetry that is perspective-flavoured in a different way: **purely lateral displacements
are ~2.5× more consistent than purely depth displacements** — δ=(+3,0) gives 0.129 and δ=(−3,0) gives 0.106, versus
δ=(0,+3) at 0.050 (GRU; RSSM 0.145 / 0.127 vs 0.054). A sideways move changes *which* rays are hit but not the
object's apparent size; a move in depth changes apparent size by an amount that depends on where it started. So the
start-dependence is concentrated in the depth axis — just not as a smooth function of depth *mismatch*. Worth a
dedicated look.

**§7 Learned from oracle Δh: memorisation, cleanly diagnosed.** Fitting `g(h0, target) → Δh_true` on 1500 disjoint
edits: MLP **train R² 0.951** → **held-out R² 0.088** [RSSM 0.948 → 0.322], held-out cosine +0.48. Applied to
held-out edits it yields Edit Index **+0.01** [+0.09] versus the oracle it imitates at +0.68 [+0.61]. So even with
*ground-truth supervision on a demonstrably working edit*, the map does not transfer. This is `learn_to_edit`'s
"memorisation signature" made precise — and it sits consistently with §5: a map with no shared structure across
edits is exactly the kind that memorises.

**§8 Probe accuracy does not buy reachability (hypothesis refuted).** Across the 8 controls GRUs (probe R²
0.19 → 0.87), the enrichment `f / chance` stays at **0.46–0.89×** with no clean trend. *The raw fraction appears to
fall steeply (0.632 → 0.079), but that is almost entirely the changing chance level* — `√(d/H)` is 0.707 at H=8 and
0.088 at H=512. Correcting for it removes the effect. A more accurate probe does **not** capture more of the edit.

**§7 Compositionality — the surprise, and it goes AGAINST my predictions.** Two tests, both with the composed
state actually applied and rolled out (the decisive readout; vector agreement alone can mislead).

| | GRU cos | GRU: % of the direct edit's Edit-Index gain | RSSM cos | RSSM % |
|---|---|---|---|---|
| **sequential** (p0→p1→p2 vs p0→p2, freeze-time) | +0.904 | **94%** | +0.742 | 77% |
| **object superposition** (counterfactual) | +0.873 | **83%** | +0.815 | 79% |
| **object superposition** (freeze-time) | +0.881 | **87%** | +0.873 | (108%, off a near-zero base — see caveat) |

**(a) The latent is substantially path-independent.** Moving the object via a detour waypoint lands within 94% (GRU)
of where the direct move lands, measured on the Edit Index. I predicted this would fail because the recurrence
retains history; it largely does not.

**(b) Object edits substantially superpose.** `[move obj0] + [move obj1]` recovers **83–87%** of what the
directly-constructed `[move both]` edit achieves, on *both* oracle mechanisms and *both* architectures. I predicted
this would "fail harder"; it is the strongest positive in the editability thread so far. Under the counterfactual
mechanism — which is memoryless, so it isolates the **configuration → latent map** — this says that map is close to
**additively separable across objects**.

**§7c The waterfalls confirm it in observation space (Fig 8a/8b).** Three *randomly drawn* samples (seed 0, not
picked for large displacements), columns: GT both-moved · unedited · move-obj0-only · move-obj1-only · direct
both · **composed**. The composed column visibly reproduces the direct one — both objects appear at their green
target locators and vacate the red-dashed ghosts — where the unedited column leaves both objects at the ghosts. The
one visible artifact matches the numbers: the composed rollout is slightly *brighter/over-shot* in places,
consistent with `‖composed‖/‖direct‖ = 1.13` (the sum overshoots by ~13%).

**How this squares with the rest of the notebook.** The relative residuals are large (0.39–0.69) while the Edit
Index retains 77–94% of the gain. That is exactly what §2 predicts: most of Δh is probe-invisible fiber content that
does not affect the render, so **the part of Δh that matters for the observation composes well even though the
whole vector does not.** The latent is *not* unstructured — it is additively organised in a way that no
position-probe-directed write can address.

> **A construction flaw I caught and fixed.** My first version used the **midpoint** as the sequential waypoint —
> under linear interpolation with matched frame counts, an 8+8 route then traverses *exactly* the same frames as the
> 16-frame direct route (verified: maximum difference 0.0), so the model saw an identical observation sequence and
> the test was vacuous. It reported cos +0.979/+0.987. Replacing the waypoint with a **2-unit perpendicular detour**
> gives the real numbers above (+0.904/+0.742) — the conclusion survives but is materially weaker, and the first
> version would have overstated it.

## Reading
The pieces fit into one statement: **the successful-edit displacement is well-defined per edit, enormous, and
lives almost entirely in the part of the latent no probe over physical state can address.** That is the mechanism
behind "readable ≠ controllable", stated as a measurement rather than an inference — and it explains every
probe-directed failure at once, without appealing to off-manifold-ness or predictor quality.

It also predicts the two learned results we already have: an amortized editor plateaus (§5 — nothing shared to
generalise), and fine-tuning wires a button (the map has to be memorised per edit).

## Caveats / open
- RSSM ±1 alignment is ambiguous at measurement precision (0.8% gap); its Δh numbers carry that uncertainty. Its
  near-zero row-space fraction (0.005 vs chance 0.112) is a 22× *depletion* and deserves a dedicated check.
- Freeze-time is far weaker on RSSM (+0.09) than GRU (+0.54); not investigated here.
- One checkpoint per architecture; N=256 edits; the Δh predictors are small (1500 training pairs) — the
  memorisation verdict would be firmer with a data-scaling curve.
- Row-space analysis is for *linear* probes. A nonlinear probe has no "row space", so the ceiling argument does not
  transfer directly to the MLP-probe-gradient editor.
- §6 uses 5 canonical displacements at n=64; a finer grid (or matching on depth as well as displacement) would
  sharpen how much of the residual is perspective versus genuinely state-dependent.
- §7 RSSM freeze-time superposition reports "108% of the direct gain", but the direct edit there scores only +0.03
  (freeze-time is weak on the RSSM, §1), so the ratio is off a near-zero base and should not be read as a result.
- §7 uses one detour geometry (2 units perpendicular) and one displacement pair; the *degree* of composability
  would need a sweep over detour size and object separation.

## Open questions for Sevan
- Artifact or signal? The below-chance row-space fraction is the load-bearing number and it replicates across
  8+2 models and two independent oracles.
- Does this fold into `findings/editability.md` as the *mechanism* behind the existing negative?
- The obvious follow-up: if Δh lives in the fiber (the 87% not explained by (pos,vel)), can we characterise *what*
  that content is — and is that the thing explicit object scaffolding would have to supply?
