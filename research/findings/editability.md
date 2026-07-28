# Finding: Editability (causal manipulability of hidden state)

*Sub-question 3 — are targeted latent edits coherent, intended behavioral changes?*
Model/data context unless noted: GRU `3_dset3_gru_persistentids_inview_400epochs`,
dataset `4_fixed_refl_inview`, 2 objects, N=500 edit samples.
Notebooks: `notebooks/experiments/editability/` (`canonical_state_editing`,
`geodesic_walk_k150`, `manifold_geometry_diagnostic`).

> **Scope (preliminary, 2026-07-09).** These claims concern the *specific trained checkpoint* under
> study — a GRU trained **purely to predict the next observation** (no state supervision), on
> `dataset 4`, at this stage of the investigation. They are **not** claims about GRUs / recurrent
> world models in general. A different training objective (e.g. an editability- or
> disentanglement-aware loss), dataset, or scale could change them. Read "the GRU" below as "this
> pure-next-step-prediction GRU."

## Current understanding (mutable summary)

The probe (decode) direction **IS causally connected to the observations** (the
earlier "decode≠generate" reading was a magnitude-scaling artifact). But editing
still fails, and the barrier is **not** "target unreachability under a manifold
constraint" (superseded — the target readout *is* substantially reachable
on-manifold via a constant-step geodesic, RMSE→0.35). The real barrier is that
**this GRU's `h` is predictively sufficient but non-canonical**: (i) ~35% of `h` is not a
(nonlinear) function of the world's minimal `(pos,vel)` sufficient statistic (the
decode fiber is not collapsed); (ii) the `(pos,vel)→h` embedding is strongly curved
(linear→MLP fiber-residual drop ~0.53), so linear/min-norm edits leave the manifold;
(iii) neither is a clean *linear* coordinate — position is linearly readable
(R² 0.84) but velocity only *nonlinearly* (single-frame MLP R²≈0.94 late-t vs linear
≈0.59); velocity is **not** a temporal feature (see 2026-07-08 log). Consequently
**completing the edit target to the full `(pos,vel)` does not fix the ghost** (obs moves ~1.4%, identical to position-only) — killing the
velocity-incompleteness hypothesis. An unconstrained obs-gradient objective *can*
render the target, but only by jumping to an **off-manifold, non-canonical latent**
(residual 15.7 vs ~1.75 for real states) that the dynamics reject within a few
steps. The probe-objective moves the readout but not the obs; the obs-objective
moves the obs but off-manifold — **readable ≠ controllable**, localized. Forcing an
on-manifold global-PCA edit still yields only ghost-ridden ~37%-of-swap change (it
moves the obs partly by *scrambling*, not clean relocation).

## Log

### 2026-07-16 — A multi-step (rollout) TRAINING objective does not induce editability (GRU) · `established`
Tested whether the editing failure is an artifact of the pure next-step training loss. Trained GRUs with a
**free-running multi-step rollout objective** (teacher-force context, then free-run `w` steps feeding the
model's own decoded predictions, BPTT through the whole imagination, MSE on all `w` frames), `w∈{1,2,5}`,
same architecture/data/hidden/epochs — only the objective changes. Data `datasets/4_fixed_refl_inview`
(**noisy**, `obs_noise_std=0.2`), `w=1` = the standard single-step baseline (`runs/gru/7_dset4_gru_400epochs`);
`w=2,5` = `runs/gru_multistep/w{2,5}_dset4_gru_400epochs`.
**Result — clean NEGATIVE.** The multi-step objective does what it's designed to on the *rollout* (open-loop
horizon RMSE 0.208→0.197→0.188; rollout total-variation sharpness moves *toward* GT, 1.28→1.07, **not** below
— so **no blurry mean-hedging / mode-collapse**) but **buys no editability and no identifiability/canonicality**:
the whole §4 pathology — a **decoder-inert** position-probe direction, **belief sluggishness** (even the
true-state swap moves the obs only ~0.12), and **off-manifold oracle collapse** — replicates essentially
unchanged across `w`. No non-oracle editor approaches the true-state swap for any `w` (best-editor GT
next-step RMSE ≈ Unsteered, ~0.27, vs swap ~0.20). The PCA-geodesic even drives the *readout* progressively
lower with higher `w` (1.20→0.99) while obs/ghost/next-step barely move → the state↔observation **decoupling
is structural, not a budget/geometry artifact**. If anything the objective mildly *degrades* canonicality
(MLP fiber residual 0.357→0.382→0.457; position-linear R² 0.84→0.82→0.76; linear hull + curvature inflate).
**Reading:** the editing difficulty here is a **structural** property of the learned code (decoder-inert probe
direction + single-frame belief inertia), **not** an artifact of a next-step-only loss that a rollout
objective would fix — refuting the "coherence-under-iterated-dynamics ⇒ editable state" intuition for this GRU.
*Scope:* this GRU family, dataset 4 (noisy), `w∈{1,2,5}` only; probes in-sample (cross-`w` deltas are the
load-bearing quantities). **RSSM replication — DONE (2026-07-16, scratch, pending Sevan's artifact-or-signal
review before folding in):** the negative **replicates** on the RSSM (latent-overshooting objective, `W∈{2,5}`;
no editor reaches the true-state swap) AND the objective is *additionally harmful* there — it blurs the decoder
(rollout TV/GT 1.23→0.43, objects fade), worsens single-step + open-loop prediction, collapses the linear hull
(36→10 dims), and reduces linear readability + canonicality. So for the architecture built for multi-step, the
objective buys no editability and costs predictive quality. See `scratch/2026-07-16-multistep-objective-rssm.md`.
Notebook `notebooks/experiments/editability/multistep/multistep_objective_structure.ipynb`; training helper
`scripts/train_gru_multistep.py`; note `scratch/2026-07-16-multistep-objective-structure.md`. *(Two metric
caveats noted in that thread — not affecting this result: the curvature/tangent-rotation number is not
distance-normalised, and any static-target-render metric inflates as the object moves; §4 here uses the
time-evolving clean GT.)*

### 2026-07-08 — Summary rewritten; velocity is instantaneously NONLINEAR, not temporal · `established`
The current-understanding summary now leads with **non-canonicality / readable≠controllable** (from
the 2026-06-24 keystone), replacing the superseded "target unreachability under manifold constraint."
**Velocity correction (resolved, `editability/diagnostic_corrections.ipynb`):** the keystone's
"velocity is a temporal feature" was a confound — it compared single-frame **linear** (0.47) against
2-frame **MLP** (0.76), changing two axes at once. The 2×2 {linear,MLP}×{single,2-frame} on both GRU
and RSSM shows single-frame MLP ≈ 2-frame MLP (Δ ≤ 0.007 late-t both models; GRU single-frame MLP R²
**0.94** late-t), and `dh` differencing is *worse* than single-frame. **Velocity is instantaneously
readable from one `h_t`, just nonlinearly** — the entire 0.47→0.76 gap is the linear→MLP axis, not
single→temporal. "Velocity is a temporal feature" is **RETIRED**. *Strategic:* this reframes the
planned dynamics-identifiability thrust — velocity lives in the **state** (nonlinear/entangled
coordinate), not deferred to the transition.

### 2026-06-24 — Supersedes "target unreachability under manifold constraint." The target IS substantially reachable on-manifold (geodesic constant-step RMSE→0.35). `established`
The real barrier is that the state is non-canonical: position is readable but velocity is not an instantaneous coordinate (lives in the dynamics), the (pos,vel)→h embedding is strongly curved, and ~35% of h is not a function of (pos,vel). Completing the edit target to (pos,vel) does not fix editing; the only h that renders the target is off-manifold and the dynamics reject it. "Readable ≠ controllable."

### 2026-06-23 — Matched-magnitude sweep overturns "decode≠generate" · `established`
σ along directions: probe obj0-x=0.26, probe obj0-y=0.22, PCA#1=2.23, PCA#2=2.22 —
PCA dirs have ~10× larger data-σ. The earlier "probe ≈ random ≪ PCA" result was a
**confound of σ-scaling**. At matched absolute ‖Δh‖=4, the probe direction produces
*more* RMS observation change than PCA or random. Decoder-Jacobian corroborates:
probe projection onto the top-8 decoder-sensitive singular subspace = 0.086 vs
random 0.034 (probe 2.5× better aligned). **The probe direction is generative.**
Caveat: the relationship is nonlinear (divergence mainly at large magnitude), and
realistic probe edits are 10× smaller in σ units — trust the Jacobian numbers over
the high-magnitude tail of the sweep.

### 2026-06-23 — Edit diagnostic table + reversion vs drift · `established`
| edit | readout RMSE | global resid | local resid |
|---|---|---|---|
| real states | — | 1.72 | 0.91 |
| pseudoinv | 0.00 | 1.80 | 1.06 |
| manifold (global PCA) | 0.02 | 0.00 | 1.59 |
| local tangent PCA | 0.62 | 1.86 | 0.93 |

Observation change vs unsteered (swap baseline = 0.317 = full state change):
pseudoinv 0.030 (9.5%), manifold 0.119 (37%), local 0.109 (34%).
Reversion vs drift: **pseudoinverse reverts** to the unsteered trajectory by step
~14 (dynamics project the off-manifold edit away); **manifold/local persistently
diverge** from unsteered but do *not* track the intended GT target — they go
elsewhere on the manifold. Local tangent: local_resid≈0.93 (genuinely on-manifold)
but readout RMSE=0.62 after 50 POCS iters → **the target is not reachable in the
local neighborhood**. → core diagnosis: target unreachability, not non-generative
direction.

### 2026-06-23 — Per-sample heterogeneity + ghost objects · `tentative`
Macro averages obscure strong per-sample structure. Many individual edits *do*
move the decoded position toward the target and persist; others revert or drift —
the mix dilutes aggregates. Qualitatively (waterfalls), the global manifold edit
sometimes places the object correctly (sample 156: bar goes to the correct far-left
GT position) but spawns a **phantom object at the original location** — incomplete
identity displacement. Object-identity *swaps* also observed. *Why tentative:* read
from individual plots, not yet quantified by a stratified per-sample metric.
*Next:* stratify by edit success and characterize what distinguishes persistent
edits from reverting ones.
