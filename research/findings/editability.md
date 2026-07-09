# Finding: Editability (causal manipulability of hidden state)

*Sub-question 3 — are targeted latent edits coherent, intended behavioral changes?*
Model/data context unless noted: GRU `3_dset3_gru_persistentids_inview_400epochs`,
dataset `4_fixed_refl_inview`, 2 objects, N=500 edit samples.
Notebooks: `notebooks/experiments/manifold_editing/` (`canonical_state_editing`,
`geodesic_walk_k150`, `manifold_geometry_diagnostic`).

## Current understanding (mutable summary)

The probe (decode) direction **IS causally connected to the observations** (the
earlier "decode≠generate" reading was a magnitude-scaling artifact). But editing
still fails, and the barrier is **not** "target unreachability under a manifold
constraint" (superseded — the target readout *is* substantially reachable
on-manifold via a constant-step geodesic, RMSE→0.35). The real barrier is that
**`h` is predictively sufficient but non-canonical**: (i) ~35% of `h` is not a
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

### 2026-07-08 — Summary rewritten; velocity is instantaneously NONLINEAR, not temporal · `established`
The current-understanding summary now leads with **non-canonicality / readable≠controllable** (from
the 2026-06-24 keystone), replacing the superseded "target unreachability under manifold constraint."
**Velocity correction (resolved, `manifold_editing/diagnostic_corrections.ipynb`):** the keystone's
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
