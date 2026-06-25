# Finding: Editability (causal manipulability of hidden state)

*Sub-question 3 — are targeted latent edits coherent, intended behavioral changes?*
Model/data context unless noted: GRU `3_dset3_gru_persistentids_inview_400epochs`,
dataset `4_fixed_refl_inview`, 2 objects, N=500 edit samples.
Notebook: `notebooks/experiments/manifold_editing/editability_structure.ipynb`.

## Current understanding (mutable summary)

The probe (decode) direction **IS causally connected to the observations** — the
earlier "decode≠generate" reading was an artifact of magnitude scaling. The real
barrier to editing is **target unreachability under manifold constraint**: from a
given state, no local manifold-respecting edit reaches the target-reading state,
because the manifold curves away. Forcing an on-manifold edit (global PCA
projection) produces real, substantial observation change (~37% of a full
state-swap) but with **ghost/phantom-object artifacts** — the model moves the
target object yet fails to cleanly displace the old identity memory. Local tangent
PCA does not beat global in practice and can be dropped from the main editor; the
iterative local-subspace idea survives in the geodesic-walk direction.

## Log

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
