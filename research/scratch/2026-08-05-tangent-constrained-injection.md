# Tangent-constrained injection: a new direction, and it still does not edit

**Date:** 2026-08-05 · **Branch:** `orthogonal_edit_analysis` · **Direction:** `orthogonal-edits`
(`[in-frame]`, sub-Q 3) · **Status:** in progress · **Author:** orchestrator, at Sevan's request.

## The idea

`Wᵀ Δ = δ` is 4 equations in `H = 256` unknowns, so its solution set is a 252-dimensional affine subspace.
Plain readout injection takes the **minimum-norm** member, `Δ = (Wᵀ)⁺δ ∈ col(W)` — a fixed 4-d subspace
chosen by the probe with no reference to where the state is. Sevan's proposal: take a different member —
the one lying in the **local tangent space** of the visited-state manifold at `h₀`.

With `B ∈ R^{H×d}` an orthonormal local-PCA basis (k = 512 nearest visited states):
`Δ = Bc`, `WᵀBc = δ`, `c = (WᵀB)⁺δ` ⟹ **`Δ_tan = B (WᵀB)⁺ δ`**. For `d ≥ 4` this satisfies the readout
requirement exactly *and* stays on the manifold.

## Setup / provenance

`orthogonal_edits/tangent_constrained_injection.ipynb` (14 cells, 0 errors, 4 figures), PNGs
`/tmp/tangent_injection/`. GRU `runs/controls/H256` on `datasets/4_fixed_refl_inview` (the **hard**
renderer, so this is comparable to every previous §4 result). `ef = 20`, `K = 15`, **N = 256**. State bank
**58,500** visited states from `test`. Local PCA via `pim.editors.manifold_steering.fit_local_subspace`;
PCA is nested so `B_d = B[:, :d]`. Oracle `Δh_true` = counterfactual overwrite (Edit Index **+0.644** vs
unsteered **−0.644**, so it is a real target). Alignment verified: k=−1 0.1033 / k=0 **0.1031** / k=+1
0.1182 — min at k=0, though the k=−1 gap is only 0.2%.

## Results

**1. The local manifold is ~22-dimensional, not 5–10.** Local PCA needs **22** components for 90% of local
variance and **33** for 99%. Sevan's estimate was low; `d = 8` captures only 73.5%.

**2. `Δh_true` genuinely lies in the tangent space — well above chance.**

| d | local variance captured | fraction of `Δh_true` in `span(B)` | chance `√(d/H)` | ÷ chance |
|---|---|---|---|---|
| 4 | 0.552 | 0.407 | 0.125 | **3.26×** |
| 8 | 0.735 | 0.568 | 0.177 | **3.21×** |
| 16 | 0.862 | 0.724 | 0.250 | 2.90× |
| 32 | 0.937 | 0.788 | 0.354 | 2.23× |

So the tangent space is a *meaningful* constraint — unlike the probe's row space, which sits at or below
chance. This is the first subspace in the thread that is enriched for the true edit.

**3. The editor is genuinely new — and it does not edit.**

| editor | readout error | ‖Δ‖ | Edit Index | Target RMSE ↓ | Ghost RMSE ↓ | fidelity ratio |
|---|---|---|---|---|---|---|
| unsteered | — | — | −0.644 | 0.488 | 0.578 | — |
| plain injection `(Wᵀ)⁺δ` | 8.7e-07 | 0.53 | −0.633 | 0.482 | 0.575 | 1.00 |
| inject then project onto `span(B)` | 3.1e+00 | 0.08 | −0.642 | 0.488 | 0.577 | 1.00 |
| **tangent-constrained `B(WᵀB)⁺δ`** | 2.1e-06 | 9.60 | **−0.290** | 0.488 | 0.610 | **1.14** |
| **oracle projected onto `span(B)`** | 3.4e+00 | 3.16 | −0.197 | 0.453 | 0.432 | 0.91 |
| oracle: counterfactual overwrite | 1.7e+00 | 5.77 | **+0.644** | 0.107 | 0.115 | 0.60 |

It *is* a new direction: cosine **+0.069** with plain injection, +0.031 with inject-then-project. And its
Edit Index moves a long way, −0.644 → **−0.290**.

**But that movement is degradation, not editing.** Target RMSE is unchanged (0.488 = unsteered), Ghost RMSE
gets *worse* (0.610 vs 0.578), and the fidelity ratio is **1.14 > 1**. The Edit Index rises only because
the output moves *away* from the unedited world without approaching the edited one — driving the index
toward 0, not toward +1. **Edit Index alone would have scored this a success; the fidelity ratio is what
catches it.** (My first pass at the summary logic did exactly that and had to be rewritten.)

**4. The scale sweep confirms it.** Over `α ∈ [0.25, 32]` the tangent editor's Edit Index climbs to
**+0.007** at `α = 32` — but at that scale Target RMSE is **3.997** (vs 0.488 unsteered) and the fidelity
ratio is **16.7**. The index approaches 0 from below and stops there, which is the signature of an output
degenerating to garbage, not of an edit landing.

**5. The decisive control — the tangent-space ceiling.** Project the *working oracle* onto `span(B)`: the
best any tangent-constrained editor could possibly do. It keeps 57% of `Δh_true`, has cosine **+0.568
(55.4°)** with it, does not degrade (fidelity **0.91**), slightly improves Target RMSE (0.453) — and still
scores only **−0.197**. **Keeping 57% of the true edit produces essentially none of the effect.**

## 6. What it generates — the waterfalls settle the failure mode

Fig 5 (canonical spec: gray on dark, 6 noisy context frames above the edit line, every column free-running
from step 0, green target / red-dashed ghost) shows all seven arms plus the tangent editor at ×4 and ×32:

- **unsteered, plain injection, inject-then-project** — visually indistinguishable; the object sits on the
  red ghost locator and stays there.
- **tangent-constrained (α = 1)** — visible vertical **streaking**; the object does not reach green.
- **oracle projected onto span(B)** — the object's band washes out and blurs but stays near the ghost.
- **oracle: counterfactual overwrite** — the object is cleanly *at* the green target. This is what a
  working edit looks like.
- **tangent-constrained ×4 and ×32** — vertical-stripe garbage.

This fixes the failure-mode word. Per `CLAUDE.md`'s precise-language rule the tangent editor **collapses**
(the output degenerates off-distribution); it does not *revert* (return toward the unsteered trajectory) or
*drift*. Plain injection is the one that is inert. Four scalar figures had not distinguished these; the
waterfall does it at a glance.

## What this means

Two things, and the second is the more useful.

> **The tangent space is the first subspace we have found that is genuinely enriched for the true edit
> (3.2× chance), and constraining to it is still not enough.** The binding constraint on probe-directed
> editing is not manifold membership.

> **The edit appears to be close to all-or-nothing.** Retaining 57% of `Δh_true` in the right subspace,
> pointing 55° away, with no degradation, yields an Edit Index of −0.197 against an oracle's +0.644. There
> is no partial credit: a partially-correct displacement does not produce a partially-correct edit.

That second point is new and worth testing directly — it predicts a sharp threshold in a sweep of
`h₀ + β·Δh_true` for `β ∈ [0, 1]`, which is a cheap next experiment and would turn "appears to be" into a
measurement.

## Caveats

- One model (GRU H=256), one world (hard renderer), one probe family (linear, position-targeted).
- `d` was swept for the span fraction (4…32) but the editor itself was only run at `d = 8`. Given that the
  local manifold is ~22-d, running the editor at `d = 16` and `d = 24` is an obvious gap.
- The alignment check passes but k=−1 is within 0.2% of k=0, the same near-tie flagged for the RSSM.
- Local PCA uses a fixed `k = 512` neighbours; the tangent estimate's sensitivity to `k` is untested.

## Follow-ons

1. **Partial-credit sweep** `h₀ + β·Δh_true`, `β ∈ [0, 1]` — is the edit really all-or-nothing, and where
   is the threshold? Directly tests the claim above and needs no new machinery.
2. Run the tangent editor at `d ∈ {16, 24, 32}` to match the measured local dimension.
3. The same construction with a *nonlinear* readout constraint, since the linear probe's row space is the
   thing that is at chance.
