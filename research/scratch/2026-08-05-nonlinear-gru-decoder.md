# Nonlinear GRU — the superposition *evidence* was a decoder artifact; the *result* holds, and everything else survives

*Scratch, 2026-08-05. Notebook: `notebooks/experiments/editability/nonlinear_gru/nonlinear_gru_findings.ipynb`
(21 cells, 0 errors, 10 figures). Registry: `notebooks/experiments/editability/nonlinear_gru/NONLINEAR_GRU_RUNS.md`.
Dataset `4_fixed_refl_inview`, N=256 edits (§3–§4), 67 in-frustum samples (§5).*

## Where this came from

Sevan's question about `delta_h_analysis` §7: *isn't the object-superposition finding just an artifact of a
linear decoder?* `decode(h0+d1+d2) = decode(h0+d1) + decode(h0+d2) − decode(h0)` holds **identically** for any
vectors when `decode` is a single `nn.Linear`. He also noted the erasure worry — "the previous object locations
of the unmoved object need to be erased" — which the `−decode(h0)` term does automatically.

He was right, and the follow-up question was his too: **train a GRU with a nonlinear encoder and decoder and
see which findings survive.**

## What was built

`pim/world_models/gru/model.py` gained `enc_hidden_layers` / `dec_hidden_layers` / `mlp_activation`, all
defaulting to the original architecture. The extra blocks live in separate `enc_trunk` / `dec_trunk`
submodules that are **absent from `state_dict` at depth 0**, so every pre-existing checkpoint loads unchanged
and produces bit-identical output (asserted: `forward`, `step`, `decode` all max-diff exactly `0.0`).
Encode/decode were routed through single `_enc` / `_dec` choke-points so depth cannot apply on some code paths
and not others. `tests/test_gru_mlp_depth.py`, 10 tests. Four runs, identical recipe, differing only in depth
and seed — see the registry.

## Results

**The artifact is confirmed — as a statement about the *evidence*, not the *result*.** On both affine-decoder
models the composed decode equals the affine prediction to **6.6e-08 / 9.0e-08** — machine precision. Their
composed Edit Index (**+0.46**) is algebraically determined, so on those models the number could not have
falsified anything, whatever it read. It also sits at 90% of the **+0.51** ceiling that the *model-free* render
identity (`GT_A + GT_B − GT_BASE` vs `GT_AB`) scores by itself. That identity is leaky rather than exact —
RMSE 0.177 against an RMS signal of 0.306, because the two objects share rays in **42%** of samples — which is
why the ceiling is +0.51 and not +1.0.

**On the nonlinear models compositionality is REAL, object-specific, and holds.** Their decoders depart from
affine by **8.5e-02…8.8e-02** (about half their own total error), so the composed index is a genuine
measurement. Against the proper null models:

| | unedited | random Δ, matched norm | composed w/ **wrong** object B | **composed** | direct |
|---|---|---|---|---|---|
| nonlinear enc+dec s0 | −0.74 | −0.46 | −0.18 | **+0.44** | +0.72 |
| nonlinear enc+dec s1 | −0.74 | −0.46 | −0.20 | **+0.43** | +0.71 |
| nonlinear dec only s0 | −0.74 | −0.45 | −0.19 | **+0.43** | +0.72 |

Summing two independently-built per-object displacements renders **both** objects approximately where they
belong. Substituting another sample's object-1 delta collapses it to the unedited side (−0.18), so it is
**object-specific**. This is a *stronger* result than the linear models could support, where the identical
number was unfalsifiable.

**My first framing of this was wrong and is corrected here.** I led with "composed does not beat the affine
prediction, so the nonlinearity is a tax rather than evidence of object structure." The affine prediction is
**not a null model** — it already presupposes each single edit works. "Composed ≈ affine" means the nonlinear
decoder is near-affine along these particular directions; it says nothing against compositionality. The real
nulls are random-Δ and wrong-object-B, and composed clears both by a wide margin.

**It is partial, not exact.** Composed recovers ~82% of the index gain from unedited to `direct` (+0.72) and
lands slightly below the **+0.51** render-identity ceiling. Fig 7 sample 47 shows the shortfall: COMPOSED
develops banding where DIRECT stays clean.

**In state space it is real everywhere but *weaker* with a nonlinear read-out** — the opposite of what a "the
shallow decoder was hiding the structure" story predicts. `cos(composed, direct)` falls **+0.873 (29°) →
+0.784…+0.801 (37–38°)**, relative residual **0.52 → 0.74–0.78**, against floors **+0.31…+0.37**
(wrong-object-B) and **~+0.05** (fully shuffled).

**Everything else survives.**

| finding | baseline (affine dec) | nonlinear enc+dec | verdict |
|---|---|---|---|
| next-step RMSE vs clean | 0.1041 | 0.1029–0.1033 | nonlinear is *slightly better*; no quality confound |
| linear position R² (held-out) | 0.815 | 0.803–0.805 | still linearly readable |
| readout injection Edit Index | −0.66 (unsteered −0.67) | −0.65 (unsteered −0.67) | **inert on every model** |
| counterfactual overwrite | +0.68 | +0.68 | identical across all five models |
| row-space fraction ÷ chance | 0.76× | 1.05–1.24× | still chance; ≤15% of a successful edit is visible |

**One genuinely new result.** The **decoder-gradient oracle weakens sharply on a nonlinear decoder**: +0.97
(baseline) / +1.00 (H512) → **+0.68…+0.72**. Against an affine decoder that oracle solves a *convex*
least-squares problem in `h`; through an MLP it descends a nonconvex objective. Its near-perfect score on the
linear models was partly the decoder's convexity, not evidence that the state was that reachable. Worth
remembering wherever that oracle is quoted as an upper bracket.

## Reading

The editability negative is now known to be independent of the **read-in/read-out nonlinearity**, on top of
capacity (H=8…512), observation noise, action training, and architecture (GRU/RSSM). That is a fifth
independent axis. The barrier remains the **reachability of the edit map**: same model, same decoder, same
rollout, oracles at +0.68 and the structural editor at its own unsteered floor.

Fig 4 makes it visible — readout injection is pixel-indistinguishable from unsteered, object parked on the red
ghost locator, while all three oracles put it on the green target. Fig 7 sample 47 shows the composed state
degrading into banding where direct stays clean, a difference the Edit Index compresses away.

## Method notes (two real errors caught in-flight)

1. **Linear probe R² was in-sample while the MLP's was held-out** — not like-for-like, and it produced the
   nonsense of velocity MLP R² (0.371) *below* linear (0.502). Both now fit on the same 80% of sequences and
   score on the same held-out 20%; velocity reads 0.366 linear / 0.371 MLP, and position drops 0.842 → 0.815.
2. **The §5 waterfall's GT column went black** on samples where the displaced objects leave the frustum,
   making the comparison vacuous. Displayed samples are now restricted to ones that stay in frustum for the
   whole rollout — a stated filter affecting only what is drawn, since every §5 number is scored at step 0 on
   the full set.

Also fixed a harness bug this surfaced: `METRICS_AND_EDITORS.md` still mandated the **shared teacher-forced
`ef` row** that `CLAUDE.md` banned on 2026-07-30, and `CLAUDE.md` explicitly names that file as the path by
which the error leaked back into the `controls/` notebooks. The registry now carries the corrected spec.

## Follow-ons

- Depth/activation were fixed at 2 blocks / ReLU. No sweep. A deeper decoder might move the state-space cosine
  further; the trend so far points *down*, which would be worth confirming.
- §5 uses **one** displacement pair (obj0 `+2,+1`, obj1 `−2,+1`) and N=67. The state-space cosine deserves a
  displacement sweep before it is promoted.
- The baselines carry one seed each; only the nonlinear variants have two.
- `delta_h_analysis` §7's Edit Index columns are **not evidence on that (affine-decoder) model** and must not be
  cited without the §7b control. The claim they were making is independently confirmed here, on decoders where
  it can be tested — so the finding is upgraded, not retired.
