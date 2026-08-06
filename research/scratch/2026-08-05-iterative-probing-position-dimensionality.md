# Iterative probing — the linear position code is 116 dimensions, not 4 — and writing to all of it still does not edit

*Scratch, 2026-08-05. Notebook: `notebooks/experiments/editability/iterative_probing/iterative_probing.ipynb`
(17 cells, 0 errors, 5 figures). Model: GRU `runs/controls/H256`. Dataset `4_fixed_refl_inview` — `test` split
(78,000 aligned states from 2,000 sequences, split by sequence) for Part 1, `edits` split (N=256) for Part 2.*

## Sevan's question

Fit a linear position probe, **project its 4-dim row space out of `h` entirely**, fit a fresh probe on what
remains, repeat until chance. How many probes, and how big is the whole position-decoding subspace — `4 ×
#probes`?

This is **Iterative Nullspace Projection** (INLP, Ravfogel et al. 2020), used to *measure* a linear code's
dimensionality rather than to erase an attribute.

## Answer

**29 probes, every one exactly rank 4, → 116 dimensions** (of which the **112** removed before the last probe
already kill readability). The `4 × #probes` arithmetic holds, and it holds for
a reason worth stating: `np.linalg.lstsq` returns the **minimum-norm** solution, so each new probe's rows land
inside the row space of the already-deflated design matrix, hence orthogonal to everything removed. Both the
rank and the orthogonality (max |inner product| < 1e-6) are **asserted at every step**, not assumed.

**The decay is gradual, so the "dimensionality" is threshold-dependent:**

| dims removed | 0 | 24 | 44 | 68 | 88 | 112 |
|---|---|---|---|---|---|---|
| position R² (held-out) | 0.822 | 0.479 | 0.236 | 0.091 | 0.049 | 0.020 |

Half the readability is gone by **24 dims** (6 probes) — the core — with a long thin tail out to 112.

## The controls, which is where the result gets sharp

**Random-ablation control (the decisive one).** Remove **112 random** dimensions from the same space and
position is still readable at **R² 0.767**, against **0.020** for the 112 chosen ones. The collapse is about
*which* directions, not how many. Shuffled-label floor: **−0.003**, so chance really is 0 here.

**The position directions carry BELOW-average state energy.** After removing 112 dims the iterative track
retains **63.4%** of state energy; the random track retains **56.9%** (≈ isotropic expectation 56.3%). So
position is *not* written along the state's dominant variance axes — the probe reads it off comparatively
quiet directions. Mildly surprising and worth remembering.

**116 dims is the size of the LINEAR code, not of the information.** An MLP probe refit on the fully deflated
states still reads position at **R² 0.544** (from 0.909 undeflated) while the linear probe is at 0.020.
Deflation removed the linearly-readable *form* of position, not position.

**Scale.** The states occupy **40 / 75 / 172** PCA dims at 90 / 95 / 99% variance, not 256. So the linear
position code spans **65% of the subspace the states actually live in** — most of the state, not a corner.

## Why this matters to the editability thread

Every §4 row-space number in this thread is measured against **one probe's 4 dimensions**, and that 4-dim slice
is the entire reachable set of a readout-injection editor. The linear position code is **28× larger**. The
"readable ≠ controllable" story has been quantified against an object that turns out to be a small fraction of
the structure it stood in for. That does not overturn the negative — injection is still inert, and the oracle
Δh is still at chance *in the probe's row space* — but it reframes what that chance-level number means.

## Part 2 — editing in that subspace (run 2026-08-05, same notebook)

Sevan: can we make one edit that changes the readout of *every* probe? Yes, and it is exactly solvable — the
row-space blocks are mutually orthogonal, so `AAᵀ` is block diagonal and the min-norm solution decomposes as
`Δh = Σ_k A_k⁺ δ_k` with `A_j Δh = δ_j` for every `j`. **The multi-probe editor is the sum of 29 independent
readout injections, one per orthogonal slice, with no interference.**

**On weighting — my first justification was wrong.** I said the probes' contributions had to be arbitrated;
they don't, since all 29 constraints are exactly and simultaneously satisfiable. The real reason is that the
constraint is **false** for a low-R² probe: on a genuine state with the object at the target, probe 29
(R² = 0.02) reads approximately the population *mean*, not the target. Demanding it read the target asks for a
state no real state resembles — and it is expensive, since low R² means small `A_k`, hence large `A_k⁺`, while
`δ_k` is simultaneously *larger* (the readout sits near the mean, not near the current position). So shrinkage
means `target_k = μ + R²_k(target − μ)`: ask each probe for what a real edited state would actually give.

**RESULT — it does not work, and the ceiling arm proves it is the subspace, not the targets.**

- **A successful edit is not preferentially in the position code**: `‖P_S·Δh_true‖/‖Δh_true‖` = **0.567** vs
  matched chance `√(116/256)` = **0.673** — **0.84× chance**, i.e. slightly *below* random.
- **The ceiling.** Setting every probe's target to what it reads on the oracle post-edit state yields, by an
  identity verified to 2.7e-15, exactly `P_S(Δh_true)` — the projection of the *working* edit onto the position
  subspace, and the best any editor confined there could do. It scores **−0.23** (unsteered −0.67, oracle
  +0.68): **33% of the oracle's gain from 57% of its vector**, fidelity 0.88. No target-choosing scheme confined
  to this subspace can do better.
- **The `K=29` uniform arm is a trap.** Best-looking structural index (**−0.06**) but fidelity **1.57**, Target
  RMSE 0.787 (unsteered 0.488), Ghost 0.774 (0.589), Collateral 0.610 (0.125), ‖Δh‖ = 2.63×‖h0‖ (oracle 0.97).
  Every zone worse than doing nothing — the index moved toward 0 by **degrading** the output, which is exactly
  what the Edit Index is built to read as ≈0. The waterfall shows smeared bands and vertical striping.
- **Shrinkage works as designed but does not rescue it**: displacement 0.51×‖h0‖, fidelity **0.96** (no damage),
  index **−0.40** — the best non-degrading structural arm, a real move from −0.67, still far from editing.
- **Nothing persists.** Every structural arm is flat or drifting to 0 across the rollout; only the counterfactual
  oracle holds (+0.68 → +0.44).

**A striking replication.** The tangent-constrained experiment projected the *same* oracle onto a completely
different **22-dim local-PCA** subspace: kept **57%** of `Δh_true`, scored **−0.197**. Here a **116-dim** subspace
built on an unrelated criterion also keeps **57%** and scores **−0.23**. Two unrelated subspaces, same retained
fraction, same near-total loss of effect — strong evidence for the **all-or-nothing** reading: the edit does not
decompose into a part that works and a part that doesn't.

**On the 116 ≈ 128 coincidence Sevan spotted.** The decoder is `Linear(256 → 128)`, so `h` splits into 128 dims
reaching the observation immediately and 128 acting only through the recurrence. The position code is
**substantially aligned** with the observation-reaching half (median principal angle **41°**, 62/116 below 45°),
and `Δh_true` has 0.635 of its norm there vs chance 0.707. Position is *not* hiding in memory-only directions —
that is not why writing to it fails.

## What this leaves open

The obstacle is now bounded from an additional side: it is not the *probe's* 4-dim slice, and it is not the
linear position code either. Two subspaces containing 57% of a working edit both yield ~1/3 of its effect. The
next question is what the missing 43% is — it is not manifold-tangent (tangent experiment) and not
position-coding (this one). A direct characterisation of `Δh_true`'s complement, rather than another candidate
subspace to project onto, looks like the more efficient move.

## Caveats

- The deflation is **greedy**: the count is a property of this removal order as well as of the model, and is best
  read as an upper bound on a minimal spanning set.
- The `R² < 0.02` stop is arbitrary; the notebook's Table 2 gives 24 / 44 / 68 / 88 / 112 at thresholds
  0.50 / 0.25 / 0.10 / 0.05 / 0.02.
- One model (GRU H256), one seed, **position only** — velocity not tested. Probes use all timesteps, including
  the early frames where the filter has not converged.
