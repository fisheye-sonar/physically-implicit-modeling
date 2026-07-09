# Direction: PCA Component → Decoded Position Analysis

**Tag:** `[in-frame]` · **Sub-question:** 1 (geometry) + 3 (editability) ·
**Status:** in progress (2026-06-23) · **Complexity:** low (no new `pim/` code)

> **Interim — NOT promoted; under active investigation.** First pass in
> `notebooks/experiments/editability/pca_component_position.ipynb`; numbers in
> `research/scratch/2026-06-23-pca-component-position.md`. The agent's *decoded-position*
> table read PC0 as a global x-shift (both objects move, R²≈1.0). **But Sevan's read of the
> observation waterfalls disagrees:** across many (not all) samples PC0 moves only the **dim**
> object (obj0, min reflectivity) while the **bright** object stays. The decoded-position table
> and the observation space *disagree* — which is the lead, not a closed result.
>
> **Open question (the actual investigation):** is the bright object's decoded motion *real*
> (the model generates it) or a *probe artifact* (the linear probe reports motion the dynamics
> don't produce)? If the latter, PC0 is a selective handle for the dim object and a local
> decode≠generate case — directly relevant to sub-Q2/sub-Q3 and to `geodesic-walk.md`.
>
> **Extension underway:** add observation-space visualizations (1D intensity scans + waterfalls
> overlaid across the PC0 α-sweep) and a per-object observation-change attribution, to see *in
> observation space* which object actually moves — then compare against the decoded-position
> slopes. Resolved-during-run defs to keep: σ_i = data-std along PC_i; "selective" = slope ratio
> ≥ 3 AND larger |d| > 0.02; renderer bonus needs the α=0 control (mismatch floor ~0.238, a sub-Q2 lead).

## Motivation

The PCA-component waterfall explorer in `editability_structure.ipynb` shows PC0 ±3σ
visibly shifting *one* object while apparently leaving the other in place — exactly
the selective edit we want. But the waterfall shows intensities, not positions. Do
the PCA components map onto decoded object positions, and which component moves
which object? This also probes the σ puzzle (probe-dir σ≈0.26 vs PCA σ≈2.23, see
`findings/editability.md`): if PC0 strongly moves position, the small probe σ means
the probe direction is nearly orthogonal to the high-variance PCA directions.

## What to run

In new cells of `notebooks/experiments/editability/editability_structure.ipynb` (reuses
`states_tf`, `subspace`, `warm`, `linear` already computed there; `N_OBJ=2`,
`USE_HUNGARIAN=False`):

1. For PCA components i=0..5 and magnitudes α ∈ {-3,-2,-1,0,1,2,3}σ: add
   `α·σ_i·PC_i` to each of the 64 warmed-up `h_base` states, roll out 10 steps,
   `decode_pos` the rollout states.
2. **Sensitivity table** (print, don't just plot): slope `∂decoded_pos / ∂α` per
   object per coordinate, at step 0 — `np.polyfit(alphas, pos_obj_j_coord, 1)[0]`.
   Rows = PC index, cols = obj0-x, obj0-y, obj1-x, obj1-y.
3. **2D scatter** of decoded positions across the α sweep for PC0 and PC1, colored
   by object — does one component move one object along a clean line?
4. **Persistence:** repeat the slope at step 0 vs 5 vs 10 — does the displacement
   hold or revert (same reversion question, applied to PCA directions)?

## Questions to answer

- Does any single PC selectively displace one object and not the other?
- Is per-PC position sensitivity consistent with the waterfall impression?
- How does it relate to the probe-σ-vs-PCA-σ gap?
- Does the PCA-direction displacement persist across the rollout?

## Bonus (if the renderer is wired)

For PC0, α ∈ {-3,0,3}: run decoded positions through `pim.simulator.renderer` and
compare the *rendered-from-decoded-positions* waterfall against the actual
autoregressive rollout waterfall. Match ⇒ decoded positions are geometrically
consistent with what the model generates; mismatch ⇒ the probe reads a quantity
that isn't physical position. (This rendering helper is generally useful — see the
"simulation rendering of decoded positions" parked idea.)

## Deliverables

- Printed sensitivity table (PC × object-coord).
- 2D decoded-position scatter for PC0, PC1.
- A one-paragraph promoted-or-not note in `scratch/` → flag for the findings gate.

## Context

- Checkpoint `runs/gru/3_dset3_gru_persistentids_inview_400epochs/best_model.pt`,
  data `datasets/4_fixed_refl_inview`.
- Reuse from the notebook: `decode_pos`, `plot_waterfall_grid`, `subspace`,
  `linear`, the `PC_INDEX/PC_ALPHA/PC_SAMPLE` explorer cell, the Exp-2 sweep cell.
