# full_rowspace_edit — what if the probe row space is the WHOLE hidden state?

**Origin:** Sevan, 2026-08-13. **Branch:** `rogerio_controls`.
**Notebook:** `h8_full_rowspace_edit.ipynb` (8 code cells, 3 figures, 0 errors).
**Scratch note:** `research/scratch/2026-08-13-history-editing.md` (§ *Full row space at H=8*).

## The question

Every editability negative in this thread carries the same escape hatch: readout injection writes inside a
**4-dimensional** probe row space, and a successful edit provably lies mostly outside it
(`delta_h_analysis`: row-space fraction 0.096 vs a chance level of 0.125). So — remove the excuse.

At **H = 8** it can be removed *exactly*. Fit a linear position probe (rank 4), project its row space out
of the state, refit (INLP) → a second rank-4 probe orthogonal to the first. **4 + 4 = 8 = H.** The two
probes span the entire hidden state, the reachable fraction of any target `Δh` is **1.0**, and the stacked
map is **square**, so the target readouts determine the new state uniquely.

## Result

**Reachability was never the binding constraint.**

| | probe 1 alone | both probes |
|---|---|---|
| reachable fraction of `Δh_true` | 0.5897 (chance 0.7071 → **0.83×**) | **1.0000** (chance 1.0000 → 1.00×) |
| cos(write, `Δh_true`) | −0.011 (**91°**) | **+0.040 (88°)** |

Raising the ceiling from 0.59 to 1.00 moves the write direction from 91° to 88° — still orthogonal, still
exactly where every probe-derived direction in this thread sits.

At **identical displacement** (`‖Δh‖/‖h‖ = 0.791 = ‖Δh_true‖`): full-row-space injection **−0.277**,
probe-1 injection −0.210, **random direction −0.172**, unsteered −0.489, and the observation-channel oracle
**+0.529 at fidelity 0.81**. The full-rank probe write is *not better than noise of the same size*, and all
three degrade the rollout (fidelity 1.12–1.25).

The literal proposal is also **numerically degenerate**: the stacked map has condition number **14,373**
(smallest singular value 8.8e-04), so demanding both readouts equal the target needs a write of
**‖Δh‖/‖h‖ = 2137** → Target RMSE 660, fidelity 261, saturated garbage. The mechanism is stateable: the two
probes disagree by **0.923 sim units** on real states, so demanding exact agreement asks for a state more
self-consistent than any the model visits.

## How to read it with the rest of the branch

`../history_editing/` closes the same door from the other side: the un-edited complement **is** history, but
*observation-shaped* history, and writing the position history in does nothing a matched-norm random write
does not. Together: it is not that the editor cannot **reach** the right state — it is that a position
readout does not **know where** the right state is.

## Caveats that matter

`H=8` is a much weaker model (unsteered Edit Index −0.489 vs −0.670 at H=256; visibly blurry rollouts in
Fig 3; its own oracle reaches only +0.529). Every comparison here is internal to it. The **H=256 analogue is
not run** — spanning that state needs 64 rank-4 probes — and is the direct follow-on that would generalise
the claim beyond a model small enough for two probes to exhaust.
