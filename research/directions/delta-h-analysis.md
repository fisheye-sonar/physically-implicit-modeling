# Direction: Δh analysis — what does a *successful* edit look like in latent space?

**Tag:** `[in-frame]` · **Sub-question:** 3 (editability) · **Status:** active (2026-08-03) ·
**Complexity:** low-medium (no retraining except small Δh predictors; two oracle constructions × 2 models) ·
**Models:** GRU **and** RSSM. Branch `delta_h_analysis`.

Notebook: `notebooks/experiments/editability/delta_h_analysis.ipynb`. Origin: Sevan's request.

## The gap this closes

Every §4 negative so far is stated in terms of what *fails*: probe-directed writes do not move the object. But we
have never characterised what a **successful** edit actually *is* in latent space — so "the edit map is
unreachable" has been an inference, not a measurement.

Two mechanisms do reliably work, and both need oracle access:
1. **Counterfactual state overwrite** — invent a history in which the object always travelled through the target,
   render it, teacher-force along it, overwrite the pre-edit state.
2. **Freeze-time teacher forcing** — freeze the world, teacher-force ~8 rendered interpolation frames, resume.

They share something worth stating as the thread's through-line: **no successful edit is free of dynamics.** Every
mechanism that works operates by making the model *consume observations over time*; none writes to `h` directly.

Because they succeed, they hand us ground truth for the displacement `Δh = h_post − h_pre`. This direction
characterises it.

## The framing that makes it sharp

Readout injection produces `Δh_pinv = A⁺(target − (A h + b)) ∈ row(A)` **by construction**. So the fraction of
Δh_true lying in `row(A)` is not descriptive — it is the **hard ceiling on how much of a successful edit that
editor could ever achieve**, and `‖P_row Δh‖ / ‖Δh‖` is exactly the best cosine any injection-style edit could
reach with the truth. That converts "readable ≠ controllable" into a measurement.

## Questions

1. **Do the two oracles actually succeed** — on the canonical Edit Index, at the edit frame *and across the
   rollout*? (Landing an edit and holding it are different: the decoder-gradient oracle scores +0.94 then decays
   to −0.12.)
2. **Where does Δh live** relative to the linear probe — row space vs null space, against the `√(d/H)` chance level?
   Does adding velocity to the probe capture more?
3. **Do the two oracles agree** on the displacement? And how does Δh_true compare with what the failing editor does?
4. **How big is it** — relative to the state, to the injection it replaces, and to one ordinary dynamics step?
5. **Is the direction consistent across edits**, or does every edit have its own?
6. **Can it be learned** from oracle demonstrations — and does that generalise to held-out edits?
7. **Does probe accuracy buy reachability?** (x-axis borrowed from the 8 `runs/controls/` GRUs, probe R² 0.19–0.87.)

## Design notes that matter

- **Baselines are mandatory, or the numbers invert.** The null space is `H−d` of `H` dimensions, so a random vector
  already scores `√(d/H)` in the row space; and two random vectors in `H` dimensions have cosine ≈ `1/√H`. Both are
  reported everywhere. *The chance level varies enormously across the H=8…512 sweep (0.707 → 0.088), so that panel
  must plot **enrichment** `f/chance`, not the raw fraction — plotting raw manufactures a trend that is not there.*
- **Per-instance then average**, for every cosine and fraction. Averaging the vectors first would measure agreement
  of the means and wash out the per-edit structure under study.
- **Dynamics/rendering controls.** Both raw Δh's are confounded: freeze-time also advances 8 steps of ordinary
  dynamics, and the counterfactual is rendered *clean* while `h0` comes from *noisy* observations. Matched controls
  (`hold the object at its pre-edit position`; `render the true history`) give an "edit-only" Δh alongside the raw
  one. Both are reported — the raw version is the honest "state we know works minus where we started".
- **±1 alignment.** GRU predicts-next, RSSM reconstructs-current: the RSSM needs one extra prior step to reach the
  same convention, and each probe must be fit on the **same state type it is applied to**. Verified by measurement
  on non-edit sequences (the check must *not* use the edits split — there the pre-edit state legitimately fails to
  predict frame `ef`, which is the effect under study, so the check would be confounded).

## What would count as an answer

- **Δh sits mostly in the row space** → probe-directed editors are the right idea and just need better probes.
- **Δh is at or below chance in the row space** → the failure is structural: any linear-probe-directed write is
  confined to a subspace the edit avoids, and no amount of tuning fixes it. *(This is what was found.)*
- **The oracles disagree on Δh** → many latent states render the same scene and "the" edit direction does not exist.
- **Δh is learnable and transfers** → an edit map exists and the whole thread's negative is a search problem.

## Deliverables

The notebook above, PNGs to `/tmp/delta_h_analysis/`, and a dated `research/scratch/2026-08-03-delta-h-analysis.md`.
Metrics from the canonical set only (`METRICS_AND_EDITORS.md` §4 / `scripts/editability_metrics.py`).
Do NOT edit `findings/` or `RESEARCH.md`.
