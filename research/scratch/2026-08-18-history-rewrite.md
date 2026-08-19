> ## ⚠ SUPERSEDED THE SAME DAY by `2026-08-18-history-rewrite-renderer-free.md`
>
> This note reported the **render-based** history rewrite (+0.626) and read it as "coherence of the
> evidence, not precision of the write". That reading was **under-determined**: it rested on the one
> arm that uses the simulator, with no renderer-free arm to compare against. When the edit Sevan
> actually asked for — the Othello MLP write applied at every history frame — was run, it landed at
> **−0.544**, i.e. indistinguishable from the single-frame write. Consistency across frames is
> **not** the missing ingredient; **frame validity** is. Numbers below are correct; the
> interpretation is not. Kept as provenance.

# 2026-08-18 — Rewriting the whole observed history CROSSES ZERO and HOLDS, with no ground truth

**Direction:** none open — Sevan-directed follow-up to `2026-08-18-othello-gpt-method-port.md`.
**Thread:** `notebooks/experiments/editability/othello_gpt/history_rewrite.ipynb` (+ `history_edit.py`).
**Model:** `runs/transformers/W16` (no new models). **Data:** `datasets/4_fixed_refl_inview`, edits
split, `ef=20`, `K=15`, **N=256** — the same episodes as the sibling notebook.
**Findings updated:** `findings/editability.md` (2026-08-18, `observed`).

## The method

To teleport object `k` by `δ` at the edit frame, apply **the same `δ` to every prior frame**,
rebuilding each observation from the model's **own decoded positions**, then teacher-force on that
history. It should work because velocity is constant here: translating a whole track by a constant
`δ` is *itself a valid trajectory*, so the rewritten history is a consistent world rather than one
inconsistent frame the dynamics must absorb.

**No ground truth is used by the method.** Positions come from the probe reading the model's own
residual stream (point 3, R² 0.934). `δ` is computed from decoded quantities only. Rendering needs
only radius and reflectivities, which on a `fixed_reflectivities` dataset are **world constants
identical for every episode**, not per-episode state.

## Results

| arm | EI step 0 | EI step 14 | fidelity |
|---|---|---|---|
| Unsteered | −0.684 | −0.439 | 1.000 |
| Reconstruction control (`δ=0`) | −0.569 | −0.375 | 1.039 |
| Latent write (Othello method) | −0.538 | −0.428 | 0.994 |
| Oracle observation *(leads by one)* | +0.126 | −0.030 | 0.858 |
| **History rewrite (`δ`, clean)** | **+0.626** | **+0.351** | **0.674** |
| History rewrite (`δ`, matched noise σ=0.2) | +0.629 | +0.370 | 0.676 |
| Oracle history rewrite (`δ`, GT positions) | +0.640 | +0.364 | 0.603 |

**1.** Gain over its **own** reconstruction control: **+1.195** at step 0, **+0.727** at step 14.
The latent write gains +0.146 → +0.010 on the same episodes.

**2. Not a degradation artefact.** Fidelity **0.674** — the rollout ends **33% closer** to the true
post-edit world than doing nothing. Every index-moving-by-wrecking arm in this thread has fidelity > 1.

**3. Decode error is nearly irrelevant.** The GT-position oracle reaches +0.640 vs the decoded
+0.626 — a gap of **0.014**, despite a decoded-position RMSE of 0.49 sim units and a displacement
error of 0.486. **The method needs a CONSISTENT read-out, not an accurate one.**

**4. Render/noise mismatch is a non-issue** (+0.626 clean vs +0.629 noise-matched).

**5. Depth sweep — placing and holding need different amounts of history.** Step-0 index rises
+0.265 (depth 1) → +0.594 (depth 5) and is flat from depth 8. Step-14 index keeps climbing:
+0.080 (1) → +0.302 (8) → **+0.355 (16)**, flattening near the model's **16-frame per-layer
attention window**. Suggestive, **untested** — needs W2/W4 where the window differs.

**6. One rewritten observation frame beats every latent write**: depth 1 = **+0.265** vs the best
latent write's −0.538, same model, same episodes.

## Reading (not established)

Sharpens the thread's standing conclusion rather than overturning it: the negative was never "this
world state cannot be changed" but **"the latent is not the surface on which it can be changed."**
Everything that has ever worked here (counterfactual overwrite, freeze-time TF, now this) writes
through the **observation channel**. What is new is that this needs **no oracle**.

Result 3 is the pointed one and it is testable: an *inconsistent* write is rejected however
accurately it hits the probe target (the sibling notebook's whole finding), while a *consistent*
write is honoured even when substantially inaccurate. That points at **coherence of evidence**, not
precision of the write, as the barrier. Falsifiable: corrupt the rewritten history's internal
consistency while holding its accuracy fixed and the effect should die.

## Limits

- **Uses the renderer** — not a pure latent intervention, must never be quoted beside the latent
  editors as if it were one. The observation function is treated as known.
- Translating a track pushes it out of frustum on **10.0% of frames / 46.1% of episodes**. The
  effect survives it; a cleaner version would reject or clip those.
- `δ` derives from a known target, as every editor's does. What is new is that nothing *else* is.
- One model, one dataset, N=256. The window correspondence in (5) is suggestive only.
- Constant velocity is what makes a constant `δ` valid. A world with forces/collisions would need a
  dynamics-consistent replay, not a translation.

## Harness work done here

- `editability_metrics.sim_config_from` / `object_constants` extracted as the **one** place the
  reference-render `SimConfig` and world constants are built (previously inline in
  `build_edit_zones`; `history_edit.py` needed the same thing and duplicating it would have been
  the exact drift `harness/ANALYSIS.md` §1 forbids). Verified: zones render identically, 178 tests pass.
- **Bug caught by a gate:** a hand-rolled frustum test halved `x_near`, which is *already* a
  half-width, and reported **28% of ground-truth frames** as out of view on an always-in-frustum
  dataset. Replaced with the simulator's own `frustum_half_width`; GT now reads exactly 0.0%. The
  notebook asserts this. The bogus version would have reported 84% of episodes as compromised.
