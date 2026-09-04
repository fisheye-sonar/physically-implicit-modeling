# Spec — Bayes-mean references for the discworld Edit Index

**Status: planned, not started (2026-09-02).** Agreed between Sevan and Claude; parked so the
day's analysis could proceed. Everything needed to start is here; nothing in the codebase
has been changed for it. Estimated 1–1.5 h to implement, ~2.5 h unattended to re-score.

## Why

On Othello the two Edit Index references are Bayes-optimal by construction (uniform over the
legal set, before and after the flip). On discworld they are **clean renders**, which are not
the Bayes-optimal prediction under the instance's noise: the observation is
`clip(render(p) + ε, 0, 1)` with `ε ~ N(0, σ_o)` (`renderer.py:118-120`), so the mean observation
at a background ray is `σ_o/√(2π) ≈ 0.08` at σ_o = 0.2, not 0; and positions take
`p_{t+1} = p_t + v·dt + η`, `η ~ N(0, σ_p)` (`sim.py:207-208`), so a counterfactual position is
a distribution, not a point. An MSE-trained model predicts the conditional **mean**, so it is
scored against a slightly wrong target in every noisy row. The **noiseless instance is the
control**: there clean = mean, and the editability negative stands — so this is a refinement
of the numbers, not of the conclusion.

## Definitions (headline index is step-0, the edit frame `ef`)

Let `m_σ(c)` be the mean of a normal clipped to [0, 1]: for `X = c + σZ`,
`m_σ(c) = c·(Φ(b) − Φ(a)) + σ·(φ(a) − φ(b)) + (1 − Φ(b))`, with `a = −c/σ`, `b = (1 − c)/σ`;
`m_0(c) = c`. Applied per ray to a clean render.

**Post-edit reference** `gt_edited_mean = m_σo(render(p_ef^edit))`. All four coordinates at
`ef` are exact, realised values — the editor writes both objects' positions (the `pos` dim set
is every object's xy; `bench.DIM_SETS`), and the collateral object's `η` at `ef` is a realised
draw recorded in the dataset. **No position averaging.** (Sevan's correction, 2026-09-02.)

**Unedited reference** `gt_unedited_mean = E_η[ m_σo(render(p_edited^cf(η), p_other^ef)) ]`
where `p_edited^cf(η) = p_{ef−1} + v·dt + η` is the edited object's counterfactual position and
`p_other^ef` is the collateral object's **realised** position at `ef` — held fixed, identical in
both worlds. The only expectation anywhere is over the edited object's one-step `η`.

*Why the collateral object is held, not averaged, in the unedited reference too:* the two
worlds must differ **only in the edit**. Averaging the collateral object's `η` in one reference
but not the other would put its edges into the differing support — a difference the edit did
not cause. The unedited reference is therefore the Bayes-mean prediction conditioned on the
exact state at `ef−1` *and* on the collateral object's realised step; a deliberate, symmetric
oracle choice, stated as such.

**Supports are unchanged.** `target`, `ghost`, `collateral` and `differing` masks keep coming
from the **clean** renders (geometry), never from blurred references. Only the reference
*values* change.

**What moves:** `edit_index` (both distances), `fidelity_ratio` (`edit_frame_rmse` is against
`gt_edited`, whole frame), `target/ghost/collateral_rmse`. Expect both distances to shrink
toward what the model emits and the noisy runs' unedited floors to slide slightly toward −1.
Noiseless rows: **bit-identical** (σ_o = σ_p = 0 ⇒ `m_0 = id`, `η = 0`).

**Not changed:** `gt_edited_traj` / `gt_unedited_traj` (the by-step diagnostics) stay clean
single samples of the noisy future — the Bayes mean at later steps would need the noise
averaged over a trajectory; the headline is step-0 where the mean above is exact. Document
this in the docstring.

## Relation to the true Bayes optimum (for the write-up)

The true optimal pre-edit prediction conditions on the noisy observation *history* — a
nonlinear filtering posterior over `(p, v)` (the render is nonlinear in position; occlusion) —
no closed form. The reference above conditions on the exact previous state instead: an oracle
upper bound on any history-conditioned predictor. It should be the same object as the
registry's "state oracle" MSE floor (0.022171 vs the obs-noise-only floor 0.018866) — verify
that derivation in `research/GOTCHAS.md` / the Bayes-floor notes before citing them together.

## Implementation (all in `pim/metrics/editability.py` + one setting)

1. `clipped_normal_mean(c, sigma)` — vectorised closed form; `sigma == 0` returns `c`
   unchanged (exact, not approximately).
2. `build_edit_zones(..., reference="clean")` → add `"mean"`:
   - render the clean references as today → masks from them (unchanged);
   - `gt_edited = clipped_normal_mean(clean_edited, σ_o)`;
   - `gt_unedited`: for `M` samples (default 256, `rng = default_rng(0)`), draw `η` for the
     edited object only, render with the other object fixed, apply `clipped_normal_mean`,
     average. The existing per-case loop (`editability.py:211-216`, `render_frame`) gains an
     inner loop; ~192 × 256 renders ≈ 1 min per bench. `σ_o`, `σ_p` come from the `sim` dict
     already passed in (`obs_noise_std`, `position_noise_std`).
3. `bench.load_bench(..., reference=...)` passes it through; `master_eval` cell [2] gains
   `"dw_reference": "mean"`; **bump `EVAL_VERSION`**.
4. Tests (`tests/test_metrics_canonical.py` or a new file): closed form vs Monte-Carlo (1e-3);
   `sigma=0` bit-identical to clean; `reference="mean"` with a zero-noise `sim` dict reproduces
   `reference="clean"` exactly (the noiseless gate); masks identical between the two modes.
5. REGISTRY: the Edit Index row says which reference is canonical; GOTCHAS: the 0.08 floor.

## Re-scoring plan

Everything is probe-cached, so the version bump costs editors + benches only: ~5 min per
transformer discworld run (2 canonical + 8 `training_curve` points), ~18 min per Recurrent-L
run (GS descent at d = 1024), ~1 min of MC per bench, and Othello's 9 runs at ~3 min (rescored
by the bump, numerically unchanged). **≈ 2.5 h unattended**, then `build_full_table`. Run as a
capped `systemd-run` unit (45G), one job at a time, with a watcher — the pattern in
`experiments/training_curve/drivers/training_curve.sh`.
