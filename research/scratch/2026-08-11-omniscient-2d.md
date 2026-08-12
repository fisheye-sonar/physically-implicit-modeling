# 2026-08-11 — Omniscient 2D: does `readable ≠ grabbable` survive full observability?

> `scratch/` is ungated — nothing here is "true" yet. Promotion to `findings/` is Sevan's call.

## The question

Every editability result in this repo has been measured through a **1D perspective scan**, an
observation that is lossy twice: it **projects** (2D world → 1D signal) and it **occludes** (first-hit
only). `orthogonal_edits` (2026-08-05) moved the thread's central negative from the models to the
**world**, via `∫gg' = 0`: a linear probe reads an object's *plateau*, moving the object changes only
its *edges*, and a plateau is nearly perpendicular to the spikes at its own edges — predicted
cos ≈ −0.125, measured −0.151.

That argument is stated for a 1D scan. **This thread swaps the observation channel for an omniscient
one** — a top-down orthographic raster where nothing is projected away and nothing is hidden — and
asks which results survive. Sevan's framing: *"train the GRU not on the 1D observations but on the
whole omniscient 2D world which is fully observable — everything is known."*

## What was built

**Renderer** `pim/simulator/render2d.py` — additive, default-off, following the `soft_render.py`
precedent. 48×64 grid over `x∈[-6,6] × y∈[3,12]`, row 0 = near plane, flattened **row-major** so the
whole downstream stack still sees `(N, T, R)` with `R = 3072` and needs no changes. Hard discs, no
occlusion, no perspective, no depth falloff. `obs_dim(cfg)` in `config.py` is the single source of
truth for `R`. 18 tests incl. `test_defaults_are_bit_identical`.

**Dataset** `datasets/12_omniscient2d` (10 GB, 5 min, 30k/3k/3k/3k). Split base seeds matched to
`4_fixed_refl_inview` **explicitly** (`--seed-val/--seed-test/--seed-edits`, new flags) rather than
derived sequentially. Verified on 200 rows of test and edits: positions, velocities, reflectivities,
edit objects and edit values are **bit-identical** to dataset 4. One variable changed.

**Runs** `runs/omniscient_2d/{2D_H256_s0, 2D_H256_s1, 1D_H256_30k_s0}` — the `controls/H256` recipe
verbatim. The 1D arm is **sample-matched**: `--n-train-limit 30000` takes seeds 0–29999, precisely
dataset 12's train scenes, and the split RNG depends only on count+seed, so the train/val partition
matches scene-for-scene. Registry: `notebooks/experiments/editability/omniscient_2d/OMNISCIENT_2D_RUNS.md`.

**Waterfall analogue** — `frame_grid.py` (`frame_grid` + `frame_trails`), spec proposal in
`WATERFALL_SPEC_2D.md`. **Awaiting Sevan's sign-off**; a literal waterfall cannot be drawn when a
frame is 2D. Validated against known answers: the unedited-world arm scores exactly **−1.00**, a
synthetic collapse **+0.16**.

## The measurement caveat that governs every cross-channel claim

An object covers ~13 % of a 1D scan at mid-depth but **0.73 %** of the omniscient grid — an **~18×
dilution** (measured: 22 px/object of 3072; occupancy 1.45 % for two objects). So any average over
*all* rays/pixels — next-step RMSE, Edit-frame RMSE, GT-traj RMSE — is dominated by background to a
wildly different degree in each channel and is **not** cross-channel comparable. Cross-channel
reading is restricted to: the **Edit Index** (bounded, defined on differing pixels only), **R² and
fiber residual** (dimensionless / a fraction of ‖h‖), and **ratios to each arm's own reference**.
Recorded in the runs registry, the notebook definitions table, and the `editability_metrics` module
docstring so it cannot drift.

Measured reference scales: 2D noise floor 0.1422 (1D: 0.1539).

## Bugs and traps caught while building

1. **`build_inmemory_dataloaders` moved the whole tensor to GPU then split**, allocating the split
   copies while the full tensor was still live — peak ~2× the dataset (29.5 GB here), which OOMs a
   32 GB card that comfortably holds the 14.8 GB it needs. Now splits on the host. Same `perm`, so
   the split is unchanged.
2. **`generate_dataset.py` derives split seeds sequentially.** With 30k/3k/3k/3k that would have put
   the 2D test/edits scenes inside dataset 4's *train* seed range — leaking against the 1D baseline.
   Fixed with explicit per-split seed overrides plus an overlap check that refuses to generate.
3. **A silent no-op notebook.** Building an `.ipynb` from a generator script with
   `source = s.split("\n")` drops the trailing newlines; `.ipynb` joins `source` with `""`, so every
   cell collapsed onto **one line** — and because every cell starts with a `# [N]` comment, the whole
   cell became a comment. Result: nbconvert reported success, execution counts incremented 1→20,
   **zero outputs, zero errors**. Use `splitlines(keepends=True)`.
   **Generalisable check: an executed notebook with no outputs has not run.** Worth asserting after
   any programmatic notebook build — "it ran without error" is not the check (`CLAUDE.md` already
   says this about figures; it applies to whole notebooks too).
4. `render_scene` sized its arrays from `cfg.obs_res` directly; now from `obs_dim(cfg)`.

## Results (2026-08-11)

Notebook `omniscient_2d_world_state.ipynb`, N=256 held-out edits, K=15. Full table in the runs
registry.

**0. The 1D control validates the whole pipeline.** `1D_H256_30k_s0` reproduces the published
`controls/H256` numbers (90k sequences) to within 0.03 index points on *every* editor —
injection −0.63/unsteered −0.66 (published −0.66/−0.68), Counterfactual +0.68 (+0.70), Decoder Grad
k=1 +0.96 (+0.97), k=15 +0.81 (+0.83), Freeze-time +0.52 (+0.52). The 30k restriction costs nothing,
so no 1D↔2D difference is a sample-size effect.

**1. THE NEGATIVE SURVIVES FULL OBSERVABILITY.** Best standard (training-free) editor gain over its
own unsteered row: **+0.11 / +0.14** on the two omniscient arms vs **+0.13** in 1D. Pseudoinverse
Injection inert in both (+0.02/+0.02 vs +0.03). Removing *both* projection and occlusion does not
make the latent grabbable.

**2. The surprise: the omniscient latent is WORSE on every axis.** Position R² 0.634/0.686 (linear),
0.752/0.762 (MLP) vs 0.797/0.877 in 1D. Fiber residual 0.881/0.836 vs 0.583 — much less canonical.
Every oracle weakens: Counterfactual +0.38/+0.24 vs +0.68, Decoder Grad k=1 +0.62 vs +0.96,
Freeze-time +0.34/+0.27 vs +0.52. Strictly more information → a strictly worse world state.

**3. Geometry moves the other way:** PCA hull @90 % 74/70 dims vs 44, while TwoNN intrinsic dim is
2.3/2.5 vs 3.2 — a wider linear hull around a lower-dimensional estimate.

**4. Seeds agree** to ≤0.14 index points and ≤0.05 R².

### Interpretation (mine, not established)

The leading candidate for (2) is **occupancy dilution acting through the objective**, and it is
measured, not assumed: an object covers **25.5 %** of the 1D scan but **1.45 %** of the omniscient
frame. With a plain per-pixel MSE, ~98.5 % of the omniscient gradient is about background, so the
pressure to encode object position precisely is ~**18× weaker per unit of loss**. The model reaches a
*lower absolute* next-step RMSE (0.0875 vs 0.1051) at a similar ratio to its own noise floor (0.62×
vs 0.68×) largely by predicting empty space well, and Fig 5 shows the consequence: relative to object
size the omniscient generations are soft blobs where the 1D model's are crisp. The same blur explains
the less-negative unsteered index (−0.54/−0.52 vs −0.66) — `d_unedited` is inflated by the model's own
blur on exactly the pixels the index scores.

**This makes result 1 provisional in a specific way.** It shows the negative is not caused by
projection or occlusion. It does **not** show independence from observation *sharpness*, because the
arms differ in effective blur as well as channel.

### The pre-registered follow-on (interpretation guide fixed BEFORE running)

Train an **occupancy-matched** omniscient arm — reweight the per-pixel loss by object occupancy, or
enlarge the objects / shrink the world until occupancy approaches 25 %.
- If position R² and oracle strength recover toward the 1D values **while the standard editors stay
  inert**: result 1 stands, and "omniscient is worse" is an objective-weighting artifact rather than a
  fact about observability.
- If the **standard editors improve too**: result 1 was confounded by blur and must be re-run at
  matched occupancy before it means anything.

Not parameter-matched, by necessity: hidden size (the object under study) is matched at 256, but the
encoder/decoder scale with the observation, so the 2D arms carry 1.97 M params vs 0.46 M. The
asymmetry runs *in favour* of the 2D arm, which still reads worse.

## Owed / not done

- Waterfall spec is **unapproved**; figures use the draft.
- One hidden size (256), one grid resolution (48×64), one world.
- 30k training sequences vs the published 1D GRU's 90k — the sample-matched 1D arm controls for this,
  but `controls/H256` (90k) should be quoted alongside so the sample-size effect is visible.
- No RSSM / transformer / DiT arm on the omniscient channel.
- The `∫gg' = 0` geometry argument has **not** been re-derived for a 2D disc. In 1D the object's
  image is a plateau with two edge rays; in 2D it is a disc with a 1D boundary *ring*, so the
  edge-to-interior mass ratio differs. That derivation is the natural follow-on and would say whether
  a 2D replication of the orthogonality result is predicted or surprising.
