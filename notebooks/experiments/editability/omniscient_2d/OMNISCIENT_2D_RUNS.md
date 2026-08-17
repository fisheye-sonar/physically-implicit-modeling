# Omniscient-2D thread — canonical run registry

Every checkpoint and dataset used by this thread, with its **full** config. A notebook that
mentions a run copies its row into its own definitions table, so the notebook still stands alone
(`CLAUDE.md`). **Adding a run means adding its row in the same commit.**

Thread question: the whole editability thread's central negative — *readable ≠ grabbable* — has
only ever been measured through a **1D perspective scan**, an observation that both **projects**
(2D world → 1D signal) and **occludes** (nearest surface only). This thread replaces that channel
with an **omniscient** one and asks whether the negative survives. `orthogonal_edits` relocated the
negative from the models to the *world* (the `∫gg' = 0` argument: a probe reads an object's
plateau, moving it changes only its edges). That argument is stated for a 1D scan; an orthographic
2D raster is where it is tested rather than assumed.

---

## Datasets

| name | observation | grid / dim | splits (n, base seed) | notes |
|---|---|---|---|---|
| `datasets/4_fixed_refl_inview` | 1D perspective scan | `obs_res` 128 | train (90000, 0) · val (10000, 90000) · test (10000, 100000) · edits (10000, 110000) | the canonical eval dataset for the whole editability thread |
| `datasets/12_omniscient2d` | **omniscient 2D orthographic raster** | 48×64, flattened row-major → **3072** | train (30000, **0**) · val (3000, **90000**) · test (3000, **100000**) · edits (3000, **110000**) | **NEW 2026-08-11** |

**The two suites are the same worlds.** Split base seeds are matched to dataset 4 exactly (not
derived sequentially, which is the script default), so scene generation — deterministic in the seed
— produces **bit-identical positions, velocities, reflectivities, edit objects and edit targets**.
Verified on 200 rows of `test` and `edits`. Only the observation channel differs. Two consequences:

1. Any 1D↔2D difference is attributable to the observation, not to sampling.
2. Dataset 12's `test`/`edits` scenes are drawn from dataset 4's *test*/*edits* seed ranges, **not**
   its train range — so a 1D model trained on dataset 4 has not seen them either.

Shared sim config (both suites): `n_objects=2`, `n_frames=40`, `dt=1`, `radius=0.5`,
`boundary="open"`, `always_in_frustum=True`, `fixed_reflectivities=True` (→ 0.4, 0.8),
`obs_noise_std=0.2`, `position_noise_std=0.04`, direction/speed noise 0, `edit_frame=20`,
`edit_always_in_frustum=True`, world `x∈[-6,6] × y∈[3,12]`.

**Omniscient-2D grid.** Rows span depth `y∈[3,12]`, columns `x∈[-6,6]`; **row 0 is the near plane**.
Pixels are exactly square at 0.1875 world units, so a radius-0.5 disc is **5.33 px across ≈ 22 px**.
No occlusion, no perspective, no depth falloff, hard silhouettes (values are only {0, 0.4, 0.8}
before noise). Implementation `pim/simulator/render2d.py`; frames flattened row-major so the whole
downstream stack still sees `(N, T, R)` with `R = 3072`.

**Measured reference scales** (dataset 12 test split): noise floor RMSE(`obs`, `clean_obs`) =
**0.1422** (dataset 4's 1D value: 0.1539); mean occupied-pixel fraction **1.45%** for two objects.

> ### ⚠ Whole-frame errors are NOT comparable between the two suites
> An object covers ~13% of a 1D scan at mid-depth but **0.73%** of the omniscient grid — an ~18×
> dilution — so any average over all rays/pixels (next-step RMSE, `edit_frame_rmse`) is dominated by
> background to a wildly different degree in each. **Cross-channel comparisons ride on
> zone-restricted metrics** (Target / Ghost / Collateral RMSE) **and the Edit Index**, which are
> computed only on the pixels the zone or the two ground-truth worlds pick out. Within one channel,
> every metric is comparable as usual.

---

## Checkpoints — `runs/omniscient_2d/`

All three are the **`controls/H256` recipe verbatim**, changing only the dataset and seed:
`hidden_size=256`, `num_layers=1`, `dropout=0`, affine decoder (`dec_hidden_layers=0`),
`batch_size=256`, `lr=1e-3`, AdamW `weight_decay=1e-4`, 400 epochs, `--in-memory`,
`val_fraction=0.1` (an internal split of the train file; the separate `val.h5` is unused by
training). Best-val checkpoint selection.

| run name | observation | train file | train seqs | seed | `input_dim` | params | what it is a control for |
|---|---|---|---|---|---|---|---|
| `2D_H256_s0` | omniscient 2D | `12_omniscient2d/train.h5` | 30000 (→27000 train / 3000 val) | 0 | 3072 | ~2.0 M | **the main arm** |
| `2D_H256_s1` | omniscient 2D | `12_omniscient2d/train.h5` | 30000 | 1 | 3072 | ~2.0 M | seed robustness for the main arm |
| `1D_H256_30k_s0` | 1D perspective scan | `4_fixed_refl_inview/train.h5`, `--n-train-limit 30000` | 30000 | 0 | 128 | ~0.46 M | **sample-matched 1D control** |

**Why the 1D control exists and why it is exact.** The published 1D GRU (`controls/H256`) trained on
**90k** sequences; the 2D arm has 30k, so a 1D↔2D difference would otherwise be confounded by
training-set size. Samples are stored in seed order, so `--n-train-limit 30000` takes seeds 0–29999
— *precisely* the scenes in dataset 12's train file — and the internal split RNG depends only on
the sample count and the seed, so the train/val partition is identical scene-for-scene. The control
differs from the main arm in **exactly one variable**: the observation channel.

`controls/H256` (90k, 1D) remains the reference for the published 1D numbers and should be quoted
alongside `1D_H256_30k_s0` so the sample-size effect is visible rather than assumed.

### Naming
`{channel}_H{hidden}[_{trainsize}]_s{seed}` — `2D` = omniscient 2D raster, `1D` = perspective scan;
`30k` marks a run deliberately limited below its file's size; `s0`/`s1` are training seeds.

---

## Figure/spec assets in this directory

| file | what |
|---|---|
| `frame_grid.py` | the single implementation of the 2D waterfall form (`frame_grid`, `frame_trails`, `frame_animation`) |
| `WATERFALL_SPEC_2D.md` | the 2D adaptation of `CLAUDE.md`'s waterfall spec — **approved 2026-08-12, now binding for any 2D-raster observation** and pointed at from `CLAUDE.md` and `../METRICS_AND_EDITORS.md` |
| `SPEC_fig{S1,S2}_*.png` | spec review figures, rendered from simulator data only (no model) |
| `anim1_editors_2d.gif` | `Anim 1`, produced by notebook cell [19] — the optional animated view (3 fps, ~1 s holds on the edit frame) |

---

## Results

### Current results (updated 2026-08-11)

Source: `omniscient_2d_world_state.ipynb`, N=256 held-out edits, K=15, best-val checkpoints.

| | `2D_H256_s0` | `2D_H256_s1` | `1D_H256_30k_s0` |
|---|---|---|---|
| best epoch / val loss | 397 / 0.01498 | 393 / 0.01511 | 150 / 0.02391 |
| next-step RMSE vs clean *(within-channel)* | 0.0875 | 0.0885 | 0.1051 |
| … ratio to its own noise floor | 0.62× | 0.62× | 0.68× |
| occupied fraction of the frame | 1.45 % | 1.45 % | **25.54 %** |
| PCA hull @90 % | 74 | 70 | 44 |
| intrinsic dim (TwoNN) | 2.3 | 2.5 | 3.2 |
| position R² linear / MLP | 0.634 / 0.752 | 0.686 / 0.762 | **0.797 / 0.877** |
| fiber residual (MLP g) | 0.881 | 0.836 | **0.583** |
| Edit Index — Unsteered | −0.54 | −0.52 | −0.66 |
| Pseudoinverse Injection | −0.52 (**+0.02**) | −0.50 (**+0.02**) | −0.63 (**+0.03**) |
| Global PCA Projection (PI) | −0.43 (+0.11) | −0.38 (+0.14) | −0.53 (+0.13) |
| MLP Grad Steering | −0.47 (+0.07) | −0.47 (+0.06) | −0.59 (+0.07) |
| First Obs. TF *(oracle)* | −0.13 | −0.20 | −0.11 |
| Freeze-time Interp. TF @8 *(oracle)* | +0.34 | +0.27 | **+0.52** |
| Counterfactual Overwriting *(oracle)* | +0.38 | +0.24 | **+0.68** |
| Decoder Grad Steering k=1 *(oracle)* | +0.62 | +0.62 | **+0.96** |
| Decoder Grad Steering k=15 *(oracle)* | +0.60 | +0.60 | **+0.81** |

Parenthesised values are the gain over that arm's **own** unsteered index.

**The 1D control validates the pipeline.** `1D_H256_30k_s0` reproduces the published `controls/H256`
values (90k sequences, `../METRICS_AND_EDITORS.md`) to within 0.03 index points on every editor, so
the 30k restriction costs essentially nothing and no 1D↔2D difference here is a sample-size effect.

**Headline.** The `readable ≠ grabbable` negative **survives full observability** — best standard
editor gain +0.11/+0.14 (2D) vs +0.13 (1D), injection inert in both. But the omniscient latent is
**less** readable (position R² 0.63–0.69 vs 0.80), **less** canonical (fiber residual 0.88/0.84 vs
0.58) and **less** editable by every oracle. Leading candidate mechanism, stated as interpretation in
the notebook's §5: **occupancy dilution acting through the per-pixel MSE objective** (1.45 % vs
25.54 % occupancy ⇒ ~18× weaker gradient pressure on object position), which shows up directly as
blurrier generations relative to object size. The pre-registered test is an **occupancy-matched**
run — reweight the loss by occupancy, or enlarge the objects — which should recover readability and
oracle strength while leaving the standard-editor result unchanged.

⚠ Until that control is run, result 1 establishes that the negative is not caused by *projection or
occlusion*; it does **not** establish independence from observation *sharpness*.
