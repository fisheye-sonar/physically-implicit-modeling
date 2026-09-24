# Ray-count ablation: 8 rays, radius 1.0 (dw-8ray, L-dw-8ray-20m)

> **Rescored 2026-09-12** under the PRE-dynamics write target (GOTCHAS 2026-09-12; `scratch/2026-09-12-alignment-rescore.md`): every discworld editor number quoted below moved by ≤ 0.06 (median 0.005), floors unchanged, no conclusion changed. `scores.json` holds the current values.


**Date** 2026-09-04 · **Instance** `dw-8ray` (= dw-noiseless with disc radius 1.0 and 8
usable rays: 10 cast, the two frustum-wall rays dropped) · **Run**
`runs/ray_ablation/L-dw-8ray-20m` (Transformer-L, 780k steps, matched recipe, `Linear(8,
512)` in/out, 25.25M params) · **Driver** `scripts/drivers/dw_8ray.sh` · **Logs**
`logs/ray_ablation/dw_8ray/` · **Chain** generation 2 h 08 min (20M sequences, 26 GB;
slower than the 128-ray corpus because radius-1 discs make the in-frustum acceptance loop
reject more), training 7 h 53 min, scoring + baselines + tables 27 min.

## Question

Does discworld's editability change when the observation is coarse (8 rays) and the
objects large — the regime in which each disc is a few-ray blob rather than a 20-ray
profile?

## Numbers (canonical scoring, EVAL_VERSION 2026-09-01.4; dw-noiseless beside it)

| | dw-8ray · frustum | dw-8ray · cartesian | dw-noiseless · frustum |
|---|---|---|---|
| val MSE (best) | 0.00575 | | 0.00106 |
| Probe Skill LIN / MLP-128 | 0.950 / 0.981 | 0.886 / 0.932 | 0.959 / 0.996 |
| random-init floor LIN / MLP | 0.955 / 0.975 | 0.883 / 0.919 | 0.825 / 0.987 |
| observation floor MLP (250k) | 0.925 | 0.847 | 0.906 |
| unedited Edit Index | −0.888 | −0.888 | −0.924 |
| PI (best arm) | **+0.297**, fid 1.11 (pos·pt3·α175) | +0.242, fid 0.96 (pos·pt3·α175) | +0.233, fid 1.95 |
| GS (best arm) | −0.097, fid 0.95 | −0.156, fid 0.92 | −0.099, fid 0.99 |

Per-component LIN skill (frustum, best point): positions 0.94 / 0.89 / 0.96 / 0.90,
velocities 0.58 / 0.47 / 0.63 / 0.48 — velocity decodability drops with 8 rays (dw-noiseless
frustum: 0.73 / 0.54 / 0.84 / 0.69 on L-dw-20m's scale), position barely.

## Findings

1. **Editability is unchanged in kind, slightly better in degree.** PI reaches +0.30 on the
   frustum basis (+0.24 cartesian) against +0.23 on dw-noiseless, GS stays at −0.10, ND
   remains inapplicable. Still far below Othello's +0.6, and the same shape: a single
   PI regime around +0.2–0.3 that no amount of write can push further.
2. **The canonical α grid is pinned at its edge here, but the index has plateaued.** The
   best arm is the last grid value (α 175) with the index still rising (frustum pt3: α100
   +0.264 → α175 +0.297). Re-running the identical canonical arm with α up to 1500
   (`experiments/ray_ablation/alpha_check/`, cached probes) shows every point saturating at
   +0.2–0.3: the best anywhere is pt1·α1500 at +0.323 (fid 1.10), the best with fidelity
   ≤ 1 is pt1·α1000 at +0.311. So the reported +0.297 is a lower bound by ~0.02, not a
   qualitatively different number.
3. **A non-destructive positive index appears for the first time on discworld.** With 8
   rays, moderate writes land with the fidelity guard *below* 1: frustum pt3·α100 gives
   +0.26 at fidelity 0.90, cartesian pt3·α175 gives +0.24 at 0.96, and the fidelity of the
   best arms is 1.0–1.1 instead of the 1.6–2.3 of every 128-ray run. The edit no longer
   wrecks the frame it lands in; it just does not land more than a third of the way.
4. **The trained model is no more decodable than a random reservoir on this instance.**
   Random-init Transformer-L probes at 0.955 (LIN) / 0.975 (MLP) on the frustum basis
   against the trained model's 0.950 / 0.981, and the observation probe alone reaches
   0.925 — with 8 rays and radius-1 discs the state is almost a linear function of the
   recent observations, so decodability says nothing about training here. The binding
   floor is random-init, as on every discworld instance, and it is now within 0.006 of the
   trained model.
5. **Edit-Index support is coarse.** ~15 % of the 192 bench cases have no differing ray
   between edited and unedited worlds at 8 rays (the teleport moves the disc within the
   same rays); those cases carry no index and the mean rests on the rest. Same
   construction, wider error bars.

## What it says for the programme

The 128-ray/small-disc geometry was not what kept discworld un-editable: coarsening the
observation to 8 rays and doubling the discs leaves the Edit Index ceiling at ~+0.3 with
the same one-regime shape, while making the writes non-destructive. Whatever blocks the
edit is in how the model uses the state, not in the resolution of the input.

## Canonical changes made for this run (all behaviour-preserving for earlier instances)

`SimConfig.drop_edge_rays` + `obs_dim` (config.py); `render_frame` keeps the interior rays
via `_keep` (renderer.py); `--radius`, `--drop-edge-rays` (generate_dataset.py);
`build_edit_zones` / `sim_config_from` size by `obs_dim` (editability.py); per-instance
`OBS_RES` and the `dw-8ray` registry entry with fresh seed blocks (bigcorpus.py); the
waterfall sized by `obs_dim` (viz.py); tiny-split HDF5 chunking (dataset.py).
`tests/test_drop_edge_rays.py` pins the kept rays to rays 1–8 of the 10-ray render.

## Addendum 2026-09-04 — Recurrent-L on the same instance (`ray_ablation/R-dw-8ray-20m`)

The architecture pair: 4 × 1024 GRU with carried edits, identical data and recipe, 5.1 h of
training. Best val MSE 0.00595 at step **40k**, drifting to 0.00625 by 780k (the same
early-peak behaviour as the two 128-ray recurrent runs, without their divergence spikes).

| frustum basis | Recurrent-L 8-ray | Transformer-L 8-ray |
|---|---|---|
| Probe Skill LIN / MLP-128 | 0.967 / 0.981 | 0.950 / 0.981 |
| random-init floor LIN / MLP | 0.975 / 0.977 | 0.955 / 0.975 |
| unedited Edit Index | −0.886 | −0.888 |
| PI best | +0.191, fid 1.41 (pos·pt4·α175) | +0.297, fid 1.11 |
| GS best | −0.608, fid 0.92 | −0.097, fid 0.95 |

Same picture from the other architecture: decodability sits at (LIN below) the random-init
floor, PI lands a fifth of the way at fidelity 1.4, GS is destructive. Carrying the edited
state forward instead of recomputing it from the history does not help here either.

⚠ Reading Table 1b for this run: its per-component row is taken at the LIN best *point*,
which for the recurrent model is point 0 (aggregate 0.967, position-dominated) — a point at
which no velocity is linearly present (0.00–0.03) although points 1–4 read velocity at
0.4–0.7 (LIN) / 0.45–0.80 (MLP). The rule "per-component at the aggregate-best point"
hides that; the per-point profile is in the run's scores.json.


## Addendum 2026-09-16 — the ray axis closed at the top: `ray_ablation/L-dw-128ray-20m` (dw-128ray)

**Why.** dw-noiseless (128 rays cast, radius 0.5, no edge-ray drop) and the coarse-ray family
(5 / 8 / 16 kept rays, radius 1.0, wall rays dropped, `--max-edit-attempts 2000`) differed in
TWO things, ray count and disc size. dw-128ray is the family's geometry with 130 rays cast → 128
kept (Sevan, 2026-09-15 night; `scripts/drivers/dw_128ray.sh`; built and trained on the lab box —
generation 2 h 14, training 8.0 h, best val 0.000929 vs dw-noiseless 0.00107, dw-16ray 0.0040).
The chain was interrupted by two hard freezes during scoring (07:22, ~10:20; a failing PSU,
replaced) and finished under a 400 W GPU power cap; every stage is idempotent and nothing was
re-fitted.

**Result: dw-128ray lands on dw-noiseless, not on dw-16ray — the coarse-ray gains are a ray-count
effect, not a disc-size effect.** Cartesian regression block (PI / GS / IM, Edit Index / fidelity):

| run | skill LIN / MLP | PI | GS | IM |
|---|---|---|---|---|
| dw-noiseless (128 cast, r 0.5) | 0.872 / 0.973 | +0.20 / 1.71 | -0.06 / 1.07 | +0.59 / 0.34 |
| **dw-128ray (128 kept, r 1.0)** | 0.914 / 0.981 | +0.24 / 1.57 | -0.03 / 1.19 | +0.57 / 0.32 |
| dw-16ray (16 kept, r 1.0) | 0.892 / 0.965 | +0.22 / 1.33 | -0.10 / 0.90 | +0.66 / 0.29 |
| dw-8ray (8 kept, r 1.0) | 0.886 / 0.932 | +0.21 / 1.00 | -0.07 / 0.88 | +0.71 / 0.27 |
| dw-5ray (5 kept, r 1.0) | 0.850 / 0.883 | +0.14 / 0.85 | -0.10 / 0.86 | +0.81 / 0.23 |

Factorised categorical target (appearance-fac):

| run | skill LIN / MLP | PI | ND | GS | IM |
|---|---|---|---|---|---|
| dw-noiseless (128 cast, r 0.5) | 0.433 / 0.786 | +0.01 / 1.93 | +0.61 / 0.80 | +0.33 / 0.71 | +0.60 / 0.34 |
| **dw-128ray (128 kept, r 1.0)** | 0.261 / 0.705 | -0.00 / 1.80 | +0.54 / 0.96 | +0.31 / 0.79 | +0.57 / 0.32 |
| dw-16ray (16 kept, r 1.0) | 0.879 / 0.918 | +0.10 / 1.39 | +0.42 / 0.90 | +0.28 / 0.85 | +0.68 / 0.28 |
| dw-8ray (8 kept, r 1.0) | 0.935 / 0.943 | +0.38 / 0.95 | +0.49 / 0.98 | +0.46 / 0.54 | +0.70 / 0.31 |
| dw-5ray (5 kept, r 1.0) | 0.926 / 0.932 | +0.51 / 0.70 | +0.55 / 0.58 | +0.60 / 0.46 | +0.84 / 0.26 |

On both targets every column of dw-128ray is within a few hundredths of dw-noiseless: the
categorical read-out is barely linearly decodable (0.26 vs 0.43), PI is destructive, GS is weak,
IM lands at the late points only (IM by point: −0.22 −0.25 −0.02 +0.29 +0.47 +0.55 **+0.57**
+0.55 +0.54; the same ramp as dw-noiseless and dw-blink, where the coarse-ray runs are flat at
+0.55 … +0.81 across every point). Doubling the disc radius at 128 rays changed nothing.
Registry: instance row `dw-128ray`, run row `ray_ablation/L-dw-128ray-20m`; the run is in both
table notebooks (between blink and 16-ray, so the family reads 128 → 16 → 8 → 5).
