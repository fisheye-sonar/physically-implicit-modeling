# 2026-09-15 — near-teleport pilot: a hole in the joint support does NOT break IM on discworld

**Question.** IM fails on oth-adjacent exactly on the cases whose flipped board no history can produce
(scratch `2026-09-15-othello-ceiling-symdiff.md`, addendum). Is "the target state is outside the
training support" the operative condition? Test on discworld without training: the generator never
lets two discs come within collision_margin × 2r = 1.6 units, so a teleport target at 1.0 < d < 1.6
from the other disc (no overlap, never seen) is outside the joint support of every training frame.

**Method.** `experiments/near_teleport_pilot/scripts/near_pilot.py --run runs/noise_ablation/L-dw-noiseless-20m`:
the canonical 192 selected cases, the edited disc re-teleported to a uniformly random point in the
annulus 1.0 < d < 1.6 around the other disc (in-frustum, non-overlapping for the rest of the
sequence; 191/192 found), post-edit frames re-rendered into a pilot `edits.h5`, scored through the
canonical `bench_arrays` / arms / scorecards (protocol α grids, full-state writes, cached probes and
inverse maps). Paired control: the SAME cases with their canonical far targets.

| dw-noiseless (191 paired) | unedited | PI | GS | IM | IM-NN |
|---|---|---|---|---|---|
| near targets (mean d 1.31) | −0.934 | +0.259 / 1.52 (pt2 α60) | −0.002 / 1.08 | **+0.626 / 0.32** (pt5) | +0.429 / 0.68 |
| canonical far targets | −0.925 | +0.230 / 1.61 (pt3 α60) | −0.049 / 1.09 | **+0.601 / 0.33** (pt6) | +0.382 / 0.61 |

IM by point, near: −0.04 0.20 0.40 0.51 0.61 0.63 0.61 0.59 0.57; far: −0.21 0.10 0.30 0.40 0.53 0.58 0.60 0.59 0.56.

**Reading.** Off-support targets edit exactly as well as on-support ones, for every editor and at
every point. So "outside the training support" is not what makes IM fail on oth-adjacent; the
inverse map extrapolates freely into the exclusion zone here because the discworld code is
(evidently) additive over discs, whereas the adjacent board code ties colour to placing-parity
features that no latent can satisfy for an unreachable board. Working criterion: IM lands off-support
iff the latent code factorises over the edited variable. Consequence for the paper's missing cell
(a discworld world where nothing edits): a forbidden REGION in position space will not do it; the
coupling has to be one the model's computation exploits (an equality the model can shortcut through,
e.g. a fixed inter-disc distance), with the "reparameterisable state" objection stated and accepted.
Also a robustness statement for IM on discworld: its success is not confined to the training support.

Files: `experiments/near_teleport_pilot/{scripts,scores,data}` (data gitignored).

## Addendum — the same cases through the `appearance-fac` target (`near_pilot_fac.py`)

Same pilot file, factorised categorical probes (cached, GRID_PROBE_RECIPE), protocol grid α's, ND
included for reference (best discworld editor on this target), IM as the state write; 191 paired
cases (every near and far teleport changes a run cell).

| dw-noiseless, appearance-fac | unedited | PI | ND | GS | IM | IM-NN |
|---|---|---|---|---|---|---|
| near targets | −0.934 | +0.02 / 1.95 | **+0.64 / 0.77** (pt3 α3) | +0.36 / 0.69 | **+0.63 / 0.32** | +0.43 / 0.68 |
| canonical far targets | −0.925 | +0.02 / 1.90 | **+0.61 / 0.79** (pt2 α6) | +0.35 / 0.87 | **+0.60 / 0.33** | +0.38 / 0.61 |

Identical picture: the categorical read-out edits an off-support target exactly as well as an
on-support one (ND +0.64 vs +0.61, GS +0.36 vs +0.35, PI dead on both). The factorised code, like
the regression one, extrapolates into the exclusion zone without loss. File
`scores/near_pilot_fac_L-dw-noiseless-20m_1.0_1.6.json`.
