# Pair ablation — a rigid two-disc world is MORE readable and LESS writable, but the edits do not fail (2026-09-16)

**Status:** `observed` (one run, one seed, canonical scoring under the unified protocol; no waterfall built yet —
the qualitative panel is owed before any generation claim is quoted). ⚠ The main caveat is structural, not
statistical: see *The confound this run cannot separate*.

**Question.** Sevan (2026-09-15): take dw-noiseless and constrain the two discs to a FIXED centre-to-centre
distance, which forces them to share a velocity — a world whose state has two free position dimensions plus an
orientation instead of four. Then score it on the UNCHANGED single-object teleport bench, whose target moves one
disc and therefore breaks the constraint the model learned. Sevan's expectation, stated when he asked for the run:
*"the fact that the edits will likely fail is actually the reasoning behind this."* `dw-pair` is dw-noiseless with
`SimConfig.pair_separation = 2.0` (object 1 placed at distance 2.0 from object 0 at a random orientation, same
velocity; `sim.py`, `--pair-separation`, `tests/test_sim.py::test_pair_separation_rigid_pair`). Everything else —
128 rays, radius 0.5, no noise, 2 objects, 40 frames, open boundary, always-in-frustum, 20M sequences,
Transformer-L, the matched recipe, the probe targets, the bench rule — is identical.

**Answer: the edits degrade; they do not fail.** `runs/pair_ablation/L-dw-pair-20m` (780k steps, 666 min on the
4090, best val MSE **7.80e-4** at 765k against dw-noiseless's 1.06e-3 — the constrained world is *easier* to
predict) scored under `EVAL_VERSION_BY_ENV` discworld `2026-09-12.2` on its own 1000-case bench (1001 scanned,
1 dropped as same-cell; dw-noiseless: 1000 scanned, 0 dropped — the two benches are equally unfiltered).

| block | skill LIN / MLP (rand-init · observation) | unedited | PI | ND | GS |
|---|---|---|---|---|---|
| frustum (canonical) | 0.944 / 0.998 (0.90 / 0.99 · 0.43 / 0.96) | −0.951 | +0.243 / fid 1.63 | −0.114 / 1.87 | −0.093 / 1.02 |
| cartesian | 0.922 / 0.994 (0.80 / 0.97 · 0.31 / 0.86) | −0.951 | +0.229 / 1.86 | −0.113 / 1.72 | −0.109 / 1.04 |
| appearance-fac (262 classes) | 0.448 / 0.798 (0.31 / 0.62 · 0.12 / 0.53) | −0.951 | −0.022 / 1.72 | **+0.490 / 0.91** | **+0.294 / 0.86** |

Against dw-noiseless under the same protocol: frustum PI +0.231 / 1.54 → **+0.243 / 1.63**, ND −0.135 → −0.114,
GS −0.082 → −0.093 (inert on both, as on every 128-ray instance); appearance-fac skill 0.433 → **0.448**,
ND **+0.612 / 0.80 → +0.490 / 0.91**, GS **+0.326 / 0.71 → +0.294 / 0.86**, PI ≈ 0 on both.

**Reading.** Three things, in decreasing order of confidence.

1. **The constraint does not make the state unwritable.** ND still lands at +0.490 inside the fidelity guard
   (0.91), and GS at +0.294 / 0.86, on a bench whose every target is a configuration the world forbids and the
   model never saw. Whatever the model holds, a nullspace write moves it to an off-manifold state and the
   generation follows — it does not snap back to the constraint.
2. **But both landing editors land lower and damage the frame more.** ND −0.12 with the guard worsening
   0.80 → 0.91, GS −0.03 with 0.71 → 0.86. The ND drop is an order of magnitude larger than the run-seed SD on
   the Edit Index (0.005–0.017, `seed-variance.md`), so the direction is not seed noise, though this is one seed
   per instance and two different worlds rather than replicates.
3. **Decodability moves the OTHER way, and the velocity rows say why.** The factorised target is read slightly
   better (0.433 → 0.448 LIN, floors unchanged at 0.30 → 0.31), and in the frustum block the per-dimension linear
   skill at the best point goes 0.864 / 0.830 / 0.966 / 0.934 (positions) and 0.482 / 0.398 / 0.696 / 0.625
   (velocities) on dw-noiseless → 0.908 / 0.942 / 0.948 / 0.951 and **0.920 / 0.761 / 0.933 / 0.788** on dw-pair.
   Positions are read about as well; VELOCITIES are read far better, which is what a shared velocity estimated
   from two discs' worth of evidence should look like. So the constrained world is more readable and less
   writable at once — the two axes move in opposite directions under a world toggle, which is the cleanest case
   of that we have.

**The confound this run cannot separate.** A lower Edit Index here has two possible causes and the run does not
distinguish them: (a) the model's state is genuinely less writable, or (b) the state is as writable as ever but
the TARGET is off-manifold — one disc moved, the constraint broken — so rolling forward from a correctly written
state still scores badly against a rendering the world's dynamics would never produce. The control that separates
them is a PAIR-teleport bench (move both discs by the same vector, keeping the target on-manifold): if ND recovers
to ≈ +0.61 there, the effect is (b) and the state is fully writable; if it stays at ≈ +0.49, the effect is (a).
That bench is a small option in the edit generator, not a new run. Sevan chose the single-object bench
deliberately for this run; the pair bench is the obvious follow-up and the reading above should not be quoted as
"the constraint reduces writability" until it exists.

**Bets on record (2026-09-16 13:45, before the IM/IM-NN arms were scored).** The IM editor was absent from this
run for a branch reason — `pim/editors/inverse.py` lived only on `sweeps_and_blates`, which the remote was not on
— and IM is the only editor that lands in discworld's regression block (frustum: +0.52 dw-blink … +0.73 dw-8ray;
dw-noiseless +0.598 / fid 0.34, IM-NN +0.397 / 0.65). Fitting it here is the sharpest available test of the
confound above, because IM and IM-NN differ exactly on the manifold question: IM *synthesises* a residual for the
target state through a learned g, while IM-NN *retrieves* the residuals of real states, and no real dw-pair state
has a broken separation.

- **Sevan:** IM will fail on dw-pair.
- **Claude:** IM lands but is clearly weakened — **+0.40 to +0.55** against dw-noiseless's +0.598, with fidelity
  worse than 0.34. Reasoning: the teleport breaks one coordinate (the separation) while the velocities stay
  shared, so s_post is off-manifold in a single direction; g takes p₀ and p₁ as explicit inputs and the
  observation code is close to additive over the two discs' ray runs, so a ReLU MLP's piecewise-linear
  extrapolation should still put disc 1 roughly where it is asked.
- **Claude, the sharper call:** IM-NN falls much further than IM — **below +0.30** against dw-noiseless's +0.397 —
  because retrieval cannot represent an impossible configuration at all; the nearest real states are legal pairs.
  So the **IM − IM-NN gap widens beyond 0.25** (dw-noiseless: 0.598 − 0.397 = 0.20). This is the prediction to
  judge me on: it is the one that distinguishes "the state is unwritable" from "the target is off-manifold",
  and it fails cleanly if the two editors move together.
- **Sevan wins outright** if IM's guarded Edit Index is ≤ +0.10, or if it has no arm inside the fidelity guard.

**Not yet done.** The pair-teleport bench (above); a waterfall panel for the ND appearance-fac arm; a second seed;
the joint-cell `appearance` and fixed-grid targets on this run; a second separation (3.0) if the effect is worth
a dose-response.

**Provenance.** Instance `datasets/discworld/dw-pair/` (seeds train 300e9, eval 325.2e9, edits 325.3e9,
probe 1100e9 / 1110e9); built, trained and scored on the WSL remote by `scripts/drivers/dw_pair.sh`
(unit `dw_pair`, 2026-09-15 20:42 → 2026-09-16 12:20: generate 64 min for the 410 GB corpus, train 666 min,
canonical scoring in both bases 61 min, appearance-fac probes + both floors + rescore 118 min), logs
`logs/pair_ablation/dw_pair/` + `logs/dw_pair_fac/`. Run, probes, scores, floors, logs and the instance's
probe / eval / edits splits pulled to the lab box; the 410 GB training corpus stays on the remote.
