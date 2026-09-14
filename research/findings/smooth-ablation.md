# Smooth ablation — anti-aliasing the observation does not make the geometric read-out editable (2026-09-13)

**Status:** `observed` (one run, one seed, canonical scoring under the unified protocol; no waterfall built yet — the
qualitative panel is owed before any generation claim is quoted).

**Question.** dw-noiseless renders each disc as a flat plateau: a covered ray reads the full reflectivity, so the
image is piecewise constant in position and only the two silhouette rays carry positional information — the
observation is aliased. Sevan's hypothesis (2026-09-12): the geometric read-out (frustum position/velocity) is
decodable but not editable on dw-noiseless because the model never needs a continuous position to predict a
quantised frame; give it a continuous observation and the position may become a used, writable variable.
`dw-smooth` is dw-noiseless with the disc's radial profile changed to the power dome intensity = reflectivity ×
(1 − (perp/r)²)² (`SimConfig.soft_shading="power"`, `soft_profile_power=2`; profile H of
`experiments/antialias_pilot`), so every covered ray's value varies continuously with sub-ray position. Everything
else — geometry, 128 rays, radius 0.5, no noise, 20M sequences, Transformer-L, the matched recipe — is identical.

**Answer: no.** `runs/smooth_ablation/L-dw-smooth-20m` (780k steps, 516 min, best val MSE 5.9e-5 at 735k — 18× below
dw-noiseless's 1.06e-3; the Bayes floor is 0 for both) is scored under `EVAL_VERSION_BY_ENV` discworld `2026-09-12.2`
on its own 1000-case bench (every teleport changes ≥ 9 rays — no case had to be filtered).

| block | skill LIN / MLP (rand-init · observation) | unedited | PI | ND | GS |
|---|---|---|---|---|---|
| frustum (canonical) | 0.982 / 0.997 (0.828 / 0.984 · 0.229 / 0.950) | −0.971 | **+0.109 / fid 2.62** | −0.092 / 1.09 | −0.024 / 1.16 |
| appearance-fac (233 + 29 classes) | 0.545 / 0.676 (0.252 / 0.488) | −0.972 | +0.001 / 2.67 | **+0.376 / 0.94** | +0.317 / 0.80 |

Against dw-noiseless under the same protocol (`scratch/2026-09-13-protocol-rescore.md`): frustum PI **+0.231 / 1.54 →
+0.109 / 2.62** (worse on both axes), ND −0.135 → −0.092, GS −0.082 → −0.024 (still ≤ 0); appearance-fac ND
**+0.612 / 0.80 → +0.376 / 0.94**, GS +0.326 → +0.317, PI 0 → 0. Decodability is unchanged in kind: the frustum state
is read at ≈ 1.0 by the MLP on the trained model AND on the random-init model (0.984), so — as on every 128-ray
instance — the geometric variable is available from the input alone and training adds nothing a probe can see;
the linear probe gains over its random-init floor (0.83 → 0.98) exactly as on dw-noiseless (0.83 → 0.96).

**Reading.** A continuous observation makes the *prediction* far more precise (the 18× lower MSE) without making
the *position read-out* any more causal for the model's output: the model still predicts the next frame from
the frame, not from a position register it would consult. The categorical read-out (appearance-fac), which
does edit on dw-noiseless through ND, edits *less* here — plausibly because the dome makes each factorised
class (run centre, run length) a coarser description of a now-continuous frame, so a categorical write
underdetermines the target rendering more than it did on the plateau. Together with dw-blink (carried state, not
editable) and dw-5ray / dw-8ray (quantisation *raises* editability), this closes the "aliasing hides the register"
explanation: editability in discworld tracks how coarse the observation is, not how continuous.

**Not yet done.** Waterfall panel for the frustum PI arm; a second seed; the appearance (joint-cell) and grid
targets on this run; a Haufe-corrected PI row in Table 3 (the run is not in the Table 3 run list).

**Provenance.** Instance `datasets/discworld/dw-smooth/` (`scripts/drivers/dw_smooth_gen.sh`, unit `dw_smooth_gen`,
2026-09-12 19:56–22:03; seeds train 200e9, eval 225.2e9, edits 225.3e9, probe 1040e9 / 1050e9); run and scoring by
`scripts/drivers/rescore_2026-09-12.sh` stages 7–10 (unit `rescore_protocol`, 00:58–11:10 2026-09-13). Renderer
knob pinned by `tests/test_soft_render.py`. Rows in `build_full_tables` (long list) only.
