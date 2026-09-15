# inlp_sweep — INLP cascades and multi-probe shrink writes on four discworld runs (2026-09-14)

**Question (Sevan).** At every residual point of `L-dw-noiseless-20m`, `L-dw-8ray-20m`, `L-dw-smooth-20m`
and `L-dw-8ray-tok-20m`: how many orthogonal deflations before the full state (frustum position +
velocity) stops being linearly decodable — iterations to held-out R² < 0.4, to < 0.05, to exhaustion
(< 0.02) — and what Edit Index / fidelity guard the write through the first K cascade probes reaches at the
best K when every probe's target is shrunk toward the population mean by its own R² (the
`pim.editors.nullspace` shrink rule; uniform weighting blows up the weak tail), against the canonical PI / GS.

**Method.** `scripts/inlp_dw_sweep.py` — the construction of `experiments/inlp/scripts/inlp_dw.py` (the
2026-09-01/04 discworld INLP; probe 1 ≡ PI[zspace]) with the fit done from moment matrices on the GPU in
float64 as in `experiments/adjacent_flip_ablation/scripts/inlp_othello.py`; the probes are packed into the
canonical `NullspaceCascade` and written through the canonical `multiprobe_delta`. 20k probe sequences,
canonical split by sequence, canonical standardisation; matched random-direction control per point; K ∈
{1, 2, 4, 8, 16, 32, all} × the run's canonical PI α grid; the canonical 1000-case bench and scorecards
(ray-zone Edit Index for the frame models; the token bench's frame-set construction for the token model, †).
Wiring check per point: the K = 1 uniform write against the canonical PI[zspace] step.

`scripts/table.py` prints the summary table; `scripts/plot.py` draws `outputs/inlp_r2_by_iteration_<run>.png`
and the combined panel, in the style of the Othello INLP figure. Driver `drivers/sweep.sh`, unit `inlp_sweep`,
logs `logs/inlp_sweep/`. Cascades under `probes/<run>/cascade_pt*.pt`; scores `scores/inlp_<run>.json`.

**Bets on record (2026-09-14 16:30, before any row landed; measure = deflations of the mean-over-variables
curve to R² < 0.4 and < 0.05 at points 1–8, plus copies per variable to exhaustion).**
- Sevan: 8-ray is slightly LESS redundant than the other regression runs (noiseless, smooth); the appearance-fac
  8-ray variables are MEANINGFULLY less redundant.
- Claude: 8-ray is at least as redundant as noiseless and smooth (the 2026-09-04 whole-probe cascade on this run
  found 29–40 probes vs 15–20 on L-dw-20m, position read-outs above 0.8 for eight deflations); appearance-fac
  copies about equal to the position variables' (run centre ≈ quantised lateral position, run length ≈ quantised depth).
- Addendum 16:40 (Sevan adds `L-dw-5ray-20m` appearance-fac; the 5-ray full state is added as its own baseline): Sevan —
  5-ray appearance-fac is MEANINGFULLY less redundant (less certain now about 8-ray appearance-fac). Claude — unchanged:
  5-ray full state at least as redundant as noiseless; 5-ray appearance-fac about equal to 5-ray's own position variables.

**Write correction (2026-09-14 ~17:00, Sevan's catch).** The first write summed each variable's K-copy steps
independently (the Othello script's form, which writes ONE tile per case, so it had no cross-talk); with eight
variables written at once the non-orthogonal per-variable directions disturb each other's read-outs, and K = 1
sat far below canonical PI (8-ray: −0.01 vs +0.26). The write is now a JOINT step over the stacked copies of all
variables (multi-output least squares is separable, so at K = 1 the rows are the joint lstsq probe's and the
step is PI in z-space — verified: 8-ray pt 4 K = 1 +0.23 / 0.98 = canonical PI's guarded arm there). The
unregularised joint pseudo-inverse blows up from K ≈ 8 (guards 3–4 at every α: near-dependent, small-norm rows);
the default solver is a weighted ridge (rows weighted by R², λ = 1e-2 × mean diagonal), the truncated
pseudo-inverse (rtol 1e-2) is the alternative — on 8-ray pts 2 / 4 they give best +0.33 / 0.96 (K 32) and
+0.35 / 0.93 (K 85) respectively, K = 1 +0.25 on both. α grid extended to the canonical top (175). The three
runs that had completed under the independent-sum write (noiseless, 8-ray, smooth) are rescored writes-only
from their saved cascades (`drivers/rescore_writes.sh`); their first score files are kept under
`scores/_superseded/*_independent-sum-write.json`. Redundancy numbers are unaffected.
