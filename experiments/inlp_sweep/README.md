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
