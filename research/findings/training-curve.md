# Editability tracks decodability in time, in both environments — the gate is not training

**Status:** measured overnight 2026-09-01/02 on the canonical runs' own log-spaced checkpoints
(steps 1k, 4k, 16k, 64k, 128k, 256k, 512k, 780k), laid out as `runs/training_curve/<run>_s<step>/`
by `experiments/training_curve/scripts/make_training_curve.py` and scored by the unchanged `master_eval.ipynb`
(`EVAL_VERSION 2026-09-01.4`; fresh probes at every checkpoint, both floors reused).
Figure: `build_full_table.ipynb` **Fig 1**. Driver: `experiments/training_curve/drivers/training_curve.sh`.

## The question

Sevan's observation was that Othello became editable "only at scale with a lot of training".
Three outcomes were registered before the run: (1) Othello's editability emerges *late*, after
decodability saturates — a second phase discworld never enters; (2) discworld's editability is
*trending* — it is undertrained; (3) both are structural from early on.

## Discworld (`L-dw-20m`, frustum basis)

| step | LIN | MLP | unedited | PI EI / fid | GS EI / fid |
|---|---|---|---|---|---|
| 1k | 0.831 | 0.971 | −0.596 | +0.034 / 2.06 | −0.081 / 1.70 |
| 4k | 0.940 | 0.990 | −0.663 | +0.137 / 1.98 | −0.104 / 1.14 |
| 16k | 0.968 | 0.995 | −0.691 | +0.184 / 4.05 | −0.155 / 1.01 |
| 64k | 0.980 | 0.996 | −0.692 | +0.197 / 1.96 | −0.180 / 0.96 |
| 128k | 0.981 | 0.996 | −0.698 | +0.214 / 1.91 | −0.163 / 0.95 |
| 256k | 0.983 | 0.996 | −0.696 | +0.214 / 2.07 | −0.192 / 0.94 |
| 512k | 0.984 | 0.996 | −0.700 | +0.190 / 2.09 | −0.262 / 0.96 |
| 780k | 0.984 | 0.997 | −0.700 | +0.199 / 2.04 | −0.228 / 0.94 |

MLP decodability is **at the random-init floor (0.960) from step 1k** and saturates by 16k.
Linear decodability climbs 0.83 → 0.97 by 16k and is flat after 64k. PI's Edit Index rises
with it and then sits at **0.19–0.21 from 64k to 780k** — twelve-fold more training moves
nothing — with fidelity pinned near 2.0 throughout: every edit that lands is bought by
degrading the prediction, at every checkpoint. GS drifts *more* negative with training while
its fidelity settles just under 1: it becomes harmless and ineffective at once.

**Outcome (2) is dead.** Discworld is not trending; it is not undertrained in any way that
editability responds to.

## Othello (`L-oth-20m`, mine/theirs)

| step | LIN | MLP | unedited | PI EI / fid | ND EI / fid | GS EI / fid |
|---|---|---|---|---|---|---|
| 1k | 0.766 | 0.786 | −0.076 | −0.033 / 0.96 | +0.043 / 1.10 | −0.017 / 1.00 |
| 4k | 0.833 | 0.843 | −0.217 | +0.049 / 0.85 | +0.132 / 0.88 | +0.001 / 1.00 |
| 16k | 0.905 | 0.905 | −0.427 | +0.258 / 0.57 | +0.286 / 0.57 | +0.112 / 0.85 |
| 64k | 0.953 | 0.952 | −0.591 | +0.435 / 0.40 | +0.452 / 0.38 | +0.383 / 0.47 |
| 128k | 0.954 | 0.961 | −0.632 | +0.489 / 0.36 | +0.497 / 0.34 | +0.483 / 0.36 |
| 256k | 0.965 | 0.966 | −0.686 | +0.578 / 0.25 | +0.585 / 0.26 | +0.575 / 0.27 |
| 512k | 0.973 | 0.973 | −0.690 | +0.606 / 0.23 | +0.601 / 0.26 | +0.624 / 0.23 |
| 780k | 0.977 | 0.977 | −0.710 | +0.615 / 0.23 | +0.620 / 0.23 | +0.639 / 0.21 |

Editability rises **continuously with decodability from the start**: fidelity is below 1 by
4k steps (the edit already helps the model predict the post-edit board), and by 16k — with
decodability at 0.90 — the edit index is +0.26 at fidelity 0.57. There is no plateau in
decodability followed by a late rise in editability; both keep climbing together to 780k.
(Read the index against its floor: the unedited index moves from −0.08 to −0.71 as the model
learns to tell boards apart at all.)

**Outcome (1) is dead.** Nothing emerges late on Othello; there is no second phase for
discworld to be missing.

## Conclusion — outcome (3)

In *both* environments editability tracks decodability **in time**. The difference between
them is not *when*; it is that on Othello the write is non-destructive from 4k steps on and
becomes strongly beneficial, while on discworld the write is destructive at every checkpoint
and its non-destructive version does nothing (`inlp-redundancy.md`: no fidelity ≤ 1 arm
exceeds EI +0.012 at any K). Whatever gates discworld is present at step 1k and unchanged at
step 780k. Training length, and by extension "scale", is not the variable — the candidates
that remain are about the task and the representation it induces (target sharpness / CE vs
MSE; a nonlinear consumer of a linearly-readable code; recomputation across positions; the
edit being off the data manifold). The Othello-as-regression flip is the next test.

Also settled in passing: the discworld Edit Index is not an artefact of a metric that cannot
move — it moved with decodability from 1k to 16k and the fidelity guard moved with it.

Related: `inlp-redundancy.md`, `decodability-baselines.md`, `editability.md`.
