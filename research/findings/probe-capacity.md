# Probe capacity: the random reservoir plateaus below the trained model in BOTH worlds — at 0.975 on discworld, at 0.60 on Othello

**Status:** measured 2026-09-02. `experiments/probe_capacity/` — one-hidden-layer probes of
width h ∈ {LIN, 16, 64, 128, 512, 1024, 2048}, one residual point per environment
(discworld point 3, Othello point 7 — each the canonical LIN argmax), three sources
(trained model, random-init model, observation history), on **~5× the canonical probe rows**
so that width, not memorisation, is the binding constraint: discworld 250k sequences
(`dw-pn04/probe_250k`, 9.75M rows), Othello 170k games (`oth-uniform/probe_large`,
10.0M rows). 50 epochs (≈2× the canonical step count), batch 4096, 80/20 by sequence,
seed 0. Every fit persisted (`experiments/probe_capacity/probes/`, 42 files); results
`scores/probe_capacity_{discworld,othello}.json`; figure **Fig 2** in `build_full_table.ipynb`.
**Added 2026-09-20:** the corpus-size control (last section) — EDITABILITY through instruments fitted on 2–3× the
canonical rows; flat, except the discworld inverse map (+0.012 on IM per doubling).

## The question

Table 3 showed an untrained network already supports 0.96 of the trained model's 0.996
MLP-128 decodability on discworld, but only 0.58 of 0.98 on Othello. Sevan's hypothesis: the
probe class is *overpowered* for discworld and *right-sized* for Othello — widen the probe and
Othello's random reservoir should catch up too, just later. Alternatively the two worlds
differ in what random features *contain*, and no width closes the gap.

## Result — held-out Probe Skill vs width (in-sample gap in brackets where it exceeds 0.01)

| | LIN | 16 | 64 | 128 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| **discworld** trained | 0.983 | 0.986 | 0.996 | 0.997 | 0.998 | 0.998 | 0.998 |
| discworld random-init | 0.671 | 0.886 | 0.942 | 0.957 | 0.970 | 0.975 | **0.975** |
| discworld observation | 0.251 (.02) | 0.690 (.02) | 0.844 (.02) | 0.875 (.02) | 0.913 (.03) | 0.917 (.03) | 0.918 (.03) |
| **Othello** trained | 0.975 | 0.578† | 0.941† | 0.976 | 0.978 | 0.980 | 0.980 |
| Othello random-init | 0.569 | 0.429† | 0.551† | 0.577 | 0.589 | 0.596 | **0.599** |
| Othello observation | 0.530 | 0.447† | 0.628† | 0.737 | 0.790 (.01) | 0.801 (.02) | 0.805 (.02) |

† h below the EFFECTIVE output dimension, 64 tiles × (3 − 1) free logits = 128 (the third logit
per tile is softmax gauge): a rank-h bottleneck on the read-out, not a weaker function class —
width 128 reproduces the linear probe to three decimals. Discworld's d_out is 8, so no width
bottlenecks it.

**Neither reservoir catches up.** On discworld the random-init curve rises 0.67 → 0.975 and
stops, 0.023 short of the trained model at every width ≥ 1024, with in-sample gaps ≤ 0.001 —
so it is not data-limited and not width-limited: that is what random features of the
observation history contain, and training adds a real 0.02 on top. On Othello the random
curve moves 0.577 → **0.599** across a 16× widening on 8.5× the data (gap ≤ 0.005), against a
trained plateau of 0.980. The hypothesis of a *later* saturation is refuted; the board is not
in random features at any probe capacity we can reach, whereas position on discworld nearly
is. **The difference between the worlds is in what the random network computes, not in the
probe class.**

Two further facts the sweep settles:

- **Plateaus are ordered and never cross.** Discworld: observation 0.918 < reservoir 0.975 <
  trained 0.998 — a random transformer over the ray history is a better feature map for
  position than the raw history, by 0.06 at every width. Othello inverts the first two:
  observation 0.805 > reservoir 0.599 — raw moves beat random features of moves, and a wide
  probe on raw moves keeps climbing (it is learning fragments of the flip rules; its gap grows
  with width accordingly), while the reservoir does not. Random mixing *helps* a smooth
  target and *hurts* a recursive one.
- **The canonical corpus was fine for the model probes and not for the observation floor.**
  Trained and random-init values at LIN and 128 reproduce the canonical 30k/20k numbers to
  ±0.005 (hollow markers on Fig 2); the observation MLP-128 moves from 0.696 (gap 0.25) to
  0.875 (gap 0.02) on discworld and 0.724 → 0.737 on Othello. Table 3's observation rows are
  under-estimates at the canonical size, as flagged there; the 5× values are the honest floor.

## What it means for the programme

Together with `decodability-baselines.md`, `training-curve.md`, `inlp-redundancy.md` and
`recurrent-l.md`: decodability provenance is a property of the *task*, robust to probe class,
probe data, training length, architecture and write strategy. Discworld's state is nearly a
free function of its input; Othello's is manufactured by training. The editing results line
up with that split, and the remaining candidate gates are about the task's dependence on
state (target sharpness — the Othello-as-regression flip) and the edit's manifold.

## 2026-09-20 — corpus-size control: decodability and editability are flat in the amount of probe data at 2–3× the canonical rows; the one mover is the discworld inverse map (IM +0.012 per doubling) — `observed`

**Why.** Methods has to state that the regression probes and the inverse map are fitted on 30k discworld
sequences / 20k Othello games (1.17M / 1.18M rows — matched in rows) while the categorical discworld read-outs stream
200k sequences. The sweep above settles DECODABILITY at ~8× the rows (one residual point, streamed fits); nothing had
measured EDITABILITY through instruments fitted on more data. Sevan (2026-09-19): measure it rather than argue it.

**Set-up.** `experiments/probe_corpus_size/` — the canonical pipeline with the corpus size as the only knob. One run per
environment: `initial_othello_comparison/L-oth-20m` on the first n games of `oth-uniform/probe_large`, and
`noise_ablation/L-dw-noiseless-20m` (cartesian basis) on the first n sequences of `dw-noiseless/probe_250k`. Seed 0,
80/20 by sequence, every residual point, each run's own bench (1,000 cases on discworld), alpha grids and GS layers; the
run's canonical `scores.json` numbers — fitted on the canonical split, a DIFFERENT corpus — sit beside them as the
reference. Best arm per editor = the scorer's unguarded argmax in every row including the reference, so the rows are like
for like (see the caveat on selection). Cells are Edit Index / fidelity ratio; Othello's index is the symmetric-difference one.

| Othello, games (rows) | LIN skill | PI | ND | IM | IM-NN | g R² (max) |
|---|---|---|---|---|---|---|
| canonical: 20k of `probe` | 0.9751 | +0.818 / 0.30 | +0.749 / 0.33 | +0.806 / 0.38 | +0.036 / 2.54 | 0.884 |
| 20k of `probe_large` (1.18M) | 0.9747 | +0.819 / 0.29 | +0.748 / 0.34 | +0.810 / 0.38 | +0.036 / 2.14 | 0.886 |
| 40k (2.36M) | 0.9750 | +0.824 / 0.28 | +0.746 / 0.34 | +0.808 / 0.37 | +0.024 / 2.45 | 0.892 |
| 60k (3.54M) | 0.9752 | +0.819 / 0.28 | +0.746 / 0.34 | not run | not run | not run |

| discworld, sequences (train rows) | LIN | MLP-128 | PI | GS | IM | IM-NN | g R² (max) |
|---|---|---|---|---|---|---|---|
| canonical: 30k of `probe_120k` | 0.8718 | 0.9726 | +0.197 / 1.71 | −0.056 / 1.07 | +0.590 / 0.34 | +0.327 / 0.76 | 0.741 |
| 30k of `probe_250k` (0.94M) | 0.8718 | 0.9727 | +0.197 / 1.73 | −0.063 / 1.07 | +0.592 / 0.34 | +0.321 / 0.77 | 0.742 |
| 60k (1.87M) | 0.8725 | 0.9736 | +0.197 / 1.72 | −0.081 / 1.06 | +0.604 / 0.33 | +0.364 / 0.72 | 0.755 |

**Reading.**
- **Decodability is flat**: LIN moves by ≤ 0.0007 and MLP-128 by 0.0010 across sizes in both worlds — inside the
  training-seed SD of the same quantities (dw-noiseless, n = 3 at 512k: LIN ± 0.002, MLP ± 0.000) and consistent with
  the ~8× sweep above.
- **The linear editors are flat**: PI picks the same point and alpha at every size (Othello pt 4, α 3, within 0.006 —
  half a case-level SE of 0.011; discworld pt 2, α 100, within 0.001). ND within 0.003 (at 40k it picks a near-tied
  neighbouring arm, pt 4 α 0.5 instead of pt 5 α 0.35).
- **GS (discworld only)** drifts −0.056 → −0.063 → −0.081. That is inside the training-seed SD of this family's GS
  (± 0.027) and the refit noise of the MLP grid (same-checkpoint lab-vs-remote GS differs by up to 0.06 per arm), the
  case-level 95% intervals overlap ([−0.080, −0.045] and [−0.100, −0.062]), and the direction is DOWN — more probe data
  does not rescue the editor. It is a failing arm at every size (negative index, guard > 1).
- **The inverse map is the one instrument that moves, and only on discworld**: IM +0.592 → +0.604 at 2× its data (g R²
  0.742 → 0.755), against a training-seed SD of ± 0.005 — a small, real gain, so discworld's canonical IM is a slight
  UNDER-estimate of what a larger inverse-map corpus would give. It is 3% of the 0.40 gap to the best probe-based
  editor (PI +0.197) and changes no ordering. On Othello g's R² also rises (0.884 → 0.892) but the edit does not (+0.810 → +0.808).
- **IM-NN depends on corpus size by construction** (the retrieval bank IS the corpus): +0.321 → +0.364 on discworld
  as the bank doubles. On Othello it is a failed arm at every size (index ≈ +0.03, guard 2.1–2.5).

**Scope and caveats.** `observed`: one run per environment, one probe seed, one basis, and **2–3× the rows, not the 6.7×
(30k vs 200k) the question named**. The larger sizes are out of reach on this box: the dense regression fit needs 46.8 GB
of anonymous memory at 100k sequences (OOM-killed at the 45 GB cap) and the Othello inverse map 40.7 GB at 60k games
(OOM-killed at 40 GB) — `GOTCHAS` 2026-09-20; a 200k editability point needs a STREAMED regression fit, which is new
`pim` code and was not added mid-queue. Selection is the UNGUARDED argmax, not the tables' guarded `best_arm`: on
Othello and for discworld IM the winning arm passes the guard, so the two selections coincide; discworld's unguarded PI
and GS arms fail it (1.72, 1.06), so the guarded PI / GS cells the tables quote for this run (−0.096 / −0.162) were not
re-measured — with skill and the unguarded arm equal to 0.001 there is no reason to expect them to move, but it is not
measured. Falsified by: a streamed fit at 200k that moves skill by more than 0.005, or PI / IM by more than ~3
training-seed SDs.

**What the paper can say.** Refitting every instrument on two to three times the canonical probe corpus changes
decodability by at most 0.001 and the best linear-editor Edit Index by at most 0.006 in both environments; the
inverse-map editor gains 0.012 on discworld and nothing on Othello. (Decodability at ~8× the rows: the sweep above.)

**Evidence.** `experiments/probe_corpus_size/scripts/corpus_size_{oth,dw}.py`; results
`scores/oth_L-oth-20m.json`, `scores/dw_L-dw-noiseless-20m_cartesian.json` (every best arm with its case-level SE and
guard CI); fitted probes cached in `probes/` (gitignored). Queue jobs `ctrl_corpus_oth` (lab, 2026-09-20 19:58–20:13)
and `ctrl_corpus_dw` (lab, 20:13–20:28), logs `logs/paper_ci/ctrl_corpus_{oth,dw}/`; the earlier killed attempts and what
each left behind are dated in the experiment's README. Training-seed SDs: `experiments/paper_ci/dashboard/ledger.md`
(dw-noiseless, n = 3 at 512k; Othello's are not pooled yet — `rep_oth-standard_s1/s2` are in the queue).
