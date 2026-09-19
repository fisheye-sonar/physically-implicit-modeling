# 2026-09-15 — the Othello Edit-Index ceiling under the symmetric-difference headline

**Why.** The honest ceiling (the model run on a real history whose board IS the flipped board,
kept only when the model treats that history as ordinary) was measured 2026-09-11
(`honesty_check_v2.py`) on the UNION index only. The Othello headline has been the symmetric
difference since 2026-09-12 and the benches changed the same day (1000 single-tile flips at a
fixed 20-move prefix per instance, from a dedicated `edits` split; Li's 1001 retired to the
appendix anchor). The paper's Metrics section needs the ceiling on the headline axis.

**Script.** `experiments/adjacent_flip_ablation/scripts/ceiling_symdiff.py --run <run>` — the
v2 ordinariness filter verbatim (rmse to uniform-over-own-legal within the held-out 95th
percentile for the prefix length; seed 0, 900 sampled cases), the current canonical best arms
read from each run's `scores.json` (eval 2026-09-12.1, selected on symdiff), PI / ND / GS scored
on the SAME cases under both constructions plus the move fidelity ratio. Output
`experiments/adjacent_flip_ablation/scores/ceiling_symdiff_<run>.json`. GPU 4 min per run.

## Result (Edit Index symdiff / union · fidelity)

| run | exact boards | ordinary | unedited | **true counterfactual (ceiling)** | PI | ND | GS |
|---|---|---|---|---|---|---|---|
| L-oth-20m (standard) | 31 | 13 | −0.941 / −0.709 | **+0.909 / +0.692** · 0.16 | +0.751 / +0.531 · 0.29 (pt4 α3) | +0.826 / +0.637 · 0.18 (pt4 α0.35) | +0.399 / +0.119 · **2.89** (pt4 α0.2) |
| L-oth-adjacent-20m | 176 | 29 | −0.979 / −0.673 | **+0.916 / +0.674** · 0.18 | +0.078 / +0.074 · 2.99 (pt1 α60) | +0.668 / +0.327 · 1.55 (pt0 α12) | −0.658 / +0.006 · 7.45 (pt2 α1.5) |
| L-oth-adjacent-flip-20m | 91 | 32 | −0.954 / −0.643 | **+0.908 / +0.644** · 0.20 | +0.190 / +0.086 · 0.64 (pt2 α5) | +0.381 / +0.225 · 0.56 (pt1 α2) | +0.183 / +0.004 · 4.91 (pt0 α0.35) |

Fraction of the symdiff ceiling recovered on the same cases: standard PI 0.83, ND 0.91, GS 0.44;
adjacent ND 0.73 (but fid 1.55); adjacent-flip PI 0.21, ND 0.42.

## Reading

1. **The ceiling is ≈ +0.91 on every Othello instance under the headline** (+0.69 under union),
   against unedited floors of −0.94 … −0.98. It is not +1 because a Bayes-optimal model's own
   deviation from uniform-over-legal is a real fraction of the gap between the two references
   when the symmetric difference is 1–2 squares. The three instances agree to 0.01, as they did
   under union on 2026-09-11 (+0.66 … +0.70), so the ceiling is a property of the metric on
   these boards, not of the model. **Quote Othello indices against +0.91, not +1.** Discworld's
   overwrite-oracle ceiling is also +0.91 (ray-zone, 2026-09-07, 192-case pre-protocol bench —
   re-measure on the 1000-case bench before pairing the two in the paper).
2. **On standard Othello ND recovers 91% of the symdiff ceiling, PI 83%** — the same 91% as the
   union measurement of 2026-09-09 (+0.592 of +0.650, n = 16). The editors sit below the true
   counterfactual on every instance, as they must.
3. ⚠ **GS is destructive on this subset of standard Othello** (symdiff +0.40 at fidelity 2.89
   on 13 cases) although it is the best full-bench arm (+0.828 / 0.28 at pt 4, α 0.2). n = 13
   and the subset is selected for reachable boards (parity tiles), so this is a flag, not a
   finding: check GS's per-case fidelity distribution on the full bench before quoting GS as
   uniformly clean.
4. **Case counts differ from 2026-09-11 (42 → 31 exact, 16 → 13 ordinary on standard)** because
   the bench changed on 2026-09-12 (1000 fixed-prefix cases vs Li's 1001 with the 5–30 mix),
   not because the search is nondeterministic (`search_cf` seeds its RNG by the square).

## Open

- Bootstrap the ceiling over cases (n = 13 is thin) and re-measure the discworld oracle ceiling
  on the protocol bench so the paper can state "the index reaches +0.91 on both environments".

## Addendum (same day) — IM by case reachability

Same script, `--im` section added: the canonical IM arm (cached inverse maps, every residual
point, best point by symdiff) on (a) the ordinary reachable cases above and (b) a matched sample
of cases for which the search finds NO exact counterfactual board. Symdiff / union · fidelity:

| run | IM on ordinary reachable | IM on no-counterfactual cases |
|---|---|---|
| L-oth-20m (standard) | n=13: **+0.76** / +0.50 · 0.34 (pt 5) | n=30: **+0.70** / +0.45 · 0.38 (pt 5) |
| L-oth-adjacent-flip-20m | n=32: **+0.70** / +0.37 · 0.51 (pt 5) | n=32: **+0.70** / +0.36 · 0.57 (pt 5) |
| L-oth-adjacent-20m | n=29: **+0.53** / +0.42 · 0.49 (pt 1) | n=30: **−0.17** / −0.08 · 1.41 (pt 3) |

**Reading.** On the two flip models IM lands identically whether or not the flipped board is
reachable by any legal game: the inverse map generalises to boards no game produces, because
colour varies independently of square and move parity in their training data. On
adjacent-noflip IM lands ONLY on cases whose flipped board some history can produce (+0.53) and
fails on the rest (−0.17, destructive) — the full-bench −0.03 is the mixture. So IM's failure on
adjacent-noflip is a case-level reachability effect, and the operative property is whether the
training distribution varies the edited variable independently of the variables the output
consumes: it does with flips, it does not without them (colour = placing-move parity), and there
the state map g covers only the manifold of history-producible boards. Prediction for a
discworld analogue: couple the two discs' positions through the generator (fixed separation, or
a mirror) — single-disc teleports become off-manifold and IM should fail on them while
on-manifold paired edits still land.

Files: `experiments/adjacent_flip_ablation/scores/ceiling_symdiff_*.json` (`IM_ordinary_reachable`,
`IM_no_counterfactual`).
