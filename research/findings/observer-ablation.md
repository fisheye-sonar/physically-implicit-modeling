# Observer ablation — five observers make the categorical read-out MORE decodable and LESS editable (2026-09-14)

**Status:** `observed` (one run, one seed, canonical scoring under the unified protocol; no waterfall yet).

**Question.** dw-8ray is the discworld instance where the categorical read-out edits best (appearance-fac: PI +0.38,
ND +0.49, GS +0.46). Two explanations were live: the observation is coarse (8 rays, 30 realisable runs), or the
output is small (8 values) so an edit has little to get right. Sevan's control (2026-09-13): keep every ray as
coarse as dw-8ray's but render the same two discs from FIVE observers on a ring about the frustum's depth
midpoint, discs confined to the circle every observer's near/far planes are tangent to
(`pim/environments/discworld/observers.py`; instance `dw-8ray-obs5`). The observation becomes 5 × 8 = 40
values; position is more determined (five partial views; as with anti-aliasing) but through a different pressure
— integrating views — and the output is five times wider. Read-outs: world-frame **Cartesian** regression (the
frustum basis is observer-0-relative) and the **per-view factorised appearance** (5 × (15 + 1, 5 + 1) classes per
disc, 110 logits, 20 tiles; `grid_target._view_cells`). Every arena position lights ≥ 1 kept ray in every view;
a disc is invisible to a given observer ~7 % of the time through occlusion, never to all five.

**Answer.** `runs/observer_ablation/L-dw-8ray-obs5-20m` (780k steps, 476 min, best val MSE 0.00302 vs dw-8ray's
0.00574 per entry — five views constrain each next frame), scored on its own 1000-case bench (from 1040 scanned;
teleport mean 1.9 world units, the round arena is tighter than the frustum's 3.1).

| block | skill LIN / MLP (rand-init · observation) | unedited | PI | ND | GS |
|---|---|---|---|---|---|
| cartesian (regression) | 0.979 / 0.997 (0.934 / 0.990 · 0.420 / 0.959) | −0.889 | +0.189 / fid 1.60 | −0.126 / 1.41 | −0.065 / 1.10 |
| appearance-fac (per view, 110 classes) | 0.964 / 0.989 (0.913 / 0.981 · 0.730 / 0.978) | −0.885 | +0.161 / 1.52 | **+0.418 / 1.06** | **+0.371 / 0.62** |

Against dw-8ray under the same protocol: geometric read-out PI +0.26 / 0.99 → **+0.19 / 1.60** (inert both ways, the
guard now failing); categorical read-out PI +0.38 → +0.16, ND +0.49 / 0.98 → +0.42 / 1.06, GS +0.46 / 0.54 →
+0.37 / 0.62 — **every editor lands lower**, while the categorical read-out became MORE decodable (MLP 0.94 →
0.99; the random-init transformer already reads it at 0.98, the observation MLP at 0.98).

**Reading.** Five observers did not restore the geometric register and cost the categorical one about a third of
its editability. The numbers land almost exactly on dw-smooth's (ND +0.38, GS +0.32, PI 0.00): two different ways
of making position better determined — a continuous profile, five views — produce the same outcome, a better
predictor with a less writable state. The per-view run codes are now redundant (five views of the same position),
and a write to one object's twenty tiles asks the model for a coordinated change it does not implement as a
single variable. Read with dw-5ray (coarser, +0.60) and dw-8ray (+0.46): editability in discworld tracks how
few distinct frames the observation can take, not how well position is pinned down. The output-dimensionality
explanation survives in weakened form — the 40-value output is where editing got harder — but it cannot be
separated from the redundancy here, since both grew together by construction.

**Not yet done.** Waterfall for the categorical GS arm; the single-view control on the same arena (N = 1, circle),
which separates "arena" from "observers"; a second seed; the fac floors' entry in `runs/_baselines/dw-8ray-obs5/`
(fitted in cartesian, recorded 2026-09-14 morning after the baselines cell learned the per-instance basis).

**Provenance.** Instance `datasets/discworld/dw-8ray-obs5/` (seeds train 230e9, eval 255.2e9, edits 255.3e9,
probe 1060e9 / 1070e9); chain `scripts/drivers/dw_8ray_obs5.sh` (unit `dw_8ray_obs5`, 2026-09-13 17:50 →
2026-09-14 07:29, one relaunch after the edits-sampler fix, GOTCHAS 2026-09-13). Renderer + arena pinned by
`tests/test_observers.py`; per-view target by `tests/test_multiview_target.py`. Long-list tables only.
