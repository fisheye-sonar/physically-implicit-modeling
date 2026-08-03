# Does a multi-step (rollout) training objective change the latent world structure? — GRU

**Date:** 2026-07-16 · **Direction:** `multistep-prediction-objective` · **Status:** → **FLAG FOR PROMOTION** (clean negative result)
**Author:** worker subagent · **Scope:** GRU (primary). **RSSM leg: CUT** (rationale below).

## TL;DR (headline)
A **free-running multi-step rollout training objective** (`w∈{2,5}`: teacher-force context, then free-run `w`
steps feeding the model's own decoded predictions back in, BPTT through the whole imagination, MSE on all `w`
frames) **does NOT make the GRU's latent world state more editable**, and does not meaningfully improve any of
the structural properties it was hypothesised to. The core §4 editing pathology — state↔observation
**decoupling**, a **decoder-inert** position-probe direction, **belief sluggishness**, and **off-manifold
oracle collapse** — replicates **essentially unchanged** across `w=1` (single-step baseline), `w=2`, `w=5`.
If anything the multi-step objective **slightly degrades canonicality** (fiber residual ↑, position-linear R² ↓)
and **inflates the linear hull / curvature** — the *opposite* of the "more coherent/canonical" intuition.
**No blur catastrophe:** rollout sharpness moves *toward* GT (not below it), and open-loop rollout accuracy
*improves*, at a tiny next-step cost.

This directly tests, and does **not** support, the brief's intuition that a next-step-only loss "never forces the
state to be coherent under its own iterated dynamics, which may be exactly the property editing needs." For this
GRU / this task, forcing that coherence changes the *rollout* behaviour (as designed) but leaves editability and
identifiability where they were.

## Setup / provenance
- **Models — identical architecture / data / hidden (256) / epochs (400); only the objective (`w`) changes.**
  - `w=1` baseline: `runs/gru/7_dset4_gru_400epochs/best_model.pt` (reused; best ep170, val 0.0236). With our
    loss, `w=1` reduces *exactly* to the single-step teacher-forcing objective (verified analytically + the metric matches).
  - `w=2`: `runs/gru_multistep/w2_dset4_gru_400epochs/best_model.pt` (best ep110; 1-step val ≈ 0.0238).
  - `w=5`: `runs/gru_multistep/w5_dset4_gru_400epochs/best_model.pt` (best ep69 by multi-step val 0.0271; 1-step val ≈ 0.0249).
- **New training helper (no existing file touched):** `scripts/train_gru_multistep.py` (in-memory GPU training,
  vectorised free-running overshoot loss sliding the start index over the whole T=40 sequence). w=2 ≈ 5 min,
  w=5 ≈ 10 min on the RTX 5090.
- **Data:** `datasets/4_fixed_refl_inview` (T=40, R=128, edit_frame=20, 2 objects); teacher-forced **test** split
  (10k) for §1–§3, **edits** split (first 64) for §4. All probes in-sample (comparisons load-bearing).
- **Notebook (executed, 0 error cells):** `notebooks/experiments/editability/multistep/multistep_objective_structure.ipynb`.
- **Figures:** `/tmp/multistep_objective/` — fig0_sharpness, fig1_geometry, fig2_recoverability, fig3_fiber,
  fig4_editor_metrics, fig5{a,b,c}_waterfalls_w{1,2,5}, fig6{a,b,c}_scans_w{1,2,5}.

## Results by section (w=1 / w=2 / w=5)

**§0 Sharpness & next-step quality (the blur watch-item).**
- Teacher-forced next-step RMSE vs clean: **0.104 / 0.106 / 0.109** — multi-step slightly *worse* on the 1-step metric (small).
- Open-loop horizon RMSE (warm 10, free-run 30): **0.208 / 0.197 / 0.188** — multi-step *improves* rollout accuracy (as designed).
- Rollout total-variation sharpness (÷ GT TV): **1.28 / 1.22 / 1.07**. `w=1` is *over-sharp* (speckly, 1.28×GT); multi-step
  moves TV *toward* GT (1.07 at w=5), **not below 1** ⇒ **no blurry mean-hedging / mode-collapse.** Watch-item cleared.

**§1 Geometry.** Intrinsic dim (TwoNN) **5.2 / 5.3 / 5.2**, MLE **6.9 / 7.2 / 7.7** — ~flat, near the physical 8.
Linear hull @90% **39 / 51 / 68** (inflates). Tangent rotation **19.5° / 22.2° / 23.1°** (slightly *more* curved).
→ multi-step does **not** simplify/straighten the manifold; it spreads the linear hull and adds a little curvature.

**§2 Recoverability.** Position linear R² **0.84 / 0.82 / 0.76** (drops — less *linearly* readable), position MLP
**0.97 / 0.97 / 0.97** (flat). Velocity single-frame MLP (late-t) **0.94 / 0.94 / 0.95** (stable); two-frame−single-frame ≈ 0
(instantaneous, not temporal) for all w. → the physical statistic is no more, and slightly less-linearly, readable.

**§3 Canonicality / fiber-collapse.** MLP fiber residual ‖h−g(pos,vel)‖/‖h‖ **0.357 / 0.382 / 0.457** — **rises**
with w: `h` becomes **less** a function of the 8-dim physical state under the multi-step objective (more history /
scaffolding), i.e. **less canonical**, opposite the hypothesis. Linear residual ~0.88 for all (curved embedding, unchanged).

**§4 Editing head-to-head (headline).** Same edit set, all three GRUs. Numbers below are GT next-step RMSE (obs) /
obs-change (% of true-state swap) / ghost-ray ratio:
| state | w=1 | w=2 | w=5 |
|---|---|---|---|
| Unsteered (no-edit) | 0.280 / 0 / 1.00 | 0.278 / 0 / 1.00 | 0.277 / 0 / 1.00 |
| **True-state swap (model ceiling)** | **0.207 / 100 / 0.68** | **0.201 / 100 / 0.64** | **0.207 / 100 / 0.73** |
| Readout injection (readout→0.000) | 0.276 / 19 / 0.98 | 0.274 / 19 / 0.99 | 0.271 / 28 / 0.98 |
| MLP-probe gradient | 0.276 / 50 / 0.99 | 0.273 / 44 / 0.98 | 0.272 / 59 / 0.99 |
| Global-PCA projection | 0.269 / 79 / 0.93 | 0.272 / 45 / 0.97 | 0.266 / 47 / 0.95 |
| PCA geodesic | 0.277 / 62 / 0.89 | 0.275 / 62 / 0.89 | 0.271 / 74 / 0.86 |
| Decoder gradient (ORACLE) | 0.221 / 230 / 0.09 | 0.217 / 216 / 0.09 | 0.225 / 246 / 0.09 |

- **No non-oracle editor moves the observation toward the true post-edit trajectory, for any w:** every non-oracle
  editor's GT next-step RMSE (0.266–0.277) sits **at the Unsteered level** (0.277–0.280) and **far above** the True-state
  swap (0.201–0.207). Best non-oracle editor: **0.269 / 0.272 / 0.266** vs unsteered **0.280 / 0.278 / 0.277** — the
  ~0.01 gap is identical across w. **Editability is unchanged by the objective.**
- **Decoupling is structural, not a budget/geometry artifact:** the PCA-geodesic drives the *readout* progressively
  lower with higher w (readout RMSE 1.20 → 1.11 → 0.99), yet its obs-change/ghost/next-step barely move — a *better*
  readout still fails to move the decoder/dynamics.
- **Readout injection stays decoder-inert** (readout 0.000, obs-change ~19–28% of swap, ghost ~0.98) and the **oracle
  decoder-gradient** still nails step-0 obs (0.010–0.024 vs static target) only by parking off-manifold (leave-out
  local-PCA resid 0.99, global hull resid 16) and then collapsing — **all unchanged across w**.
- **Belief sluggishness ceiling unchanged:** even the True-state swap moves the obs only 0.11–0.13 (ghost 0.64–0.73).

## Reading (interpretation)
The multi-step objective does exactly what it is designed to on the *rollout* (lower open-loop horizon RMSE,
GT-matched sharpness) but this **buys no editability and no extra identifiability/canonicality** — and on these
measures mildly *hurts* canonicality (fiber residual ↑, position-linear R² ↓, hull/curvature ↑). This strengthens the
emerging reading that the editing difficulty in these world models is **structural** — a decoder-inert probe
direction plus the model's own single-frame belief sluggishness — and is **not** an artifact of a next-step-only
training loss that a rollout objective would fix. It is a genuinely informative **negative** for the "coherence
under iterated dynamics ⇒ editable state" hypothesis.

## Caveats / open questions
- **`w=5` "best" checkpoint is early (ep69)** by its own multi-step val loss (it plateaus early; 1-step val stays ~flat
  to ep400). A quick sanity re-run from `latest.pt` (ep400) would confirm the trends are not a checkpoint-selection
  artifact. (I expect they hold — the §1/§3 trends are monotone in w.)
- Only `w∈{2,5}` tested. A much larger `w`, a scheduled-sampling/curriculum variant, or annealing teacher-forcing
  might differ, but the monotone "canonicality ↓, hull ↑" trend argues against a hidden reversal.
- In-sample probes throughout (absolute R²/residual optimistic; cross-w comparisons are the load-bearing quantities).
- **RSSM leg CUT.** Rationale: GRU is the primary deliverable and produced a clean, complete §1–§4 result across
  `w∈{1,2,5}`. RSSM "latent overshooting" (imagining in the KL-regularised latent) is materially more involved — a new
  RSSM training variant — and would risk the ≤2–3h RSSM cap the brief set. Left as a clean follow-up if the GRU null
  is judged important enough to test for architecture-independence.

## Pointers
- Training helper: `scripts/train_gru_multistep.py` (new; nothing existing modified).
- Checkpoints (gitignored): `runs/gru_multistep/w2_dset4_gru_400epochs/`, `runs/gru_multistep/w5_dset4_gru_400epochs/`.
- Notebook: `notebooks/experiments/editability/multistep/multistep_objective_structure.ipynb` (executed, 0 error cells).
- Figures: `/tmp/multistep_objective/*.png` (11 PNGs; see §0/§1–§4 above).
