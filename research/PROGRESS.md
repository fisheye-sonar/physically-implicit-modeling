# PROGRESS.md — Session Handoff

> Agent-owned, rewritten freely each session. Answers **"where is the work right
> now?"** — *not* "what's true" (that's `findings/`). Git history is the backstop.

_Last updated: 2026-07-09 (learn-to-edit launched + nbstripout terminal fix)_

## 2026-07-09 (pm) — learn-to-edit launched, nbstripout flood fixed

**Branch:** `learn_to_edit` (Sevan made it; the 2026-07-09 promotions/rename are committed+merged via PR #7).

**HARNESS UPGRADE (durable, from Sevan's learn_to_edit review — points 1 & 5):** added a **"Notebook
legibility" hard standard** to `CLAUDE.md` (workers read it) + a pointer in `WORKER.md`. Requires, in every
experiment notebook: a **definitions table up front with each metric's explicit formula** (not buried); the
**same metric set + units across anything compared** (RMSE, not MSE); **tables for dense value sets**; inline
**data-source provenance** for borrowed constants; and a **GT/reference column in every comparison figure**.
This is the fix "for the long run, not just this notebook."

**learn_to_edit v2 REVISION — DONE + VERIFIED.** `editability/learn_to_edit.ipynb` revised in place (27 cells,
0 errors, 8 figs; RMSE now used throughout; definitions table added; note addendum in
`scratch/2026-07-09-learn-to-edit.md`). Verified on disk incl. the GT column now in the FT waterfalls (Fig 5d)
and the new Variant-B data-scaling figure (Fig 4B). **Verdict UNCHANGED** — the new B fine-tune budget sweep
reinforces v1: held-out d_gt improves only slowly (0.287→0.273 over 128→1024), ghost drops modestly,
**sel_err gets monotonically WORSE with budget** (all worse than ORIG's 0.129), h_edit stays off-manifold
(~2.7–2.9 vs real ~1.75), and fine-tuning slightly **de-canonicalizes** (fiber 0.382→0.407). Editability
still not cleanly induced. Deeper follow-ups (heavier FT, λ sweep, RSSM) remain parked for Sevan's call.

**Note (not acted on):** a ~35-min-old ipykernel (7 procs, ~part of 4.6 GB GPU) persists — most likely
Sevan's own VSCode review kernel (predates the worker; stable kernel file), so NOT killed. GPU has headroom;
kill it only if it's a stray.

**nbstripout terminal flood — FIXED.** The `BrokenPipeError` was flooding Sevan's terminal (git prompt kept
re-invoking the clean filter, which printed a Python traceback on every early-closed pipe). Fix (local
`.git/config`, persists across branches): clean filter now runs python with `signal.signal(SIGPIPE,
SIG_DFL)` so a broken pipe dies silently instead of printing a traceback, and `filter.nbstripout.required
= false` so a filter hiccup can't hard-fail git. Verified: early-closed pipe exits with no traceback;
stripping still works (0 outputs in cleaned stream). Also killed an orphaned 88-min Jupyter kernel
(leaked from earlier `setsid nohup` worker runs) — GPU now clear (1.5/32 GB). Discarded a stray
kernelspec-only diff on `editability_structure.ipynb`.

**Learn-to-edit — DONE + VERIFIED (both variants working end-to-end).** `editability/learn_to_edit.ipynb`
(15 cells, 0 errors, 7 figs), note `scratch/2026-07-09-learn-to-edit.md` (→ FLAG FOR PROMOTION). Verified
on disk (numbers present, no orphaned kernels). **RESULT: NEGATIVE — editability could NOT be cleanly
induced on this GRU**, neither by a frozen learned editor (A) nor a light fine-tune (B); both show the
**memorization signature** (train obs-loss collapses, held-out barely beats unsteered, selectivity gets
WORSE). Nuance: the info IS present (A overfits train; the obs-gradient oracle solves per-sample but
off-manifold at resid 6.8; more data helps d_gt/ghost *slowly*) — it's just **not reachable by a fixed/
amortized function few-shot, and only off-manifold per-sample**. B also **failed to canonicalize** (fiber
flat 0.382→0.383; readability down; dims up) → doesn't falsify editability⟺canonical, just fails to
support it. Strength: **medium**, not "impossible." **HELD for Sevan (judgment call on a negative
result):** the main threats-to-validity / flip-tests are (i) a **heavier Variant B fine-tune** (current was
light, 1.5k iters — a stronger intervention could still induce editability = a positive result), (ii) a
**λ sweep** mapping the on-manifold↔reach-the-edit tension, (iii) the **RSSM pass**. Did NOT auto-launch
these — interpreting/extending a negative result is the human judgment call. Offer stands to launch on request.

**Master notebook — REVISION FEEDBACK from Sevan (deferred behind learn-to-edit; captured in
`directions/master-editability-notebook.md` REVISIONS section):**
- §4 waterfalls are disliked AND possibly **wrong** — the "Unsteered" panel "looks like a model's output,"
  not an unsteered rollout. **Investigate as a potential bug**, not just aesthetics.
- Drop the purplish colormap → **classic academic style** (the light/Okabe-Ito theme).
- **Add the next-step line plots** (the 1D-line style Sevan liked from `geodesic_walk_k150`).
- Sevan will give fuller notebook feedback later.

## 2026-07-09 — Promotions, folder rename, git/nbstripout triage

**Folder rename DONE:** Sevan renamed `notebooks/experiments/manifold_editing/` → `editability/`.
Swept **all** downstream path references (findings, directions, scratch, PROGRESS, folder README) —
`grep manifold_editing` now returns 0. Notebook internals unaffected (relative paths, same depth).

**4 candidates PROMOTED to `findings/` (Sevan-approved), with preliminary/scoped hedging:**
- `findings/editability.md` (was too conclusive about "the GRU" → now scoped to *this
  pure-next-step-prediction GRU checkpoint*, not GRUs in general) ← `candidate-editability`.
- `findings/state-geometry.md` ← `candidate-state-geometry`.
- `findings/architecture-independence.md` (NEW) ← `candidate-rssm-replication`.
- `findings/predictive-quality.md` (NEW) ← `candidate-predictive-quality`.
Each opens with a **Scope (preliminary)** banner: claims are about *these trained checkpoints* at this
stage, not the architectures in general. Candidates marked ✅ PROMOTED, kept as backing detail.
`findings/README.md` index updated.

**git / nbstripout triage (Sevan's error):** nothing corrupted. (1) The `BrokenPipeError` is benign —
git ran the `nbstripout` clean filter on the large notebooks during the branch switch and closed the
pipe early; the checkout completed (clean tree, right HEAD `2dc6b4f`). (2) **Real consequence:**
`nbstripout` (clean filter, `required=true`) strips notebook outputs on commit, so after the
main→branch roundtrip the working-copy notebooks now have **0 embedded figures** (`00_master_editability`
+ `diagnostic_corrections`). The figures survive only as `/tmp` PNGs → **copied to gitignored
`runs/_review_figures/{master_editability,diagnostic_corrections}/`** so they're not lost to a /tmp wipe.
DECISION FOR SEVAN: to review the master notebook *with* figures you must either re-run it, view the
saved PNGs, or (if you want persistent inline figures) exempt presentation notebooks from nbstripout /
export an HTML. Nothing to fix in the repo itself.

**learn-to-edit: HELD** (Sevan's call). Brief stays `proposed`; not launched.

**Dynamics thrust — reframed (Sevan):** velocity lives in the *state* (nonlinear/entangled), so the
next question is **how the GRU *uses* positions and velocities to update its state** (mechanism of the
transition), not "state vs dynamics." This is the natural successor thread once editability is banked.

**Uncommitted now:** the rename sweep + 4 promotions + 2 new findings + folder README fix +
`runs/_review_figures/` (gitignored). Ready to commit on request.

## 2026-07-08 — Editability reorganization session

**Branch:** `editability_reorganization` (off the merged RSSM work; HEAD 6bcc3a9). NB: the prior
`2026-07-02` RSSM-investigation PROGRESS section lived on `editability_rssm_replicate`'s working tree,
not this branch — but the substantive artifacts are all HERE (notebook `rssm_structure/
rssm_state_geometry.ipynb`, restored scratch note, and `candidate-rssm-replication.md`).

**Corrections worker — DONE (2026-07-08):** `directions/diagnostic-corrections.md` → notebook
`editability/diagnostic_corrections.ipynb` + note `scratch/2026-07-08-diagnostic-corrections.md`
(verified on disk). Results folded into findings + candidates:
1. **Velocity 2×2 → "velocity is temporal" RETIRED (both models).** single-frame MLP ≈ 2-frame MLP
   (Δ ≤ 0.007 late-t; GRU sf-MLP R² 0.94), `dh` worse than single-frame. Velocity is instantaneously
   readable, just **nonlinearly** — the old 0.47→0.76 gap was the linear→MLP axis, not single→temporal.
   *Strategic:* undercuts the planned "velocity-in-the-dynamics" thrust — velocity is in the STATE
   (nonlinear/entangled), not the transition. Reframe that thrust before running it.
2. **RSSM det-only fiber = 0.368 ≈ GRU 0.337** (full-320 0.602 was the stochastic `s` at 0.891). The
   "RSSM less canonical" claim is DEAD — det cores are on par; KL structure buys no canonicity.
3. **Small-k geodesic — SHAKY (worker-flagged), overturned my brief's expectation.** Walked states hug
   the manifold MORE than real states (honest leave-out resid 0.11–0.30 < real 0.58–0.79); obs moves a
   lot but readout only 30–46% reached. Reinterpretation: bottleneck is reachability *along* the curved
   manifold, not off-manifold ejection. Weak swap denominator → exploratory, NOT promoted.

**Master notebook worker — DONE + VERIFIED (2026-07-08):** `directions/master-editability-notebook.md`
→ `editability/00_master_editability.ipynb` (primary/entry notebook; note
`scratch/2026-07-08-master-editability.md`; PNGs `/tmp/master_editability/fig0–7`). Verified on disk:
33 cells, **0 error outputs**, 8 embedded figures; every corrected number present in outputs (velocity
0.944/0.951, fiber 0.337/0.368/0.602/0.891, reversion 0.011→0.275). Visually spot-checked Fig 5 (unified
5-editor waterfall, dark theme, green=target/red=ghost — reads clearly) and Fig 6 (the reversion: GT
sticks, MLP-gradient reaches target @step0 then climbs back by ~step4 with the quantitative curve). Repo
clean — the worker's OOM-recovery runner scripts stayed in `/tmp`, none leaked into git. Only nit: RSSM
scree @90% recomputed = 35 vs cited 34 (subsample; noted in-notebook). **Aesthetic caveat for Sevan:**
§4 uses **waterfalls**, not the 1D-line overlay you said you liked from `geodesic_walk_k150` — waterfalls
show more, but the 1D-line version can be added if you prefer it.

**Directory reorg (Sevan item 4d) — judgment call:** did NOT rename `editability/`. Sizing showed
~17 markdown files reference the path (incl. provenance scratch notes, which shouldn't be rewritten to a
new path). Instead expressed structure via a **primary/working/scratch convention** documented in
`notebooks/experiments/editability/README.md` (primary = `00_master_editability.ipynb`), plus a
naming fix (local-tangent projection [one-shot] ≠ PCA geodesic [iterative]). A full pillar rename remains
available as a coordinated reference sweep if Sevan wants it.

**Done this session (orchestrator, CPU, in parallel with the worker):**
- **Findings corrected** (Sevan-authorized, NOT promotions): `editability.md` summary → non-canonical /
  readable≠controllable (supersedes "target unreachability") + 2026-07-08 log entry flagging the
  velocity 2×2 in-progress (do not cite "velocity is temporal" until it lands); `state-geometry.md`
  summary → intrinsic dim ~5–7 + curvature ~56° + local-resid tautology retraction; fixed stale
  editability notebook ref.
- **Scratch consolidated** → 4 self-contained candidates (kept separate, not squished):
  `candidate-editability`, `candidate-state-geometry`, `candidate-rssm-replication`,
  `candidate-predictive-quality`. Raw dated notes retained as provenance; scratch/README points at them.
- **Learn-to-edit brief** written (`directions/learn-to-edit.md`, `[reframe]`, status **proposed**):
  Variant A frozen editor (information-presence test), Variant B light fine-tune (inducibility +
  re-measure canonicality). Ready for Sevan to mark active and kick off next turn.
- **Recovered** `scratch/2026-07-02-rssm-state-geometry.md` (was untracked + disturbed during a scratch
  tidy; restored from commit 7719825, now staged/tracked — no longer at risk).

**Promotions HELD for Sevan's post-lunch read** — the 4 candidates. Recommendation: promote editability
(after the velocity 2×2), state-geometry, rssm-replication (hold the fiber *magnitude* pending the
det-only refit), and the RSSM generative-quality gap. Each candidate ends with its own recommendation.

**RSSM eval refinement (Sevan item 2):** specific case (non-canonicality measured on the full 320-d
incl. stochastic `s`) handled NOW by the worker's det-only refit; broader note (RSSM evals should
report h-only / s-only / full consistently, not default to full-state) captured in
`candidate-rssm-replication`.

**Session tasks — all DONE** (Sevan's 6 items): (1) small-k geodesic ✓, (2) velocity-MLP-on-h_t ✓
[temporal retired], (3) master notebook ✓ + reorg [light-touch, flagged], (4) learn-to-edit brief ✓
[proposed], (5) unified waterfall comparison ✓ [Fig 5/6], (6) git [Sevan's earlier push/merge/branch].

**Uncommitted:** a clean body of finished work on `editability_reorganization` (master + corrections
notebooks, 4 candidates, 3 briefs, findings corrections, folder README, restored 2026-07-02 note staged).
NOT committed — waiting on Sevan (harness rule: commit only when asked). Ready to commit on request.

**HELD for Sevan (decisions):** promotion calls on the 4 `candidate-*.md`; mark `learn-to-edit` active to
launch next; the pillar-rename; the §4 waterfall-vs-1D-line aesthetic choice; and — the strategic one —
**reframe the dynamics-identifiability thrust** now that velocity is shown to live in the state
(nonlinear/entangled coordinate), not the transition.

## 2026-06-29 — RSSM refinement (engineering, branch `rssm_refinement` off main)
Good-faith predictor-tuning of the RSSM (item #4 of the 2026-06-24 sequence). Full write-up:
**`research/scratch/2026-06-29-rssm-refinement.md`**. Headline: best RSSM now competitive —
near-horizon clean-obs MSE 0.01726 vs GRU 0.01515 (~14% gap), **beats GRU at long horizon**,
recoverability fell 0.55→0.32 as a byproduct (no position supervision). Fixed a real bug
(best-checkpoint was selected by total ELBO → froze on an undertrained warm-up epoch; now
by recon loss). Levers = lr(3e-4)+free_nats(3)+epochs; architecture is NOT the lever (plateau).
Qualitative gap confirmed (Sevan's eye-test): RSSM-mean fades the 2nd object; RSSM-**sampled**
rollout jitters/forks — analyze in prior-mean mode. Best ckpt: `runs/rssm/4_dset4_refined_best/`
(gitignored — reproducible from config+seed0). NEXT: parallel GRU tuning pass for a fair compare.
New substrate (committed): `scripts/sweep_rssm.py`, `scripts/compare_rollouts.py`, RSSM `sample`
toggle + enc/dec depth, recon-based ckpt selection. Watcher-heartbeat used for monitoring (see
auto-memory `feedback-watcher-heartbeat`; ScheduleWakeup did not fire in this env).

## Current state

- **Branch:** `edits_investigate_structure`
- **Active thread:** causal editability of GRU hidden states (sub-Q3), reframed around the
  **canonical sufficient statistic**. Verified from the sim: dynamics are constant-velocity
  (no accel/vel-noise; tiny pos noise 0.04), so the **minimal sufficient statistic is
  `(positions, velocities)` = 8-dim** for 2 objects — which sits right at the variance elbow.
- **Synthesis reached this session (the "why can't we edit it" answer):** the GRU state is
  **predictively sufficient but non-canonical**, and the world state is embedded in `h` as a
  **curved, history-entangled, non-snapshot** manifold. The *readable* code ≠ the *controllable*
  code: a probe reads position off a linear slice, but rendering a moved object needs an `h` that
  is on the curved ~6-dim manifold, carries the consistent ~35% "extra" state, and encodes
  velocity *temporally* — no low-dim probe-targeted edit produces such an `h`; the only state that
  renders the target is off-manifold and the dynamics reject it. Two independent experiments
  (keystone + geometry) corroborate this.
- **Candidate unification (framing, not a finding):** **editability ⟹ canonical (snapshot,
  factored, on-manifold) state ⟹ recoverability + coherent rollout + persistence.** This GRU has
  the *dimensionality* (~6–8) but not the *canonicality* — the gap explicit physical scaffolding
  would close. Language to ground in: causal representation learning (observational vs
  interventional identifiability; Locatello disentanglement impossibility); observability-vs-
  controllability (control theory). Read into these before committing vocabulary.

## Done this session (2026-06-24) — all 4 verified on disk

All notebooks under `notebooks/experiments/editability/`. Each worker wrote its own scratch
note + numbered notebook (plots + printed tables). Orchestrator verified every headline number
against the notebooks' printed outputs (not the sign-offs).

1. **Canonical-state keystone** — `canonical_state_editing.ipynb`; note
   `scratch/2026-06-24-canonical-state-editing.md`; PNGs `/tmp/canonical_state/`. **Hypothesis held
   strongly.** (A) Position linearly readable (R² 0.84, MLP 0.96); **velocity NOT readable from a
   single `h_t`** (R² 0.47) — it's a **temporal feature** (2-frame MLP → 0.76). (B) **Fiber NOT
   collapsed:** best `g(pos,vel)→h` leaves **34.7% residual**; linear→MLP drop 0.53 ⇒ strongly
   curved embedding. (C) **Completing the target to `(pos,vel)` does NOT fix editing** (1.4% gap,
   ghost 0.99, identical to position-only) ⇒ kills the velocity-incompleteness hypothesis. (D)
   **Obs-driven edit = readable≠controllable, localized:** reaches the target obs but lands 15.7
   off-manifold / 16.7 from canonical and reverts by ~step 4 (sequence target sticks better).
2. **Geometry diagnostic** — `manifold_geometry_diagnostic.ipynb`; note
   `scratch/2026-06-24-manifold-geometry-diagnostic.md`; PNGs `/tmp/manifold_geometry/`.
   (i) **Intrinsic dim ~5–7** (TwoNN 5.2, MLE 6.9) — brackets the physical 8 DOF; 38–73-dim global
   hull is the curved embedding, not DOF. (ii) **Strongly CURVED**: tangents rotate ~56° at the
   nearest-neighbor spacing; local tangent never aligns with global. (iii) **The geodesic's
   "strictly on-manifold" local-resid ≈0.0002 was a projection tautology** + coarse `LOCAL_K=512`;
   honest local residual never collapses (~0.75–0.84 at all k).
3. **Geodesic K=150 confirmation** — `geodesic_walk_k150.ipynb`; note
   `scratch/2026-06-24-geodesic-walk-k150.md`; PNGs `/tmp/geodesic_k150/`. K=30 "curvature barrier"
   was a **schedule artifact** (fractional step decays geometrically): constant-step control
   descends ~2× faster (RMSE→0.35), readout *is* reachable; obs still doesn't move. NOTE: its
   "stays strictly on-manifold (local resid 0.0002)" sub-claim is **retracted** by experiment 2
   (tautology). Its core point (readout reachable, obs unmoved) is now subsumed by the keystone.
4. **PCA component → position** (earlier) — `pca_component_position.ipynb`; note
   `scratch/2026-06-23-pca-component-position.md`. Metric-dependent; parked for interactive refine.

## Awaiting Sevan (human-gated — I did NOT touch `findings/` or `RESEARCH.md`)

- **Promotion calls** on 4 scratch notes: canonical-state-editing (proposed new *core* editability
  finding), manifold-geometry-diagnostic, geodesic-walk-k150, pca-component-position.
- **Finding corrections** (these contradict established entries):
  - `findings/editability.md` current-understanding — *target unreachability under manifold
    constraint* is **superseded** by non-canonicality / curved embedding / readable≠controllable.
  - **Local-residual numbers** in `findings/editability.md` + `findings/state-geometry.md` are
    **projection tautologies**; honest local residual floors ~0.75–0.84 (dated correction owed).
  - Trivial: `findings/editability.md:6` still has the pre-move notebook path (left for you).
- **Caveats for the artifact-or-signal calls:** N=64 edits; tiny |v|≈0.05 depresses velocity R²;
  "canonical" reference is teacher-forced (soft oracle). Curvature + fiber-collapse use the full
  200k–390k bank, so those are robust.

## Proposed sequence (saved 2026-06-24 EOD — pick up tomorrow)

Agreed direction, in order. Revised by the EOD discussion (velocity-in-dynamics + the
"don't-integrate-yet" decision).

1. **Velocity probe check (cheap, do first):** retrain the velocity probe on **late timesteps
   only** (t≥~15, where velocity is actually inferrable) and plot **probe-R²-vs-rollout-step** (as
   `world_model_eval` does for position). The current keystone probe used ALL timesteps incl. early
   frames where velocity is undetermined — a real confound. Prediction: single-frame R² rises but
   plateaus below the 2-frame 0.76 ⇒ velocity is encoded temporally, not as a snapshot coordinate.
2. **Velocity-in-the-dynamics (the bridge to dynamics-identifiability):** probe the GRU **update-
   network activations** (gate/candidate pre-acts `z,r,n`), not just `h`. Hypothesis (Sevan): the
   state stores *position*; the update recomputes effective velocity from `obs_t` vs `h_{t-1}` and
   discards it. If velocity is decodable from the update activations but not `h`, velocity is
   identified in the **dynamics**, not the representation — a dynamics-identifiability result, and
   the natural zoom-out from editability. NB: this reinterprets the keystone's 34.7% residual — it
   conflates *spurious history* with *legitimate dynamics scaffolding* (a reason it's underbaked).
3. **Sevan's promotion + finding-correction calls** on the 4 scratch notes (still owed) + the doc
   edits below.
4. **RSSM refinement (engineering, autonomous, NEW BRANCH off `main`):** the RSSM works but predicts
   worse than the GRU and its probe is weaker — diagnosing it now would confound undertrained-vs-
   architecture. Refine training/hyperparameters largely autonomously, **with a defined target +
   compute budget** (e.g. match GRU rollout-prediction quality within X%, or N trials) so it can't
   spin. Branch off `main` (not this diagnostic branch); checkpoint lands in gitignored `runs/`.
   Decoupled from integration — can run in parallel.
5. **Re-run the diagnostic on the refined RSSM** → the generalization result for the editability /
   canonical-state story.

**Integration decision (revised this session): DO NOT integrate the day's instruments into main
yet.** Metric-bloat is the anti-goal — the story should live in a *few principled values*, not 30
metrics. The fiber-collapse residual (conflates history vs dynamics-scaffolding) and the geometry
diagnostics (curvature metric needs a cleaner definition; not yet deeply owned) stay **exploratory**
until they are (a) formalized into something principled and (b) understood well enough to present.
Integration bar = **principled + deeply understood + paper-worthy.** Manifold-projected editing may
later be kept as the *reference editor* (honestly captioned: best of a set that all largely failed).

## Docs to edit from this session (OWED)

- **`RESEARCH.md` (Sevan):** add the organizing-principle *hypothesis* — affordances may be
  downstream of a single property, a **canonical, factored, predictively-sufficient state**;
  editability is its sharpest test. Human-authored; mark as hypothesis not result.
- **`findings/state-geometry.md` (correction owed):** local off-manifold residual ≈0 was a
  **projection tautology**; honest local residual never collapses (~0.75–0.84 all k). Intrinsic dim
  ~5–7 brackets the physical 8 DOF; 38–73-dim hull = curved embedding, not DOF.
- **`findings/editability.md` (correction owed):** supersede "target unreachability under manifold
  constraint" → non-canonical state / curved `(pos,vel)→h` embedding / velocity-in-dynamics /
  "readable ≠ controllable." Fix stale notebook path on line 6 (→ `…/editability/`).

## Meta / strategy (in discussion, 2026-06-24 EOD — not yet ratified)

- **Depth-first per criterion** (dig into one affordance, make it precise, then zoom out) — endorsed.
  **Zoom-out triggers:** (i) understanding plateaus (experiments refine numbers, not the mental
  model); (ii) you have a paper-section-worthy claim; (iii) the live leads point *outward* (to
  another architecture or another sub-question). **Editability is at a zoom-out point now** — its
  remaining threads (RSSM generalization, velocity-in-dynamics) are already outward moves; likely
  next criterion = **dynamics-identifiability.**
- **Scaling stance:** the bottleneck is **Sevan's understanding (serial)**, not compute. So:
  parallelize the **engineering substrate** (training/infra/datasets — objective targets) across
  autonomous branches; keep **diagnostic science serial + interactive**; use worker agents for
  legwork *within* a criterion, synthesis stays human. Automation should *feed* understanding, not
  flood it.
- **Educational gate (proposed):** nothing promoted/integrated without a short "mechanics & meaning"
  explainer (how computed, what it means, assumptions, failure modes) that Sevan has read and could
  present. Exploration ungated; *integration* gated on *understanding*. This is the prerequisite for
  scaling automation safely.
- **Findings-gate evolution — RATIFIED + DONE (2026-06-25):** the bright line moved from *typing*
  to *commitment*. The **orchestrator may now draft `findings/` edits as a diff for Sevan's
  approval** (workers stay scratch-only); the promotion decision + approval stay human;
  `RESEARCH.md` stays fully human-authored. Encoded in `research/README.md`, `ORCHESTRATION.md`,
  `WORKER.md`. (A PreToolUse hook to enforce remains a future upgrade, still prose-only.)

## Substrate / harness state

- **Notebooks reorganized** into `notebooks/experiments/editability/` (Sevan's move). All
  internal relative paths normalized to the new 3-deep location; KB markdown refs updated. New
  convention in CLAUDE.md: **number every cell (`# [N]`) and every figure (`Fig K`)**.
- **Briefs written this session:** `directions/canonical-state-editing.md` `[reframe]`,
  `directions/manifold-geometry-diagnostic.md` `[in-frame]`; backlog index updated.
- **Multi-agent orchestration used for the first time (worked):** 3 background workers executed
  end-to-end this session; ownership boundaries held (wrote only `scratch/` + own notebooks); the
  verify-on-disk discipline caught nothing fabricated but is the reason we trust the numbers.
  Restraint still applies — one execution-heavy worker at a time, judgment-heavy work stays
  interactive.

## Open decisions / parked

- **Background-agent Edit/Write — FIXED 2026-06-23** (`settings.local.json`: added
  `Write`/`Edit`/`NotebookEdit` to `permissions.allow`; `worktree.bgIsolation: "none"` because
  `datasets/`+`runs/` are gitignored so a worktree has no data). Verified this session — workers
  used `NotebookEdit` cleanly.
- **`Read` token cap on figure-heavy notebooks:** a fully-executed notebook with embedded PNGs can
  exceed the `Read` limit (hit on `geodesic_walk_k150.ipynb`), which blocks `NotebookEdit`'s
  read-precondition. Workaround used: surgical JSON edit via Bash for that one file; otherwise keep
  outputs lean / edit setup cells before outputs accrue.
- **Harness enforcement:** PreCompact hook wired (reminds to update PROGRESS before compaction).
  Still prose-only (deferred until a failure demands): scratch→findings promotion gate, the
  `RESEARCH.md` write-block, the worker-reads-orchestrator-files guard.
- **Shared notebook setup → `pim/`:** `rollout_from_flat`, `decode_pos`, `sigma`, the
  load/teacher-force/subspace/warm bootstrap are duplicated per notebook. Factoring into `pim/`
  would kill the cold-start burden. (Code change — Sevan's call.)
