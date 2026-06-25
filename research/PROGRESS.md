# PROGRESS.md — Session Handoff

> Agent-owned, rewritten freely each session. Answers **"where is the work right
> now?"** — *not* "what's true" (that's `findings/`). Git history is the backstop.

_Last updated: 2026-06-24 (end of day)_

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

All notebooks under `notebooks/experiments/manifold_editing/`. Each worker wrote its own scratch
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
  "readable ≠ controllable." Fix stale notebook path on line 6 (→ `…/manifold_editing/`).

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

- **Notebooks reorganized** into `notebooks/experiments/manifold_editing/` (Sevan's move). All
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
