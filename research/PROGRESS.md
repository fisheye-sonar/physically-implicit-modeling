# PROGRESS.md — Session Handoff

> Agent-owned, rewritten freely each session. Answers **"where is the work right
> now?"** — *not* "what's true" (that's `findings/`). Git history is the backstop.

_Last updated: 2026-07-16 (editability_multi_exploration — 3 experiments DONE + VERIFIED, awaiting Sevan's review)_

## 2026-07-16 (later) — NEW BRANCH `editability_multi_exploration`: 3 parallel experiments briefed

Sevan opened branch `editability_multi_exploration` to run **three editability lines at once**. Master
notebook `00_master_editability` is OFF-LIMITS (no new edits/results); all work goes in NEW notebooks,
scratch-only, promotion deferred to Sevan's review. Focus: **GRU primary, RSSM where cheap** (RSSM: examine
deterministic `h` — primary world-state carrier — and stochastic `s` separately). NOT the DiT.

**Key feasibility fact:** a GRU trains 400 epochs in **~8.5 min** on this GPU (dataset 4, 256 hidden) →
retraining for Exp 2/3 is cheap; the RSSM is the only expensive leg. Dataset 4 = `4_fixed_refl_inview`
(T=40, R=128, edit_frame=20, 2 obj, 90k train). Master baselines: GRU
`3_dset3_gru_persistentids_inview_400epochs`, RSSM `4_dset4_refined_best`; matched dataset-4 GRU baseline =
`7_dset4_gru_400epochs`.

**Three briefs written (status `proposed`, awaiting Sevan to mark active):**
1. `directions/multistep-steering.md` `[in-frame]` — Exp 1: (1a) interleaved closed-loop latent steering
   (push a little → decode → feed back → push, re-asserting the unedited object's target) vs one-shot;
   (1b) freeze-time teacher forcing (interpolate the edit over N∈1..15 frames, TF, then unfreeze). No
   retraining. Deliverable = editability success/failure only (NOT full master spread).
2. `directions/action-conditioned-structure.md` `[reframe]` — Exp 2. **Reframed by Sevan:** the question
   is whether **training on (random) discrete-token actions** with real causal effect **induces
   causal/editable latent structure**, tested by **discarding the action channel (no-op) and re-running
   the master latent editors** — NOT editing via the action channel (that's a secondary completeness
   check, expected limited since nudges ≪ edit teleports). Discrete tokens {no-op, obj0±x/±y, obj1±x/±y},
   no-op dominant/sparse actions. Requires new sim nudge + action-augmented dataset + action-conditioned
   GRU (must conform to `HiddenStateModel` protocol at no-op so the master suite runs unchanged) + train.
   Proposed optional control (perturbed-passive: same nudged trajectories, token withheld) to separate
   "perturbation diversity" from "action-knowledge" — the enactivist crux. Replicates master §1–§4.
3. `directions/multistep-prediction-objective.md` `[reframe]` — Exp 3: multi-step rollout training
   objective (free-running w-step BPTT), w∈{2,5} vs single-step baseline. Watch blur/mode-collapse.
   GRU primary; RSSM nice-to-have, **≤2–3h cap, cut if slower**. Replicates master §1–§4.

**ALL THREE COMPLETE + VERIFIED (2026-07-16). Nothing promoted — all scratch, awaiting Sevan's review.**
Three workers launched in parallel (Sevan approved incl. the Exp 2 perturbed-passive control). Each verified
on disk (0 error cells, notes, PNGs). Consolidated results:

- **Exp 1 — `notebooks/experiments/editability/multistep_steering.ipynb`** (10 cells, 0 err; note
  `scratch/2026-07-16-multistep-steering.md`; PNGs `/tmp/multistep_steering/`). **Freeze-time TF (1b) is a
  clean WIN on GRU+RSSM** — rendering the edit in over N frames (sweet spot N≈3–8) monotonically lands the
  edit + removes ghost (GRU ghost 0.333→0.123; RSSM 0.485→0.130), deployable (we render the target). **Interleaved
  latent steering (1a) does NOT win** — closed-loop push only eats ghost by dragging BOTH objects (collateral
  explodes); one-shot latent inject is inert → reproduces *readable≠controllable*. Velocity artifact from
  freezing is real (bends GRU RMSE→GT back up past N≈5) but degrades dynamics, not placement. Caveat: N=64.

- **Exp 3 — `notebooks/experiments/multistep/multistep_objective_structure.ipynb`** (15 cells, 0 err; note
  `scratch/2026-07-16-multistep-objective-structure.md`; script `scripts/train_gru_multistep.py`; ckpts
  `runs/gru_multistep/w{2,5}_dset4_gru_400epochs`; 11 PNGs `/tmp/multistep_objective/`). **Clean NEGATIVE:** a
  free-running w-step rollout objective (w∈{2,5}) buys open-loop rollout accuracy (0.208→0.188) and GT-matched
  sharpness (**no blur** — watch-item cleared) but **no editability and no canonicality gain** — §4 pathology
  (decoder-inert probe, belief sluggishness, off-manifold oracle collapse) replicates unchanged across w; if
  anything canonicality mildly *degrades* (fiber resid 0.357→0.457, pos-linear R² 0.84→0.76). RSSM leg CUT
  (per cap). Refutes the brief's "coherence-under-iterated-dynamics ⇒ editable state" intuition.

- **Exp 2 — `notebooks/experiments/actions/action_conditioned_structure.ipynb`** (22 cells, 0 err; note
  `scratch/2026-07-16-action-conditioned-structure.md`; substrate `pim/simulator/actions.py` +
  `pim/world_models/action_gru.py`; dataset `datasets/5_action_augmented`; ckpts `runs/gru/8_action_cond_gru_400ep`
  + `runs/gru/9_perturbed_passive_gru_400ep`; 8 PNGs `/tmp/action_conditioned/`). **NUANCED (partial positive):**
  three GRUs on byte-identical trajectories (baseline / perturbed-passive control / action-cond). **Action-training
  improves the PASSIVE latent's identifiability + canonicality — localized to action-KNOWLEDGE (3→2), not
  perturbation (1→3):** pos-linear R² 0.838→0.890, vel-linear R² 0.582→0.659, MLP fiber resid 0.379→0.324.
  **BUT editability did NOT follow** — §4 editors still fail on all three (readable≠controllable persists); the
  canonicality gain is necessary-direction but not sufficient-magnitude. Side result: *unexplained* perturbations
  (model 3) reduce belief inertia (true-swap obs-change 0.121→0.202, ghost 0.680→0.347) — a coherent-rollout effect.
  **Worker FAILURE (recovered):** the Exp 2 worker built the pipeline + dataset and launched the full nbconvert
  but **backgrounded it and stopped** (the recurring orphan-the-run failure — 3rd occurrence) *and* never wrote
  its scratch note. Orchestrator **adopted the running nbconvert** (rather than kill+restart), watched it to
  completion (0 err), verified all 3 models/figures, and **wrote the scratch note by reconstructing from the
  notebook's printed tables** (per ORCHESTRATION "reconstruct from artifacts"). Harness upgrade candidate: the
  synchronous-execution rule in WORKER.md is being ignored a 3rd time → escalate to enforcement.

**AWAITING SEVAN:** (1) read-through + artifact-or-signal calls on all three (esp. Exp 2's identifiability/
canonicality-yes-but-editability-no nuance, and whether Exp 1's freeze-time win + Exp 3's clean null warrant
`findings/` entries); (2) promotion decisions; (3) **commit** — the branch `editability_multi_exploration` holds
all of it, uncommitted (3 notebooks, 2 pim modules, 1 script, 3 briefs, 3 notes, PROGRESS/README edits). Master
notebook untouched throughout (per Sevan's constraint). One long-lived GPU kernel (PID 946778, ~3.4h) left
alone — predates the session, likely Sevan's VSCode review kernel.



## 2026-07-15 (later) — master-notebook S4/S5 review (Sevan, 31 items)

**Bugs CONFIRMED by code inspection (Sevan caught both):**
- **Fig 5 "GT" column was NOT ground truth** — it plotted the model rollout from the teacher-forced
  post-edit state `h_gt` (hence ghost traces / extra streaks). Fix: GT column = sim `edits.clean_obs`;
  the model rollout from `h_gt` stays as its own labeled "True-state swap (model rollout)" column.
- **"MLP-gradient" was a misnomer** — it is the DECODER/obs-gradient editor (Adam on h vs GT obs). The
  repo's actual MLP-probe steering primitive (`pim.editors.gradient_steer`, from the mlp_steering PR) was
  never in the line-up. Renamed → "Decoder gradient"; "MLP-probe gradient" ADDED as a new editor.
- Also: the per-step `→target` metric compared against the STATIC edit-frame target render (so even the
  true-state swap "drifts" from it) — redefined vs the time-evolving sim clean obs at ef+s.
- Sevan's read of the decoder-gradient failure is right: it **collapses off-distribution**, it does not
  "revert" — language fixed everywhere + a revert/collapse/drift precision rule added to CLAUDE.md.

**HARNESS (durable, CLAUDE.md):** mechanism-based method names (no repo-name collisions); reference scale
+ units for every magnitude; PCA-prefixed estimator names ("honest" banned); revert-vs-collapse-vs-drift
language; calibrated claims (quantities in body, interpretation only in Summary); comparison sets grow
(new editor/model = added column/row, not a redesign).

**v4 (S4 rebuild, both models) — DONE + VERIFIED** (26/26 cells, 0 errors, sync execution, S0–S3
byte-identical; Fig 5a visually verified: GT column = clean sim render, 8 full-size cols, 6 shared
context frames, top legend, decoder-gradient collapse visible). Editor line-up per model: Readout
injection / MLP-probe gradient (new, `gradient_steer`) / Global-PCA projection / PCA geodesic / Decoder
gradient (renamed, oracle) + GT(sim)/Unsteered/True-state-swap refs. Figs: 4 (row/model), 5a/5b, 6a/6b
(step-0 scans), 6c (geodesic budget). **NEW SCIENCE from v4 (held for Sevan; feeds candidates/findings
after his read):**
- **The true-state swap itself is sluggish** — obs-change only 0.129 (GRU) / 0.059 (RSSM) with ghost-ray
  ratio 0.665 / 0.884: a single-frame belief update barely moves the rendered scene, so *every* editor's
  ceiling is low. Reframes "editing fails": even reality's own state, injected, doesn't visually teleport
  the object in one frame.
- **Geodesic K=600: ASYMPTOTES** (GRU 1.75→1.03 plateau by ~iter 135, flat to 600; RSSM no descent).
  Resolves Sevan's "did it just need longer?" — NO. And GRU's plateau readout (1.03) is *better* than the
  true-swap's readout (1.61) while its obs stay ≈unsteered → readout and obs accuracy nearly decoupled.
- **No non-oracle editor beats the true-state swap on GT next-step RMSE, on either model.**
- **Old "reverts by ~step 4" was partly a metric artifact** (static-render target); decoder-gradient on
  GRU **collapses off-distribution** (distance-to-unsteered stays flat ≈0.31 — never returns); RSSM's is
  milder (best next-step 0.131, smears by ~step 12).
- Worker-flagged caveats: ±1-frame decode-convention offset (GRU predicts next / RSSM reconstructs
  current) footnoted; geodesic's tiny leave-out residual partly tautological (last op is a local-PCA
  projection); RSSM geodesic non-descent may be step-size-limited (no sweep run).
**v5 (S5 Summary redesign) — DONE + VERIFIED** (0 errors, sync ~9 min, S0–S4 code unchanged — diff-checked;
only print-string fixes in two §0/§1 cells). §5 = "Summary — what these experiments say about the learned
state" with a clearly-marked "Our reading (interpretation)" block; calibrated phrasing ("≈34% of ‖h‖ not
explained by any g(pos,vel) we fit — largely but not fully a function of the physical state"; "close to
the 8 physical DOF (GRU slightly below, RSSM above)"); collapse-not-revert throughout. **Fig 7 — Summary**
visually verified: (a) capability bars (both models, values on both bars), (b) ONE cross-architecture
scatter (readout RMSE symlog × GT next-step RMSE; color=editor, circle=GRU/square=RSSM, legend outside) —
replaces the old 7b (negative-% bars) and 7c (Fig-3a duplicate). Consolidated summary cell → demarcated
markdown tables under "Current results (updated 2026-07-15)". `fig7_summary.png` (stale fig7_synthesis
removed).

**ENTIRE S4/S5 31-item feedback batch: COMPLETE.** Master notebook now fully review-passed §0–§5.

**2026-07-16 — correction + proposed experiment (from discussion):** Sevan refuted my "true-state swap =
editing ceiling" claim, and he's right: the one-frame-evidence state is the optimum of *observation-
mediated single-frame* belief updating — a LOWER bound for latent editing, not a ceiling (editors have
direct write access to `h`, unconstrained by filter dynamics). **Proposed: counterfactual-history state**
— back-extrapolate the edited object from the target with preserved velocity (other object true history),
render clean obs 0..ef, teacher-force → `h*`; inject at the edit frame. Since rollout is fully determined
by `h`, this should render the teleport cleanly and persist → existence proof that a clean-edit state
exists in h-space; the failure then localizes entirely to the **edit map's reachability**, sharpening the
learn-to-edit negative result. Caveat: back-extrapolation may exit the frustum early (train data was
always-in-frustum) — teacher-force only last ~10 counterfactual frames as mitigation. Also re-files the
sluggish-swap result as a **belief-inertia** measurement (dynamics/coherent-rollout thread; natural
K-frames-of-evidence convergence curve).

**LAUNCHED as a SEPARATE reference notebook** (Sevan: keep it out of the master to avoid bloat; expected
to succeed ~tautologically; promote only if surprising). Worker RUNNING →
`notebooks/experiments/editability/counterfactual_history_state.ipynb`; brief
`directions/counterfactual-history-state.md`; note will be `scratch/2026-07-16-counterfactual-history-state.md`.
The belief-inertia / K-frames-of-evidence convergence curve remains a separate future idea (dynamics thread).
**Awaiting Sevan:** (1) read-through of the rebuilt §4/§5 (esp. the sluggish-swap + geodesic-asymptote
results and the ±1-frame decode-convention caveat); (2) promotion calls — the v4 results likely warrant
updating `candidate-editability` / `candidate-rssm-replication` (sluggish swap reframes the editing story;
RSSM intrinsic-dim 9.6–10 still unpromoted); (3) **commit** — the branch holds all master-notebook
revisions (v2–v5), learn_to_edit v1+v2, harness upgrades, briefs, scratch addenda — none committed yet.

## 2026-07-15 — master-notebook review continues (Sevan; S2/S3)

- **HARNESS (durable):** strengthened `CLAUDE.md` "Notebook legibility" — (i) **clearly-demarcated tables**
  (display'd DataFrame / markdown, NOT aligned-monospace prints); (ii) **plain language, no shorthand**
  (`~=`/`=>`/`<<`/ALL-CAPS jargon banned; titles state what's shown, not the result); (iii) **define every
  implementation detail** (thresholds/subsets like "late-t = t≥15") where used.
- **NOTEBOOK v3 (S2/S3) — worker RUNNING (with the new synchronous-execution rule):** `directions/
  master-editability-notebook.md` "REVISION PASS v3". S2: loose print-tables → demarcated DataFrames; plain
  titles + `Current results` block; **switch all-t → early-t (t<15) vs late-t (t≥15)** with definitions;
  simplify Fig 2 bars (2a single-frame {lin,MLP} × GRU/RSSM early/late; 2b single-frame only; 2c same),
  keep single-vs-2-frame in the table. S3: `Current results` block; demarcated fiber table; Fig 3 plain
  title + value labels on BOTH bars + headroom so legend doesn't overlap. S0/S1/S4/S5 untouched.
- **Answered (chat):** item-4 early-t/late-t definition; flagged the all-t→early-t interpretation for Sevan.
- §4/§5 review still pending Sevan's continued pass.

## 2026-07-09 (eve) — master-notebook section-by-section review (Sevan; intro/S0/S1)

Sevan is reviewing `00_master_editability.ipynb` section by section (today: intro, S0, S1; **S2+ tomorrow**).
Two kinds of action taken:
- **HARNESS (durable):** added a **"Synthesis notebooks (source-of-truth tier)"** standard to `CLAUDE.md`:
  separate the invariant spine from dated **`Current results (updated YYYY-MM-DD)`** blocks; build every
  figure/table to hold **N world models** (color-coded per WM, no results-in-titles, compute don't-"~same");
  keep lightweight. This tier = provisional proposals for `pim`.
- **NOTEBOOK v2 (intro/S0/S1) — DONE + VERIFIED.** `00_master_editability.ipynb` re-ran clean (0 errors, 8
  figs; S2–S5 intact). Verified: Fig 0 redrawn (clean architecture-agnostic pipeline, no colinear arrows);
  Fig 1 rebuilt for N models; S0 belief-state/CM note; dated `Current results` blocks. **NEW RESULT (held):
  RSSM intrinsic dim COMPUTED — TwoNN 9.6 / MLE 10.0, HIGHER than GRU (5.2/6.9) and ABOVE 8** (GRU brackets
  8; RSSM above) — updates the old "geometry ~same" hand-wave; consistent with the belief/stochastic-latent
  view. NOT promoted — `findings/architecture-independence.md` should gain this once Sevan judges it.
- **Recovery:** the v2 worker **orphaned its notebook execution** (`run_in_background` nbconvert) and stopped
  early — orchestrator watched it to completion, fixed unfilled `@TOKEN@` placeholders it left (surgical JSON
  edit, figures preserved), and wrote the scratch addendum it skipped.
- **HARNESS FIX (root cause):** added a hard rule to `WORKER.md` — workers must run notebook execution to
  completion **synchronously in-turn**; NEVER `run_in_background`/`setsid nohup` the execution and stop
  (orphans the run + GPU kernels). This is the 2nd such failure; now explicit.
- §2–§5 review + §4 waterfall items still pending Sevan's continued pass.
- **Answered (in-chat):** the computational-mechanics question (causal/belief state) and the tangent-rotation
  method — see chat; the CM point may later refine RESEARCH.md framing (Sevan's call).

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
