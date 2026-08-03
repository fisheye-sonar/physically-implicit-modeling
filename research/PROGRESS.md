# PROGRESS.md — Session Handoff

> Agent-owned, rewritten freely each session. Answers **"where is the work right
> now?"** — *not* "what's true" (that's `findings/`). Git history is the backstop.

_Last updated: 2026-07-30 (branch `more_trained_editability`: trained-editability thread built + run; §4 metric set redesigned earlier the same day)_

## 2026-07-30 (night) — NEW branch `more_trained_editability`: can editability be INDUCED BY TRAINING?

Sevan asked for a much more extensive test of **trained** editability, including fine-tuning the world model to
induce it. New topical dir `notebooks/experiments/editability/trained_editability/` with `learn_to_edit.ipynb`
**moved into it** (paths repointed) and a new notebook alongside. Registry `TRAINED_EDITABILITY_RUNS.md`; brief
`directions/trained-editability.md`; note `scratch/2026-07-30-trained-editability.md` (**FLAG FOR PROMOTION**).
**This pays the "heavier fine-tune still OWED" debt** that has sat in `METRICS_AND_EDITORS.md` since `learn_to_edit`.

**New: `scripts/train_editable_gru.py`.** Two mechanisms, one evaluation, 5 arms, all from `runs/controls/H256`:
- **fine-tune the world model** so a **fixed, frozen** readout-injection probe works — nothing about the editor is
  learned, so all adaptation is in the model, which must learn to honour writes along `A⁺`. Loss
  `edit + λ·retention`, where retention is ordinary next-step prediction; λ is what separates "became editable"
  from "was destroyed and now renders whatever it is asked for".
- **amortized editor** `E_θ(h,target)→Δh` against a **frozen** world model.
Arms: `FT_light` (300 steps) · `FT_heavy` (3000) · `FT_heavy_noret` (λ=0) · `FT_heavy_obj0` (object-0 edits only,
the content control) · `AMORT`. Trained on `edits[2000:]`; **everything reported on the held-out `edits[:64]`**, the
same samples the `controls/` notebooks use. `eval_controls.py` gained `--root` so the identical §4 suite scores them.

**RESULT — training moves the edit, and wires a BUTTON.** Δ Edit Index of the *trained interface* vs each arm's own
unsteered: base **+0.01**, light **+0.04**, heavy **+0.13**, no-retention +0.10, object-0-only +0.10,
**amortized +0.54**.
- **The light-budget negative was partly a budget artifact** (+0.04 → +0.13 from 300 → 3000 steps) — worth knowing,
  since that negative is currently cited in `findings/editability.md`.
- **But even the best arm only reaches "equidistant".** Amortized absolute Edit Index **−0.14**, against its own
  unsteered −0.68 and the decoder-gradient oracle's +0.94. It never arrives at the edited world.
- **No mechanism generalisation.** The *same* mechanism with a freshly-fit probe moves only **+0.01…+0.04** on every
  arm; the other standard editors are unmoved. The model obeys the interface it was trained for, not the class.
- **No content generalisation.** `FT_heavy_obj0` has an obj1−obj0 gap of **−0.08** vs the both-objects control's
  **+0.09** — withholding an object costs ≈0.17 index points. A per-object button.
- **Cost.** Fine-tuning costs 13% of next-step prediction even with retention (0.1041 → 0.1173). Without retention
  it degrades to **0.1486**, essentially the observation noise floor (0.1539) — the world model is destroyed —
  **and editing gets no better**.
- **The cleanest new fact is an asymmetry:** learning a bespoke editor for a *frozen* latent (+0.54, zero cost to
  the model) works far better than making the latent obey a *fixed* editor (+0.13, 13% prediction cost). Consistent
  with the standing reading that the obstacle is the **reachability of the edit map**, not the representation.

**Also answered (Sevan's question about the Edit Index).** Why is unsteered ≈ −0.7 rather than −1 for `H256`?
Computed: the references are **clean** renders (not noisy), and the index is evaluated **only on the differing
rays**, so shared background is already excluded (`d_edited` = 0.547 there, near full object contrast). The gap is
entirely the model's own blur — `d_unedited` = 0.090, which is its one-step prediction error rather than 0.
Decisive detail: **the split is by observation noise, not model quality.** At obs noise 0.2 the index *saturates at
−0.72* from H=128 on (H128/256/512 all have `d_unedited` ≈ 0.09 — the noise-limited floor), while the noise-free
models reach −0.84. So `H256` is at its best achievable value; more capacity cannot lower it. Matching the *true*
unedited world still gives exactly −1.0 (asserted).

**Owed / next:** train-from-scratch with an edit objective in the loss is the one version of "train for editability"
this does not cover; RSSM untested; one seed per arm.

## 2026-07-30 (latest) — Sevan's review: a real rendering bug, a rename, and the Edit Index over the rollout

**BUG (mine, and it came from a stale spec): every controls waterfall painted a shared teacher-forced `ef` row
across ALL columns.** Only the **Oracle observation** reference actually sees that frame, so every other column
looked teacher-forced when it wasn't — and it **hid the exact frame the §4 scorecard scores** (step 0). It also
displayed the *clean* render while the model that legitimately sees that frame is fed the **noisy** `edits.obs[ef]`.
Root cause: `CLAUDE.md`'s waterfall spec *mandated* the shared row — but that convention had already been caught and
removed in `eval_editability_endogenous.py` v2 (2026-07-28) for exactly this reason. I followed the stale spec.
**Fixed:** every column now shows its own free-run from step 0, GT column = `clean_obs[ef:ef+K]`, and the
`CLAUDE.md` spec now carries an explicit **⛔ never paint a shared teacher-forced row** block explaining why.

**RENAMED `True-state swap` → `Oracle observation`** (Sevan: the old name doesn't match what it does). It is not a
state swap — the model is teacher-forced **one extra frame**, the real **noisy** `edits.obs[ef]`, i.e. it simply
gets to *see* the teleport. Renamed in the registry, master notebook, controls notebooks, eval script and this
thread's notes; historical notes on the retired metric scale were left alone.

**Confirmed the metric is computed correctly (Sevan asked why unsteered isn't −1).** It is correct, and the offset
is interpretable: `d_unedited` is the model's **own one-step prediction error**, not 0, so a perfect predictor
would score exactly −1 and a real one falls short by its blur. Boundary controls asserted: scoring `gt_unedited`
returns exactly **−1.0**, scoring `gt_edited` exactly **+1.0**. And across the 8 controls models the unsteered
index tracks next-step RMSE with **Pearson r = +0.987** (−0.85 for the best predictor, −0.52 for the worst) — so
the unsteered row is effectively a readout of predictive quality and **must appear in every table**. Also verified
the counterfactual render is built on the right frame: velocity is constant in this sim
(`velocities[ef-1] == velocities[ef]`), so there is no off-by-one; the residual is one step of position diffusion.

**NEW — Edit Index over the whole rollout (Sevan's suggestion, and it pays off immediately).** Added
`edit_index_by_step` to `scripts/editability_metrics.py`: the counterfactual world is now rendered **forward**
(edited object continuing along its own velocity, other object on its true trajectory), so the bounded index can be
evaluated at every step. Sevan's prediction was right — **the decoder-gradient oracle's success is a single-frame
success**: on `H=256` it scores **+0.94 at step 0, +0.15 by step 5, −0.12 by step 14**, i.e. it decays past
"neither world". A step-0 scorecard alone would have called that a clean win. New **Fig 3b** in all three controls
notebooks plots it; **GT-traj RMSE** was also added as a panel to Fig 3 as requested.

**Plot fixes:** noise-ablation Fig 3 now uses rotated (tilted) editor labels like the other notebooks, and the
`1.0` reference line was removed from its RMSE panels — 1.0 is not a meaningful level for an RMSE (it was a
leftover intuition from the percentage metrics). It is kept **only** in the hidden-size sweep, where editors
actually cross it and it marks a real threshold (observation intensity is bounded in [0,1], so RMSE > 1 means the
scan was pushed out of range) — flagging that as a judgement call to overrule if you'd rather it go everywhere.

All re-run: `00_master_editability` (0 errors, 11 figs) and the three controls notebooks (0 errors, 6–7 figs each).

## 2026-07-30 (later) — §4 EDITABILITY METRICS REDESIGNED; master + controls re-run on the new set

**Sevan asked to replace `reach %` / `collateral %` with plain RMSE-vs-GT, which surfaced a deeper problem, and we
designed the replacement together before implementing.** The old §4 metrics measured **change away from the
unsteered rollout**, normalised by the oracle observation. Two fatal flaws, both visible in this thread's own data:
(1) they scored *change*, not *correctness* — a scrambling editor posted `reach` of **400–440%** at `H=8`/`H=32` and
the decoder-gradient oracle posted 209–327%, where 100% was supposed to be the ceiling; (2) the denominator was a
**soft, model-dependent** reference whose own strength varied widely (swap ghost ratio 0.315–0.868 across the noise
cells), so the same physical edit scored differently on different models — fatal for cross-model sweeps.
Sevan also noted `selectivity` becomes meaningless once both terms are errors, and that ghost ratio is really just
a zone-restricted RMSE.

**THE NEW CANONICAL SET** — prose in `notebooks/experiments/editability/METRICS_AND_EDITORS.md` §4, implemented
**once** in **`scripts/editability_metrics.py`** (imported everywhere, never re-derived; that drift is what produced
five incompatible versions of "reach"):
- **Layer 1 — absolute error vs ground truth, decomposed by ray zone**, all at rollout step 0, all lower-is-better,
  no normalisation: **Target RMSE** / **Ghost RMSE** / **Collateral RMSE** / **Edit-frame RMSE**, plus **GT-traj
  RMSE** over the rollout and the **fidelity ratio** (`GT-traj RMSE(editor)/GT-traj RMSE(unsteered)`; > 1 = the edit
  left the rollout further from the truth than doing nothing).
- **Layer 2 — the Edit Index ∈ [−1,+1]**, the calibrated headline. Both ground-truth worlds at the edit frame are
  *rendered*: `gt_edited` (the teleport happened) and `gt_unedited` (the counterfactual where it did not — the
  edited object continued along its own velocity). On the rays where they differ,
  `(d_uned − d_edit)/(d_uned + d_edit)`: **+1** = the output *is* the edited world, **−1** = the unedited world,
  **0** = equidistant. **It cannot be gamed by destroying the output** — garbage is far from both worlds and scores
  ≈ 0. "Dim everything toward background" also cancels (the differing rays include target rays as well as ghost
  rays). And the repo's dominant failure — *paint a copy at the target while keeping the ghost* — correctly reads
  ≈ 0 where the old reach reported >100%.

**Everything re-run on the new set: `00_master_editability.ipynb` (0 errors, 11 figs) and all three
`controls/` notebooks (0 errors).** The eval script re-ran across all 8 checkpoints. Retired-metric numbers
anywhere in the repo are flagged as not comparable.

**The redesign sharpened, and in one case corrected, the readings:**
- **Master.** Readout injection now reads in one line: readout RMSE **0.000** (the probe reads the target
  *exactly*) with Edit Index **−0.66** (GRU) / **−0.64** (RSSM) — indistinguishable from doing nothing. Unsteered
  −0.68/−0.64; no probe-directed editor escapes the unedited end (−0.50 to −0.66); oracle **+0.97/+0.87**. The
  "Current results" block was rewritten and re-dated.
- **Hidden-size sweep — the old metric had inverted the low-capacity reading.** At `H=8`/`H=32` the structural
  editors' zone RMSEs exceed **1.0** (intensity is bounded in [0,1]) with fidelity up to 2.2× — they destroy the
  observation. The Edit Index scores that ≈ 0 ("neither world"), not 400%. At `H ≥ 128` structural editors sit
  within 0.08 of unsteered while the oracle reaches +0.87…+0.99. New clean trend: **the oracle's Edit Index rises
  monotonically with capacity (+0.58 → +0.99)** — a bigger latent makes the target state more precisely reachable
  by decoder optimisation, though not by probe-directed writes.
- **Noise ablation — conclusion unchanged, plus a new incidental result.** Structural editors −0.63…−0.67 vs oracle
  +0.91…+0.97 in all four cells. New: **belief inertia is governed by sensing noise** — the oracle observation (no
  editing at all, just one frame of real evidence) reaches Edit Index **+0.54** with clean observations but
  **−0.40** with sensing noise. Suggestively the world-noise-only cell accepts the jump furthest, as if training on
  a jittery world loosens the prior over motion. Flagged as n=1-per-cell, worth a dedicated test.
- **Encoder editing — headline softened and made precise.** Hidden-state injection **−0.67** (1% of the achievable
  span) vs the same pseudoinverse at the encoder port **−0.43** (21%); best probe-directed **−0.08** (50%); render
  oracle **+0.52**. So the interface genuinely matters — but no probe-directed write crosses to the edited side,
  and the best one **triples the collateral error** (0.127 → 0.335) with fidelity 1.15: it repaints rather than
  relocates, exactly as Fig 6's intermediates show.

**Also fixed: `notebooks/experiments/controls/` was created as a SIBLING of `editability/` — wrong.** Migrated to
`notebooks/experiments/editability/controls/`. `CLAUDE.md` now carries the protocol: every experiment lives inside
the research thread it serves (controls/ablations/side-quests as subdirectories), never beside it.

> **Note on mechanics:** the master notebook could not be opened with the `Read` tool (54k tokens, over the cap), so
> its cells were patched via a JSON round-trip with exact-match assertions rather than `NotebookEdit`, then verified
> by AST-parsing every cell and executing the notebook end to end. Flagging because it deviates from the standing
> "never touch .ipynb outside NotebookEdit" rule.

## 2026-07-30 — branch `michael_controls`: Michael's three controls, all COMPLETE

Sevan relayed three control/side experiments from a conversation with his postdoc Michael. All three are **built,
trained, evaluated and written up**; all three scratch notes are **FLAGGED FOR PROMOTION** and awaiting Sevan's
artifact-or-signal call. Notebooks + registry live in `notebooks/experiments/editability/controls/`
(`CONTROL_RUNS.md`). **Uncommitted** (held per commit-only-when-asked).

**ENABLING INFRASTRUCTURE (this is what made the thread runnable).** GRU training was **CPU-bound on gzip HDF5**:
68 s/epoch at H=256 with the GPU idle → 7.5 h per 400-epoch run, and this thread needs 8 runs. Added
`build_inmemory_dataloaders` + `InMemoryLoader` (`pim/world_models/dataloader.py`) and a `--in-memory` flag to
`scripts/train_gru.py`: the observation tensor (1.8 GB) lives on the GPU, identical split/batching/optimizer.
**0.50 s/epoch — 136× faster**, loss curves matching the lazy path (epoch-2 train loss 0.0267 both). 400 epochs is now
~3.5 min. Also new: `scripts/eval_controls.py`, one pass per checkpoint computing all four affordance families
(predictive / recoverability / canonicality / editability) into `runs/controls/eval/<code>.json` + `_rollouts.npz`, so
the notebooks only load, plot and tabulate — which is how they stayed short.

**3 new datasets** (matched to `4_fixed_refl_inview` except the noise flags): `9_obsnoise0_posnoise0`,
`10_obsnoise0_posnoise004`, `11_obsnoise02_posnoise0`. **8 new runs** in `runs/controls/`.

### D1 — encoder-space editing (`directions/encoder-space-editing.md`, `encoder_editing.ipynb`)
Michael's premise: all world information enters the latent through one channel, `x_t = relu(W_enc·obs_t + b_enc)`; every
editor so far writes to `h` instead. So probe `x`, edit `x`, and spread the write over `N` frozen steps the way
freeze-time teacher forcing does — but **without the renderer**.
**RESULT — the interface is a real variable, but the write still repaints rather than relocates.** The *same* linear
pseudoinverse edit is inert on `h` (ghost **0.996**) and moves the needle at `x` (ghost **0.803**); the best
probe-directed encoder write reaches **0.670**, i.e. 27–45% of the way from unsteered to the render oracle. Spreading
helps (0.838 at N=1 → 0.650 at N=12). **But every probe-directed encoder write fails the fidelity guard** (GT-traj RMSE
1.15× unsteered) while the **freeze-time render oracle through the identical port passes** (ghost 0.266, fidelity 0.72).
**Fig 6 (the intermediate decoded observations, Sevan's explicit ask) is the money panel:** the oracle shows one
coherent object *translating*; the probe-directed write shows a *cross-fade* — a new blob brightening while the old one
stays. Also confirmed exactly as predicted: **velocity R² at the port is 0.005 vs 0.474 at `h`** — the encoder output
has no memory, so there is no velocity there to write.

### D2 — hidden-size sweep (`directions/hidden-size-sweep.md`, `hidden_size_sweep.ipynb`)
`H ∈ {8, 32, 128, 256, 512}`, one variable, dataset 4. (`H=8` = the world's true state dimensionality; `H=128` = the
observation resolution.)
**RESULT — capacity moves prediction and readability a lot, grabbability not at all.** Prediction saturates by `H=128`
(next-step RMSE 0.1495 → 0.1167 → 0.1054 → 0.1041 → 0.1042). Linear readability rises **monotonically** — position R²
0.175 → 0.855, velocity 0.002 → 0.531 — **refuting my pre-registered guess** that a squeezed latent would be more
linearly readable; it simply fails to represent the state. Canonicality moves the *opposite* way (MLP fiber residual
0.215 → 0.601), so capacity trades canonicality for readability. §4 numbers restated on the canonical metric set —
see the section above.

### D3 — noise ablation (`directions/noise-ablation.md`, `noise_ablation.ipynb`)
The 2×2 of observation noise (sensing) × position noise (the world itself), at `H=256`.
**RESULT — neither noise source is what blocks editing**; the negative holds in the fully deterministic,
perfectly-sensed world (§4 numbers restated on the canonical set above). **Both pre-registered
recoverability predictions refuted**, and one clean positive: **observation noise is a linearising regulariser** —
position R² (linear) 0.596 → 0.819 when sensing noise is turned on. Velocity readability is invariant to both sources
(0.451–0.471 across the whole 2×2). Canonicality: the linear and MLP fiber estimators **disagree in sign** (sensing
noise off moves linear −0.026 but MLP **+0.193**), so both are reported — reporting only the linear one would have
produced the opposite headline.

### METHODOLOGICAL ADDITION — the fidelity guard (now part of the canonical set)
Ghost ratio alone is **not sufficient** and can invert a conclusion: at `H=8`/`H=32` structural editors reported
"good" ghost values while their GT-traj RMSE was up to **2.2×** unsteered — the edit destroys the observation and the
vacated rays dim as a side effect. The **fidelity ratio** was introduced here and is now part of the canonical §4 set;
finding it is what led to the full metric redesign recorded in the section above.

**Awaiting Sevan:** artifact-or-signal + promotion call on all three notes; whether D2+D3 merge into one "the §4
negative is robust to capacity and to stochasticity" findings entry; whether to re-run the full §4 editor line-up at the
encoder port on the `object-individuation` models; whether to fold the fidelity guard into the metrics registry.

## 2026-07-28 — Waterfall cleanup tied off; NEW experiment thread opened (endogenous actions)
**Waterfall honesty pass DONE (source edits only, no full re-runs per Sevan).** Added the **teacher-forced edit-frame
row** to every relevant waterfall — one shared row = the TRUE post-edit obs / edit target (`clean_obs[ef]`, identical
across columns), each column's model rollout below it = its **free-run from `ef+1`**. Applied to
`actions/action_space_object_individuation.ipynb` (Fig 5), `actions/action_conditioned_structure.ipynb` (Fig 5), and
converted `learn_to_edit.ipynb` (Fig 3b/5d) from `magma`→gray + added context frames + figure-top legend (kept GT
column). **Caught + fixed a real off-by-one** I'd introduced pre-compaction: `warm_up_to_edit` teacher-forces
`obs[0..ef-1]` so the predict-next GRU rollout **step-0 ↔ `clean_obs[ef]`** (confirmed by the §4 scorecard's
`gt_traj_obs=clean_obs[ef:]`); the shared true-`ef` row therefore requires **dropping each model column's step-0**
(`ROLL[...][1:]`) and slicing GT to `clean_obs[ef+1:ef+K]` (both length `K-1`) so columns align — earlier version left
GT `ef+1`-aligned while model was `ef`-aligned. All 4 cells AST-parse + shape-check clean; the 4 edited waterfall cells
have outputs cleared (need a re-run to regenerate — the stale images contradicted the new code). **Harness locked:**
CLAUDE.md waterfall spec + `editability/METRICS_AND_EDITORS.md` now make the shared true-`ef` row mandatory with the
exact alignment rule. Also fixed the E2 `render_scene` NameError.

**NEW THREAD (proposed, design-only so far): endogenous-action interactive world** →
`directions/endogenous-action-interactive-world.md`. The promised follow-up to object-individuation (which was scoped
to EXOGENOUS actions and left endogenous open). Hypothesis (strong enactivist): actions must be **generated by the
latent world and self-predicted** (closed sensorimotor loop / efference copy), not merely observed, to induce a
factored/editable latent. Central control = **actor** (action-OUTPUT head, acts on the world) vs **observer** (identical
arch, action-INPUT, same obs+actions, never acts) — actor-yes/observer-no would isolate the effect to agency. Three
levels: L1 random position-shift, L2 force/momentum, **L3 goals** (collision/wall avoidance — the anticipated payoff).
Key design reality surfaced: the sim is non-differentiable, so **L1/L2 give no learning signal to the action head
(efference-copy ablation only); L3 needs a policy-learning loop (REINFORCE / Dreamer-in-imagination)** — this is why L3
is where agency actually enters. First concrete build (after discussion): a **stateful `InteractiveWorld.step()`** (the
one genuinely new primitive — everything today is offline `simulate()`) + a **keyboard `play.py` emulator** (2D ∥
waterfall, key overlay) that a human plays now and the trained model drives later through the SAME view. Design doc
addresses Sevan's A–D + additions E–I (embodied-vs-god-hand agent, degeneracy/anti-freeze, discrete-vs-continuous,
on-policy nonstationarity, efference-copy sanity probe).

**Discussion round 2 done + decisions locked** (in the direction doc's "Decisions locked" block): god's-hand first
(embodied later), GRU-only first, discrete keys, action head decodes from `h_t`, no action-input on the actor except
the efference copy of its own *sampled* action (needed once RL makes the policy stochastic), start REINFORCE then the
SMiRL-style survival-from-unpredictable-death variant, empowerment steered away from (circular / teaching-to-the-test).
Clarified the "dark room problem" (prediction alone rewards boredom → freeze/no-op, so L1/L2 are efference-copy
ablations and L3 needs a policy objective).

**BUILD 1 DONE + VALIDATED (2026-07-28, branch `endogenous_actions`): the interactive sim + keyboard emulator
(human-playable; no models yet).** New files, all reuse `pim/simulator` and touch no existing path:
- `pim/simulator/interactive.py` — `InteractiveWorld` (the one genuinely new primitive: stateful `reset`/`step`; there
  was no online world before) + `InteractiveConfig`. Two dynamics modes: **`shift`** (L1 position-delta, drift base,
  frustum/collision-guarded) and **`force`** (L2 F/m momentum, intrinsic anti-freeze drift, friction, speed clamp,
  bounce/clamp/death walls). God's-hand per-object actions `(n,2)` (also accepts the `(n,3)` `[active,a1,a2]` model
  schema). **Death → rebirth** built in (reset to fresh IC, optional pure-noise frames = the SMiRL substrate). Key
  design property (tested): deaths are a **force-mode** phenomenon — the `shift` guard makes collisions/frustum-exit
  impossible by action (matches the prior collision-free datasets), so the L3 avoidance game lives in force mode.
- `scripts/play.py` — the emulator: `Driver` protocol (`HumanKeyboardDriver` via WASD=obj0 / arrows(or IJKL)=obj1;
  `RandomDriver`; `HeuristicAvoidDriver`), a 2D-world ∥ grayscale-waterfall dual panel + **keyboard overlay** (pressed
  keys highlighted; a model/driver's continuous action is discretised back onto the same keys — the "see what it's
  doing as key-presses" feature) + status (frame/deaths/survived). Live `plt.show()` loop; headless `--save` GIF path.
  Reuses viz.py's frustum/waterfall style. Toggle dynamics live with `M`, reset `R`, pause `SPACE`.
- `tests/test_interactive.py` — 12 tests (both modes, determinism, bounce-containment, shift-moves/force-accelerates,
  `(n,3)` coercion, shift-guard-prevents-collision, force death+rebirth). **Full suite 43 passed; ruff clean; black
  formatted.** Validated the render pipeline headlessly (Agg) → demo GIFs for force+avoid and shift+random look correct
  (frustum, discs+reflectivity labels, action arrows, gray waterfall bands, key overlay). Live keyboard path is
  untested here (no display) — Sevan runs `python scripts/play.py` locally to play.
**BUILD 1 fixes (2026-07-28, after Sevan playtested):** (1) matplotlib keymaps cleared so `s`/etc. no longer trigger
toolbar actions (the save dialog was eating key-releases → stuck keys); (2) `--death-on-collision` now defaults ON so
deaths register; (3) collision threshold fixed from the offline generator's `collision_margin·2r=1.6` to true disc
contact `2·radius=1.0` (added `collision_slack`/`spawn_clearance`, split `_contact` vs `_spawn_sep`); (4) `M` now
toggles dynamics IN PLACE (keeps positions) instead of calling reset; (5) wall-death decoupled from bounce — walls
always bounce, `death_on_wall` is an independent toggle (`--death-on-wall` / live `B`; `C` toggles collision-death).
2 new tests (contact-distance, wall-death); **45 passed, ruff/black clean;** re-validated headlessly (deaths increment,
M in-place, `s` freed, C/B toggle). Sevan's verdict: "simulator looks good and ready to go."

**BUILD 2 DONE + RUN COMPLETE (2026-07-28 overnight, branch `endogenous_actions`): actor-vs-observer L1→L3 trained +
evaluated.** Sevan dispatched with two difficulty tweaks (wall-death on, `init_speed=0.28` momentum). New:
`pim/world_models/actor_gru.py` (`EndogenousActorGRU` = obs-only-encoder GRU + categorical policy head {−1,0,+1}/axis +
value head + action-conditioned decoder; HiddenStateModel-conformant passive no-op decode — the OBSERVER is the same
class fed the actor's actions); `scripts/train_endogenous.py` (batched on-policy rollout in `InteractiveWorld`;
predictor loss for actor+observer; **REINFORCE+value baseline into the actor's SHARED trunk** at L3 — the mechanism
under test); `scripts/eval_endogenous.py`. Runs `runs/endogenous/{L1,L2,L3,L3b}` (L3b=seed 1), ~55 min GPU, launched
detached + watcher (fired clean). Notebook `notebooks/experiments/editability/actions/endogenous_actor_observer.ipynb`
(0 err, 4 figs); scratch `2026-07-28-endogenous-action-actor-observer.md` (**FLAG FOR PROMOTION**).

**RESULT — clean positive on identifiability, localized to GOAL-DIRECTED agency:** L3 actor learned the survival goal
(survival 12→~1536–3072, deaths ~250→0–2, reward −0.03→+0.10; both seeds). **Passive-latent recoverability: L3 actor ≫
observer — pos R² lin 0.76 vs 0.59 (Δ+0.17), vel R² lin 0.56 vs 0.39 (Δ+0.17), replicated (L3b Δ+0.14/+0.17).** L1
(shift, no goal) actor≡observer (Δ≈0.00); L2 (force, no goal) actor marginally worse (Δ≈−0.01) → it is **not**
self-generating actions nor momentum, but **acting toward a goal** (policy grad into the shared trunk) that reshapes the
latent. Velocity (historically hard-to-read) gains most — collision-avoidance forces motion-tracking. Gain is
**legibility, not prediction/canonicality**: L3 actor is a slightly worse predictor (next-step RMSE 0.131 vs 0.109) and
LESS canonical (fiber MLP 0.40 vs 0.34 — carries extra control state). The **observer is a strong control** (same
obs+actions, no agency → no gain) = the enactivist prediction; extends object-individuation's "readable ≠ grabbable"
with a big *readability* gain from agency.

**§4 GRABBABILITY FOLLOW-UP (same night, Sevan asked): is the more-readable L3 latent an editable object HANDLE? Mostly
NO** (`scripts/eval_editability_endogenous.py`; passive latent, foreign latent-surgery editors → object-0 teleport
target; N=64; `editability_metrics.json` + waterfalls `runs/endogenous/edit_figs/`). The genuine structural editor
(MLP-probe gradient) reaches **2× further on the actor than the observer (75–83% vs 35–45%)** — readability *does* buy
obs-space reach — **BUT the object-handle hallmarks fail for both:** ghost **0.91–1.16** (vs oracle 0.01, true-swap ≈0 —
the object never *leaves* its old spot) and non-selective (collateral ~100%, selectivity ~0.45 — drags the other
object). Waterfalls confirm the reach is *painting a copy at the target while keeping the ghost*, not moving the object;
readout injection is inert; only true-swap + the off-manifold decoder-gradient oracle move it cleanly. **Verdict:
agency buys legibility + steerability, NOT a clean grabbable handle — "readable ≠ grabbable" holds under endogenous
goal-directed action, sharpened.** Keeps pointing at explicit object scaffolding (RESEARCH.md endgame). Note updated
(nuanced FLAG FOR PROMOTION). **Still owed:** action-interface controllability (edit *through* the trained action
channel); non-action auxiliary-task control; embodied; RSSM. **Uncommitted** (branch `endogenous_actions`, held per
commit-only-when-asked). **Awaiting Sevan:** artifact-or-signal + promotion call; next-move pick.

## 2026-07-28 (late) — IN FLIGHT: stronger-predictor rerun of the §4 grabbability test
Sevan reviewed the §4 negative and pushed back on two things — **(i) a waterfall bug** and **(ii) "the predictions are
so messy it's hard to tell whether it's really failing or just a bad predictor."** Both were legitimate:
- **Waterfall bug (FIXED).** v1 injected the TRUE target-obs row into *every* column and dropped each editor's own
  step-0 decode (`ROLL[...][1:]`), so every column looked teacher-forced on the edit frame **and the exact frame the
  scorecard scores was hidden**. Only True-swap legitimately sees that frame. v2: each column shows **its own free-run
  from step 0**, GT is its own column.
- **Predictor quality was genuinely poor (CONFIRMED).** Measured: weak-model free-run RMSE **0.24**, sharpness
  **TV ratio 0.59** (only ~60% of GT sharpness). A new **quality gate** (free-run RMSE + TV ratio + next-step) now runs
  for every model, so editability is only interpreted for models that pass.
- **Off-distribution rollout (checked, NOT the driver).** v1 rolled out with no-op actions though the actor always
  acts; v2 adds a `self` mode (model's own policy acts on its imagined world). Results are near-identical to `noop`.
- **Editor line-up widened:** + Global-PCA projection, + PCA geodesic (reusing `pim/editors/manifold_steering`).
- **NEW action-channel control (informative already):** a PD controller in the REAL sim closes **94%** of the distance
  to the target (the channel genuinely has authority), but the weak model's *imagination* of those same actions barely
  moves the object (imagined reach **2.1%**, ghost **0.987**; model-vs-real RMSE 0.29) → the weak model cannot even
  simulate its own action channel off-policy, i.e. its editability failure IS partly a predictor failure. This is the
  cleanest evidence that Sevan's objection was right and the v1 verdict must be re-tested at higher model quality.
- **Stronger models TRAINING (detached + watcher):** `runs/endogenous/{L3s0,L3s1}` (L3, seeds 0/1) and `L2s0` —
  hidden **512**, **2-layer MLP encoder + residual MLP decoder** (added to `EndogenousActorConfig` as `enc_layers` /
  `dec_layers`, defaults preserve the old architecture so **old checkpoints still load strictly**), a **5-step free-run
  (multistep) objective** to fight rollout blur, 25k iters. Early signal: at it 1000 the strong model already matches
  the weak model's *final* prediction RMSE and reaches survival 768.
- Built + validated: `scripts/eval_editability_endogenous.py` (v2, all of the above) and the comparison notebook
  `notebooks/experiments/editability/actions/endogenous_grabbability.ipynb` (9 cells, valid, 0 syntax errors).
**COMPLETE (2026-07-29 00:05).** All 3 strong runs trained (`L3s0`,`L3s1` 25k it; `L2s0` 12k it), both evals re-run
across all 7 checkpoints, notebook `endogenous_grabbability.ipynb` executed (0 err, 7 figs), scratch note revised.

**RESULT 1 — §4 grabbability CONFIRMED, and now NOT a predictor artifact (the control the first pass lacked).**
Structural editors are inert on the strong models: ghost **0.998–1.010** (1.0 = the object never leaves), reach 0.3–6%.
But on the **same model / decoder / rollout**, the **decoder-gradient oracle** (ghost 0.004–0.012, reach 89–93%) and
the **oracle observation** (ghost ≈ 0, reach 100%) succeed completely. **If blur caused the failure the oracle would fail
too** → a state rendering the target exists and rolls out fine; probe-directed writes cannot reach it. Failure = the
**edit map's reachability**, not the predictor. Replicated 2 seeds × 2 rollout modes. Counter-intuitively the editors
got *more* inert as the predictor improved (PCA geodesic reach 28% → 4%).

**RESULT 2 — the 2026-07-28 identifiability headline is DOWNGRADED (do not cite the old magnitudes).** Δ(actor−observer)
position R² **+0.155 → +0.017** at strength — **identical to the no-goal control (+0.018)**, so the position advantage is
no longer goal-specific at all (the observer catches up, 0.589 → 0.863). Velocity survives but ~3× smaller (**+0.052**
vs control −0.015); canonicality **flips sign** to a cleaner positive (fiber MLP **−0.074**, actor now *more* canonical;
control −0.026). Revised reading: goal-directed agency mainly **accelerates** the emergence of readable structure; what
durably survives is a modest velocity-readability + canonicality gain.

**Honest limitations found:** (a) the stronger models did **not** fix the blur (sharpness 0.607 → 0.633; free-run RMSE
slightly worse) — capacity + multistep were insufficient; (b) the **action-channel control is not a clean "button"
result** — the real sim closes 93–95% of the distance but the model's imagination of those (OFF-POLICY) actions barely
moves the object (reach 2–6%), conflating "doesn't transfer to the state" with "poor off-policy generalization". The
earlier "button, not a handle" phrasing **overclaimed and is retracted** pending an on-policy action-intervention test.
Corrected framing: the model is an **on-policy predictor, not an intervention-supporting simulator** — no tested
intervention route works in imagination except decoder optimization and fresh observational evidence.

**Awaiting Sevan:** artifact-or-signal + promotion call on the revised note; next-move pick (on-policy action-
intervention test / non-action auxiliary-task control / embodied / RSSM / go constructive with explicit scaffolding).

## 2026-07-29 — Sevan's notebook review (12 items): legibility fixes, metric corrections, animations
Sevan reviewed both endogenous notebooks. Two **methodological corrections**, one **harness fix for a recurring
failure**, and a new qualitative notebook.

**HARNESS (recurring failure — Sevan: "you are still reintroducing terms and inconsistent idiosyncratic naming
conventions that I can't follow").** Added a hard `CLAUDE.md` rule: every experiment thread keeps a **canonical run
registry**; every notebook copies the rows it uses into its own definitions table; **figures use descriptive labels,
never bare codes** (`L3 force+goal · strong · seed 0`, not `L3s0`); a suffix encoding a variable must state what it
encodes; adding a run means adding its registry row in the same commit. Created the first registry:
`notebooks/experiments/editability/actions/ENDOGENOUS_RUNS.md` (every run + role + level + architecture + seed +
purpose, plus the metric caveats below). Both notebooks now carry full definitions tables.

**METRIC CORRECTIONS (both were real):**
- **`survival` is capped + quantized at 3072** = `batch·rollout / max(deaths,1)`, i.e. bounded by the **measurement
  window** (64×48 frames/iteration), NOT by the world (episodes are unbounded; only death ends one). 0 or 1 deaths both
  read 3072. This is why the curve looked spiky/saturated. Fig 1 now leads with **deaths per 1000 frames** (unbounded,
  linear) and marks the 3072 cap explicitly on the survival panel.
- **`mean reward` is per STEP, not per episode** (+0.1 survive / −1.0 death) so **+0.1 is the ceiling**, not "survives a
  few frames". Documented, with the return-scale note (γ=0.99 ⇒ survival stream ≈ 10, so death ≈ −11 in return terms).
- **Dropped the sharpness/TV metric** from the grabbability notebook per Sevan's preference; Fig 1a is now next-step
  RMSE with the repo's standard **dashed baselines** (`pim/eval/baselines.py`): copy-previous-frame 0.160 and
  observation noise floor 0.066 — models sit at 0.10–0.13, so below the trivial baseline but well above the floor.
- **Added the per-step "does the edit land and hold" curve** (RMSE vs the post-edit target render vs rollout step),
  replacing the old panel; **observer waterfalls** now rendered alongside the actor's for every run.

**Answered in-notebook (Sevan's Q5/Q7/Q9):** (a) the actor's loss is a fixed weighted sum
`pred + 1.0·policy + 0.5·value + 0.01·entropy` and **those weights were NEVER swept** — the prediction-vs-control
balance is an arbitrary unvalidated hyperparameter and the contrast is by construction sensitive to it (flagged as the
most obvious missing control); prediction is not strictly needed for survival — it is there because it is the research
subject and to keep actor/observer objectives comparable (the Dreamer/RSSM pattern). (b) **The "death = unpredictable"
idea does not remove the need for RL**, but the clean version is to keep REINFORCE and make **reward = −(prediction
error)** (the SMiRL / free-energy formulation) — one self-consistent objective instead of an arbitrary λ; recommended
as the next experiment. (c) The **static GT column** is deliberate: each editor changes the latent → changes the policy
→ would induce a *different* true future, so there is no single common reference; the frozen target is editor-
independent but only correct at step 0 (step-0 metrics unaffected; the per-step curve reads as "how long does the edit
keep resembling the intended scene", not prediction error).

**NEW notebook `endogenous_agent_animations.ipynb`** (Sevan's item 4): play.py-style animations of every trained agent —
2D world + **keyboard overlay showing the model's actions as key presses** + white force vectors + real observation
waterfall + **the model's predicted-observation waterfall**. Built by **importing the same `Emulator` class
`scripts/play.py` uses** (extended with a `ModelDriver` and support for N predictor panels), with world settings read
from each checkpoint so the visualisation matches training (death-on-collision/wall, death noise frames, momentum).
Covers L1/L2/L3 weak + both strong seeds + the strong no-goal control, an **actor-vs-observer** 4-panel comparison, and
**three training stages** (barely trained → partway → trained) from a checkpointed rerun (`L3s0_ckpt`, `--ckpt-every
2500`, running). GIFs → `runs/endogenous/animations/`.

## 2026-07-29 (later) — Sevan's second review: two real bugs + a hygiene failure
- **BUG (mine): the "deaths per 1000 frames" panel was INVERTED.** I plotted `1000/deaths` instead of
  `1000·deaths/frames`, so 0–1 deaths rendered as **1000** and 252 deaths as **4** — the curve rose as the agent
  *improved*. Sevan caught it ("all of the plots are going UP over time, even reaching 1000"). Fixed to use the raw
  per-iteration death counts (`deaths_curve`, now exported by `eval_endogenous.py`). Corrected values: L3 strong ends at
  **0.33 deaths/1000 frames**, the strong no-goal control at **79.1**.
- **Why 3072 is a cap, explained properly:** training is on-policy with a **fixed budget of 64 worlds × 48 steps = 3072
  frames per iteration**; `survival` is estimated inside that budget as frames ÷ deaths, so zero deaths is
  indistinguishable from immortality and reads 3072 — **right-censoring**, plus quantization (3072/1536/1024…). The world
  has no frame limit. The rate statistic is unbiased; the notebook now leads with it and marks the censoring limit.
- **Plot bloat (my regression):** adding all 7 runs to every figure made them unreadable. Reverted to a **3-run main
  comparison** (L3 goal weak · L3 goal strong · L2 no-goal strong control) with short two-line labels; L1/L2-weak and the
  second seeds are footnotes appearing only in the full table. Fig 1 now shows **one seed each** (weak vs strong).
- **HYGIENE FAILURE (Sevan was right):** every endogenous run used **`obs_noise_std=0.05`** while the repo standard —
  every dataset 0–8, including dataset 4 behind the exogenous-action work — is **0.2**. It leaked from a `play.py`
  *display* default into the science. Internal comparisons remain valid (all runs share it) but **absolute RMSE / noise
  floor / probe R² are not cross-citable with earlier notebooks**. `train_endogenous.py` now exposes `--obs-noise` and
  **defaults to 0.2**; the deviation is documented at the top of `ENDOGENOUS_RUNS.md` and both notebooks. **A matched
  re-run at 0.2 is OWED** before any cross-thread numeric comparison.
- **Q6 answered with evidence:** the action-channel test finds a PD-controller action sequence in the **real** sim
  (closes 93–95%), then replays those exact actions in the model's imagination with the policy head **bypassed** (the
  action enters via the decoder conditioning, the same pathway used in training). Its poor showing is now explained by a
  new per-step panel: model-vs-real RMSE is **0.12 at step 1** (matching the teacher-forced animations) rising to
  **0.35 by step 15** — i.e. the animations show *one-step* prediction, the test is a *15-step closed-loop* rollout, and
  the controller's actions are additionally off-policy (the two are confounded — why "button, not a handle" stays retracted).
- **New + a genuinely informative RESULT:** autoregressive ("dreaming") animations (`AutoregressiveModelDriver`) for L3
  weak + strong — after a 15-frame warm-up the model consumes only its own predictions while still acting on the real
  world. **Quantified: the goal-trained actor dies 2.8 times per 1000 frames teacher-forced, but 87.8 (weak) / 85.0
  (strong) closed-loop — ≈31× worse, and essentially the same as the NO-GOAL control's 79.1.** So *acting inside its own
  imagination is about as bad as having no policy at all*, and the strong configuration does not help. This is the
  cleanest statement yet of the thread's through-line: **the model is an on-policy predictor, not a simulator you can
  act inside** — and it is the same regime the editability + action-channel tests operate in, which is why their numbers
  look so much worse than the one-step-ahead panels suggest. Animation notebook's training-stage cell also made robust
  to a missing final checkpoint.
- **Animation notebook size:** embedding 12 GIFs made it 284 MB; regenerated at dpi 55 / 100 frames → **70 MB** with
  legibility preserved (GIFs also on disk in `runs/endogenous/animations/`, ~4 MB each, for easy saving). Note
  `nbstripout` strips outputs on commit, so the on-disk GIFs are the durable artifact.

## 2026-07-29 (evening) — action-in-transition ablation: a real bug, but NOT the dominant cause
**What was wrong.** The endogenous actor fed its action **only to the decoder**, never to the recurrence:
`h_t = GRU(enc(o_t), h_{t-1})`, `ô_{t+1} = dec([h_t, proj(a_t)])`. Measured consequence: feeding *opposite* actions
produced a **bit-identical** next state (‖Δh‖ = 0.0000) — the action could not influence the imagined state at all,
only the decoded observation, so its effect had to re-enter via decoder→predicted-obs→encoder (a lossy bottleneck).
Every standard action-conditioned world model (including this repo's own `action_gru_continuous`) puts the action in
the transition. This was my design error, not a property of endogenous action.

**Fix (`action_in_transition`, default False so old checkpoints load strictly).** The **previous** action is
concatenated to the GRU input — `h_t = GRU([enc(o_t), proj_trans(a_{t-1})], h_{t-1})` — using a *separate* projection
from the decoder's, so decoder behaviour is untouched. Previous (not current) because `a_t = π(h_t)` is produced *from*
`h_t`; `a_{t-1}` is what caused the transition into `t`. Threaded through `collect()` (tracks `prev_a`),
`predict_sequence` (right-shifts the action sequence), and the multistep free-run loss. Verified: ‖Δh‖ = 1.008 with the
flag on, 0.0000 off.

**A second bug caught while writing this up (would have invalidated the whole comparison).** Every *eval* path
(`ModelDriver`, `AutoregressiveModelDriver`, `AutoregressivePredictor`, the Emulator's predictors, `collect_eval`,
`warm`/`rollout`/`quality_gate`/`action_interface_test`) called `gru_step` **without** `prev_action`, so the new model
would be evaluated with a **no-op in its transition** (‖Δh‖ = 0.345 vs correct). The completion watcher ran the
comparison 28 s **before** the fix landed and reported a spurious teacher-forced rate of 23.3 for `L3s0_ait`. All paths
are now patched (harmless for flag-off models — verified `L3s0` numbers unchanged) and the comparison was re-run.

**RESULT (`L3s0_ait` = `L3s0` + action-in-transition, single variable, 25000 it):**
| | teacher-forced | closed-loop | imagined-vs-real RMSE @ step 1 / 10 / 20 |
|---|---|---|---|
| `L3s0` (action NOT in transition) | 2.8 | **85.0** | 0.159 / 0.397 / 0.457 |
| `L3s0_ait` (action IN transition) | 2.8 | **72.2** | 0.186 / 0.319 / 0.391 |
*(deaths per 1000 frames; no-goal control = 79.1; copy-previous-frame baseline RMSE 0.160, random-frame 0.393)*

**Verdict: the missing action pathway was a genuine bug but is NOT the dominant cause of the closed-loop collapse.**
Fixing it buys ~15% fewer closed-loop deaths (85.0 → 72.2) and slightly slower drift, but 72.2 is still barely better
than having **no policy at all** (79.1), and the imagination still reaches **random-frame-level error (≈0.39) by ~10–20
steps** — i.e. the dream decouples from reality rather than merely degrading. Teacher-forced metrics are identical
(2.8 both), confirming teacher forcing is blind to this change. Remaining suspects, in order: **no latent-space
consistency objective** (nothing ties the imagined latent to observation-informed latents — this is exactly RSSM's
KL(posterior‖prior)), the **hidden-state reset every 48 frames** (still present here, so this null is partially
confounded — flagged before the run), and a **5-step imagination horizon trained vs 100+ evaluated**.
→ Strengthens the case that the fix needed is a *training signal*, not more plumbing. **Next: RSSM**, aligned with
standard practice (free bits / KL balancing, actor trained in imagination, state carried across boundaries), keeping
the actor/observer contrast *inside* RSSM so agency and architecture stay separable (Sevan's constraint).

**Throughput profile (measured, batch 64):** simulator stepping ~39% (over half of it rendering), model forward during
collection ~45%, gradient update only ~16%. So my earlier "the Python loop is the bottleneck" claim was **wrong** —
collection is 84% of wall-clock but the *model's* 48 sequential latency-bound GPU calls are the largest slice. Batch
scaling: model forward is ~flat (16× batch for 1.6× time) while the per-world Python simulator is strictly linear —
so **vectorizing the sim is what would unlock large batches** (~10× env-frames/s), not the ~1.6× direct saving.
Recommended *after* the RSSM build (Dreamer-style imagination training reduces real-env demand).

## 2026-07-29 (evening 2) — VECTORISED (GPU) SIMULATOR + parity suite
Sevan asked to make training faster before the RSSM build, and to validate the change by
re-running an existing training run and checking the results are unchanged.

**Why the simulator was the right target (measured, not assumed).** Per iteration at batch 64:
simulator ~39 %, model forward during collection ~45 %, gradient update ~16 %. So the simulator is *not* the biggest
slice — but it is the only **linear** one. Batch scaling: the model forward is latency-bound and nearly flat (16× batch
for ~1.6× time) while the per-world Python simulator is strictly linear. The simulator is therefore what *prevents*
using the large batches the GPU is idle-waiting for.

**New: `pim/simulator/interactive_batched.py` — `BatchedInteractiveWorld`.** World state as `(B, n_obj, 2)` tensors,
device-agnostic (CPU or **CUDA**, observations stay on-device). Vectorises physics, wall handling, the collision test,
the death→noise→rebirth state machine, and the ray-casting renderer. Two scalar-world subtleties preserved
*deliberately*: shift-mode's accept-guard is **sequential over objects** (object 1 sees object 0's already-shifted
position — kept as an inner loop over `n_obj`), and wall handling **resolves y before x** (the x half-width uses the
updated y). The scalar `InteractiveWorld` is untouched and remains the parity reference.

**Speed (48 steps, obs_res 128, 2 objects):**
| batch | scalar Python loop | batched CPU | batched GPU | GPU speedup | GPU env-frames/s |
|---|---|---|---|---|---|
| 64 | 165 ms | 30 ms | 51 ms | 3.2× | 60k |
| 256 | 655 ms | 56 ms | 58 ms | 11.2× | 211k |
| 1024 | 2641 ms | 84 ms | 65 ms | **40×** | 752k |
| 4096 | 10592 ms | 227 ms | 72 ms | **148×** | **2.7M** |
GPU time is nearly **flat** in batch size (51 → 72 ms for 64× the worlds), i.e. the simulator is now latency-bound like
the model instead of linear — which is exactly what unlocks large batches.

**Parity suite: `tests/test_interactive_batched.py` (11 tests; whole suite now 56 passed).**
- **Bit-exact in float64 with noise off** (`drift_force_std=0`, `obs_noise_std=0`), given the same initial state and
  actions: positions and velocities for **both** dynamics modes; observations exact after the scalar world's own
  float32 cast (asserted as *equality*, stronger than a tolerance); shift-mode `blocked` flags and positions.
- **Event parity** (collision / wall / died / alive) with `reset_on_death=False`.
- **Death→rebirth TIMING parity** — compared only up to each world's first rebirth. Writing this test surfaced a real
  property (not a bug): **after a rebirth the two implementations legitimately diverge**, because each resamples fresh
  initial conditions from its own RNG stream. Trace confirmed identical behaviour through death and all noise frames,
  divergence starting exactly at the rebirth frame. Consequence: **any training comparison can only be statistical**,
  never bit-identical, once a death occurs.
- **Statistical parity with noise on**: matched noise σ and matched death rate.
- **CUDA path matches the CPU path** in float64.

**Integration:** `scripts/train_endogenous.py --batched-sim` adds `collect_batched()`, which keeps the whole rollout
on-device (no numpy round trip). Default off, so every existing result is reproducible by the original code path.

**Validation run COMPLETE — the vectorised simulator reproduces the training outcome.**
`runs/endogenous/L3s0_ait_batched` (identical to `L3s0_ait` except `--batched-sim`), 25000 iters in 1942 s.
| metric | `L3s0_ait` (scalar) | `L3s0_ait_batched` | seed-noise reference (`L3s0` vs `L3s1`) |
|---|---|---|---|
| final train pred RMSE (actor/obs) | 0.0825 / 0.0747 | 0.0815 / 0.0737 | — |
| position R² linear (actor) | 0.781 | 0.803 | 0.783 vs 0.869 (Δ 0.086) |
| velocity R² linear (actor) | 0.526 | 0.551 | 0.537 vs 0.452 (Δ 0.085) |
| fiber residual MLP (actor) | 0.492 | 0.463 | 0.453 vs 0.451 |
| next-step RMSE (actor) | 0.1252 | 0.1203 | 0.1188 vs 0.1015 |
| deaths/1000 frames, teacher-forced | 2.8 | 3.9 | — |
| deaths/1000 frames, **closed-loop** | **72.2** | **72.8** | — |
**Verdict: every difference is smaller than the seed-to-seed variation of the same config** (e.g. position R² differs
by 0.022 between simulators vs 0.086 between seeds), and the headline closed-loop failure is unchanged (72.2 vs 72.8).
Bit-identical agreement is impossible by construction — the two implementations diverge the moment a rebirth resamples
initial conditions from different RNG streams — so this is the correct form of validation, and it passes.

**Speed: 2.81x end-to-end at batch 64** — 25000 iterations in **1942 s vs 5455 s** for the identical scalar-sim run
(same config, same iteration count; the cleanest available comparison). Implied simulator share of the scalar iteration:
141/218 = **~64%**, consistent with the standalone sim benchmark (165 ms at batch 64).

> **Two of my own measurements were wrong and are retracted.** (1) The profile claiming "sim 39% / model forward 45% /
> update 16%" is invalid — its model-forward reading (196 ms at batch 64) was warmup/contention noise, and the true value
> is ~20-25 ms; a 39% share is also arithmetically incompatible with an observed 2.8x speedup (Amdahl caps it at 1.6x).
> (2) A "controlled interleaved" benchmark reporting only 1.24x was **not** controlled: the scalar path is CPU-bound and
> the batched path is GPU-resident, so running it while another job held the GPU penalised the batched path far more.
> Interleaving equalises *exposure* to contention, not *sensitivity* to it. **Rule going forward: quote full-run
> comparisons, not micro-benchmarks taken while other jobs hold the GPU.**

**Batch size — both framings, since only one was given earlier.** For a **fixed frame budget** a larger batch is much
faster: 76.8M frames needs 25000 iterations at batch 64 (~32 min) but only 1562 at batch 1024 (~7 min, estimated) —
roughly **5x** on top of the 2.8x already banked. For a **fixed number of gradient updates** a larger batch instead costs
modestly more wall-clock and sees ~16x more data. The genuine caveat is update count (1562 policy updates vs 25000), but
**the survival task was solved by iteration ~1000 in both runs**, so there is large headroom and large-batch training is
very likely sufficient. Per-iteration costs at batch 256/1024 still need a clean measurement on an idle GPU.

**Next run STARTED automatically (2026-07-29 16:00): `runs/endogenous/L3s0_ait_state`** — the **fair GRU baseline**,
differing from `L3s0_ait_batched` by **exactly one flag** (`--carry-state`), so it is a clean single-variable test of the
hidden-state-reset flaw. The recurrent state is now carried across iteration boundaries (detached => truncated BPTT)
instead of being zeroed every 48 frames while the world continues; `predict_sequence` gained an `h0` argument so the
update starts from the state collection started from, and the actor/observer carry separate states (the actor's from
collection, the observer's from its own teacher-forced pass). The state is deliberately **not** reset on death — the
worlds are one continuous stream and rebirth is already observable through the noise frames.

## 2026-07-29 (night) — RSSM build: world model WORKS, imagination-based actor DOES NOT (yet)
Brief written and approved: `research/directions/endogenous-action-rssm.md` (hypotheses stated up front; Sevan agrees
he'd *like* emergent editability but doesn't expect it). Built:
- **`pim/world_models/rssm_actor.py`** — subclasses `RSSMModel` (base untouched, verified: its `gru_cell` input size is
  still stoch-only). Adds (a) **action in the transition**: `h_t = GRUCell([s_{t-1}, proj(a_{t-1})], h_{t-1})` — the base
  RSSM had no action input at all; verified opposite actions now change the next state (‖Δh‖ = 1.70 vs the GRU's
  historical 0.0000); (b) policy + value heads on `[h,s]` (same factored discrete space, so the `play.py` key overlay
  still works); (c) **reward + continue heads** — required because training the actor inside imagination has no simulator
  to query; (d) `imagine_for_actor` (differentiable imagination; verified gradients reach policy, reward head and the
  **prior net**).
- **`scripts/train_rssm_endogenous.py`** — online loop on `BatchedInteractiveWorld`, standard objective: recon +
  KL-balanced (0.8/0.2, DreamerV2) with **free bits**, reward/continue heads on real data, actor via **λ-returns over
  imagined rollouts** (REINFORCE + value baseline, discrete actions), critic regressed on the same returns. Observer twin
  trained on the **world-model loss only**. State carried across chunks with dead worlds cleared (GRU-thread lesson).
  **`obs_noise_std=0.2`** — the repo standard, clearing the 0.05 debt.

**WORLD MODEL: healthy.** recon RMSE 0.37 → 0.22, KL rises 0.008 → 0.18 and sits **above** the free-bits floor (0.094),
so the KL term is active and there is **no posterior collapse** — I checked this explicitly because an early KL of 0.029
looked like collapse and turned out to be an untrained-model transient.

**ACTOR-IN-IMAGINATION: fails, and not merely by entropy collapse.** Sweep at 1500 iters:
| ent_coef | final entropy | final reward | imagined return |
|---|---|---|---|
| 0.003 | 0.04 (dead) | −0.058 | −0.72 (falling) |
| 0.03 | 1.21 (alive) | −0.053 | −0.73 (falling) |
Reward ends **worse than initialisation** (−0.024 → −0.058) and the **imagined return falls monotonically**, so this is
not just under-regularised exploration — the policy is optimising a bad objective. Policy-gradient sign verified correct.
**Diagnosis:** imagined latents drift off the visited-state distribution, the reward head extrapolates nonsense there,
and the actor faithfully maximises that nonsense. (Note this is a *different* failure from the GRU's carry-state
collapse, which was stale dead-world state.)

**Overnight hedge launched** (`runs/endogenous_rssm/`): (1) **`R2s0`, level 2, 10000 it — no actor loss at all**, so it
cannot hit the bug; guarantees a trained action-conditioned RSSM world model, which makes **closed-loop coherence and
§4 editability testable in the morning regardless**. Then (2) `R3s0_warm` / (3) `R3s1_warm`, level 3 with
`wm_warmup=4000` (imagination trustworthy before the policy optimises against it) and `ent_coef=0.05`.

**OVERNIGHT RESULTS (all three runs completed).**

**World model: trains well.** recon RMSE 0.166–0.168 at `obs_noise_std=0.2`, KL 0.15 (active, above the free-bits
floor), no posterior collapse. The long warm-up + `ent_coef=0.05` **did fix the entropy collapse** — policy entropy
ends at 3.93–4.02 instead of 0.04.

**Actor: still does not learn the task.** `R3s0_warm` reward −0.016, `R3s1_warm` −0.022 versus the no-goal control's
−0.033; deaths 72.5 / 76.0 per 1000 frames versus the no-goal 83. So the policy went from *actively worse than nothing*
to *marginally better than nothing* — nowhere near the GRU actor, which solved survival outright (2.8 deaths/1000
teacher-forced). Imagined return still drifts negative. **Hypothesis 3 (agency effect) cannot be tested until this works.**

**Closed-loop coherence: hypothesis 1 is NOT supported.** Warm on real observations (posterior), then imagine forward
with the prior under the model's own actions while the real world receives the same actions. Absolute RMSE is not
comparable across threads (GRU ran at noise 0.05, RSSM at 0.2), so compare each to its OWN baselines:
| model | step-1 error ÷ copy-previous-frame | late error ÷ random-frame |
|---|---|---|
| GRU `L3s0` | 0.99 | **1.16** (worse than a random frame) |
| RSSM `R2s0` | 0.98 | **0.77** |
| RSSM `R3s0_warm` | 0.91 | 0.82 |
| RSSM `R3s1_warm` | 0.87 | 0.87 |
**Reading:** the RSSM is *relatively* better — its imagination stays below the random-frame baseline out to 40 steps,
whereas the GRU's exceeded it by step ~20 — but it is still only **≈ copy-previous-frame quality from step 1 onward**.
That is not a usable simulator. Note the sharp **prior/posterior gap**: the same model reconstructs at 0.166 from
observations but its prior-only imagination sits at 0.30–0.34. The KL term did not close that gap.

**So: adding latent consistency (KL) + a proper imagination path did not rescue closed-loop rollout — in these runs.**

> ### ⚠ RETRACTED OVERREACH (2026-07-30, Sevan pushed back and he is right)
> I originally wrote here that suspicion should move "OFF the objective and ONTO the observation channel", speculating
> that a 1D 128-ray scan may be too impoverished for long-horizon self-consistency. **That conclusion is not supported by
> this evidence and is withdrawn.** It generalises from two *under-engineered* attempts to a claim about what is
> *achievable* — precisely the "bug reframed as insight" failure mode `RESEARCH.md` names as the one to guard against.
> Against it: (1) **teacher-forced next-step prediction is good** in both architectures, so the observation demonstrably
> carries the needed information — the failure is that our models do not *propagate* it; (2) the RSSM actor **never
> learned the task at all**, which indicates implementation/tuning problems, not an information limit; (3) the
> prior/posterior gap (recon 0.166 vs imagination 0.30–0.34) is a **classic symptom of an undertrained/under-tuned
> RSSM**; (4) Dreamer-class models routinely achieve long-horizon imagination on far harder, more ambiguous
> observations, on training budgets far larger than our ~40-minute first attempt.
> **Correct status: "not achieved by our implementation yet", NOT "not achievable".** Separating those two would need a
> working reference implementation or an information-theoretic argument, and we have neither. Sevan's read — that the
> task should be achievable and the open question is how much engineering it takes — is the better-supported one.

**Owed / next:** consolidated notebook (predictive + animations + editability — Sevan's explicit request) still to
build; §4 editability on the RSSM latent not yet run (the editor script is written against the GRU API — `gru_step`,
`decode_action` — and needs an RSSM adapter, though `RSSMActor` does satisfy `HiddenStateModel`). Actor fixes if we
continue that line: if the warm-start actor still fails, the candidate fixes are (a) train the reward head on *imagined*
as well as real latents, or regularise imagination to stay near the visited-state manifold; (b) shorter imagination
horizon early, annealed up; (c) fall back to REINFORCE on real rollouts for the policy while keeping RSSM's KL for the
world model — a hybrid that abandons "actor in imagination" but keeps the latent-consistency term that motivated RSSM.
Consolidated notebook (predictive + animations + editability, per Sevan's request) still to build.

> **⏳ OWED / REMINDERS FOR SEVAN (deferred — surface these in catch-ups):**
> 0. **Re-run the endogenous thread at the standard `obs_noise_std=0.2`** (deviation found 2026-07-29; see above).
> 1. **Pure-latent-overshooting RSSM re-run** — our RSSM-multistep result used a HYBRID objective (latent-overshoot
>    KL + an added observation-overshoot reconstruction term that pure PlaNet/Dreamer omits, and which drives the
>    blur). The RSSM-multistep finding is **HELD** until we re-run with **pure** latent overshooting to confirm the
>    "objective harms the RSSM" sub-claim isn't our added term. Brief: `directions/multistep-objective-rssm-pure-overshoot.md`.
>    (The §4 editability null is structural and robust to this.) **PING Sevan to schedule.**
> 2. **Tangent-curvature metric not distance/scale-normalized** — absolute degrees are a density/scale artifact
>    (56° vs 20° across notebooks is not real). Deferred fix + options: `directions/curvature-metric-normalization.md`.
>    Does not change any finding's conclusion (intrinsic dim + hull are load-bearing).

## 2026-07-27 — Catch-up after Sevan's week away; loose ends
Everything from 2026-07-17 was committed by Sevan (branch `action_conditioning`, clean). This session: renamed the
head commit to describe the experiment; **PROMOTED the object-individuation finding** →
`findings/object-individuation.md` (Sevan-approved), **scoped explicitly to EXOGENOUS actions** (endogenous action
untested — flagged as the natural next question), with Exp-2 (`action-conditioned-structure`) **folded in** as the
earlier/weaker version. Set up the two OWED reminders above. **Still HELD (Sevan):** RSSM-multistep promotion (pending
the pure-overshoot control). **Ready to draft on request:** counterfactual-history-state (metric fixed) + multistep-
steering (freeze-time-editing win) findings. **Infra:** `nvidia-smi` fails (NVML driver/library mismatch from the
week's update) but **torch/CUDA compute WORKS** — only the monitoring tool is broken; re-runs are fine.
**multistep_steering notebook cleaned (2026-07-27):** clarified η/S/+manifold/N definitions; **all waterfalls
rewritten to the master Fig-5a spec** (gray cmap, 6 noisy context frames, edit-frame line, figure-top legend) via a
reusable `waterfall_grid` helper (they had shipped `magma` + no context frames); added two **behind-the-scenes**
expository waterfalls (fig0a interleaved self-decoded-obs process; fig2b freeze-time teacher-forced-frames process).
**CLAUDE.md waterfall spec strengthened** (hard requirement, "one helper, route every waterfall through it",
recurring-violation warning). Re-ran clean (0 err). Note for the eventual freeze-time finding: 1b (freeze-time WINS)
replicates on RSSM; 1a (interleaved latent steering FAILS) is GRU-only.

## 2026-07-17 — NEW branch `action_conditioning`: action-space → object-individuation experiment
Prior editability_multi_exploration work is **committed + merged** (PR #9; RSSM multistep negative replicated).
Sevan set up clean branch `action_conditioning`. After a long design discussion (the reframe below), launched the
follow-up to Exp-2's actions.

**The reframe (important — this is the through-line now):** the real target is **object individuation**, not
"editability" per se. Question: does training a world model on an **interaction affordance** (moving objects)
reorganize its *passive* latent into a **separable, grabbable object handle** that **generalizes to interventions it
was never trained on** — vs just wiring a trained "button"? "Realism" of the latent world (structural-realist /
pragmatic stance) = the structure supports untrained interventions + persistence. Editability was always a probe for
objecthood. Sevan's framing shift: treat the GRU+latent **as the world** ("the latent world"), not a model *of* one.

**Brief:** `directions/action-space-object-individuation.md` (active). Independent variable = **action-space type**:
`dxdy` (large relative), `teleport` (absolute in-frustum placement — saturates content, forces ghost-removal),
`axis_x` (x-only restricted — the **content-generalization** probe: train x, test y). Confound triad kept (baseline
`7_dset4` / perturbed-passive-teleport control / action-conditioned). **All eval on the PASSIVE latent (action OFF)**
with the master §4 editors — so the test is **interface generalization** (does the affordance live in the *state*,
grabbable by a foreign write-mechanism, or only in the input→dynamics pathway?). Headline readouts: **object-handle
selectivity** (reach / collateral / ghost / persistence), **content generalization** (M_axis y-vs-x), interface
generalization, + light §1–§3 + an exposition (show the affordances; confirm they're perceptually large this time).
A **clean negative** is a strong result (motivates explicit scaffolding — RESEARCH.md endgame). GRU only first pass;
RSSM later. **Worker LAUNCHED** (uses the fixed WORKER.md decoupled-execution rule: train via foreground script
calls, keep the notebook light). Awaiting completion → verify artifacts → scratch note review with Sevan.

**RESULT — DONE + VERIFIED (0 error cells, 4 ckpts, 14 figs; note `scratch/2026-07-17-action-space-object-individuation.md`,
FLAG FOR PROMOTION). Worker did NOT orphan (decoupled-execution fix held).** CLEAN NEGATIVE on the primary readout:
**no action space individuates a grabbable object handle in the passive latent.** With the canonical structural editor
(PCA geodesic, an untrained write-mechanism) targeting object k on the passive/no-op latent: **ghost never clears
(0.90–0.93 for ALL five models** vs oracle-observation 0.44–0.67, decoder-gradient oracle 0.09), and edits are **non-selective
(≈0.56–0.58** — the other object is disturbed nearly as much). Holds for every affordance (dxdy/teleport/axis_x) + the
confound triad; **baseline actually has the best reach (36.7%)** so the affordances don't help the handle at all. Actions
were genuinely large this time (|Δobs| 0.19–0.22, 2–7× Exp-2). Content generalization moot (M_axis ≈ baseline; the y>x
reach asymmetry is a lateral-vs-depth geometry artifact in baseline too). **Weaker POSITIVE, localized to action-knowledge:**
large affordances make the passive latent more canonical / linearly-readable (fiber Pert-pass 0.488 → M_teleport 0.316 →
M_axis 0.282 vs baseline 0.395; vel-linear R² up) — replicates+strengthens Exp-2, but that's representation *legibility*,
not *manipulability*. **Interpretation:** objecthood lives in the input→dynamics pathway (a button), not the state (a
grabbable handle) — the affordance doesn't transfer across write-mechanisms. Readable ≠ grabbable. This is the
"you-can't-lose" negative that **motivates explicit object scaffolding** (RESEARCH.md endgame). Awaiting Sevan: read +
artifact-or-signal + promotion call; whether to (a) probe a manipulation-type reach / persistence test, (b) go
constructive (explicit-slot architecture), or (c) an RSSM check. Caveats: GRU only, N=48 edits, in-sample probes.

> **⏳ OWED / REMINDER FOR SEVAN (deferred, do when back — needs thought):** the **tangent-rotation
> "curvature" metric is not distance/scale-normalized**, so its absolute degrees are a sample-density &
> latent-scale artifact (this is why master says 56° and the newer notebooks say ~20° — NOT a real
> difference). Fix spec + options in `directions/curvature-metric-normalization.md`. Does **not** change any
> current finding's conclusion (intrinsic dim + linear hull are the load-bearing geometry numbers). Also
> OWED: the same **static-target-render / target-fill metric inflates** as the edited object moves away
> (frozen target rays) — fixed in counterfactual + multistep_steering this round, but `00_master_editability`
> likely has it too and is under a no-edit hold until Sevan re-opens it.

## 2026-07-16 (review round) — Sevan's feedback on the 3 experiments: promotion, fixes, harness

**Clearance delivered (gates promotion): ALL models trained on NOISY obs (`obs_noise_std=0.2`).** Verified:
dataset 3 (counterfactual/master GRU), dataset 4 (multistep + action baseline), dataset 5 (action models 2/3,
confirmed noise-matched to dataset 4 — the `obs_noise_std=0.0` in the action notebook is only cell 18, a
demo/edit render, NOT training data). So multistep + counterfactual are cleared, and there is **no noise
confound** in the action baseline-vs-treatment comparison.

**Sevan's two methodological catches — both CONFIRMED correct:**
- **Static-target / target-fill metric confound.** `target-fill(s)=mean(rollout@targetrays)/mean(GT@targetrays)`
  with **target rays frozen at the edit frame** → as the object moves away, `GT@targetrays→background≈0`, the
  ratio inflates >1 and trends upward for every method (explains h*_shared 1.4, unsteered 0.655). Same flaw in
  `multistep_steering`'s "RMSE vs static target render" (`s_target`) and probably in master. Sound metrics
  (obs-RMSE vs the **moving** clean GT, ghost ratio) are unaffected. → fixing the metric to track the object.
- **Curvature not normalized** (see reminder above).

**PROMOTED (Sevan's explicit approval):** multi-step-objective **NEGATIVE** result → `findings/editability.md`
2026-07-16 entry (multi-step rollout training buys rollout accuracy + GT-matched sharpness/no-blur, but NO
editability/canonicality gain — editing failure is structural, not a next-step-loss artifact). RSSM
replication noted as OWED in the entry.

**HARNESS FIX (root cause of the worker orphan-the-run failure).** Diagnosis: a **subagent is NOT re-invoked
when a background job finishes** (that notification goes to the parent), and the **10-min Bash cap** makes a
30-min/3-training notebook impossible to run as one foreground blocking call → workers are structurally pushed
into "background it and stop → orphan." `WORKER.md` rewritten: (1) **decouple training into standalone
foreground script calls** (a GRU is ~9 min < cap) + keep the analysis notebook light (loads checkpoints only);
(2) if a run must exceed the cap, **poll in-turn with back-to-back foreground calls, never return while
pending** (ending the turn early = task failure). This is a design fix, not a sterner warning.

**RSSM multistep replication:** brief written `directions/multistep-objective-rssm.md` — **HOLDING for Sevan's
go-ahead** (he'll greenlight an overnight run after the small fixes land).

**NOTEBOOK-EDIT PASS — DONE + VERIFIED (all 4 re-ran, 0 error cells, no retraining):**
- (a) **counterfactual** — frozen-target metric fixed to **track the object per-step** (target-fill now →1
  sanely, no >1 inflation) + **h*_shared W-sweep (W=1..10)** added (`fig4_Wsweep.png`): more counterfactual
  context monotonically lowers RMSE→GT (0.240→0.183), ghost (0.77→0.39), raises target-fill (0.53→0.88); only
  **~7–9% of the displacement is reachable by linear position injection** (the reachability point, quantified).
- (b) **multistep_objective** — "Fig S"→"Fig 0", §S→§0, 30° reference line + panel-c legend removed; stale
  `figS_sharpness.png` deleted.
- (c) **multistep_steering** — confounded static-target curve replaced with an object-tracking metric
  (panel 1a; 1a conclusion unchanged — interleaved doesn't beat one-shot, heavy collateral).
- (d) **action** — **exposition section E1–E3 inserted before §1**: E1 actions↔obs effect (0.7-unit nudges
  visible as step-jumps in object x/y + marked waterfall), E2 **change-the-action sanity** (flip the token at
  t0 → rollout shifts, mean|Δobs| 0.027 obj0+x / 0.134 obj1+x → **the action channel is causally used**,
  answers the item-12 leakage question), E3 2D world **GIF** (`action_demo.gif`). Action scratch note updated
  with a validity addendum (noise-matched, no confound; causal-use confirmed; shallow-shortcut caveat).

**Action promotion still HELD by Sevan** pending his read of the exposition. Tangent-curvature fix still OWED
(deferred, see top-of-file reminder + directions brief). Master notebook untouched throughout.

## 2026-07-16 (late) — RSSM multistep replication RUNNING (Sevan green-lit; he's out)
Sevan green-lit the RSSM multistep run + left. Executing autonomously (orchestrator-driven for reliability, not a
worker). **Objective:** PlaNet-style **latent overshooting** — new script `scripts/train_rssm_multistep.py`
(standard ELBO + imagine W steps through the prior from each posterior state, obs-recon of the future + KL(sg(post)‖
imagined-prior); starts subsampled n_start=8). **W∈{1(pure ELBO),2,5}**, matched **150-epoch** budget (reduced from
the refined RSSM's 500 to fit the 2-3h cap; det 256 / stoch 64; ~11s/ep baseline). Training 3 RSSMs sequentially in
bg (~112 min) → `runs/rssm_multistep/w{1,2,5}_dset4`. Analysis notebook **built + validated**:
`notebooks/experiments/editability/multistep/multistep_objective_rssm.ipynb` (adapted from the GRU multistep notebook: RSSM
checkpoints, `sample=False` prior-mean eval, §0/§1/§2/§3/§4 + a NEW **§3b det-vs-stoch split**; all cells compile;
core RSSM editor pipeline + det/stoch logic validated against the refined RSSM — det carries ~all pos/vel code &
is far more canonical than stoch, as expected). Pending: training finish → run notebook → verify → scratch note.
Caveat baked into the notebook: 150-epoch undertraining (cross-W is the load-bearing comparison) + the un-normalized
curvature metric.

**RSSM RESULT — DONE + VERIFIED (0 error cells, 12 figs; note `scratch/2026-07-16-multistep-objective-rssm.md`).**
Training done (w1 recon 0.0247 / w2 0.0323 / w5 0.0365; 109 min). Notebook ran clean after a one-line ckpt patch
(added `val_loss` key the loader needs). **Verdict: the GRU negative REPLICATES on the RSSM, and the objective is
additionally HARMFUL there** — no editor reaches the oracle observation for any W (readable≠controllable, unchanged);
AND multi-step overshoot **blurs the decoder** (rollout TV/GT 1.23→0.43 — objects fade; OPPOSITE the GRU's no-blur),
**worsens** single-step (next-step RMSE 0.113→0.166) and open-loop (0.204→0.247) prediction, **collapses the linear
hull** (36→10 dims @90%), and reduces linear readability (pos 0.82→0.64) + canonicality (MLP fiber 0.42→0.52). det
h carries ~all (pos,vel) (det≈full, intrinsic ~4); overshoot de-canonicalises the det core. Caveat: overshoot
best-recon ckpts are early (w2 ep64, w5 ep25) → harm understated if anything. Finding `editability.md` OWED-RSSM
line updated (marked done-scratch, pending Sevan's review). **This completes ALL work Sevan assigned; awaiting his
return** — promotion calls (multistep RSSM leg; action-conditioning still held pending his exposition read; the
counterfactual metric is now fixed and ready to draft as a finding on his word) + the deferred curvature-metric fix.



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

- **Exp 3 — `notebooks/experiments/editability/multistep/multistep_objective_structure.ipynb`** (15 cells, 0 err; note
  `scratch/2026-07-16-multistep-objective-structure.md`; script `scripts/train_gru_multistep.py`; ckpts
  `runs/gru_multistep/w{2,5}_dset4_gru_400epochs`; 11 PNGs `/tmp/multistep_objective/`). **Clean NEGATIVE:** a
  free-running w-step rollout objective (w∈{2,5}) buys open-loop rollout accuracy (0.208→0.188) and GT-matched
  sharpness (**no blur** — watch-item cleared) but **no editability and no canonicality gain** — §4 pathology
  (decoder-inert probe, belief sluggishness, off-manifold oracle collapse) replicates unchanged across w; if
  anything canonicality mildly *degrades* (fiber resid 0.357→0.457, pos-linear R² 0.84→0.76). RSSM leg CUT
  (per cap). Refutes the brief's "coherence-under-iterated-dynamics ⇒ editable state" intuition.

- **Exp 2 — `notebooks/experiments/editability/actions/action_conditioned_structure.ipynb`** (22 cells, 0 err; note
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
  the model rollout from `h_gt` stays as its own labeled "Oracle observation (model rollout)" column.
- **"MLP-gradient" was a misnomer** — it is the DECODER/obs-gradient editor (Adam on h vs GT obs). The
  repo's actual MLP-probe steering primitive (`pim.editors.gradient_steer`, from the mlp_steering PR) was
  never in the line-up. Renamed → "Decoder gradient"; "MLP-probe gradient" ADDED as a new editor.
- Also: the per-step `→target` metric compared against the STATIC edit-frame target render (so even the
  oracle observation "drifts" from it) — redefined vs the time-evolving sim clean obs at ef+s.
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
- **The oracle observation itself is sluggish** — obs-change only 0.129 (GRU) / 0.059 (RSSM) with ghost-ray
  ratio 0.665 / 0.884: a single-frame belief update barely moves the rendered scene, so *every* editor's
  ceiling is low. Reframes "editing fails": even reality's own state, injected, doesn't visually teleport
  the object in one frame.
- **Geodesic K=600: ASYMPTOTES** (GRU 1.75→1.03 plateau by ~iter 135, flat to 600; RSSM no descent).
  Resolves Sevan's "did it just need longer?" — NO. And GRU's plateau readout (1.03) is *better* than the
  true-swap's readout (1.61) while its obs stay ≈unsteered → readout and obs accuracy nearly decoupled.
- **No non-oracle editor beats the oracle observation on GT next-step RMSE, on either model.**
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

**2026-07-16 — correction + proposed experiment (from discussion):** Sevan refuted my "oracle observation =
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
