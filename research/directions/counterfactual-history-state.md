# Direction: Counterfactual-History State — existence check for a clean-edit hidden state

**Tag:** `[in-frame]` · **Sub-question:** 3 (editability) · **Status:** active (2026-07-16) ·
**Complexity:** low-medium (one notebook, GRU; separate reference experiment)

> A **dedicated sanity-check experiment in its own notebook** —
> `notebooks/experiments/editability/counterfactual_history_state.ipynb`. **Do NOT touch the master
> notebook** (keep it un-bloated). Sevan's call: expected to succeed ~tautologically, so this is a
> *reference* existence-proof, not a headline; **only flag for promotion if the result is surprising**.

## Why this exists
Latent editing has write access to `h` and is **not** constrained by the model's observation-mediated
belief update. The master notebook's "true-state swap" (`h_gt` = teacher-force the actual post-edit obs
— which contain a **teleport discontinuity** at `ef` — up to `ef`) gives the model only **one frame** of
teleport evidence, so belief inertia leaves a ghost. That is a **lower bound** for editing, not a ceiling.
This experiment asks the existence question directly: **does a hidden state `h*` exist that, injected at
the edit frame, renders the teleport cleanly (object at target, ghost-free, persists)?** If yes, the
editing failure localizes to the **edit map's reachability** (not the target's existence, nor missing
information) — sharpening the learn-to-edit negative result. If `h*` does NOT render cleanly, that is
**surprising and important** — flag it loudly.

## Construction (per edit sample i)
- `o = edits.edit_object[i]`; `target = edits.positions[i, ef, o]`; `vel_o` = preserved velocity of the
  edited object (from the edits HDF5 `velocities[i, ef, o]`); `dt` from sim config.
- **Counterfactual trajectory of the edited object** (constant velocity through target at `ef`):
  `pos_o(t) = target − vel_o · (ef − t) · dt`, for `t = 0..ef`.
- **Other object(s): true history unchanged** — `edits.positions[i, t, other]`.
- **Render clean obs** (obs_noise=0, fixed reflectivities/radii) for frames `0..ef` of this counterfactual
  world (per-frame single-frame Scene/`render_scene`, exactly as master §4 / `canonical_state_editing`).
- **`h* = teacher-force`** the GRU on these `ef+1` counterfactual obs (state after frame `ef`), same
  `tf_hidden_at` loop the master §4 uses.
- **Frustum caveat:** back-extrapolated positions may exit the frustum in early frames (training used
  `always_in_frustum`). Report the fraction of samples with the edited object out-of-frustum early; if
  material, ALSO run a **shared-context variant** (real pre-edit obs for `0..ef−W`, counterfactual obs for
  the last `W≈10` frames) and report both — note which is cleaner and whether it changes the verdict.

## Test / metrics (define each in a definitions table; mirror master §4 obs-space metrics)
Roll out from `h*` for `K=15`; **GT = `edits.clean_obs[i, ef:ef+K]`** (the sim's true post-edit obs — same
target world, since the counterfactual world from `ef` onward equals the post-edit world).
- **Head-to-head references, same samples:** GT (sim) · **counterfactual-state `h*` rollout** ·
  one-frame-evidence `h_gt` (true-state swap) · Unsteered (`h0`) · Readout injection (probe pseudoinverse,
  a known-failing contrast).
- Per state: per-step **obs RMSE vs GT**, **ghost ratio**, persistence over `K`.
- **Decision rule:** `h*` rollout matches GT (obs RMSE ≪ the one-frame-evidence state's; ghost ≈ 0;
  persists) ⇒ **a clean-edit state exists in h-space** ⇒ editing failure = reachability of the edit map,
  not existence/information. Otherwise ⇒ surprising; investigate (injection-vs-teacher-force equivalence;
  can the model even represent the counterfactual?).
- **h-space geometry (the reachability point):** `‖h* − h0‖`, `‖h* − h_gt‖`, `‖h* − readout-injection‖`;
  and the fraction of `(h* − h0)` that lies **along the position-probe direction** vs its full norm.
  Argument: `h*` differs from `h0` in **many** coordinates (history-laden), so a low-dim probe-aligned
  edit cannot reach it — ties to the ~35% history-entangled fiber and the learn-to-edit negative result.

## Figures (follow CLAUDE.md legibility: definitions table, demarcated tables, plain language, GT column)
- (a) **Waterfall** (dark, `world_model_eval` style): columns GT | counterfactual `h*` | one-frame `h_gt`
  | Unsteered | Readout injection, ~3 samples, green target / red ghost lines, a few pre-edit context
  frames. (b) per-step obs-RMSE-to-GT curves per state. (c) h-space distance bars + probe-aligned fraction.

## Deliverables
- Executed notebook `notebooks/experiments/editability/counterfactual_history_state.ipynb` — run
  **synchronously in-turn** (0 error cells). PNGs → `/tmp/counterfactual_history/`.
- Dated note `research/scratch/2026-07-16-counterfactual-history-state.md`: the verdict (does `h*` render
  cleanly?), the h-space geometry, caveats. Mark `→ FLAG FOR PROMOTION` **only if surprising**; otherwise
  "reference sanity check — not for promotion." Do NOT touch the master notebook, `findings/`, or `RESEARCH.md`.

## Bootstrap
GRU `runs/gru/3_dset3_gru_persistentids_inview_400epochs/best_model.pt`, data
`datasets/4_fixed_refl_inview`. Mirror `canonical_state_editing.ipynb` / master §4 for `warm_up_to_edit`,
`Scene`/`SimConfig`/`render_scene`, `_rollout`, `tf_hidden_at`, probe (`LinearExtractor` + `inject_state`).
Paths 3-deep (`../../..`, `../../../runs`, `../../../datasets`). GRU-only is sufficient; add RSSM only if cheap.
