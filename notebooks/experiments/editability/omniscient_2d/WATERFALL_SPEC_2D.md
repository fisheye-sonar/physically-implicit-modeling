# Omniscient-2D waterfall spec — FOR REVIEW (drafted 2026-08-11)

**Status: awaiting Sevan's sign-off.** `CLAUDE.md` fixes the waterfall spec as a hard,
recurring-violation rule and makes a waterfall **mandatory** for any claim about an effect on the
generations. That spec assumes a **1D** observation. This thread's observation is a **48×64 2D
raster**, where a single frame already uses both image axes and a literal waterfall cannot be
drawn. This file is the proposed adaptation. It is **not** a licence to drift — the intent is that
it becomes the thread's binding spec once approved, and that `CLAUDE.md` gains a pointer to it.

Implementation: `frame_grid.py` in this directory — **one definition, imported by every notebook in
the thread**. (`CLAUDE.md` says "define one `waterfall_grid(...)` helper in the notebook"; a shared
module is a stricter reading of the actual instruction, which is *don't re-implement per notebook*.
Flagging the deviation rather than burying it.)

Review figures, rendered from real simulator data (no model involved):
- `SPEC_figS1_frame_grid.png`
- `SPEC_figS2_frame_trails.png`

---

## What is preserved, unchanged in meaning

| 1D spec requirement | how it is met in 2D |
|---|---|
| `cmap="gray"` on dark background | identical — never magma/viridis/pink-purple |
| GT (sim clean-obs) reference column | GT is an **arm**, always present, always first, styled apart |
| ~6 noisy pre-edit context frames — the **actual observations teacher-forced on** (`edits.obs`), not the clean render | preserved, subsampled to 3 (see below); still the noisy `edits.obs`, still per-arm |
| marked edit-frame line | a thick **orange left edge** on the first post-edit cell of every row |
| below the line, **every column shows its OWN free-run from step 0**; no shared teacher-forced `ef` row | preserved exactly — this ban is fully intact |
| GT column shows `clean_obs[ef:ef+K]` | identical |
| alignment: `ROLL[:, 0:K]` ↔ `clean_obs[ef:ef+K]`, no slicing, no dropped step | identical; `Arm(leads_by_one=True)` labels a First-Obs-TF arm rather than re-aligning the others |
| green = target, red-dashed = ghost locators | preserved, as **circles** (see below) |
| figure-top legend | identical |
| single what-is-shown title, no results | identical; `fig_num` carries the required figure number |
| arm's headline metric in its label | identical (`Arm.metric`) |
| wide enough that every cell stays full size | `cell=` grows the figure; `CELL_ASPECT` is fixed so cells can never be squashed to fit more arms |

Added beyond the 1D spec: **fixed `vmin=0, vmax=1` on every cell.** Per-cell autoscaling would make
a collapsed arm look normal — exactly the failure these panels exist to catch.

## What necessarily changes, and why

**1. Axes swap — arms are rows, time is columns.**
In 1D, arms had to be columns because time already owned each image's vertical axis. Here both axes
of the grid are free, and time-left-to-right is the reading order a viewer expects.

**2. Time is subsampled — 3 context frames, 5 rollout steps (0, 3, 7, 11, 14).**
This is the one substantive loss. A 1D waterfall shows all ~21 frames because a frame costs one
pixel row; here a frame costs a whole cell, so 21 columns would either shrink cells below
legibility or make the figure ~2 m wide — both already forbidden. The chosen steps preserve what
the frame series is *for*: what the model was fed, whether the edit **lands** (step 0), and whether
it **holds** (steps 11/14). Every displayed step is labelled with its true frame index, and
`steps=` makes the choice explicit and auditable per figure.

**Mitigation — `frame_trails` composites EVERY step**, so nothing is hidden by the subsample. The
proposal is that the two ship **together** whenever a claim concerns persistence, and that this
pairing is the thread's replacement for "the waterfall".

**3. Locators are circles, not vertical lines.**
In 1D a position projects to a ray index, so a locator is a line. In 2D the target and ghost are
*places*: circles of the true object radius at true world coordinates, drawn through
`imshow(extent=...)` in world units. `aspect="equal"` is mandatory — with `aspect="auto"` the
circles render as ellipses and apparent object shape becomes a lie. (Caught in the first render.)

---

## Reading the two panels

- **`frame_grid`** — raw model output, unprocessed. This is the one that catches **degradation**: a
  collapsed arm shows as noise or saturation in cells that should hold a clean disc.
- **`frame_trails`** — each arm's whole rollout composited into one image, later steps brighter.
  This is the one that shows **where the object went**: a landed edit is a trail arriving inside the
  green circle; a failed one sits on the red dashed circle; a collapsed one is a smear with no disc
  structure. Note the per-cell scale here is **relative**, unlike the grid's fixed 0–1.

## Validation of the review figures

Rendered from `edits` data only — no trained model — so the arms are exact known quantities and any
layout error is visible against a known answer:

| arm | what it is | Edit Index | expected |
|---|---|---|---|
| GT (sim) | `clean_obs[ef:ef+K]`, the edited world | — | reference |
| Unedited world | the counterfactual rolled forward (`build_edit_zones.gt_unedited_traj`) — literally what a perfectly inert editor produces | **−1.00** | exactly −1 |
| SYNTHETIC collapse | GT scaled to 0.35 + N(0, 0.28) + 0.25, **a fabricated illustration, not a measurement** | **+0.16** | ≈0, not spuriously good |

Both land where the metric's definition says they must, which is the point: −1 for "did nothing",
≈0 for "destroyed the output". The collapse arm is labelled SYNTHETIC in the figure itself so it can
never be misread as a result.

---

## It has now been used on real results

`omniscient_2d_world_state.ipynb` Figs 5–6 run both panels on the finished models across six arms.
They earned their place: the **MLP Grad Steering** arm posts a respectable-looking Edit Index
(**−0.47** vs unsteered −0.54) but the frame grid shows visible ringing artifacts at step 0 and the
trail is a smear — consistent with its **fidelity 1.11**, i.e. degradation rather than editing. The
scorecard alone would not have separated that from **Global PCA Projection**'s +0.11 at fidelity
1.04. That is exactly the failure `CLAUDE.md` requires a waterfall to catch, and the 2D form catches
it.

## Open questions for you

1. **Is the 5-step subsample acceptable**, given `frame_trails` shows all 15? If you want more
   steps, the cost is figure width — 8 steps is still readable at ~16 in.
2. **3 context frames vs the 1D spec's ~6.** Six is affordable if you'd rather stay literal; it
   costs ~4 in of width and the frames are near-identical at 0.2 noise.
3. **Should this pairing be added to `CLAUDE.md`** as the sanctioned 2D form once approved, or stay
   thread-local until the thread proves out?
4. **GIF option.** `CLAUDE.md` already specifies numbered animations (~3 fps, hold on edit frames),
   and 2D frames are the case where an animation genuinely beats a static grid. Worth adding an
   `Anim` builder to this module, or is the grid + trails pair enough?
