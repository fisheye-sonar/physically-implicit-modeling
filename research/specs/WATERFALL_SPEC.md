# WATERFALL_SPEC.md — the 1D observation waterfall (canonical qualitative panel)

This is **this project's instantiation** of the canonical qualitative panel required by
`harness/STYLE.md` §2. Every universal rule there still binds — reference column, ≥3 sample
rows, every method gets a column, fixed scaling, marked event, metric in each column title,
one shared helper. What follows is the project-specific *form* those rules take for a **1D**
observation, where a frame is one row of pixels and a whole sequence fits in one image with
time on the vertical axis.

(The 2D-raster variant and its `frame_grid` implementation were retired with the
omniscient-2d thread in the 2026-08-31 housecleaning; recover both from the
`pre-cleanup-2026-08` tag if that observation channel returns.)

---

## The fixed spec — do NOT invent a colormap

Observation waterfalls use **`cmap="gray"` on the dark background** — the
style the retired `00_master_editability.ipynb` Fig 5a established (the canonical
reference renders live in `pim/figures/waterfall.py`). **Never magma, viridis, or the pink-purple scheme.**

Every waterfall comparison has:

- a **ground-truth reference column** — the simulator's clean observations (`clean_obs`);
- **~6 pre-edit context frames** — the **actual (noisy) observations the model was
  teacher-forced on** (`edits.obs` / `test.obs`), **not** the clean render; only the GT column
  is clean — placed above a marked **edit-frame line**;
- below that line, **every column shows its OWN free-run starting at step 0**, with the GT
  column showing `clean_obs[ef:ef+K]`;
- vertical **green = target** and **red-dashed = ghost** locators;
- a **figure-top legend**, never inside a panel;
- a **single what-is-shown title** — no results in the title;
- width sized so **every column stays full size** (add columns by widening, never by
  shrinking);
- **≥ 3 sample rows, always** — 3–4 examples per method or configuration is the minimum for a
  qualitative judgement; two rows is not a comparison;
- **every method under discussion as its own column**, including trained and fine-tuned
  editors and the plain baseline model. A waterfall showing only the training-free editors,
  when the notebook's subject is the trained ones, reads as "the trained ones were never run".

## ⛔ Never paint a shared teacher-forced `ef` row across all columns

*(Corrected 2026-07-30.)* An earlier version of this spec mandated one shared row =
`clean_obs[ef]` in *every* column. **That is wrong and is banned.** It makes every column look
as though it were teacher-forced on the post-edit frame when only the **Oracle observation**
reference actually was, and it **hides the exact frame the §4 scorecard scores** (step 0). It
also displayed the *clean* render while the model that legitimately sees that frame is fed the
**noisy** `edits.obs[ef]` — a second inconsistency.

Seeing the post-edit frame is a **property of one editor**, never a display convention.

*(The same error was caught and fixed in `eval_editability_endogenous.py` v2, then leaked back
in via `CLAUDE.md` into the `controls/` notebooks. It is a recurring violation.)*

## Alignment rule — get this exactly right

`warm_up_to_edit` teacher-forces `obs[0..ef-1]`, so the predict-next GRU's rollout **step 0 is
`ef`** (`decode(h_edit) ≈ obs[ef]`, i.e. `ROLL[:, 0] ↔ clean_obs[ef]`).

So plot **`ROLL[:, 0:K]` against `clean_obs[ef:ef+K]`** — no slicing, no dropped step.

The one exception is the **Oracle observation** column, which was fed `obs[ef]` and therefore
**leads by one frame**. Label it as such rather than re-aligning the other columns to it.

## One helper, always — `pim.figures.waterfall_grid`

```python
from pim.figures import waterfall_grid
```

**This is the implementation. Do not write another one.** It bakes in the whole spec: gray
colormap on dark, noisy context frames above the edit-frame line, each column's own free-run
from step 0 below it, green target / red-dashed ghost locators, a figure-top legend, fixed
`vmin`/`vmax`, and the metric in each column title.

Two rules are enforced **structurally**, so the common violations cannot be expressed:

- `columns` maps each arm name to **that arm's own rollout**, so a shared teacher-forced row
  painted across every column is not representable.
- `vmin`/`vmax` default to 0/1 and apply to every cell, so per-cell autoscaling — which makes a
  collapsed arm look normal — cannot happen by accident.

Fewer than three sample rows raises a warning rather than failing silently.

> ### Migration debt — RESOLVED by deletion (2026-08-31)
> Eighteen separate `waterfall` implementations once existed across the retired experiment
> notebooks — the drift this spec was written to prevent. The housecleaning removed every
> copy along with its notebook; `pim.figures.waterfall_grid` is now the only implementation
> in the tree, and new work has nothing else to reach for.

## Eyeball check before committing

Gray colormap? Noisy context frames above the edit line? **Each column its own free-run from
step 0** (no shared teacher-forced row)? Figure-top legend? GT column? ≥3 sample rows? Every
method present? Metric in each column title?

If not, it is not done.
