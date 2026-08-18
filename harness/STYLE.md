# STYLE.md — presenting work so it is actually read

Read this **before** writing any figure, table, notebook, or reader-facing deliverable.

Work has two readers with different strengths. The **human** does the scientific judgment
and reads best from pictures. The **agent** cannot see figures and self-verifies from
numbers. Always produce **both** — rich visuals and printed metric tables. Never tables
alone; never figures alone.

---

## 1. The reading order — design for it

This is the observed reading path, not a style preference. Deliverables that ignore it get
misread. **Build so that the top 20% of the reading path carries 100% of the findings.**

1. **The canonical qualitative panel — always first.** Every project has one primary
   artifact that shows the raw output of every method side by side; it is the first thing
   opened, because success or failure is visible instantly. See §2.
2. **Then the figures**, skimmed for the main points. Each must stand alone.
3. **Then, only when something is unclear**, the definitions and run tables at the top.
   This is why those tables must be complete and near the top.
4. **The opening text**, when the work matters or is revisited after time away.
5. **Per-section prose**, only when confused about what was done.
6. **Printed numbers are rarely read**, and **the summary is essentially never read** by the
   human — write it anyway, as your own consistency check and the handoff to the record.

**Consequences.** Put the finding in the *figure*, not the prose. Never let a result exist
only in a printed table or only in the summary. If a method is worth discussing, it is worth
a column in the qualitative panel. If a figure needs a paragraph to be interpretable, fix the
figure.

---

## 2. The canonical qualitative panel

Every project defines one, and it is the single most load-bearing graphic. Whatever its
form, these requirements are universal:

- **Every method under discussion gets a column.** Including baselines, including trained or
  tuned variants, including the plain unmodified reference. A panel that omits the arms the
  work is actually about reads as "those were never run."
- **A ground-truth / reference column, always.** A comparison with no reference is
  uninterpretable.
- **At least three sample rows — always.** The panel is judged qualitatively, so it needs
  3–4 examples per method to distinguish a real effect from a lucky sample. Two rows is not
  a comparison.
- **Each column shows its own output.** Never paint a shared row sourced from one privileged
  arm across all columns; it makes every method look as though it had access that only one
  of them had, and it hides the exact quantity being scored. Where one arm legitimately has
  extra information, label *that arm*, do not re-align the others to it.
- **Fixed scaling across cells** (explicit limits, no per-cell autoscaling). Autoscaling makes
  a degenerate output look normal, which is the exact failure the panel exists to catch.
- **Mark the event.** Where there is an intervention, boundary, or transition, draw it.
- **Include each arm's headline metric in its column title**, so the picture and the number
  are read together.
- **Include the degenerate and extreme settings as their own columns** — that is where
  collapse becomes visible.
- **Implement it exactly once**, as a single helper that bakes in the whole spec, and route
  every instance through it. Re-implementing the panel per notebook is where drift happens,
  reliably.

**Mandatory:** any claim about an effect on a model's outputs ships with this panel. A
scorecard compresses everything to one number and routinely hides the difference between
"the intervention worked" and "the output degraded" — the two look identical in a metric
that moved.

**Before committing one, eyeball the rendered image:** reference column present? every
method present? ≥3 rows? fixed scaling? event marked? legend at the top? If not, it is not
done.

---

## 3. Figure mechanics

These are recurring, expensive mistakes. All of them have been made.

- **Never use ALL-CAPS for emphasis** — figures, titles, legends, prose, or docs. It reads as
  shouting. Use **bold** or *italics*.
- **Every figure stands alone:** axis labels, units, and a legend entry for **every line
  drawn**, including dotted, reference, and baseline lines. An unexplained dotted line is a
  bug — the reader will ask what it is, which means the figure failed.
- **Legends are labels, not explanations.** A few words. If it needs a sentence, the sentence
  goes in the subtitle or caption and the legend entry stays short. A legend wide enough to
  run off the figure is a bug.
- **Legend handles must show the line style.** Default handle length can make solid and
  dashed indistinguishable; set `handlelength` so they are separable.
- **Name the exact quantity on every axis and in every metric label.** Any metric with a step,
  a window, or a subset baked into it must say so — not "Index" but "Index — at the
  intervention step (step 0)".
- **Plot the absolute quantity, not a gain,** unless the gain is itself the subject. A
  "gain over baseline" axis hides that a method may only have moved the output toward
  garbage. Plot the absolute value and mark each arm's own reference **on the same axis**, so
  both are read in the same units.
- **One quantity per graphic element.** Never print a different metric as text on the bars of
  a chart whose axis is something else. A second quantity gets its own panel or table column.
- **Panels compared side by side share identical category order and position.** Horizontal
  scanning is the entire point of a multi-panel comparison; a category present in one panel
  and absent in another shifts every row below it. Keep one canonical category list and show
  absent entries explicitly (an empty slot labelled *n/a*, reason in the caption).
- **One quantity per axis — a shared axis claims the bars mean the same thing.** If the
  "same" metric is computed from structurally different constructions, it does not belong on
  one axis however similar the column header looks. Test: *could a reader subtract two bars
  and get something meaningful?* If not, split the figure. Where a genuine outcome metric is
  common to both, that panel may stay shared — say so explicitly.
- **Guardrail metrics stay, as a mark rather than a number.** Where a quality guard
  distinguishes a real success from one that scored by degrading the output, it cannot be
  dropped. Put the *number* in the table; in the figure, mark **only the failing arms** with
  one visual cue explained in a single legend entry.
- **Axis labels must be legible — check the rendered image, not the code.** Long series names
  in vertical bar charts overlap into a smear at 4+ categories. Use **horizontal bars** for
  anything with long names. Never shrink a label below ~7pt to make it fit. "It ran without
  error" is not the check.

---

## 4. Numbering and addressability

Every artifact gets a number, because the work is referenced later in discussion and a claim
must be citable without hunting.

- Prefix each code cell with a sequential `# [N]` tag.
- Give every figure a number in its title (`Fig 3 — convergence`), sub-panels lettered
  `(a)/(b)/(c)`. A claim should be citable as "cell [7] / Fig 3a".
- **Animations are numbered the same way** — a *persistent* figure-level title carrying the
  number, separate from any per-frame caption, and the saved file named to match. Playback
  must be legible: slow enough to read (~3 fps, not the library default), and **hold on the
  key frames** by repeating them so the viewer can register the effect.
- An animation is an **addition, never a substitute**. It cannot be read in a committed diff
  or a paper, so the claim still ships with the static panel.

---

## 5. Legibility standards

A deliverable is read later by the human (from plots) and re-derived by agents (from
numbers). It must stand alone and be followable top to bottom.

- **Definitions table up front.** Right after setup: a table defining every non-obvious term
  and **every metric with its explicit formula**, units, and better-direction (↑/↓). A
  metric's definition lives in that table, not in a print sidenote or a code comment. When a
  term first appears, it must already be defined. Where the project has a canonical metric
  registry, copy the exact name/formula/units from it rather than re-inventing terms.
- **Every run, model, and variant name is defined where it is used — no bare short codes.**
  This is a recurring, high-friction failure: labels like `L3` vs `L3b`, "weak" vs "strong",
  shipped with no expansion, leaving the reader unable to tell what was compared. The rules:
  1. Each thread keeps a **canonical run registry** — one table listing every run name with
     its full configuration (objective, data, architecture, training length, seed, and what
     it is a control *for*).
  2. Every deliverable that mentions a run **copies the rows it uses** into its own
     definitions table, so it still stands alone.
  3. **Figures and tables use self-describing labels**, never raw codes.
  4. A suffix encoding a variable must state the variable it encodes.
  5. Adding a run means adding its registry row **in the same commit**.
- **When a comparison varies along more than one dimension, the label carries both** — and
  neither may hide inside the other's naming slot. Arms testing different things must not
  look like one family because they share a parenthetical. Prefer labels that make the tested
  dimension unmistakable, and if one dimension has only one level, say *why* rather than
  letting the asymmetry look like an omission.
- **Name methods by their mechanism.** The name must say what the method actually does. Never
  name a method after an incidental implementation detail, and never reuse a name that
  already means something else in the project.
- **Tables for dense values, clearly demarcated.** When a step emits many named scalars,
  render a real table with visible row/column structure — not an aligned-monospace `print`
  block. Targeted use: do not duplicate every plot as a table.
- **Data-source provenance.** Each section states the exact model / checkpoint / dataset /
  split it uses. When a number is imported as a comparison from other work, **cite the source
  inline** rather than dropping a bare constant.
- **Define every implementation detail where it is used.** Any threshold, subset, or cutoff a
  reader would ask about must appear in the definitions table or a clearly identifiable note
  — never left as an unexplained label on an axis.
- **Plain language, not shorthand.** Titles, print headers, and prose are for a human. Write
  "≈", "→", "much less than", "≠", or plain words — not `~=`, `=>`, `<<`, `!=`. A **title
  states what is shown, not the current result**; results belong in a dated results block,
  never in a figure or section title.
- **Keep one question in one deliverable.** Variants a reader would naturally compare belong
  together, even when produced by different scripts on different days. Splitting them makes
  the reader conclude the missing arm was never run. If a split is genuinely unavoidable,
  **every** deliverable on both sides carries a prominent pointer — at the top and at the
  figure that would otherwise look incomplete — naming its sibling and what it holds.

---

## 6. Visual aesthetic

Two themes, chosen by what is being shown:

- **Results, metrics, analysis → light academic theme.** White background, the colorblind-safe
  Okabe-Ito palette, consistent axis styling. `harness/theme.py` provides `PALETTE`,
  `style_ax`, and `style_ax_dark`; copy it into the project's figure module.
- **Raw data artifacts → dark theme.** Anything showing the data as it actually is — sensor
  output, imagery, the system's own generations — uses the dark background.

When in doubt: metrics light, raw data dark. Grayscale intensity data is displayed on the
dark theme with a gray colormap; do not invent a decorative colormap for data whose absolute
values matter.

---

## 7. Structure of a top-tier synthesis deliverable

Some work sits above one-off experiments: a single source of truth consolidating a thread
across models, proposing the language and metrics that may later be folded into the codebase.
Extra standards for that tier:

- **Separate the invariant spine from dated results.** Definitions, formulas, and the
  pipeline are stable. Every *result* lives in a clearly marked **`Current results (updated
  YYYY-MM-DD)`** block — never woven into a section header, a definition, or a figure title.
  The reader must be able to tell "what this measures" from "what it reads now".
- **Build every figure and table to hold N models.** No two-model hardcoding: categories on
  one axis, one colour-coded series per model, shared legend; never put a model's result in a
  panel title. Adding a third model must be a data change, not a re-layout. Report the *same*
  estimator for every model — compute it, do not write "about the same".
- **Comparison sets grow.** Lay out so a new method is an added column (wider figure, single
  legend at the top), not a redesign.
- **Calibrated claims.** Body sections state quantities without verdict adjectives and do not
  binarize graded quantities. Interpretation is confined to a clearly marked summary section
  and stays quantified.
- **Lightweight:** recompute only the cheap things; cite the rest with provenance.

---

## Local instantiations (this project — not portable)

- Canonical qualitative panel, 1D observations → the fixed specification in `../CLAUDE.md`
- Canonical qualitative panel, 2D observations →
  `../notebooks/experiments/editability/omniscient_2d/WATERFALL_SPEC_2D.md`, implemented in
  that directory's `frame_grid.py`
- Metric and method registry →
  `../notebooks/experiments/editability/METRICS_AND_EDITORS.md`
- Figure theme in use → `pim/figures/theme.py` (`style_ax`, `style_ax_dark`, `PALETTE`)
- Run registries → the `*_RUNS.md` file in each thread directory under
  `../notebooks/experiments/editability/`
