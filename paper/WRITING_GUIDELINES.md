# Writing guidelines for the paper

Standing instructions from Sevan for any help with `paper/paper_draft.tex`. Read this before
touching the draft so the guidance does not have to be repeated.

## Scope of any edit

- Edit only the sections that were asked for. Everything else in the file stays untouched,
  including the preamble, abstract, introduction, discussion, related work, and appendix
  skeleton, unless the request names them.
- Sevan's inline `%` notes inside a requested section are working notes. Convert them into
  prose and delete them. Leave a short `% TODO:` where a placeholder is deliberate.
- Restructuring sentences, paragraphs, and subsection boundaries inside a requested section is
  fine and encouraged when it strengthens the paper. Section and subsection titles are
  suggestions, not fixed.
- Sevan compiles on Overleaf. Do not compile locally, and do not add packages to the preamble
  without asking. Use only LaTeX constructs the draft already uses (`align`, `bmatrix`,
  `\operatorname`, `\mathbb`, plain `tabular` with `\hline`, `\citep` / `\citet`).

## Voice and style

Reference points for voice: the abstract and introduction of the current draft, and
`paper/writing_sample.pdf` (Brodjian, Hobley, Perona, "Single-View Seafloor Recovery from
Imaging Sonar via Differentiable Rendering").

- Clean, simple, straightforward. Confident, clear, precise. Write like a human.
- The first sentence of a paragraph states the point or claim of that paragraph. The rest of
  the paragraph argues, supports, justifies, elaborates, or discusses it.
- Paragraph lengths vary naturally, sometimes widely. Not every paragraph is the same size.
- Avoid complex sentence structure. One idea per sentence where possible.
- No em dashes. No semicolons. Colons only when they clearly help.
- Do not open a paragraph by pointing back at the previous one with "this": no "This shows",
  no "We build off this idea". Start with the new point.
- Avoid generic phrasing and language that reads as machine-written. Avoid the term
  "load bearing".
- Hedge once, in the sentence where the claim is made, and move the rest of the caveats to
  the body or appendix. Do not stack qualifiers.
- Name methods by what they do. Expand a term before its abbreviation the first time.
- American spelling (color, center, standardize), matching the abstract and introduction.

## Length budget

- The whole submission is 9 pages. Methods and Experiments together should occupy about 4
  pages including their tables and figures, so roughly 1,800 words of prose plus three compact
  tables. The first draft (2026-09-10) ran to 8 pages and had to be cut by more than half.
- Before handing over, estimate pages: about 600 words of dense prose per page, a compact
  table about 0.3 page, a display equation block about 0.1 page.

## Placeholders and uncertainty

- Use `\red{...}` (already defined in the preamble) for anything provisional in the compiled
  text: pending runs, placeholder figures, results that are not in the canonical table, notes
  to self about structure, and undefined references. The reader should be able to see every
  soft spot at a glance.
- Keep provenance notes and file paths in `%` comments, not in `\red{}`, so they never reach
  the PDF.

## Math and tables

- Keep LaTeX math clean and basic so equations can be reworked later. No elaborate or
  confusing macros. Prefer several short equations to one dense one.
- Tables must fit the text width. Test by eye at `\small` with `\setlength{\tabcolsep}{4pt}`,
  shorten row labels and headers before anything else, and never rely on `\resizebox` (the
  preamble does not load graphicx). Long unbreakable tokens such as file paths do not belong
  in a `\parbox` or a table cell.
- Tables use plain `tabular` and `\hline`. Every number in a table must be traceable to a
  `scores.json`, `baselines.json`, or a findings file in `research/findings/`.

## Accuracy

- Everything written about the method must match the code in `pim/`. Check the implementation
  before describing an editor, probe, metric, or environment rule. `research/REGISTRY.md` is
  the index of canonical objects.
- Every quoted number comes from the canonical scores (`runs/<topic>/<run>/scores.json`,
  written by `notebooks/master_eval.ipynb`), the floors (`runs/_baselines/`), or a dated entry
  in `research/findings/`. Never quote an Edit Index without its fidelity ratio.
- State scope honestly where it matters (one seed per cell, one instance per variant, the
  number of cases) without letting the caveats swamp the claim.
- The two Edit Index constructions (ray-zone for frame models, legal-set for distribution
  models) share an axis and not a formula. Mark which one a number comes from.

## Current scope decisions (2026-09-10)

- Discworld analysis is on the noiseless instances only (128-ray, 8-ray, blink). The noisy
  `dw-pn04` run is not discussed.
- The tokenized Discworld control and the Othello squared-error head control are appendix
  material and get one sentence each in the main text.
- Only Transformer-L appears. No Transformer-S, no Recurrent-L.
