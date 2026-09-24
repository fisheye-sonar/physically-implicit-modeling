Write like a man. Mimic my style. 

Use transitions, setups, flow. Engage the reader. Make it less dry. Make it less technical and jargon heavy. Introduce ideas before diving into them.

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
- On 2026-09-23 the main body stood at about 10.25 pages (Sevan's Overleaf count), with about
  5,100 words of prose, 2,100 of them in Experimental Setup. Cutting to 9 is ongoing work, so
  an edit to a main-text section should not grow it.
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
  shorten row labels and headers before anything else, and never rely on `\resizebox`. Long
  unbreakable tokens such as file paths do not belong in a `\parbox` or a table cell.
- Tables use `tabular` and `\hline` with the preamble's heat column types: `P` (Probe Skill,
  0 to 1), `E` (Edit Index, −1 to +1), `F` (Edit Fidelity, −0.25 to 1) and `G` (a gap column).
  A cell whose seed standard deviation exceeds 0.1 carries `\dg` after its number (`-0.23\dg`),
  which the heat macros parse. `\multirow`, `\shortstack` and `\rotatebox` are already in use.
- Every number in a table must be traceable to a `scores.json`, `baselines.json`, or a findings
  file in `research/findings/`. Put the source in a `% source:` comment above the table.

## Accuracy

- Everything written about the method must match the code in `pim/`. Check the implementation
  before describing an editor, probe, metric, or environment rule. `research/REGISTRY.md` is
  the index of canonical objects.
- Every quoted number comes from the canonical scores (`runs/<topic>/<run>/scores.json`,
  written by `notebooks/master_eval.ipynb`), the floors (`runs/_baselines/`), or a dated entry
  in `research/findings/`. Never quote an Edit Index without its Edit Fidelity.
- Read table values through `pim.figures.tables.collect` (the Rayworld basis is `cartesian`),
  never by picking arms out of `scores.json` by hand. `scores.json["best"]` is the scorer's own
  unguarded argmax and is not what the paper reports.
- Edit Fidelity is 1 minus the stored RMSE ratio (`pim.metrics.fidelity`): 1 reproduces the
  edited world, 0 is no better than leaving the model alone, below 0 is worse. `scores.json`
  still stores the ratio itself (`fidelity_ratio`, lower is better).
- The reported setting of an editor (`pim.metrics.selection.best_arm`, 2026-09-23) is its
  highest Edit Index among settings with Edit Fidelity ≥ 0. Where none qualifies, it is the
  setting with the highest Edit Fidelity. The appendix alternative selects on the highest
  Edit Fidelity outright (`collect(..., select="fidelity")`).
- The main tables are one training run per variant at 780k steps. The seed spread comes from
  two further seeds at 512k (n = 3; `standard-noflip` reached n = 3 last, on 2026-09-23), read from
  `experiments/paper_ci/dashboard/ledger.md` or `tables.collect(...).rep_sd`.
- State scope honestly where it matters (one seed per cell, one instance per variant, the
  number of cases) without letting the caveats swamp the claim.
- The two Edit Index constructions (ray-zone for frame models, legal-set for distribution
  models) share an axis and not a formula. Mark which one a number comes from.

## Current scope (2026-09-23)

The paper calls the moving-disc world **Rayworld**. The code, runs and findings call it
**discworld** (`dw-*`). The paper's variant names map to runs as follows.

| Paper | Run |
|---|---|
| Othello `standard` | `runs/initial_othello_comparison/L-oth-20m` |
| Othello `adjacent-flip` | `runs/adjacent_flip_ablation/L-oth-adjacent-flip-20m` |
| Othello `adjacent-noflip` | `runs/adjacency_ablation/L-oth-adjacent-20m` |
| Othello `standard-noflip` | `runs/flip_ablation/L-oth-noflip-20m` |
| Rayworld `standard` (128 rays) | `runs/noise_ablation/L-dw-noiseless-20m` |
| Rayworld `blink` | `runs/blink_ablation/L-dw-blink-20m` |
| Rayworld `128-ray` / `16-ray` / `8-ray` / `5-ray` (big discs) | `runs/ray_ablation/L-dw-{128,16,8,5}ray-20m` |
| Appendix: `smooth`, `obs5` | `runs/smooth_ablation/L-dw-smooth-20m`, `runs/observer_ablation/L-dw-8ray-obs5-20m` |
| Appendix: `8-ray` tokens | `runs/interface_ablation/L-dw-8ray-tok-20m` |

- Rayworld has two state targets: the continuous Cartesian position and velocity
  (`cartesian` block), and the categorical appearance target (`appearance-fac` block). The
  categorical IM is the categorical inverse map, which exists only on the N-ray family.
- The main text covers the ten variants above. The appendix holds `smooth` and `obs5`, the
  tokenized `8-ray` model, the grid targets, the nearest-neighbor control, legal against
  illegal Othello targets, history rewriting, and the fidelity-selected table.
- Not discussed: the noisy `dw-pn04` run, the original `L-dw-20m`, Transformer-S and
  Recurrent-L. Only Transformer-L appears. The Othello squared-error head control is not in
  the current draft.
