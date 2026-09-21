# Common brief for every paper-figure worker (2026-09-21)

You are a WORKER. You build ONE figure folder under `paper/figs/` and report. You do not orchestrate.

## Read first
1. `harness/WORKER.md`, `CLAUDE.md`, this file, your folder's `BRIEF.md`.
2. `paper/figs/paper_style.py` — import it first in every script (`ps.apply()`, `ps.save()`); it fixes fonts, sizes, colours.
3. `harness/STYLE.md` §3 (figure mechanics) and §6 (aesthetic). §1–2 are for internal notebooks; the paper's
   text rules below override them where they conflict (no metrics in titles, no figure-top legend unless needed).
4. The paper draft `paper/paper_draft.tex` — the sections your figure serves (your brief names them). The figure
   must match what the text says and what the code does (`research/REGISTRY.md` indexes the canonical objects).
5. Existing paper-figure scripts are the reference for HOW things are drawn here:
   `paper/figs/qualitative_edits/make_figure.py` (ray-world strips: `_panel`, `error`, locators, page layout),
   `paper/figs/qualitative_edits_othello/make_figure.py` (`draw_board`), `paper/figs/history_rewrite/make_figure.py`.
   Import from them or copy the few lines you need; do not re-implement them differently.

## Sevan's rules for this work (binding)
- **Minimal code.** A few clean scripts in your folder, nothing else. Go as far as you like on styling and
  aesthetics, but implement no long or complex operations. **Compute no metric**, re-implement no rendering,
  no environment rule, no probe, no editor. Every canonical quantity is an import from `pim.*`
  (`pim.environments.*` renderers / rules / benches, `pim.metrics.selection.best_arm`,
  `pim.figures.tables.collect`, …). If a figure would need new logic, stop and say so in the report.
- **Text minimal.** Short column / row labels only. No titles, no explanatory sentences, no results, no run
  codes, no file names inside a figure. Captions carry the details (put what the caption must say in your README).
  You tend to over-write and over-explain inside figures: do not.
- **One aesthetic.** White page, black text, Times New Roman (via `paper_style`), Okabe-Ito colours,
  `ps.EDITOR_COLORS` for PI / GS / IM / ND everywhere an editor is coloured. Raw 1-D observations are drawn exactly
  as the canonical waterfall draws them: `gray` colormap on the dark panel background
  (`pim.figures.waterfall.DARK_BG`), fixed 0–1 range, nearest interpolation, a thin `ps.FRAME` border, white page.
  Othello boards are drawn as `draw_board` draws them (board green, black / white discs, yellow tint, pink edited tile).
  Locators: `ps.ORIGIN_C` (cyan, where the edited disc was) and `ps.DEST_C` (pink, where it was moved to).
- **Paper-ready.** Vector PDF (text as TrueType, rasters only for observation strips / waterfalls, drawn with
  `interpolation="nearest"`), plus a 300-dpi PNG preview beside it. Design at the final printed width
  (ICLR text width 5.5 in; half width 2.65 in) with 8–9 pt text; never below 7 pt on the page. Every axis that
  exists is labelled with the exact quantity; every line drawn has a legend entry; legends are a few words.
  Colour-blind safe. No ALL-CAPS. No em dashes in any text you write.
- **Pieces AND a composite.** Sevan assembles the final figure himself from high-resolution pieces, so export
  every element as its own PDF (each board, each strip, each panel, legend keys) AND make your best attempt
  at the full composed figure. Where the brief asks for options, produce each as a separate file.
- **Selection rules are stated.** Any example you show is chosen by a rule written in the README and a sidecar
  JSON (case ids, seeds, arms, instance, run). Random with a seed is the default; an editorial rule ("the cases
  whose edit changes the most rays") is allowed for a main-text figure when the README says so.

## Machine rules
- The lab GPU is shared with a running training job of the paper queue. Load one model at a time, free it
  (`del model; torch.cuda.empty_cache()`), keep batches modest. CPU where the brief says CPU.
- `runs/`, `datasets/`, `logs/` are READ-ONLY for you. Never run `master_eval.ipynb` or any scorer, never write
  into a run directory, never `pkill -f` anything. Caches go to `<repo>/.scratch/` (already used by the figure scripts).
- Write only inside your figure folder (and `.scratch/`). Do not edit `pim/`, the paper `.tex`, or other figure folders.
- Run scripts with the venv interpreter: `.pim/bin/python paper/figs/<folder>/<script>.py`.

## Verify before reporting
Open every PNG you produced with the Read tool and look: fonts serif? text legible at print size? nothing
clipped or overlapping? fixed intensity scale on observation panels? locators / event marked where relevant?
Fix and re-render. Then write `README.md` in your folder: what each output file is, how to regenerate it, the
data / cases / seeds / arms behind it, the selection rule, and the caption facts Sevan will need.

## Report (hard requirement)
Return a tight report: the files (relative paths), what each option shows and your recommendation among them,
the selection rule, any number you placed in a figure with its source, and what you could not do and why.
