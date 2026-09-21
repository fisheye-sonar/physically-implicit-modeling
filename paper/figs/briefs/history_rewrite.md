# BRIEF — a clean paper figure for history rewriting (appendix §"Persistent Transformer Edits with History Rewriting")

Read `paper/figs/FIGURE_BRIEF_COMMON.md` first. Folder: `paper/figs/history_rewrite/` (existing). GPU for about a
minute (one model; the lab GPU is shared — one model at a time, free it after).

## What exists
`make_figure.py` computes everything: the counterfactual history integrated backwards, the sim's clean render of it
(`cf_clean`), the rewritten history (`obs_cf`, the inverse map written at every step), and four rollouts on the first
32 canonical bench cases of `noise_ablation/L-dw-noiseless-20m`: `unsteered`, `IM` (one write at EF−1), `hist`
(rewritten history, no write), `hist+IM` (rewritten history plus the write). It draws three internal waterfalls with
`pim.figures.waterfall_grid` (dark page, verbose titles) — those are NOT paper style. Read it fully first.

## Do
1. **Split compute from drawing with minimal change.** Make `make_figure.py` also save its arrays to
   `.scratch/history_rewrite_arrays.npz` (rolls, `obs_cf`, `cf_clean`, `b.obs[:, :EF]`, `b.gt_roll`, target /
   ghost x per case, the edited object and `s_post`, plus the per-arm cards as JSON) — a few lines, nothing else
   changes; keep its existing outputs. Then write `draw_paper.py` that reads the npz and draws the paper figure.
   Compute no metric in the drawing script (the cards already hold the Edit Index / fidelity; they go in the README,
   not the figure).
2. **The figure** (Sevan): three rows = three edits. Columns: **Ground truth** (the edited world: above the edit-frame
   line the clean counterfactual history `cf_clean` for the last 6–8 frames, below it `gt_roll`); **Unedited**
   (original observed frames above, the unsteered rollout below); **Single-point edit** (original observed frames
   above, `IM` rollout below); **History rewrite** (the REWRITTEN frames `obs_cf` above the line, `hist+IM` below —
   also export a variant with `hist` alone below). Time downward; one edit-frame line per panel; cyan / pink
   locators (`ps.ORIGIN_C` / `ps.DEST_C`) at the edited disc's origin and destination rays. Draw the panels as
   `_panel` in `paper/figs/qualitative_edits/make_figure.py` does (gray on the dark panel, nearest, thin frame;
   white page; short column labels only, no metrics in the figure).
3. **Row selection rule**: cases with a big, visible edit — rank the 32 by the teleport displacement in rays
   (|target_x − ghost_x|) and take the top three, or random among those above the median; state the rule in the
   README and record case indices in a sidecar JSON. Also produce a random-3 variant (seed 0).
4. **Sizes**: full width 5.5 in, four columns; try K = 15 rollout steps (what the rolls hold) and a version cropped to
   the first 10 steps if the panels get too tall. Every observation strip 128 rays wide, fixed 0–1 gray.
5. `README.md` update: the new files, the rule, the arms' Edit Index / fidelity (from the saved cards) for the caption.

## Stretch (only after the above is verified)
A predictive-quality waterfall for the appendix: the model's free-run against the ORIGINAL world's clean render on a
few held-out sequences (no edit). The bench dict's arrays / `layout.edits_file` HDF5 (`obs_intensity` and a clean
render) may give you both without new logic; if a clean unedited render is not already available from `pim`, skip and
say so. Same panel style, three rows, GT | prediction, for the standard model only.
