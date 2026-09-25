# Verifier report: tables

**Verdict: every table reproduces; one quoted number cannot be reproduced.** Every table cell of the paper is reproduced by the release, except the two known and accepted `tab:im_by_point` cells. The two known in-text mismatches are confirmed, and no new numeric mismatch was found in any table or table-derived number.
- One major gap: the Othello reconstruction-test ratios (paper line 825) have no producer in the release (finding 10).
- The rest are minor issues and nits: two inaccurate prose claims in the paper, notebook output that the edited paper no longer quotes, and two unused shipped artifacts.

Compared against the paper **as it is now on disk**. `paper/paper_draft.tex` has uncommitted edits made at 23:26 on 09-23, after the tables builder's check script was written at 23:18. Among them: "seven" became "six" SDs, the token-table dagger became `*`, and the "49 of 1000" and "480 searched" phrases were removed.

Work dir: `experiments/release/work/tables/verifier_v1/`. Nothing was written to RELEASE or STAGING (checked with `find -newer` and strace).
- `tree/` holds copies of `pim/`, `scripts/` and `notebooks/`, with `runs/` and `datasets/` symlinked into STAGING.
- `tree_core/` has only the core-bundle files, linked per the README's `--exclude` globs: 784 files, 2.85 GB.
- My own code: `paper_parse.py` (LaTeX tabular parser), `verify_tables.py` (cells), `verify_numbers.py` (in-text numbers) and `compare_private.py` (PRIVATE against STAGING, JSON only). I did not reuse the builder's `check_tables.py`.
- Logs and executed notebooks are in `logs/` and `executed/`; the rendered tables are in `png/`.

## What I checked, with evidence

1. **master_eval is a no-op.**
   - I executed a copy under `strace -f` (the kernel included): exit 0, no error or stderr output.
   - It printed 12/12 `skip baselines` and "all baselines present", then 43/43 `skip <run> (scored at 1.0)` and "all runs scored". Both return `[]`.
   - Every one of the 214 opens under `runs/` and `datasets/` was `O_RDONLY`. There were no write-mode opens, mkdirs, renames or unlinks outside `/tmp`, `/dev` and the output notebook.
   - No STAGING file is newer than the start marker, and `pim` was imported from my tree.
2. **Both table notebooks execute.**
   - Command: `nbconvert --execute`. Exit 0, with no error or stderr output in either notebook.
   - The README commands, run verbatim (no timeout flag), with `CUDA_VISIBLE_DEVICES=""`, on the **core-bundle-only** tree: 3.5 s and 4.5 s. Every text output and PNG is identical to the full-tree run. The README's "core bundle suffices for every table; CPU only" holds.
3. **Every printed cell** (`verify_tables.py`): **871 cells** across all 13 tables and 14 panels, including every ± part, `--`, n and ×10⁻³ scaling. Each cell was checked three ways:
   - (a) the paper's LaTeX, parsed by my own parser with comment lines dropped;
   - (b) the release `Table.text`;
   - (c) my independent recomputation from the raw shipped JSON. This has its own selection rule written from the paper's text, its own SDs (`np.std(ddof=1)` over the three `__seed*` members, each asserted at 512,000 steps), and its own floor and Bayes-floor arithmetic, without `pim.figures`.

   **Results:** release vs paper, 2 mismatches; raw vs paper, 2 mismatches; release vs raw, **0**. The two mismatches are the accepted `tab:im_by_point` Othello IM Index cells:
   - point 3: 0.25492 prints +0.25, paper +0.26;
   - point 6: 0.38495 prints +0.38, paper +0.39.

   PRIVATE `runs/initial_othello_comparison/L-oth-20m/scores.json` holds the same 0.25492092790460197 and 0.3849462111839557. So these are stale paper cells, not an export change.
4. **Daggers.**
   - `tab:editability`: paper = release = my SD > 0.1 set, 4 cells: adjflip PI EI and IM fid; adjacent-noflip PI EI and IM EI.
   - `tab:seed_spread`: paper = mine, the same 4 cells; they sit exactly where the Table 2 daggers are.
   - The nearest non-dagger SD is 0.085, so no cell sits at the boundary.
5. **PRIVATE vs STAGING** (`compare_private.py`): 126 JSON files covering 43 runs, `_baselines/{baselines,bayes_floor,reachability}.json` and the analysis JSONs. That is 349,490 numeric leaves, 0 differences and 0 arms missing; only the removed ND arms are absent from STAGING.
6. **Notebook structure.**
   - All three notebooks pass `nbformat.validate` (v4.5) with unique cell ids. Kernelspec is `python3`, there are 0 outputs, every execution_count is null and cell metadata is empty.
   - Each paper table has exactly one producing cell: `paper_tables` cells 5 and 7; `appendix_tables` cells 3, 8, 27, 29, 33, 34, 39, 41, 43, 45, 49. That is exactly the paper's appendix order, with no table produced twice.
7. **In-text numbers** (`verify_numbers.py`, own computations). The following all match:
   - Results and seed spread:
     - 53.6% (mean of IM/IM-NN − 1);
     - SD maxima 0.0037 and 0.0344; the same maximum for any "lands" threshold from 0.2 to 0.5;
     - four cells > 0.1;
     - 2.484;
     - IM steps: minimum 6.10 combined SDs, 7.39 in larger-SD units;
     - fixed-setting SDs 0.071 / 0.050 / 0.317;
     - one adjacent-noflip IM seed with no arm inside the cutoff;
     - adjflip IM at point 5 on all seeds, 0.623 ± 0.028, fidelity 0.40–0.59;
     - seed-mean vs main differences 0.0449 (EI) and 0.2563 (fid);
     - Othello standard PI at α 5 on every seed vs 3, fidelity 0.478 vs 0.701.
   - Refits and prediction:
     - probe refits 0.00073 / 0.0160 / 0.0042 / 0.0362, with 6 and 10 seeds;
     - gap closed ≥ 0.9914.
   - Decodability paragraph:
     - MLP skill ≥ 0.8832;
     - random-init MLP within 0.028 of trained except blink (0.109);
     - standard-noflip ≥ 0.99993 on every probe.
   - Editability and selection:
     - adjflip PI/GS at most +0.348;
     - fidelity rule: 18 steps smaller, 9 the same, 1 larger;
     - fidelity-rule rises in IM (continuous) and in PI and GS (categorical).
   - Reachability, two-flip and tokens:
     - reachability counts 441/555/4, 335/614/51, 0/1000/0 twice;
     - two-flip SE 0.113;
     - vocab_size 422;
     - token probe gap 0.0127.
   - Additional variants:
     - smooth gain 0.347, the largest;
     - smooth/obs5 PI and GS from −0.189 to +0.020 at fidelity 0.02.
   - Residual points and setup:
     - residual-point prose (PI and GS peak at point 4, fail from 6 on, fail at every Rayworld point);
     - PI at α=1: `readout_err_after` = 0 at points 1–8;
     - 780k steps, batch 256, 7.9–12.1 h and 19.2–25.7 h.
8. **Accepted mismatch confirmed:** "IM moves by at most 0.10": raw 0.1063 (adjflip, +0.664 → +0.558). The release prints 0.1063, and its header says "about 0.1".
9. **Figure and hygiene.**
   - `scripts/figures/editability_by_point.py` runs on the core tree on CPU. Its PNG is byte-identical to the style-fix render.
   - The writing-rule and anonymity grep over `pim/figures`, the three notebooks and that script is clean.
   - I viewed the rendered PNGs for Table 2, seed spread, predictive and categorical; the layout and daggers are correct.

## Findings

**minor 1: the paper overstates the seed-selection story.** `paper/paper_draft.tex:467` says "their spread comes mostly from selecting a different setting on each seed".
- Evidence (my raw arms; `T.dagger_cells` agrees):
  - adjflip PI seeds: pt2 α5, pt2 α5, pt1 α10;
  - adjacent-noflip PI seeds: pt1 α10, pt1 α10, pt1 α20;
  - adjacent-noflip IM seeds: pt2, pt0, pt1.
- So only IM picks a different setting on every seed; for each PI cell, one seed out of three moves. The builder's report row "three selection daggers | a different setting on each seed | same" overstates this.
- Fix: "...comes mostly from the selected setting changing between seeds rather than from the models."

**minor 2: "both probe-derived editors rise steadily" is not true for GS.** `paper/paper_draft.tex:348`.
- Categorical GS goes +0.308 (128-ray) → +0.280 (16-ray) → +0.458 → +0.598. The seed means also fall, +0.286 → +0.255, with seed SDs 0.014 and 0.006, about 2 combined SDs. PI does rise steadily (−0.31, −0.06, +0.38, +0.51).
- Fix: "...as the rays coarsen PI rises steadily and GS rises from 16 rays on, to +0.51 and +0.60 at 5 rays."

**minor 3: the notebooks print and announce numbers the edited paper no longer quotes.**
- Appendix cell `tokens-numbers-md` (`notebooks/appendix_tables.ipynb:330`) says "...; 49 of 1000 categorical cases removed". `T.tokens_numbers` (`pim/figures/tables.py:779`) prints "categorical cases kept 951". The paper sentence "which removes 49 of the categorical target's 1000 cases" was deleted; 49 and 951 now appear only in LaTeX comments.
- `T.two_flip_numbers` (`tables.py:720`) prints `partners_searched`: 54/104/4, 46/89/0 and 0/480/0. The caption's "(none of the 480 searched)" was deleted, and only "SE reach 0.11" remains quoted.
- Fix, either way round:
  - drop "; 49 of 1000 categorical cases removed" from the header and the `"categorical cases kept"` entry from `tokens_numbers`;
  - drop `**d["partners_searched"]` from `two_flip_numbers`, keeping `n_cases` and `max Edit Index SE`;
  - or restore the two phrases in the paper.

**minor 4: two shipped `variance.json` files are read by nothing and produced by no documented command.**
- The files are STAGING `runs/othello/adjacent-flip/variance.json` and `runs/rayworld/standard/variance.json` (`full` with 20 seeds, `appearance-fac` with 6).
- `T.probe_refit_spread` reads only othello/standard, adjacent-flip__seed0/1/2 and rayworld/8-ray.
- The README refit loop runs `rayworld/8-ray othello/standard othello/adjacent-flip__seed{0,1,2}`. The rayworld/8-ray `full` entry is regenerated but not quoted, which is harmless.
- Fix (export): leave these two files out of the bundle, or add the two runs to the README loop.

**nit 5: "at least six standard deviations" holds only under one reading.** `paper/paper_draft.tex:467`; header `steps-md` (`appendix_tables.ipynb:128`).
- The claim holds with seed means and the combined SD √(s₁²+s₂²): minimum 6.10, at 16→8 continuous.
- With the 780k main-run values, the 16→8 step is 0.051, which is 5.4 combined SDs. In units of the larger SD it is 7.39 on seed means.
- Fix: say "at least six combined seed SDs between the seed means" in the paper or the header.

**nit 6: some quoted derived numbers have no producing cell.** The appendix header promises "each followed by the numbers the text quotes from it" (`appendix_tables.ipynb:10`), but these come only from arithmetic on table cells:
- "trails IM ... by 0.17 to 0.26" (IM − IM-NN; `im_vs_nn_gain` prints ratios only);
- "within 0.03 ... except blink" (trained minus random-init MLP);
- "the largest gain of any Rayworld variant" (trained minus random-init linear).

All of them are correct. Fix: add an `IM - IM-NN` column to `im_vs_nn_gain`, or soften the header.

**nit 7: "fit on about 1.2M pairs" counts the held-out rows.** `paper/paper_draft.tex:202` and `:923`.
- Stored `n_train_rows` is 943,478 for Othello and 936,000 for Rayworld; 1.18M includes the 20% held-out rows (235,998 on Othello).
- Fix: "about 1.2M pairs, 80% of them for fitting".

**nit 8: row labels differ cosmetically from the paper.** The check matches rows by order and variant name.
- `tab:categorical`: "Appearance (30)" vs "Appearance (30 bins)"; "Appearance, factorized (30)" vs "Appearance, factorized".
- `tab:im_vs_nn` and `tab:fidelity_selected`: "standard (128 rays)" and "blink (128 rays)" vs "standard" and "blink".
- `tab:predictive_skill`: "128-ray (big discs)" vs "128-ray"; "Rayworld (8-ray-tokens)" vs "(8-ray tokens)".
- Fix: pass per-table label maps, or accept as is.

**nit 9: the master_eval summaries use a different arm than the tables, without saying so in the notebook.**
- `print_summaries` shows the unguarded top arm by union Edit Index. For example, adjacent-flip IM appears at pt4 with EI(sd) +0.558, while Table 2 reports +0.66.
- The docstring in `pim/scoring/summary.py` says so, but the notebook's `summaries-md` cell (`master_eval.ipynb:84`) does not.
- Fix: header "## Summaries (each editor's top arm, before the fidelity cutoff; the tables apply the selection rule)".

**major 10: the release cannot reproduce the Othello inverse-map reconstruction test.** `paper/paper_draft.tex:825` says: "Overwriting the latent state with g of the unedited board ... keeps the output's error within about twice the model's own only at points 4 and 5, and raises it to five to sixteen times the model's own elsewhere."
- No release file produces these ratios. `grep -rni reconstruct pim scripts notebooks README.md` finds only the Rayworld `reconstruct_clean_obs` helper.
- No shipped artifact stores them. The LaTeX comment at line 822 gives the private source, `research/findings/inverse-probe.md`.
- SPEC's list of in-scope analyses does not name this test, so the builders were not asked for it. Under the rubric it is still a paper number that the release cannot reproduce.
- Fix, one of:
  - (a) add a small `scripts/im_reconstruction.py`: for each point p on `othello/standard`, write `g(pre-edit board)` at p with no edit and report the error against the pre-edit legal set divided by the model's own error;
  - (b) or cut the two ratios from the sentence and keep the qualitative claim, which `tab:im_by_point`'s fidelity column supports.

## Not covered

- The history-rewriting numbers (figures verifier).
- Flip rates, checked in passing: STAGING `_baselines/othello/{standard,adjacent-flip}/corpus_stats.json` holds 2.2449 and 0.2687 flips per move, which matches the paper's 2.2 and 0.27.
