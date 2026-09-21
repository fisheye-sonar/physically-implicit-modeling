# predictive_quality — the trained models predict well before any edit (appendix)

Two figures, one per environment. Neither script loads a model: both read caches that other figure scripts
computed. Every PDF is vector (Arial embedded as TrueType; the only rasters are the Rayworld observation
strips, nearest interpolation at 600 ppi so all 128 rays survive), cropped to the content with no outer
padding (`paper_style.save`), with a 300-dpi PNG preview beside it. No number appears inside either figure;
the only text is the row labels, the axis words and the key.

| file | what it shows |
|---|---|
| `rayworld.{pdf,png}` | Rayworld, `runs/noise_ablation/L-dw-noiseless-20m` (dw-noiseless): two rows, Ground truth (the simulator's clean render of the UNEDITED world continuing) and Prediction (the model's free-run), by three held-out sequences as columns; in every panel the last 8 observed frames above the dashed line and the 15 free-run steps below it, time downward, 128 rays wide; 5.40 x 2.29 in |
| `rayworld.json` | sidecar: run, instance, the cases and their rule, what each row draws, the source arrays |
| `othello.{pdf,png}` | Othello, one board per rule variant, (a) standard, (b) adjacent-flip, (c) adjacent-noflip, (d) standard-noflip: the board tinted by its true legal set ("legal moves": uniform over the legal squares, the Bayes-optimal target) over the same board tinted by the trained model's unedited next-move distribution ("model"); 5.55 x 2.54 in |
| `othello.json` | sidecar: cache column, run, instance, bench case id, board length, legal-set size, tint parameters |
| `pieces/rayworld_case<idx>_{gt,pred}.pdf`, `pieces/rayworld_key.pdf` | every Rayworld panel (1.61 x 0.97 in) and the key |
| `pieces/predictive_<variant>_{legal,model,pair}.pdf` | every Othello board singly and as a legal / model pair (PNG previews beside them are gitignored) |

## Rayworld — `rayworld.py`

`.pim/bin/python paper/figs/predictive_quality/rayworld.py` (CPU, 3 s). Reads the SAME array file as
`paper/figs/history_rewrite/draw_paper.py`: `.scratch/history_rewrite_arrays.npz`, written on the GPU by
`paper/figs/history_rewrite/make_figure.py` for the 32 canonical bench cases of the run (if the file is
missing, run that script first). Arrays used: `obs_hist` the observed frames 0..19, `gt_unedited_roll` the
simulator's clean unedited continuation over frames 20..34 (the scorer's ghost trajectory), `roll_unsteered`
the model's free-run from frame 20 with no edit. The panel code (`waterfall`, the key, the geometry) is
imported by path from `draw_paper.py`, so the strips are drawn exactly as the history-rewrite figure draws
them: `gray` on the dark panel, fixed 0-1 range, nearest, thin frame, white page. Layout transposed to be wider
than tall: rows = Ground truth / Prediction, columns = the three sequences.

**Cases.** `numpy.random.default_rng(0).choice(32, 3, replace=False)`, sorted: bench cases **16, 19, 25**.
No edit is involved, so no displacement filter (the history-rewrite figure filters small edits; this one does
not need to). dw-noiseless has no observation noise, so the observed frames above the line equal the clean
render. Caption: the model's free-run tracks the clean unedited world over 15 steps on held-out sequences;
the run's prediction-skill numbers for the text come from its `scores.json` (`master_eval`), not from this
figure.

## Othello — `othello.py`

`.pim/bin/python paper/figs/predictive_quality/othello.py` (CPU; `--seed`, default 0). Moved 2026-09-21 from
`paper/figs/environments_overview/othello/make_predictive.py` (outputs renamed `predictive_composite.*` to
`othello.*` and `predictive.json` to `othello.json`; the pieces kept their names). One random case per
variant (`rng.integers` over the 1000 cached bench cases, seed 0) from `.scratch/othello_edits_guarded_cache.pkl`,
the qualitative Othello figure's cache (guarded best arms, 2026-09-19): `board_pre`, `legal_pre`,
`probs["Unedited"]` per variant. Boards are the bench's 20-move positions; the "legal moves" row is
`pim.metrics.set_editability.uniform_over_legal(legal_pre)`, the "model" row the cached `probs["Unedited"]`.
Squares are tinted exactly as `draw_board` tints them (full at 0.02 probability mass and above,
`(p / 0.02) ** 0.6` below). The board drawing and page layout are loaded by path from
`paper/figs/environments_overview/othello/make_figure.py`, which itself loads `draw_board` from
`paper/figs/qualitative_edits_othello/make_figure.py`.

| panel | run | bench case | legal squares |
|---|---|---|---|
| (a) | `initial_othello_comparison/L-oth-20m` | 850 | 18 |
| (b) | `adjacent_flip_ablation/L-oth-adjacent-flip-20m` | 636 | 24 |
| (c) | `adjacency_ablation/L-oth-adjacent-20m` | 511 | 19 |
| (d) | `flip_ablation/L-oth-noflip-20m` | 269 | 8 |

The two rows look identical at the canonical tint saturation (0.02): a model that puts 1/|L| on every legal
square looks exactly like the legal-set board, which is the intended message.
