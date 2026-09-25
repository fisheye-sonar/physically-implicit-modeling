# Verifier report: reviewer

**Status: done.** I read the release as an ICLR reviewer would on first contact. That covered README.md top to bottom, the three notebooks, every script's `--help`, and the modules listed in the brief. I ran what a reviewer would run. I found no blockers: no anonymity leak, and no wrong number in the tables. Four findings are major. Three are real gaps and one is a known issue that is still open:
- the README's rescore recipe silently drops the categorical blocks;
- Othello's `scores.json` field `edit_index` is not the paper's Edit Index, and nothing says so;
- the demo's own documented command fails;
- the Othello appendix figures still disagree with the paper's text (already reported by figures).

Most of the other findings come from one gap. The code's words are never glossed for a reader who knows only the paper: arm, block, bench, basis, guard, zone, gates, floors, instance and Transformer-L. A single "Terms" table in the README fixes most of them.

Work dir: `experiments/release/work/reviewer/`. It holds:
- `tree/`: the release copy with STAGING symlinks;
- `coretree/`: a symlink farm of the 784 core-bundle files only;
- `scratch_scripts/master_eval_dry.py`;
- the extracted table PNGs in `img/`.

I made no writes to RELEASE, to STAGING, or to PRIVATE's `runs/`, `datasets/`, `logs/` or `outputs/`, and ran no git commands.

## Checks run

| check | result | detail |
|---|---|---|
| `--help` on all 25 scripts | pass, 1 exception | `scripts/figures/editability_by_point.py --help` has no argparse, so it runs the whole figure and writes `outputs/figures/appendix/editability_over_res_point.{pdf,png}` |
| `paper_tables.ipynb`, `appendix_tables.ipynb` (README commands) | pass | 2.5 s and 3.7 s. Tables 1 and 2 match the paper cell for cell, including the four daggers. The IM gain over IM-NN is 0.53633 |
| the same notebooks with `CUDA_VISIBLE_DEVICES=""` | pass | text outputs identical to the GPU run, so "the tables need only a CPU" holds |
| the same notebooks on the core bundle only (`coretree/`) | pass | appendix text outputs identical to the full-bundle run |
| `master_eval` as a dry run (`score_all_baselines` and `score_all` with `dry_run=True`), full and core-only | pass | "all baselines present", "all runs scored", empty to-do lists. "On the shipped runs it changes nothing" holds |
| `print_summaries` | runs | labels mislead (see M2) |
| `demos/play.py --driver avoid --save` | pass | GIF written |
| `demos/demo.py` as documented (`--seed 7 --n-objects 4 --fixed-reflectivities`) | **fail** | `RuntimeError: Could not generate a collision-free scene after 300 attempts` (see M3) |
| README voice (em and en dashes, semicolons) | pass | none |
| names, paths, dates and stale-scope scan of code, notebooks and README; forbidden strings in the STAGING JSON | pass | no hits (a quick scan, not a full anonymity audit) |
| `hf download` flag syntax | not run | huggingface_hub is not installed and I did not use the network. This relies on infra's dry run |

## Major

**M1. The README's rescore recipe drops the categorical blocks.**
- **Where:** `README.md:84-86`.
- **The problem:** "To rescore a run, delete its `scores.json` (and its `probes/` to refit them) and run it again." The scorer never fits categorical probes. It only reads them from the cache (`pim/scoring/rayworld.py:66-75`, and the Settings markdown of `master_eval.ipynb`). Once `probes/` is gone, every `appearance-fac` / `appearance` / `grid-*` block is skipped with a single log line. The Table 2 categorical rows, `tab:categorical` and `tab:tokens_*` then cannot be rebuilt from the rescored run.
- **Evidence:** `_fit(model_8ray, "appearance-fac", "frustum", <empty cache>, ...)` prints `appearance-fac: SKIPPED (no cached probes for {...})` and returns `None`.
- **Fix:** after "(and its `probes/` to refit them)", add: "If you delete `probes/`, first refit its categorical probes with that run's lines from step 3 of *From scratch* (`scripts/fit_probes.py`)."

**M2. Nothing tells a reader how to read `scores.json`, and the obvious fields are not the paper's numbers.**
- Othello's top-level `edit_index` is the union construction, not the paper's Edit Index. The paper's is `edit_index_symdiff`, and `unedited.edit_index` is `null`. For example, `jq .best.PI runs/othello/standard/scores.json` gives `edit_index` 0.552 and `edit_index_symdiff` 0.818, while the paper reports +0.82.
- `best` is the top arm without the fidelity cutoff, not the reported setting.
- `fidelity_ratio` is 1 − Edit Fidelity.
- The `master_eval` summary (`pim/scoring/summary.py:18-24`) prints a column `fid` that is the ratio. For `rayworld/8-ray`, IM shows `fid 0.265` against the paper's 0.73.
- The Othello summary (`summary.py:36-37`) lists `EI(un)` first and never marks `EI(sd)` as the reported one.
- **Fixes:**
  - Add a short "Reading `scores.json`" paragraph to the README: "The tables apply `pim.metrics.selection.best_arm` to `arms`. `best` is the top arm without the fidelity cutoff. Edit Fidelity = 1 − `fidelity_ratio`. Othello's Edit Index is `edit_index_symdiff`. `bases` maps each block key (`cartesian`, `frustum`, `appearance-fac`, …) to its scores."
  - In `summary.py`, rename the header `fid` to `ratio` (or print `1 - ratio` as `Fid.`).
  - Relabel `EI(sd)` as `EI (reported)`.

**M3. The demo's documented command fails.**
- **Where:** `scripts/demos/demo.py:4`, shown in `--help`: `python scripts/demos/demo.py --seed 7 --n-objects 4 --fixed-reflectivities [--save outputs/demo.gif]`.
- **The problem:** it raises `RuntimeError: Could not generate a collision-free scene after 300 attempts`. With 4 objects and 100 frames, seeds 0–2, 4–6 and 8–11 work, while 3 and 7 do not. The scripts report tested 4 objects, but not the documented seed.
- **Fix:** change the usage line to `--seed 8 --n-objects 4 --fixed-reflectivities`.

**M4. The Othello appendix figures do not match the paper's text (known issue, still open).**
- **Where:** `scripts/figures/qualitative_othello.py:81` and `qualitative_overview.py`.
- **The problem:** both draw every editor at `best_arm`, with the highest-fidelity fallback. The paper (Additional Qualitative Visualizations) says that where no setting reaches Edit Fidelity 0 (GS on both no-flip variants, IM on `standard-noflip`), the figures show the highest Edit Index setting instead. figures.md and figures-recheck.md reported this.
- **Fix:** edit the paper sentence and its `% figure facts` comment (not a release file).

## Minor

**m1. Internal jargon is never glossed.** A reader of the README or notebooks meets these words with no definition:
- "block" (README:83,86,181; notebook headings "reported blocks");
- "arm" (`editability_by_point.py:3`, `arms.py:1`, `selection.py:1`);
- "bench" (README layout "edit benches and editor sweeps");
- "basis" (`fit_probes.py --basis`, the `master_eval` settings comment);
- "guard" (`within_guard` columns in the appendix output, `selection.py`);
- "zone", "gates" (the Othello summary and `othello/arms.py:1`);
- "floors" for the decodability baselines, where the paper says "baselines" and keeps "floor" for the Bayes floor;
- "instance" against "variant";
- "Transformer-L", which the paper never says (README:232, `train.py --help`).

`zone` (`EditZones`) and `zspace` (`pinv.py`) are glossed in their own modules.

Fix: add a "Terms" table to the README after *Repository layout*:

| term | meaning |
|---|---|
| Transformer-L | the paper's transformer: 8 blocks, width 512, minGPT body |
| instance | a variant's simulator settings and datasets (`8-ray-tokens` trains on the `8-ray` instance) |
| bench | a variant's 1000 edit cases with their ground truth |
| arm | one editor at one setting (residual point, step size) with its scores |
| block | the scores of one probe target in `scores.json` (`cartesian` is the paper's continuous state; `frustum` holds the categorical probes) |
| basis | the coordinates of the continuous Rayworld state (`cartesian`, or `frustum` = (u, 1/y)) |
| guard / cutoff | the selection rule's Edit Fidelity ≥ 0, that is `fidelity_ratio` ≤ 1 |
| zone | rays where the edited and unedited next frames differ (the Edit Index support) |
| gates | held-out next-move checks on Othello (legal mass, cross-entropy against the Bayes floor) |
| floors | the observation and random-init decodability baselines, and the Bayes floor of prediction, in `runs/_baselines/` |
| `appearance-fac` | the appearance target factorized into a center label and a length label per disc |

Also:
- In `paper_tables.ipynb` (cell `collect-md`), write "every run's reported scores" instead of "reported blocks".
- Add one clause to the `arms.py:1` docstring: "an arm is one editor at one residual point and step size".

**m2. The simulator state is called "latent state"**, which is the paper's (and the teaser's) word for the model's z.
- `pim/environments/rayworld/sim.py:1` and `:24`;
- `dataset.py:1`;
- the demo titles `viz.py:118` "2D environment (latent state)" and `scripts/demos/play.py:115` "2D world (latent state)".

Fix: "simulator state".

**m3. Othello pieces are called "disc", "tile" or "square"** where the paper says "token". "Disc" also collides with Rayworld's discs.
- `scripts/two_flip_editability.py:2,5` ("Two-disc", "disc counts");
- `scripts/othello_flip_rates.py:2,42` ("Discs recolored", "discs flipped per move");
- README:188 ("the two-disc edits");
- `pim/figures/tables.py:710` and the `appendix_tables.ipynb` heading ("Two-tile edits").

Fix: "two-token edits" and "tokens flipped per move".

**m4. The GS description does not match the code.**
- **Where:** the `pim/editors/grad_steer.py:3` docstring gives `x' <- x - alpha * dL/dx`.
- **What the code does:** `_descend` runs 100 Adam steps at learning rate `alpha * probe.act_scale`, the median activation SD.
- **The paper:** the Editors section and Implementation Details say "gradient steps of size η", with no mention of Adam or of the relative scale.
- **Fix:**
  - Docstring: "``n_steps`` Adam steps on the activation at learning rate ``alpha`` times the point's activation scale, minimizing the probe loss toward B'."
  - Paper: add one clause to Implementation Details, noting that GS uses Adam and that η is relative to the median activation SD.

**m5. `pim.figures.waterfall_grid` is dead code.**
- **Where:** `pim/figures/waterfall.py:30` onward, re-exported by `pim/figures/__init__.py:6`.
- **The problem:** no script or notebook calls it. `scripts/figures/style.py:26` imports only `DARK_BG`, `DIFF_CMAP` and `EDIT_LINE` from the module. The tables report says it was kept for the history-rewrite script, but that script uses `style.waterfall`.
- **Fix:** delete `waterfall_grid` and its `__init__` export (or move the three constants into `style.py`). Update the `pim/figures/__init__.py:1` docstring to "The paper's tables (``tables``)".

**m6. The Othello flip rates are never shown.**
- **The problem:** Section 3.1 quotes the Othello flip rates (0.27 against 2.2 flipped tokens per move). They are written to `runs/_baselines/othello/<instance>/corpus_stats.json`, but no notebook reads them. README:188-189 says the appendix tables read them.
- **Fix:** either add `T.flip_rates(OTHELLO)` (reading `corpus_stats.json`) to `paper_tables.ipynb`, or change the README sentence to "…and the Othello flip rates quoted in Section 3.1 (`runs/_baselines/othello/<instance>/corpus_stats.json`)".

**m7. `editability_by_point.py` has no `--help`.**
- **Where:** `scripts/figures/editability_by_point.py:69`.
- **The problem:** it is the only runnable script without argparse, so `--help` runs it.
- **Fix:** add `argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()` at the top of `main()`.

**m8. The README understates resource needs.**
- **Disk:** README:107 says "a 128-ray Rayworld corpus takes about 410 GB". Four instances cast 128 rays (standard, blink, smooth, 128-ray), and the eight corpora total about 1.9 TB (20M × 40 frames × float32).
- **Scoring:** the scorer also memory-maps residual stacks of about 22 GB each into `.scratch/` at the repository root.
- **Fix:** "each 128-ray corpus (standard, blink, smooth, 128-ray) takes about 410 GB, and all eight about 1.9 TB. Scoring needs about 25 GB free for residual stacks in `.scratch/`."

**m9. The unfilled URL placeholder disappears on GitHub.**
- **Where:** README:44.
- **The problem:** GitHub strips `<ARTIFACTS_URL>` in prose as an HTML tag, so the sentence renders as "are at . The download…".
- **Fix:** fill the placeholder before release, or write it as `` `<ARTIFACTS_URL>` `` until then. The HF repo id must also belong to an anonymous account, since the id is visible in the URL.

**m10. "Lands" is never defined.**
- **Where:** notebook headings (`paper_tables.ipynb:98`, `appendix_tables.ipynb:110`) and the paper's "every editor that lands".
- **The problem:** the threshold is `tables.LANDS = 0.25` (seed-mean Edit Index), but it is stated nowhere a reader sees.
- **Fix:** append "(an editor lands when its seed-mean Edit Index is at least 0.25)" to both headings, and add the same to the paper's metric-spread appendix.

**m11. The README does not say which paper figures are not generated.** Figures 1–4 (teaser, setup, and the Othello and Rayworld overviews) are illustrations, and their build scripts are not shipped.
- **Fix:** add one sentence under *Figures*: "Figures 1 to 4 are illustrations and are not regenerated."

**m12. The demos are in scope but the README never shows how to run them.**
- **Fix:** add a *Demos* line with `python scripts/demos/demo.py --seed 8 --n-objects 4 --fixed-reflectivities` and `python scripts/demos/play.py`.

## Nits

- **Size-ladder leftover.** `pim/environments/othello/corpus.py:70` has `LADDER = {"D": 20_000_000}`, used at 7 call sites. Replace it with `N_TRAIN = 20_000_000` (no numerics change).
- **British spelling.** "labelled" appears in `grid_target.py:1` and `:30`; use "labeled". The `make_othello_edits.py` recipe string must stay as it is, because it is written into the shipped manifest.
- **`appearance-fac` gloss.** The `grid_target.py:4` gloss "those cells per object" should read "each disc's appearance bin as a center label and a length label (15 and 5 classes on 8-ray)".
- **Unexplained single-value settings.** `bench.py:229-239` (`DIM_SETS = {"all": None}`) and the SETTINGS keys `rw_edit_dims: ("all",)` and `rw_target: "full"` have one value each, with no explanation. Add one comment: "full: positions and velocities, the 8-dim state".
- **Long module docstrings.** Several exceed the spec's 1–4 lines: `othello/reachability.py` (24, a useful algorithm description, keep), `othello/corpus.py` (11), `othello/__init__.py` (10), `scoring/baselines.py` (9), and the figure scripts (10–14, which double as `--help`).
- **Opaque probe files.** Shipped `probes/` directories have no `INDEX.md`, and the file names are hashes. Regenerate it with `ProbeCache(dir).write_index()` during export, or mention it in the README.
- **Tables render only as PNG.** Their `text/plain` output is `<tab:…: N rows>`. The notebook title markdown could say that `T.table_x(F).values` / `.text` give the numbers as a DataFrame.
- **Notebook metadata.** `language_info` has no `file_extension`, so `nbconvert --to script` writes `.txt`.
- **Residual point vs layer.** The paper says "residual point", while the code and settings mix in "layer" (`GS_LAYERS`, `gs_layers`, "GS@L0", "layer by layer" in `grad_steer.py:1`). A comment on `GS_LAYERS` would help: "GS start points (residual points)".
- **Opaque `--help` terms.** `bayes_floor.py --help` says "MH sweeps"; spell out Metropolis-Hastings. `probe_refit_variance.py --im-points canonical` is also opaque.
- **README setup details.** State the Python requirement before the commands. Note that `sha256sum` is `shasum -a 256` on macOS, and that the figure scripts use up to about 8 GB of GPU memory.

## Requests (exact, by owner)

- **infra (README.md):** M1, M2 (paragraph), m1 (Terms table), m3 (README:188), m6 (README:188-189), m8, m9, m11, m12, and the README nits.
- **scoring (`pim/scoring/summary.py`, `master_eval.ipynb`):** M2 labels, and in m1 the `master_eval` settings comment "basis" becomes "coordinates".
- **scripts (`scripts/demos/demo.py:4`, `two_flip_editability.py`, `othello_flip_rates.py`):** M3 and m3.
- **tables (`pim/figures/*`, both table notebooks, `editability_by_point.py`):** m1 (notebook headings), m3 (`tables.py:710` and the heading), m5, m6 (optional flip-rate cell), m7, m10.
- **core (`pim/editors/grad_steer.py`):** m4 docstring.
- **rayworld-env (`sim.py`, `dataset.py`, `viz.py`, `grid_target.py`, `bench.py`):** m2, `arms.py:1` (m1), and the nits.
- **othello-env (`corpus.py`):** the `LADDER` nit.
- **paper:** M4, m4, m10.
