# blink_ablation/waterfalls — edit panels selected BY BLINK PHASE (2026-09-08)

**Why.** The canonical panel (`runs/blink_ablation/L-dw-blink-20m/figures/waterfall_edits.png`)
draws the first 192 edit cases, which are ~79% visible / ~18% mid-blackout / ~3%
reappearance, so it shows whatever phase a case happens to be in; and it hides the 0.5
blink markers, which sit on ray 0 and ray 127 of a 128-wide strip drawn ~120 px wide and
therefore fall under the axis spine. Both are drawing problems. The data is correct: the
markers are stored (`obs_id` code −2−j) and reconstructed into `clean_obs`, and no
sequence is hidden before frame 3 (0 hidden frames among frames 0–2 in 10,000 test
sequences).

**What this adds.** `scripts/reappearance_waterfalls.py` selects cases by blink phase with
the same rule as `../scripts/subset_editability.py`, in index order (no cherry-picking),
and draws the two edge rays `--edge-width` (default 4) pixels wide inside grey fences —
the identical transform in every column, so a marker the model failed to predict stays
absent. Context is 10 frames rather than the canonical 6 so the blackout START marker is
usually on screen alongside the END marker at frame 19. Rollouts, zones and Edit Index are
the canonical `pim` ones; drawing is `pim.figures.waterfall_grid` per the spec.

    .pim/bin/python experiments/blink_ablation/waterfalls/scripts/reappearance_waterfalls.py
    # --subsets reappearance visible mid_blackout · --n 6 · --min-k 3 · --guard 1.1

**Arms.** Each editor is drawn at the arm the subset table reports for that subset, subject
to a fidelity guard (default ≤ 1.1) — the raw best PI arm on this model is a destructive
α=175 write and shows only damage. The 192-case Edit Index of the drawn arm is in the
column title next to the mean over the drawn rows, because six rows are examples, not
evidence: on the reappearance panel the drawn rows give PI −0.00 / GS +0.25 while the same
arms over the full subset give PI +0.23 / GS −0.04.

**Outputs.** `outputs/waterfall_reappearance.png` (6 of 453 cases, staleness k ≥ 3) and
`outputs/waterfall_visible.png` (the control, same arms).

**How to read them.**
- *Reappearance rows have no ghost locator, by construction*: the edited object was hidden
  at frame 19, so it vacates no rays. The green target is where it must reappear.
- *The GT column shows the reappearance*: a disc appears at the green line on the first
  frame below the dashed line. The unsteered column mostly does not put it there (unedited
  −0.85 on the subset) — the edit asks the model to reappear it somewhere else.
- *The model does not predict markers in free-run, and should not.* On 512 test sequences
  the model's value on a marker ray at a true marker frame averages 0.07 against a GT 0.5,
  landing within 0.15 of 0.5 on 8–9% of them, versus 0.03 and 0.8% on non-marker frames.
  A blackout START is a coin flip it cannot know, so the MSE-optimal prediction is close to
  the base rate (≈0.034); the small excess on marker frames is the predictable part (the
  12-frame cap makes a late blackout's END foreseeable). A dark edge track below the line
  in a model column is therefore correct behaviour, not a bug.
