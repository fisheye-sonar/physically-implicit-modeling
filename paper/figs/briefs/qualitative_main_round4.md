# BRIEF, round 4 — main-text qualitative figure: new column layout + scenario filter; appendix Rayworld figure re-seeded

Read `paper/figs/briefs/COMMON.md`, then this file, then `paper/figs/qualitative_main/README.md` and its scripts
(`common.py`, `rayworld_panel.py`, `othello_panel.py`, `composite_final.py`), and `paper/figs/qualitative_edits/make_figure.py`
(the appendix Rayworld script whose `build()` / `predictions()` / caches the main figure reuses; `build(seed, context, variants=…)`).
Folder: `paper/figs/qualitative_main/` (+ regenerate `paper/figs/qualitative_edits/`). GPU: yes, one model at a time, freed after.
Style: `paper_style` (Arial, zero padding). Keep the pruned shape: top level = `composite_final`, `composite_final_sidebyside`,
scripts, README, sidecars; everything per-element under `pieces/`; no other renderings.

## How scenarios are matched across columns (Sevan asked; put this in the README in two sentences)
One scenario = one world (positions, velocities, teleport) generated under the tightest geometry (radius 1.0) and rendered
under EACH instance's own renderer (radius 0.5 / 128 rays for dw-noiseless and dw-blink; radius 1.0 / N rays for the ray
family). Positions are identical across columns; only the rendering differs.

## The new main figure (Sevan's spec)
Panel (a), four columns, rows Context / Unedited Pred / Ground truth / PI / GS / IM (paired error strips, locators as now):
1. **Standard (continuous), Example 1** — dw-noiseless (`noise_ablation/L-dw-noiseless-20m`), cartesian block.
2. **Standard (continuous), Example 2** — same model, second scenario.
3. **128-ray (categorical), Example 3** — dw-128ray (`ray_ablation/L-dw-128ray-20m`), appearance-fac block (PI, GS, IM = the
   categorical inverse map), a THIRD scenario.
4. **5-ray (categorical), Example 3** — dw-5ray (`ray_ablation/L-dw-5ray-20m`), appearance-fac block, the SAME scenario as
   column 3 rendered under 5 rays.
Group titles over the pairs / singles ("Standard (continuous)", "128-ray (categorical)", "5-ray (categorical)"), column titles
"Example 1 / 2 / 3 / 3". Panel (b) unchanged. Same change in `composite_final_sidebyside`.

## Scenario filter (Sevan: "only show examples which change for all of them")
A scenario is eligible only if the teleport VISIBLY changes the 5-ray observation: render the scenario under the dw-5ray
config and require the clean post-edit frame at EF to differ from the clean unedited frame at EF on at least one ray (use the
existing `render_under` / scenario machinery; no new rendering code). Apply it to ALL drawn scenarios (Examples 1–3): advance
the seed until it passes; record seed, the number of changed 5-ray rays, and the changed rays on 128 rays in the sidecar JSON.
Regenerate the caches you need (`_catim` naming as before; cache hits only for probes / maps).

## Appendix Rayworld figure (`paper/figs/qualitative_edits/`)
Re-seed with the same filter: seed 0 and `more_seeds/1..5` become the first six seeds that pass (record the mapping in that
README); regenerate all five modes of the first and the `more_seeds/` set. Keep the empty (frameless) spots for Standard /
Blink categorical IM. Add the two matching sentences above to that README.

## Report
Files; the seeds chosen and the 5-ray ray-change counts; confirmation by eye that every drawn 5-ray / 128-ray edit changes the
frame; anything not done.
