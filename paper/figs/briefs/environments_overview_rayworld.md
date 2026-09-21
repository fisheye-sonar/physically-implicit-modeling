# BRIEF — Rayworld and its variants (environment figure pieces)

Read `paper/figs/FIGURE_BRIEF_COMMON.md` first. CPU only. Folder: `paper/figs/environments_overview/rayworld/`.
(Rayworld is the paper's name for the code's discworld. Nothing in a figure says "discworld".)

## What the figure must convey (paper §Experimental Setup → Rayworld; caption drafted in `paper/paper_draft.tex`,
figure `fig:rayworld_and_variants`)
An observer at the origin looks into a 2-D trapezoidal frustum; two discs move at constant velocity for 40 frames
(no collisions with walls or each other). Each frame is rendered into a 1-D observation by casting rays from the
observer: each ray returns the reflectivity of the nearest disc it hits (0.4 or 0.8, fixed per disc) or 0. Depth is
not observed; it enters only through how many rays a disc covers and through occlusion. Variants: **blink** (a disc
is invisible for 1–12 frames, start and end signalled by a 0.5 marker on one edge ray), **N-ray** for N in
{5, 8, 16} (fewer rays, disc radius doubled to 1.0 so a disc always covers at least one ray), and the
**categorical** description of state (appearance cells: two positions are in the same cell iff a lone disc lights the
same rays; the paper reports the factorised version: centre of the run and its length as two categorical variables).

Sevan's words: "showing the observer, rays, two discs. And the discs should have arrows showing their velocities. I
think we need a waterfall plot too, and something to convey that every row is a frame and the vertical axis is time.
Then in the frustum we can show a kind of faded view of the disc's trajectories so people can match it to the waterfall."

## Canonical data and objects (import, never re-implement)
- Use REAL held-out sequences, not a new simulation: `pim.environments.layout.eval_file("discworld", inst)` is an
  HDF5 with `positions (N,40,2,2)`, `velocities (N,40,2,2)`, `radii (N,2)`, `reflectivities (N,2)`,
  `obs_intensity (N,40,R)`, `is_visible (N,40,2)` (blink), and `attrs["config_json"]["dataset"]["sim"]` (the
  `SimConfig` fields: `y_near, y_far, x_near, x_far, obs_res, drop_edge_rays, radius, n_frames`). Instances:
  `dw-noiseless` (128 rays, radius 0.5 — the paper's "standard"), `dw-blink`, `dw-16ray`, `dw-8ray`, `dw-5ray`
  (`datasets/discworld/`). The frustum: half-width `x(y) = (x_far / y_far) · y` between `y_near` and `y_far`;
  `pim.environments.discworld.sim.frustum_half_width(y, cfg)`. Ray directions: read
  `pim/environments/discworld/renderer.py::render_frame` / `_fov_scale` for exactly how the rays are spread (uniform
  in the tangent of the viewing angle; edge rays dropped when `drop_edge_rays`) and draw those rays — do not invent a
  spacing. `SimConfig` is `pim.environments.discworld.config.SimConfig(**sim_dict)`.
- Appearance cells for the categorical panel: `pim.environments.discworld.grid_target.categorical_target("appearance-fac")`
  (or `"appearance"`) gives a target with `cell_of(pos, sim)`; colour a fine grid of positions inside the reachable
  region by cell id to draw the partition on the frustum (read the class docstring for the reachable-region margin).
  Use a muted categorical colouring (alternating light tones), no legend of cell ids.
- Observation strips / waterfalls: draw as `_panel` in `paper/figs/qualitative_edits/make_figure.py` does
  (gray on `DARK_BG`, nearest, `ps.FRAME` border). A waterfall is `obs_intensity[i]` (40 × R), time downward.

## Deliver (each piece its own PDF + PNG; frustum panels vector)
1. **Frustum view** (standard instance, one chosen sequence): observer mark at the origin, the frustum outline,
   the rays (128 is dense — draw them thin and light, or draw every k-th ray and say so in the README; also export
   an 8-ray version for the N-ray panel), the two discs at one frame `t*` (filled with their reflectivity grey on
   a light page, or the same grey on dark — try both), a velocity arrow on each disc, and the faded trajectory
   (earlier frames as fading discs or a dotted path) so the reader can match it to the waterfall. Mark which rays
   the discs cover at `t*` (e.g. those rays drawn brighter / thicker).
2. **Waterfall** of the same sequence: 40 rows × 128 rays, a thin "time" arrow with `t` labelled downward, the row
   `t*` marked so it links to the frustum view. Also export the single frame `t*` as a 1-row strip.
3. **Variant panels**: blink (waterfall of a `dw-blink` sequence with a blackout and its markers visible; mark the
   hidden span lightly), N-ray (frustum with 8 rays and radius-1.0 discs + the 8-ray, 16-ray and 5-ray waterfalls of
   the SAME world if you can pick matched sequences — the instances differ, so simply take one sequence per
   instance and say so), categorical (the appearance-cell partition drawn on the frustum for `dw-8ray`, the two discs
   placed, their cells highlighted).
4. **Composite attempt**: a full-width (5.5 in) figure: (a) standard frustum + waterfall side by side with the
   linking mark, then (b) blink, (c) N-ray, (d) categorical as smaller panels. Panel letters only.
5. Pick sequences by a rule (e.g. seed 0 among sequences where both discs stay visible and cross the field), record
   ids in a sidecar JSON, write `README.md` with caption facts (which instance each panel uses, `t*`, radii).
