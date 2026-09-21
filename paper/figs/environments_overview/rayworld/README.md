# Rayworld and its variants: figure pieces for `fig:rayworld_and_variants` (round 2, 2026-09-21)

Everything here is drawn from REAL held-out sequences (`datasets/discworld/<inst>/eval/test.h5`) with the
canonical geometry, by `make_figure.py` in this folder. No metric is computed and nothing is re-implemented:
ray directions are the renderer's own spread (`renderer.render_frame`: uniform in the tangent of the viewing
angle over `obs_res` rays, the two wall rays dropped when the instance's `drop_edge_rays` is set), the rays a
disc covers and where they stop come from the stored `obs_id` / `obs_depth`, the appearance cells from
`pim.environments.discworld.grid_target.categorical_target("appearance").cell_of`, and the matched N-ray
strips from `renderer.render_scene` under each sibling instance's own `SimConfig`. CPU only, about 5 s.
Style: `paper/figs/paper_style.py` (Arial, TrueType embedded, zero outer padding so every piece touches its
bounding box; rasters only for observation strips, drawn with nearest interpolation on `DARK_BG`).

Regenerate (writes every file below beside the script, deterministic):

    .pim/bin/python paper/figs/environments_overview/rayworld/make_figure.py

## Files (each a vector PDF plus a 300 dpi PNG preview)

| file | what it is |
|---|---|
| `standard_frustum_light`, `_every2`, `_every3`, `_every4` | (a) frustum view, dw-noiseless sequence 8368 at t* = 20: observer, frustum outline, the rays (all 128, or every 2nd = 64, every 3rd = 43, every 4th = 32 drawn, hits and misses alike; hits stop dark at the disc surface, misses run light to the far plane), both discs filled with their reflectivity grey, a velocity arrow each, earlier frames as fading discs. White page. **Recommended: `_every3`** (43 rays: clearly legible at 2.7 in and at the composite's 1.4 in, still reads as a dense fan; every 2nd is still a grey mass on screen, every 4th starts to look sparse). Only the drawing is thinned; the strip and the waterfall keep all 128 rays. |
| `standard_frustum_light_strip`, `_every2_strip`, `_every3_strip`, `_every4_strip` | the same with frame t* laid along the far plane: pixel k sits under ray k's far-plane crossing, so the reader sees which rays light up. `_every3_strip` is the composite's (a). |
| `standard_frustum_dark`, `_dark_every3`, and their `_strip` variants | the same on the dark observation panel (`DARK_BG`), white ink |
| `standard_waterfall` | the same sequence, 40 rows (frames, time downward, arrow labelled t with 0 and 39) by 128 columns (rays, ticks 0 and 127); row t* marked by a pointer on the right |
| `standard_frame_tstar` | frame t* alone, a 1 x 128 strip |
| `blink_waterfall` | (b) dw-blink sequence 6837: disc 1 (reflectivity 0.8) hidden in frames 16 to 22 (bracket on the right, "hidden (signaled)"), its 0.5 markers on ray 127 at rows 15 and 22 |
| `nray_frustum_8ray`, `nray_frustum_8ray_strip` | frustum view of dw-8ray sequence 2770 at t* = 20: the 8 kept rays (10 cast, wall rays dropped), radius-1.0 discs; `_strip` adds frame t* along the far plane. Piece only; no longer in the composite. |
| `nray_waterfall_16ray_matched`, `_8ray_matched`, `_5ray_matched` | (c) the SAME world (dw-8ray sequence 2770) seen through 16, 8 and 5 rays: its positions rendered by `render_scene` under the dw-16ray / dw-8ray / dw-5ray `SimConfig` (radius 1.0 on all three; the 8-ray re-render equals the stored frames exactly, asserted). 1.25 x 1.6 in, wide rectangular pixels. **Recommended.** |
| `nray_waterfall_16ray_own`, `_8ray_own`, `_5ray_own` | the alternative: one sequence drawn from each instance's own eval split by the same rule (16-ray 5725, 8-ray 2770, 5-ray 3368); the worlds differ. Same size. |
| `categorical_frustum_8ray` | (d) the appearance partition on dw-8ray (30 cells) over the reachable region, both discs of sequence 2770 at t* drawn at alpha 0.65 so the cells show through, their own cells filled stronger and outlined in black; the 8 rays drawn lightly; whole frustum with the observer. 2.2 in. |
| `categorical_frustum_8ray_crop` | the same cut just below the near plane (no observer, rays drawn from the near plane up), so the partition fills the panel. The composite's (d). **Recommended.** |
| `categorical_frustum_8ray_norays` | the whole frustum without the rays |
| `key_frustum` | key for the frustum views: disc 0.8 / disc 0.4 / velocity / earlier frames / ray that hits / ray that misses |
| `composite_v2_split` | **Recommended.** 5.5 x 3.0 in: top band (a) frustum with the t* strip + its waterfall, then (b) blink waterfall; bottom band (c) the three matched strips (0.95 in wide each) and (d) the cropped partition (1.45 in wide). Panel letters in their own margin. |
| `composite_v2_rows` | 5.5 x 3.0 in: top band (a) frustum + a 3.0-in-wide waterfall; bottom band (b) blink (no ray ticks, the t arrow only) | (c) three strips (0.62 in each) | (d) cropped partition. |
| `composite_v1`, `composite_v1_dark` | the round-1 attempt Sevan reviewed (Times New Roman, small 8-ray frustum in (c)); kept for reference, not regenerated |
| `selection.json` | sidecar: seed, t*, every sequence index and generator seed, rule survivors, radii, reflectivities, positions and velocities at t*, the blackout span, the matched-render note, the cells at t*, the rule text, the drawing constants and the recommendations |

A one-band layout (every panel in one row) was not exported: with an observer frustum at least 1.4 in tall
the remaining 3.3 in cannot hold five annotated strips at legible width (under 0.4 in per N-ray strip).

## Data and selection

Instances (all `datasets/discworld/<inst>/eval/test.h5`, 10,000 held-out sequences of 40 frames, two discs,
reflectivities 0.4 (disc 0) and 0.8 (disc 1) in every instance, no observation noise, open boundary with the
discs kept fully inside the frustum): `dw-noiseless` (128 rays, radius 0.5, the paper's standard), `dw-blink`
(128 rays, radius 0.5, blink probability 0.05 per disc per frame, mean length 7, cap 12, warm-up 3),
`dw-16ray` / `dw-8ray` / `dw-5ray` (18 / 10 / 7 rays cast, the two wall rays dropped, radius 1.0).

t* = 20 everywhere (the bench's edit frame, midway through the sequence).

**Rule for the standard and N-ray sequences** (`pick`): a seed-0 draw (`numpy.random.default_rng(0)`, drawn
in the order standard, blink, 16-ray, 8-ray, 5-ray from one generator) among held-out sequences where
(i) the two streaks cross (the discs swap lateral order between frame 0 and frame 39), with a disc fully
occluded in at most 3 frames; (ii) each disc's run centre travels at least 0.2 R rays over the sequence;
(iii) at t* the discs are at least 2 depth units apart, their runs are disjoint with a gap of at least
max(1, 0.05 R) rays, each is lit by at least max(1, 0.03 R) rays, and the two runs differ in length.
Survivors: 174 of 10,000 on dw-noiseless, 90 on dw-16ray, 74 on dw-8ray, 107 on dw-5ray. The crossing is
editorial (it shows occlusion, which the text names as the only other way depth enters); "fully occluded in
at most 3 frames" replaces "both discs visible in every frame", which no crossing sequence satisfies (the far
disc is the narrower one, so it vanishes behind the near one for at least a frame).

**Rule for the blink sequence** (`pick_blink`): the same seeded draw among sequences with exactly one blackout,
of disc 1 (its marker ray is the rightmost, so the markers and the bracket share a side), 5 to 10 frames
long, starting at frame 6 or later and ending by frame 33 (both markers present, context on both sides), the
other disc lit in every frame, the blinking disc lit whenever visible and travelling at least 0.2 R rays.
60 survivors. Drawn: sequence 6837, disc 1 hidden 16 to 22 (7 frames), markers at rows 15 and 22.

## Drawing conventions (for the caption)

- Observer: the black dot at the origin. Frustum: y from 3 to 12, half-width 0.5 y (1.5 at the near plane,
  6 at the far plane); the wall rays are the fan's edges (half field of view atan 0.5, about 26.6 degrees).
- Rays are drawn from the observer; a ray that hits a disc stops at the hit surface (`obs_depth`) and is drawn
  darker, a ray that misses runs to the far plane and is drawn light. On the 128-ray frustum views only every
  3rd ray is drawn (43 of 128) for legibility; the strip and the waterfall show all 128. Nothing in the figure
  says so; the caption should.
- Discs are filled with their reflectivity as a grey level (0.4 dark, 0.8 light), exactly the pixel value they
  produce in the strips. Earlier frames are the same disc fading in toward the present: frames 0, 3, 6, 9, 11,
  14, 17 for t* = 20 (7 ghosts, alpha 0.05 to 0.35). The arrow starts at the disc's edge and spans 15 frames of
  motion (its length is the distance the disc covers in 15 frames).
- Observation strips: `gray` colormap, fixed 0 to 1, nearest interpolation, on `DARK_BG`, thin `ps.FRAME`
  border; time runs downward (row 0 at the top). The strip along the far plane is frame t*, its pixel k
  centred on ray k's crossing of the far plane.
- Blink markers are the 0.5-grey pixels on the edge ray (ray 127 for disc 1) the frame before the blackout and
  on its last hidden frame; the bracket spans the hidden frames only.
- Appearance cells (panel d): two positions share a cell when a lone disc there lights the same rays; the
  partition is drawn over the reachable region (disc centre at least one radius from every wall,
  `sim.fully_in_frustum`), 30 cells on dw-8ray with run lengths 1 (far edge) to 5 (near edge). Tones are
  Okabe-Ito hues blended toward white with no meaning beyond making neighbours differ; the two discs' cells
  are filled stronger and outlined in black; the discs are drawn at alpha 0.65 so the cells show through. The
  factorised target the paper reports (`appearance-fac`) reads the same partition as run centre (15 classes)
  times run length (5 classes). The `_crop` piece and the composites cut the view just below the near plane.
- At t* in the dw-8ray sequence the far disc (0.4) sits in the lone-disc cell for rays 2 to 3, but the near
  disc (0.8, rays 3 to 6) occludes ray 3, so the frame shows the far disc on ray 2 only. The cell is what the
  probe target labels; the rays are what the frame shows. Worth one caption sentence if panel (d) is used.

## Caption facts, per panel

(a) dw-noiseless sequence 8368 (generator seed 52200008368), t* = 20, radius 0.5, 128 rays (43 drawn); at t*
disc 0 (0.4) is the nearer one (y 7.5 against 10.0). (b) dw-blink sequence 6837, disc 1 hidden frames 16 to
22. (c) dw-8ray sequence 2770 (seed 85200002770), radius 1.0; the 16- and 5-ray strips are the same positions
rendered under the dw-16ray and dw-5ray configurations. (d) same sequence at t* = 20; 30 appearance cells; the
discs' cells are runs 2 to 3 (disc 0, far, y 9.9) and 3 to 6 (disc 1, near, y 5.4).

## Not done, and why

- Panel (d) draws the partition, not the two factor labels (centre, length); the partition is the same
  object and the caption can name the factorisation.
- The blink panel is a waterfall only, as the brief asked; no blink frustum view was made.
- No one-band composite (see above). The 8-ray frustum piece exists but is in neither v2 composite, as asked.
- No text inside the figures beyond axis labels, the t / t* marks, "hidden (signaled)", the ray counts and
  panel letters.
