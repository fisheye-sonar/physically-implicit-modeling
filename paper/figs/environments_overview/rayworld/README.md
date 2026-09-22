# Rayworld and its variants: `fig:rayworld_and_variants` (round 5, 2026-09-22)

Everything here is drawn from REAL held-out sequences (`datasets/discworld/<inst>/eval/test.h5`) with the
canonical geometry, by `make_figure.py` in this folder. No metric is computed and nothing is re-implemented:
ray directions are the renderer's own spread (`renderer.render_frame`: uniform in the tangent of the viewing
angle over `obs_res` rays, the two wall rays dropped when the instance's `drop_edge_rays` is set), the rays a
disc covers and where they stop come from the stored `obs_id` / `obs_depth`, the appearance cells from
`pim.environments.discworld.grid_target.categorical_target("appearance").cell_of`, and the matched N-ray
strips from `renderer.render_scene` under each sibling instance's own `SimConfig`. CPU only, about 5 s.
Style: `paper/figs/paper_style.py` (Arial, TrueType embedded, zero outer padding so every export touches its
bounding box; rasters only for observation strips, nearest interpolation on `DARK_BG`).

Regenerate (deterministic):

    .pim/bin/python paper/figs/environments_overview/rayworld/make_figure.py          # both composites + pieces
    .pim/bin/python paper/figs/environments_overview/rayworld/make_figure.py --all    # also the pruned variants

## Top level

| file | what it is |
|---|---|
| `composite.pdf` / `.png` | **the figure**, 5.5 x 3.26 in, two bands: top (a) frustum view with frame t* along the far plane and its waterfall, (b) blink waterfall; bottom (c) the same world through 16 / 8 / 5 rays, (d) the appearance cells. Panel letters only. |
| `composite_onerow.pdf` / `.png` | the **alternative**: the same four panels in ONE band, 5.5 x 1.69 in. Everything is smaller (see the layout note below); use it where the figure must be short. |
| `make_figure.py` | the script |
| `selection.json` | sidecar: seed, t*, every sequence index and generator seed, rule survivors, radii, reflectivities, positions and velocities at t*, the blackout span, the matched-render note, the cells at t*, the cell-colour seed, the rule text, the drawing constants, both piece lists |
| `README.md` | this file |

### How the one-row variant differs (layout only; same data, same sequences, same conventions)

The panels are 1.34 in tall. The waterfalls put their range and their axis name on ONE row under the panel
("0   ray   127", `ticks="compact"`) instead of a tick row plus a label row, and that reclaimed height goes
to (a), whose width follows the band height at equal aspect: the frustum is 1.22 x 1.34 in, and it draws
**every 5th ray (26 of 128) at heavier weights** (hit rays lw 0.62 / alpha 0.95, misses 0.42 / 0.52) because
the two-band composite's 43 thin pale rays disappear at this size. The rest: waterfalls 0.56 and 0.50 in
wide (all 40 frames, shorter rows), the three N-ray strips 0.32 in each with their ray counts **below** them
(7 pt) so the letter row stays free, (d) the hard crop at 1.18 x 0.90 in centred in the band, and the panel
letters 0.10 in above the artwork. Same sequences, same t*, same drawing rules as the two-band composite.

## pieces/ (PDF; the PNG previews beside them are gitignored)

| piece | what it is |
|---|---|
| `standard_frustum_light` | (a) frustum view, dw-noiseless sequence 8368 at t* = 20, all 128 rays drawn: observer, frustum outline, rays (a hit stops dark at the disc surface, a miss runs light to the far plane), discs filled with their reflectivity grey, a velocity arrow each, earlier frames as fading discs |
| `standard_frustum_light_every3` | the same with every 3rd ray drawn (43 of 128), the composite's choice: legible while still reading as a dense fan |
| `standard_frustum_light_strip`, `standard_frustum_light_every3_strip` | the same two with frame t* laid along the far plane (pixel k under ray k's far-plane crossing); `_every3_strip` is the composite's (a) |
| `standard_waterfall` | the sequence's 40 x 128 waterfall: time downward (arrow t, 0 and 39), rays 0 to 127, row t* outlined in light grey with a pointer |
| `standard_frame_tstar` | frame t* alone, a 1 x 128 strip |
| `blink_waterfall` | (b) dw-blink sequence 6837: disc 1 (0.8) hidden in frames 16 to 22 (bracket "hidden" on the right), its 0.5 markers on ray 127 at rows 15 and 22 |
| `nray_frustum_8ray`, `nray_frustum_8ray_strip` | frustum view of dw-8ray sequence 2770 at t* = 20 (8 kept rays of 10 cast, radius 1.0); not in the composite |
| `nray_waterfall_16ray_matched`, `_8ray_matched`, `_5ray_matched` | (c) the SAME world (dw-8ray sequence 2770) through 16 / 8 / 5 rays: its positions rendered by `render_scene` under each instance's `SimConfig` (radius 1.0 on all three; the 8-ray re-render equals the stored frames exactly, asserted); 1.25 x 1.6 in |
| `categorical_frustum_8ray` | (d) the 30-cell appearance partition on dw-8ray over the reachable region, one colour per cell, the two discs of sequence 2770 at t* at alpha 0.65 with a dot at each centre, their own cells filled stronger and outlined; whole frustum with the observer |
| `categorical_frustum_8ray_crop` | the same cut just below the near plane (no observer, rays drawn from the near plane), the composite's (d) |
| `key_frustum` | key for the frustum views: disc 0.8 / disc 0.4 / velocity / earlier frames / ray that hits / ray that misses |

## pieces_onerow/ (PDF; PNGs gitignored by this folder's own `.gitignore`)

The same elements at the sizes `composite_onerow` uses, all 1.34 in tall: `standard_frustum_light_every5_strip`
and `standard_frustum_light_every6_strip` (1.22 in wide, the two ray densities tried — **every 5th is the one
in the composite**: 26 rays still read as a fan while every individual ray resolves, where every 6th at 21
rays starts to look sparse; `ONEROW_EVERY` switches it), `standard_waterfall` (0.56), `blink_waterfall`
(0.50), `nray_waterfall_{16,8,5}ray_matched` (0.32 each) and `categorical_frustum_8ray_crop` (1.18 x 0.90).
`key_frustum` is size independent and lives in `pieces/` only. The strips here carry no ray-count label; the
composite adds it below them.

Pruned in round 3 (regenerate with `--all`; git history keeps the committed copies): every-2nd / every-4th
ray frustums, the dark-panel frustums, the `_own` N-ray strips (one sequence per instance), the no-rays
partition, the round-1 composites and the "rows" layout (`pieces/composite_rows` under `--all`).

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
  darker, a ray that misses runs to the far plane and is drawn light. The composite's frustum draws every
  3rd ray (43 of 128) for legibility; the strip and the waterfall show all 128. Nothing in the figure says
  so; the caption should.
- Discs are filled with their reflectivity as a grey level (0.4 dark, 0.8 light), exactly the pixel value they
  produce in the strips. Earlier frames are the same disc fading in toward the present: frames 0, 3, 6, 9, 11,
  14, 17 for t* = 20 (7 ghosts, alpha 0.05 to 0.35). The arrow starts at the disc's edge and spans 15 frames of
  motion (its length is the distance the disc covers in 15 frames).
- Observation strips: `gray` colormap, fixed 0 to 1, nearest interpolation, on `DARK_BG`, thin `ps.FRAME`
  border; time runs downward (row 0 at the top). In the standard waterfall, row t* carries a thin light
  outline (the canonical neutral marker colour `pim.figures.waterfall.EDIT_LINE`) and a pointer; the strip
  along the far plane is that row, its pixel k centred on ray k's crossing of the far plane. The ray axis is
  stated under every waterfall: as ticks plus a "ray" label in `composite`, and as one compact row
  ("0   ray   127") in `composite_onerow`; in (c) the ray count is the label under each strip.
- Blink markers are the 0.5-grey pixels on the edge ray (ray 127 for disc 1) the frame before the blackout and
  on its last hidden frame; the bracket spans the hidden frames only, the word "hidden" centred on it.
- Appearance cells (panel d): two positions share a cell when a lone disc there lights the same rays; the
  partition is drawn over the reachable region (disc centre at least one radius from every wall,
  `sim.fully_in_frustum`), 30 cells on dw-8ray with run lengths 1 (far edge) to 5 (near edge). **Each cell
  has its own colour** (`cell_colours`), placed deterministically — no seed, no sampling: the hues are the
  30 evenly spaced points of the colour circle, and the cells take them greedily, most constrained cell
  first and ties by cell index, each taking the free hue whose circular distance to its already coloured
  neighbours **on the page** is largest; saturation (4 levels, 0.18 to 0.31) and value (3 levels, 0.93 to
  1.0) then cycle to separate ties. The colours carry no meaning; they exist so the reader sees many
  distinct cells rather than a repeating tiling. Measured over the partition's 59 page-adjacent cell pairs:
  minimum RGB distance **0.166**, median **0.286** (the earlier random-seed palette gave 0.056 and 0.291, so
  the worst pair is three times better separated). All 30 stay light, so the rays, the discs and the centre
  dots read on top, and every boundary also carries a thin grey line. The two discs' cells are the same hue
  further saturated and are outlined in black; the discs are drawn at alpha 0.65 with a dot at the true
  centre. The factorised target the paper reports (`appearance-fac`) reads the same partition as run centre
  (15 classes) times run length (5 classes). Both composites cut the view just below the near plane.
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
- The composite is 5.5 x about 3.3 in: the frustum's size is tied to the top band's height by the equal
  aspect, so the bigger frustum asked for in round 3 made the figure about 0.3 in taller than round 2's.
  `hf` in `composite()` is the one number to change.
- In `composite_onerow` the frustum's discs are about 0.10 in across and the partition's cells about 0.1 in
  wide: legible in print, but the two-band `composite` is the one to use where there is room. A single band
  cannot hold five annotated groups at 5.5 in without that shrink, since the frustum's width grows with the
  band's height — which is also why making (a) bigger in round 5 took the one-row figure from 1.55 to
  1.69 in tall even after the compact axis row gave back a tick row's worth of height.
