# multi_observer — several observers on a ring around a circular arena (2026-09-13)

Sevan's "increasing observers" control: the same two discs rendered into N simultaneous 1D
observations by N observers spaced equally on a ring about the frustum's depth midpoint; the
discs live in the circle every observer's near and far planes are tangent to. Same principle
as anti-aliasing (position becomes more determined) but a different pressure on the
representation; also makes 8-ray editing harder again (output = N × 8 values).

**Canonical pieces (in `pim/`, all default-off, defaults bit-identical — `tests/test_observers.py`):**
`pim/environments/discworld/observers.py` (poses, frame transform, circular arena, multi-view
render), `SimConfig.n_observers` / `SimConfig.region`, `obs_dim` scaling, and one dispatch each
in `renderer.render_frame`, `sim.sample_position`, `sim.fully_in_frustum`;
`zone_editability.sim_config_from` and `scripts/generate_dataset.py --n-observers N --region circle`
pass the knobs through, so a future instance scores under the standard bench unchanged.

**Here:** `scripts/animate.py` — top-down world beside the N observation strips, MP4 + GIF +
frame-20 PNG + a waterfall contact sheet per scene, dw-8ray geometry:

    .pim/bin/python experiments/multi_observer/scripts/animate.py --n-observers 5 --seeds 0 1 2

Outputs in `outputs/`. Measured on 40 scenes × 40 frames (dw-8ray geometry, region circle):
a disc is visible to a given observer 93 % of the time (the rest is OCCLUSION by the other disc —
every arena position lights ≥ 1 kept ray in every view, scanned 2026-09-13), an observer sees both discs 86 %, never neither; no disc is ever
invisible to every one of 5 observers. Acceptance cost of the stay-inside rule ≈ the frustum's
(19 vs 22 collision-free candidates per accepted 40-frame scene). Note that with N = 1 and the
circle the canonical observer loses a disc 8 % of the time to occlusion (the same-depth discs of a
round arena line up more often than in the frustum) — the circle is NOT dw-8ray's always-in-frustum
guarantee; that is the point of adding observers.

No instance, corpus or run exists yet.
