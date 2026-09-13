# antialias_pilot — what a "smooth dome" disc looks like on the 128-ray geometry (2026-09-12)

`outputs/dome_candidates.png`: the intensity profile of one disc at five sub-ray offsets, at three
depths, for five renderer settings on the dw-noiseless geometry (128 rays, radius 0.5), plus a
40-frame waterfall each. All settings are existing `SimConfig` knobs (`pim/environments/discworld/
soft_render.py`), so a new instance needs no renderer change:

| candidate | knobs | sub-ray shift response (mid depth, 32 covered rays) |
|---|---|---|
| A hard, flat (current) | — | 2 silhouette rays move |
| B lambert dome | `soft_shading=lambert` — intensity = reflectivity · sqrt(1 − (perp/r)²), a semicircular dome baked into the disc (a function of the ray's perpendicular distance to the centre, so identical at the frustum edge and centre) | every covered ray moves (20 of 32) |
| C dome + edge 0.05 | + `soft_edge=0.05` (10 % of the radius): sigmoid coverage at the silhouette | as B, continuous silhouette |
| D dome + edge 0.10 | + `soft_edge=0.10` | as C, softer |
| E flat + sensor blur | `soft_psf_sigma=1` — a post-hoc image blur, NOT baked into the disc | only the edge transition widens |

Rendered by the inline script in the session (2026-09-12 evening); not an experiment with results.
