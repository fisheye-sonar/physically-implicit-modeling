# edit_index_v2_pilot — the model-referenced Edit Index on paired counterfactual histories

**2026-09-11, pilot (Sevan's call).** Both Edit Index references become the MODEL'S OWN
predictions: `p_A` on the real history, `p_B` on a paired counterfactual history B whose
simulator state differs from A by one edit. Unedited scores −1 by construction, reproducing
`p_B` scores +1. Scored beside the canonical construction (simulator references) on the SAME
cases, plus the ceiling (`p_B` under the canonical index). One arm per editor at each run's
canonical best (point, α, dims) from `scores.json`; no sweep; a few dozen cases per condition.

- `scripts/pilot.py` — everything: pairs, model runs, editors (canonical `pim.editors`
  primitives; the Othello arms are re-wired for MULTI-TILE targets because a substituted move
  changes ≥ 2 tiles), both constructions (`pim.metrics.edit_index.edit_index_per_case`, the
  one formula), filters, table, figure.
- `scores/<condition>.json` — per condition: filters, arms, both indices per editor with
  per-case arrays, separation statistics, and (Othello) the tile-change breakdown by how far
  back the substituted move sits. `scores/summary.json` — everything but the per-case arrays.
- `outputs/pilot_ei.png` — per-case points and means, v2 beside v1, ceiling and floor marked.
- Log: `logs/edit_index_v2_pilot/pilot.log`.

**Counterfactuals.** Discworld: the edited object's trajectory shifted by the teleport vector
over frames 0..EF−1 (`arms.counterfactual_history`), the other object untouched, noise
matched; kept if in-frustum and collision-free at every frame. Othello: one of the last
`K_BACK = 4` moves substituted by another legal move, the original remaining moves replayed;
kept if the replay is legal, the mover unchanged, the board and the legal set differ.

**Filters.** A case counts only if `RMSE(p_A, p_B)` on the support ≥ max(absolute floor,
0.25 × the simulator's own separation there) — Sevan's rule: an edit the model's prediction
does not register is not an edit. Othello additionally requires legal mass ≥ 0.98 on BOTH
histories (the model treats them as ordinary games; GOTCHAS 2026-09-09). Discworld requires ≥
2 differing rays. Every filter's count is in `filters`.

**Not canonical.** Nothing here is in `pim.metrics` as a named construction yet; if the
design is adopted it gets a metrics module, a REGISTRY row and an `edits/v2/` bench spec
(`research/specs/DATASET_LAYOUT_SPEC.md` reserves the location).
