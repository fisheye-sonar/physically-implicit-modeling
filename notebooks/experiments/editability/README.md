# editability/ — Editability & canonical-state pillar (GRU)

Working notebooks for sub-questions 1–3 (geometry / identifiability / editability) on the GRU.
The refined-RSSM counterpart lives in `../rssm_structure/`; the two are unified in the master notebook.

## Convention (primary vs working vs scratch)
- **Primary (start here):** `00_master_editability.ipynb` — the clean, visual-heavy synthesis of the
  whole investigation, GRU **and** RSSM side by side (geometry → recoverability → canonicality →
  editing head-to-head → synthesis). This is the presentation-grade entry point; keep it linear and
  readable. Leading ideas that may later graduate into `pim`/`world_model_eval` incubate here first.
- **Working notebooks** (the source experiments the master consolidates):
  - `canonical_state_editing.ipynb` — (pos,vel) probe, fiber-collapse keystone, joint edit, obs-driven edit.
  - `geodesic_walk_k150.ipynb` — constant-step geodesic; schedule-artifact correction.
  - `manifold_geometry_diagnostic.ipynb` — intrinsic dim, curvature, projection-tautology.
  - `diagnostic_corrections.ipynb` — velocity 2×2, det-only fiber, honest small-k geodesic (2026-07-08).
  - `pca_component_position.ipynb`, `geodesic_walk.ipynb`, `editability_structure.ipynb` — earlier/parked.
- **Scratch:** new one-off experiments go here with a descriptive name; promote a stable idea into the
  master notebook once it earns a place. Keep the master clean.

## Naming (do not conflate)
- **local-tangent projection** = a *single* projection onto the local tangent subspace.
- **PCA geodesic** = the *iterative* multi-step local-tangent walk. Use these names distinctly.

## Note on directory structure
Renamed `manifold_editing/` → `editability/` on 2026-07-09 (Sevan's call); all downstream path
references across `research/` (findings, directions, scratch, PROGRESS) were swept in the same change.
The refined-RSSM diagnostics still live in the sibling `../rssm_structure/`.
