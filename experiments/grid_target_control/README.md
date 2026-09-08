# grid_target_control — is the CONTINUOUS target what separates discworld from Othello? (2026-09-08)

**Question.** Every Othello probe target is categorical (64 tiles × 3 classes) and every
discworld one is continuous (regressed positions). This control re-expresses the discworld
state as a grid of cells, each {empty, centre of object 0, centre of object 1}, fits new
probes of Othello's shape on the SAME trained model, and re-runs the canonical editors
through them. No new model.

**Prediction (Sevan, 2026-09-08):** decodability and editability do not meaningfully move.

**Status (2026-09-08):** DONE on L-dw-noiseless-20m. Mostly as predicted — nowhere near Othello — but ND +0.37 / fid 0.91 and GS +0.29 / fid 0.87 (one-frame, reverting) against GS −0.10 on the regression target: the categorical target improves the editors' conditioning, not the representation's editability. Write-up `research/findings/grid-target-control.md`; the extended-α sweep is `scores/summary_ext.md` (the canonical-grid `summary.md` had ND/GS pinned at their α edges).

**The grid** (`scripts/grid.py`): 16 lateral × 8 depth = 128 cells, uniform in the frustum
basis (normalised ray coordinate u′, inverse depth 1/y) over the reachable region, so every
cell is the same size in the observation's own coordinates and larger at the back in world
space. 384 logits vs Othello's 192; 1.56% of cell entries non-empty; the nearer object wins
a shared cell (~0.04% of frames; the exact count is in `logs/grid_target_control/a_probes.log`). Same-cell teleports (2.7% of edit cases) are dropped from
the bench because the edit is a no-op under this target.

**Run:** `runs/noise_ablation/L-dw-noiseless-20m` (dw-noiseless) — the instance every other ablation sits on and the cleaner Othello analogue (no observation or position noise, so a cell label is never ambiguous from the frame). A first launch on `L-dw-20m` (dw-pn04) was stopped after ~10 min on 2026-09-08 and its partial state parked in `_pn04_partial/` (five fitted probes, nothing deleted). Probes on the first 200k
sequences of `probe_250k`, 50 epochs (the large-corpus precedent), held out by sequence,
persisted in `probes/`. Driver `drivers/grid_target_control.sh`; logs `logs/grid_target_control/`.

- `scripts/fit_grid_probes.py` — stage A (and `--random-init` for the floor)
- `scripts/edit_grid.py` — stage B: PI (logit swap + z-space re-solve), ND (per-case
  "move" direction; applicable here), GS (cross-entropy steering), canonical grids
- `scripts/fit_grid_obs_floor.py` — the observation floor on the same target
- `scores/grid_probes.json`, `scores/grid_edit.json`, `scores/summary.md`; `scores/smoke/` is the 2-epoch smoke
