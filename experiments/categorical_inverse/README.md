# categorical_inverse — the inverse map must invert the state the block's own probes read (2026-09-20)

**Found by Sevan (2026-09-19 night).** On a categorical discworld block (`appearance-fac`, the grids) the IM editor
was never given the categorical state. `inverse_discworld` fitted one inverse map per residual point per BASIS from
the continuous full state and wrote it on every block naming that basis; the categorical blocks name `frustum`, so
their "IM" was the continuous frustum map scored on the categorical bench's case selection. Evidence: on
`L-dw-8ray-20m__seed1` the `appearance-fac` block's `inverse_map.g_r2` vector equals the frustum block's to every
digit. Every run, every categorical block (52 blocks in 22 runs).

**Sevan's specification.**
- On a categorical block, g inverts **the full categorical state the forward probes read** (the target's own labels,
  one-hot per tile) **plus the discs' continuous Cartesian velocity** — deliberately asymmetric: the forward probes on a
  categorical target do not read velocity, but the residual encodes it, and a map from labels alone would write the mean
  residual over velocities and erase the model's velocity estimate along with moving the disc.
- **Recipe = the forward probe's for the same target**: 200k sequences of `probe_250k`, 50 epochs
  (`arms.GRID_PROBE_RECIPE`), the same seeded 80/20 split by sequence. Regression blocks keep theirs (30k, 200 epochs).
- **Scope**: dw-128ray, dw-16ray, dw-8ray, dw-5ray (and the 8-ray token model, same instance) × `appearance-fac`.
  Every other categorical block carries NO IM arm; its table cell is blank. Not dw-noiseless, not dw-blink.
- **The old arms are deleted outright** from every scores.json (dated backup of each file kept in its
  `scores_backup/`, as `runs/` is never deleted from). No IM-NN on categorical blocks (not tabulated; its bank would
  be the one 12.8 GB object left at 128-ray).

**Why 200k needed new code, not new hardware.** The 200k categorical FORWARD probes stream their residual stack
from disk (`fit_probe_stream` over `MemmapRows`); the inverse map used the dense fitter written for 30k, which keeps
the rows on the GPU and then builds the retrieval bank from a second copy — 2 × 12.8 GB at 200k. The network is
~200k parameters. `fit_inverse_map_stream` gives the inverse map the same treatment with the roles swapped: the residual
stack (now the TARGET) stays on disk, the input per row is 4 labels + 4 floats, and the one-hot (1,028 wide on
dw-128ray) is built per minibatch.

## What is in this branch (`categorical_inverse`, staging clone `../pim-master-eval-refactor`)

| file | change |
|---|---|
| `pim/probes/inverse.py` | `CATEGORICAL_STATE`, `encode_categorical_state`, `CategoricalState` (rows source; one-hot columns left 0/1, velocity standardised), `fit_inverse_map_stream` (same probe body / optimiser / loss as the dense fit; `r2`, `r2_insample`, `rmse`) |
| `pim/environments/discworld/arms.py` | `iter_inverse_maps(target=…)` dispatches a categorical target to `_iter_categorical_inverse_maps` (one point's residuals on disk at a time, cached, no bank; a cache HIT needs no residuals); `inverse_arms(target=…)` writes g(one-hot `Bench.tgt`, Cartesian velocity) and REFUSES a categorical bench through the continuous map (and vice versa). **The continuous path's body is untouched.** |
| `pim/environments/discworld/token_bench.py` | the same for the frames-as-tokens model |
| `pim/scoring/discworld.py` | `inverse_discworld`: regression blocks per basis as before; categorical blocks per target, only in scope |
| `pim/scoring/blocks.py`, `driver.py` | `cat_inverse_in_scope`; `missing_inverse` never counts an out-of-scope categorical block as missing IM, and adds the in-scope one to an ALREADY-scored run only under `PIM_ADD_CAT_IM=1` (a ~30 min fit per block must be one deliberate job) |
| `notebooks/master_eval.ipynb` | new cell [2b]: `SETTINGS["dw_cat_im"]` (cell [2] byte-identical) |
| `experiments/categorical_inverse/` | `scripts/preview.py`, `scripts/clear_continuous_im.py` (dry-run by default), drivers, this file |
| `tests/test_categorical_inverse.py` | encoding; standardisation; streamed = dense fit on synthetic data; the refusal; scope + gate; the clear script |
| `research/REGISTRY.md`, `research/findings/inverse-probe.md` | the INVERSE MAP row; a dated CORRECTION entry |

## Evidence so far

- Suite: 307 pass in the clone (301 + 6 new).
- Smoke (5-ray, 3k sequences, 3 epochs, one point): the write LANDS — the run's own cached categorical probes read the
  target labels off the written residual in 80% (LIN) / 73% (MLP) of cases against 0.7% / 1.1% unedited.
- **Preview** on the four parents at the production recipe → `scores/preview_<run>_appearance-fac.json`
  (unit `catinv_preview`, 2026-09-19 23:48 →; logs `logs/categorical_inverse/`): the new arm, the old arm beside it,
  PI / ND / GS from scores.json, g's held-out AND in-sample R² per point, landing. Nothing under `runs/` is written
  (maps cached in `probes/` here).
- **Parity gate** (unit `catinv_gate`, runs after the previews): `L-dw-8ray-20m__seed1` rescored into scratch through
  this branch with the SETTINGS cell alone (categorical map out of scope) and diffed leaf by leaf against disk —
  expected: the factorised block loses its old IM / IM-NN arms, `best.IM*`, `inverse_map`; nothing else differs.
  → `scores/gate_L-dw-8ray-20m__seed1.diff.txt`.

## Deployment (NOT done — Sevan's call)

1. Merge `categorical_inverse` into `sweeps_and_blates` on the lab while no `master_eval` execution is running
   (GOTCHAS 2026-09-19); `git pull` on the 4090 during one of its training stages. No driver changes.
2. `clear_continuous_im.py --apply` on the lab (refuses while a scoring pass runs); re-run it after any job that was
   in flight at deployment has scored (idempotent). The 4090's copies are refreshed by the dispatcher's input push.
3. Catch-up job(s) at the TOP of the queue, lab first: `PIM_ADD_CAT_IM=1 PIM_ONLY_RUNS=<runs> PIM_DW_BASES=frustum,cartesian
   PIM_SKIP_TOPICS=training_curve PIM_SCORE_ONLY=1 bash scripts/drivers/score_pending.sh catim_<family>` — one per
   family so a number lands every ~30–40 min per run: parents first (one IM number per ray family), then the members.
   Replicates scored FRESH after the merge get the arm with no flag.
4. Copy this experiment's cached maps into each parent's `probes/` first (same cache key: model fingerprint, target,
   state, recipe) and step 3 costs only the arms for the four parents.
5. Record: REGISTRY run rows, `findings/ray-ablation.md`, a categorical-IM waterfall on 8-ray before any claim.
