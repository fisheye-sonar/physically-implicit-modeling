# Controls thread — canonical RUN & DATASET REGISTRY

**The single source of truth for what every run and dataset name in `notebooks/experiments/editability/controls/` means.**
Per `CLAUDE.md`: no notebook may use a run code without copying its row into its own definitions table, and
**figures use the descriptive label, never the raw code**. Adding a run means adding its row here in the same commit.

Branch `michael_controls`. Origin: Michael's controls, 2026-07-30. Directions:
`research/directions/{encoder-space-editing, hidden-size-sweep, noise-ablation}.md`.
Checkpoints live in gitignored `runs/controls/<code>/best_model.pt`.

## Shared training recipe (every run below — the only variables are the two swept ones)

`scripts/train_gru.py`, single-layer `GRUModel`, no dropout, **400 epochs**, batch 256, AdamW **lr 1e-3**,
weight decay 1e-4, seed 0, 10% of the train split held out for validation. Matches the pre-existing dataset-4
baseline `runs/gru/7_dset4_gru_400epochs` recipe exactly.

> **Note on `--in-memory`.** These runs use the `--in-memory` flag added 2026-07-30, which holds the observation
> tensor on the GPU instead of streaming it from gzip-compressed HDF5. It is a **data-delivery change only** —
> same train/val split, same batch size, same shuffling, same optimizer — measured at **136× faster**
> (68 s/epoch → 0.50 s/epoch at `H=256`), with matching loss curves (epoch-2 train loss 0.0267 both paths).
> Without it a single 400-epoch run costs 7.5 h and this thread would not be runnable.

## Shared world settings (every dataset below)

2 objects, 40 frames, `obs_res=128`, open boundary, fixed reflectivities, always-in-frustum, radius 0.5,
speed 0.05–0.12, `direction_noise_std=0`, `speed_noise_std=0`. Splits 90k train / 10k val / 10k test /
10k edits, base seed 0, **edit frame 20**, edits are in-frustum teleports of one object.

## Datasets

| code | descriptive label (use this in figures) | obs noise | position noise | notes |
|---|---|---|---|---|
| `4_fixed_refl_inview` | **both noises (obs 0.2, pos 0.04)** — the repo standard | 0.2 | 0.04 | pre-existing; the canonical eval dataset behind every earlier finding |
| `9_obsnoise0_posnoise0` | **no noise (obs 0.0, pos 0.00)** | 0.0 | 0.00 | new 2026-07-30; deterministic world, perfectly sensed |
| `10_obsnoise0_posnoise004` | **world noise only (obs 0.0, pos 0.04)** | 0.0 | 0.04 | new 2026-07-30; stochastic world, perfectly sensed |
| `11_obsnoise02_posnoise0` | **sensing noise only (obs 0.2, pos 0.00)** | 0.2 | 0.00 | new 2026-07-30; deterministic world, noisily sensed |

*What the two sources mean:* **observation noise** corrupts the 1D intensity scan — the world is exact, the
model's view of it is not. **Position noise** adds Gaussian diffusion to the discs' positions each step — the
world itself is stochastic, so it is unpredictable however well it is sensed. They are conceptually opposite.

## Runs

### Hidden-size sweep (direction `hidden-size-sweep`) — one variable: `hidden_size`
All on `datasets/4_fixed_refl_inview` (both noises). `H=8` matches the world's true state dimensionality
(2 objects × (x, y, vx, vy)); `H=128` matches the observation resolution.

| code | descriptive label (use this in figures) | hidden size | parameters | purpose |
|---|---|---|---|---|
| `H8` | **H=8 · both noises** | 8 | 5,384 | at the world's true state dimensionality — forced canonicality |
| `H32` | **H=32 · both noises** | 32 | 16,544 | below observation resolution |
| `H128` | **H=128 · both noises** | 128 | 148,352 | matches observation resolution (128 rays) |
| `H256` | **H=256 · both noises (baseline)** | 256 | 460,672 | **the shared baseline** — the repo's default, also the "both noises" cell of the noise ablation and the model used by the encoder-editing direction |
| `H512` | **H=512 · both noises** | 512 | 1,707,648 | 4× over-complete vs the observation |

### Noise ablation (direction `noise-ablation`) — one variable: which noise sources are on
All at `hidden_size=256`. The 2×2 is completed by `H256` above.

| code | descriptive label (use this in figures) | dataset | obs noise | position noise |
|---|---|---|---|---|
| `N_obs0_pos0` | **no noise (obs 0.0, pos 0.00)** | `9_obsnoise0_posnoise0` | 0.0 | 0.00 |
| `N_obs0_pos004` | **world noise only (obs 0.0, pos 0.04)** | `10_obsnoise0_posnoise004` | 0.0 | 0.04 |
| `N_obs02_pos0` | **sensing noise only (obs 0.2, pos 0.00)** | `11_obsnoise02_posnoise0` | 0.2 | 0.00 |
| `H256` | **both noises (obs 0.2, pos 0.04)** | `4_fixed_refl_inview` | 0.2 | 0.04 |

> ### ⚠ Absolute RMSE is NOT comparable across the noise cells
> A model trained and evaluated on noise-free observations has an observation noise floor of ~0, so its raw RMSE
> is lower for bookkeeping reasons, not because it is a better world model. **Every predictive number must be
> read against that model's own baselines** (copy-previous-frame, noise floor, random frame — `pim/eval/baselines.py`),
> which is why the notebook plots per-model baselines rather than one shared set. The same mistake cost the
> endogenous thread a set of cross-citable numbers (see `actions/ENDOGENOUS_RUNS.md`).

### Encoder-space editing (direction `encoder-space-editing`)
No new training — uses `H256`. The independent variable is the **edit interface** (encoder output `x` vs hidden
state `h`), not the model.

## Metric & editor definitions

This thread uses the canonical registry `../METRICS_AND_EDITORS.md` (§2 recoverability, §3 canonicality, §4
editing/object-handle) verbatim; the §4 metrics are **imported** from `scripts/editability_metrics.py`, never
re-derived per notebook. Each notebook copies the subset it uses into its own definitions table.

> **The §4 metric set changed on 2026-07-30, mid-thread.** The old `reach % of swap` / `collateral % of swap` /
> `selectivity` / `ghost ratio` were retired (they scored *change* rather than *correctness*, and normalised by a
> model-dependent soft reference) and replaced by the **Edit Index** (−1…+1, which of the two ground-truth worlds
> the output is closer to) plus **Target / Ghost / Collateral / Edit-frame / GT-traj RMSE** and the **fidelity
> ratio**. Every notebook and eval output in this directory is on the **new** set; any number quoted from before
> that date is not comparable.
