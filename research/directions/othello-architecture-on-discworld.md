# Li et al.'s Othello-GPT architecture, trained on discworld ("run A")

**Tag:** `[reframe]`
**Status:** proposed — fully specified, awaiting Sevan's two open calls (§Open decisions)
**Serves:** sub-question 3 (editability); cross-cutting (architecture independence)
**Proposed:** 2026-08-21

> **Read this with `othello_transfer/README.md`.** That thread ran *our probe and our editor on
> their model*. This runs *their architecture on our world*. They are two halves of the same
> elimination, and the conclusion of either is incomplete without the other.

## The question

Four confounds could explain why probe-derived writes edit Li et al.'s Othello-GPT and not our
world models. Two are eliminated: **our editor implementation** (2026-08-20 — our code reproduces
their intervention on their model, 2.723 → 0.016 against their published 2.68 → 0.12) and **probe
training data** (2026-08-21 — 60x more probe data buys +0.029 R², saturated). Two remain:
**model training data** (3.6M unique frames against their ~1.2B unique transitions) and
**architecture** (3.2M params / 4 layers / `d_model` 256 / banded window 16 against
25.3M / 8 / 512 / full causal).

This run collapses both at once, deliberately. Take **their architecture verbatim**, change only
what the continuous observation forces, train it on **discworld at their data scale**, and ask
whether editability appears.

**Hypothesis (Sevan's, stated in advance):** it does **not**. If the negative survives their
architecture at their data scale, then neither architecture nor data was the cause, and the
remaining explanation is the one the project actually cares about — the difference between
**rendering a continuous observation scored by pixelwise MSE** and **predicting logits over a
discrete world state**.

**If the hypothesis is wrong** — editability appears — then one of architecture or data *was* the
cause, and the ladder must be bisected (window → capacity → depth → data). Only then is run B
(their architecture at *our* current data scale, 90k episodes) worth running. **Do not run B
first.**

## Why it is worth doing now

It is the last cheap confound. Everything downstream — whether the thread's central negative is a
statement about world models or a statement about our setup — is gated on it. And the two
eliminated confounds already cost less than a day between them, so the marginal cost of closing
these two is small against the interpretive gain.

## Bootstrap (cold-start runnable)

- **Their repo (read-only):** `/home/sevan/research/PIM/othello_world`, added to `sys.path`.
  Nothing in it is modified. Its `data/othello.py` imports `seaborn`, `psutil`, `pgn` at module
  level; all three are in `.pim` (`pandas` came with `seaborn` — `research/GOTCHAS.md`, 2026-08-20).
- **Architecture source:** `mingpt/model.py` — `GPT`, `GPTConfig`. Used **verbatim**; the two
  substitutions below are made in a subclass or a thin wrapper, not by editing their file.
- **World config:** copy the `sim` block of `datasets/4_fixed_refl_inview/dataset.json` **exactly**
  — 2 objects, 40 frames, `obs_res` 128, radius 0.5, `fixed_reflectivities` true, refl 0.4–0.8,
  `always_in_frustum` true, `obs_noise_std` 0.2, `position_noise_std` 0.04, open boundary.
- **Generator:** `pim.simulator.sim.simulate` via `scripts/generate_dataset.py`. **Not**
  `BatchedInteractiveWorld` — that is the interactive setting and a different distribution.
- **Evaluation splits:** reuse `datasets/4_fixed_refl_inview` `test.h5` and `edits.h5` unchanged,
  so every editability number stays on the same axis as the GRU, RSSM, transformer and DiT results.
- **Probe and editor:** `notebooks/experiments/editability/othello_gpt/othello_probe.py`
  (`fit_probe`, `build_edit_spec`, `make_intervention_hook`, `_descend`) and
  `pim.editors.probe_steering`. Both **unmodified**.
- **Metrics:** `scripts/editability_metrics.py` (`build_edit_zones`, `edit_scorecard`,
  `fidelity_ratio`). Registry: `notebooks/experiments/editability/METRICS_AND_EDITORS.md`.
- **Reference numbers to cite, not recompute:** `W16` unsteered Edit Index −0.684, best
  probe-derived write −0.194 (`othello_gpt/`, 2026-08-18, corrected 2026-08-19); `W16` val loss
  0.02359; GRU 400-epoch reference 0.02362 (`transformers/TRANSFORMER_RUNS.md`).

## Method

### 1. Corpus

Generate **25M episodes** at the dataset-4 config. Measured: **645 episodes/s/core**, so ~30 min
across 32 cores. Store to disk — Sevan's call, on the grounds that later runs amortise it.

- **Storage:** 46.4 KB/episode for all fields = **1.19 TB**. Dropping `obs_depth` and `obs_id`
  (54% of the bytes, regenerable from the stored seeds) gives **0.54 TB**. Take the 0.54 TB option
  unless a later analysis is known to need those fields.
- ⛔ **Seed offset — the failure that would silently destroy the evaluation.** Dataset 4 uses
  `base_seed 0` for train (0–89,999) with val/test/edits in contiguous ranges after it. Generating
  25M episodes from seed 0 would **regenerate the exact test and edits episodes as training data**.
  Start the new corpus beyond dataset 4's highest seed and **assert non-overlap in code**, against
  the seed ranges read from `datasets/4_fixed_refl_inview/dataset.json`, not from memory.
- **Probe split:** a further **200k episodes** with their own disjoint seed range (~4.4 GB). 200k
  and not 140k: Li et al.'s probe corpus is ~140k games x 60 moves ≈ 8.4M rows, and matching
  **rows** (200k x 40 x 0.8 ≈ 6.4M) is what the probe actually sees. Do not let anyone "correct"
  this to 140k.
- **Val subset:** a small disjoint split for the convergence read in step 3.

### 2. Model — their architecture, two substitutions

Keep, verbatim: **8 blocks, 8 heads, `d_model` 512, `d_mlp` 2048, learned absolute positional
embeddings, full causal mask, dropout 0.1 on embedding/attention/residual, minGPT's pre-LN block
and its `_init_weights`.**

| | theirs | here | why |
|---|---|---|---|
| input | `nn.Embedding(61, 512)` | `Linear(128, 512)` | `nn.Embedding` is `Linear` applied to a one-hot; on a continuous 128-d scan there is nothing to index. This is the same operation generalised, **not** a deviation |
| output | `Linear(512, 61)` + cross-entropy | `Linear(512, 128)` + MSE | continuous observation |
| `block_size` | 59 | **39** | 40-frame episodes give 39 inputs. Do **not** regenerate discworld at 60 frames — it would move `edit_frame` and break every existing comparison |
| **no ReLU after the input projection** | — | — | `W16` uses `Linear + ReLU` so residual point 0 equals the GRU's encoder port. Theirs has no nonlinearity there. **Fidelity wins; drop the ReLU** and accept that encoder-space-editing results do not transfer (Sevan, 2026-08-21) |

⚠ **`d_model` 512 violates `TRANSFORMER_RUNS.md`'s "matches the GRU hidden size" rule**, which
exists so that row-space fractions and √(d/H) chance levels stay comparable. **Sevan authorised
this violation explicitly** (2026-08-21), on the same footing as a hidden-size ablation. State it
in every figure caption that reports a geometry metric; those numbers do **not** go in a shared
table with the GRU's.

### 3. Training — their recipe

`mingpt/trainer.py` verbatim: AdamW, decoupled weight decay 0.1, betas (0.9, 0.95), grad-clip 1.0,
**token-counted** schedule (linear warmup over `warmup_tokens`, cosine decay to a 0.1x floor at
`final_tokens`), **lr 5e-4, batch 4096**.

⚠ **Their trainer has no early stopping** — it saves the best checkpoint if given a test set and
otherwise saves every epoch, but always runs all `max_epochs`. Their own notebook passes
`test_dataset=None` and runs exactly **250 epochs**.

⛔ **You cannot early-stop this schedule.** The LR is counted against
`final_tokens = len(train) x block_size x max_epochs`; stopping at epoch 20 of a 250-epoch cosine
leaves the model at ~95% of peak LR and undertrained. **Choose `max_epochs` up front and let the
cosine complete.**

Their full recipe at our scale is ~244B presentations ≈ **4–6 days on one 5090** (they used 8
GPUs). Instead:

1. **Pilot at `max_epochs = 4`** with a val split, ~2–3 h. Read the val curve.
2. Set the final `max_epochs` from it and run the complete schedule (~5–12 h expected).

Calibration for the pilot: `W16`'s *entire* budget is 90k x 300 = **27M episode-presentations**. At
25M episodes, **one epoch already exceeds that**, with 278x more unique data. Convergence in single
digits of epochs is the expectation; if the val curve is still descending steeply at epoch 4,
reconsider.

**Training target is the noisy `obs_intensity`, including position noise** — Sevan's call
(2026-08-21). Their target has no noise, so this preserves an asymmetry deliberately: it keeps this
run to architectural and data variables only. Noise-free is the **follow-up**, and the existing GRU
noise ablation already suggests noise is not the cause.

### 4. Probe and edit

Fit **our** probes with `othello_probe.fit_probe`, unchanged, one per residual point (0…8 —
**nine** points for an 8-block model), on the 200k probe split. Fit linear and MLP families and
both targets the thread uses. Then run **our** editors, unchanged, on `edits.h5`.

## Measurement

**Metrics** — all from `METRICS_AND_EDITORS.md` §4, unchanged, so the numbers are directly
comparable to every other model in the thread:

- **Edit Index** — `(d_uned − d_edit)/(d_uned + d_edit)`, `d_· = RMSE(edited₀, gt_·)` over
  **differing rays**, −1…+1, ↑. **The headline.**
- **Edit Index by step** over the K = 15 rollout — report whenever the step-0 index is reported.
- **Target / Ghost / Collateral / Edit-frame RMSE**, observation units, ↓.
- **fidelity ratio** — GT-traj RMSE(editor) / GT-traj RMSE(unsteered), ↓; **> 1 means the edit
  degraded the rollout rather than steering it.** Report beside any success claim.
- **Probe quality** — held-out R², by residual point, **split by sequence** (`ANALYSIS.md` §2).
- **Next-step val MSE** against **clean** observations (`ANALYSIS.md` §6), for the convergence read
  and to confirm the model trained at all.

**Controls / baselines (mandatory):**
- **Unsteered** — rollout from the un-edited state, computed through the identical code path.
- **`W16`** — cited from `othello_gpt/`, not recomputed: unsteered −0.684, best write −0.194.
- **Random-init control** — the same architecture with random weights, probe-read at every point.
  2026-08-21 showed roughly half of `W16`'s linear decodability is architectural; at `d_model` 512
  and 8 layers that baseline will differ and must be measured, not assumed.

**Decision rule, stated before running.** On `edits.h5`, with the fidelity ratio inside the 1.05
guard, does any probe-derived editor reach an **Edit Index > 0** (i.e. the generation is closer to
the edited world than the unedited one)?
- **No** → architecture and data are both eliminated. The negative is about the world or the
  read-out. Proceed to the discrete/logits ablation.
- **Yes** → bisect: full-causal vs windowed, then capacity, then depth, then data. Run B becomes
  worth running.

**Interpreted magnitudes.** Edit Index is dimensionless on ±1. RMSEs are in observation-intensity
units; the reference scale is the unsteered arm's own value on the same rays. Probe R² is against
the train mean.

## Visualization

- **Canonical qualitative panel:** `pim.figures.waterfall_grid` via
  `notebooks/experiments/editability/WATERFALL_SPEC.md`, columns = GT (sim) · Unsteered ·
  every editor arm · First Obs. TF. **≥ 3 rows, samples drawn at random with the selection rule in
  the title** (an extreme-case panel cost this thread a misread on 2026-08-18).
- **Probe quality by residual point**, trained vs random-init, both probe families — the 2026-08-21
  format.
- **Edit Index by applied residual point**, absolute, with the unsteered floor and +1 both marked.
- **Edit Index across the K = 15 rollout**, with each arm's own baseline.
- **Training curve** — val MSE against the `W16` (0.02359) and GRU (0.02362) reference lines, which
  is also the sanity check that the port trained correctly.
- Both plots **and** printed tables.

## What would falsify the hypothesis

Any probe-derived editor reaching **Edit Index > 0 at fidelity ≤ 1.05** on `edits.h5`. That is the
same bar every other architecture in the thread has failed, and it is the bar `W16` misses at
−0.194.

Two ways the run could be **uninformative** rather than falsifying, both of which must be checked
before any conclusion is drawn:
1. **The model did not train.** Val MSE not reaching the `W16`/GRU reference band means the port
   is broken, not that the architecture is uneditable.
2. **The probe did not fit.** Held-out R² well below `W16`'s 0.934, or the `MLP ≥ linear` tripwire
   firing with a real gap, means the read-out is broken. Check the random-init baseline in the same
   breath — at this width and depth it will be higher than `W16`'s.

## Open decisions (Sevan)

1. **Pilot `max_epochs`** — proposed **4**, then set the final count from the val curve.

*(`always_in_frustum` is **settled: `true`**, matching dataset 4 — Sevan, 2026-08-21. It was never
a real decision; I raised it off a weak inference from the occlusion analysis that Sevan did not accept,
and then kept carrying it as open. Removed here so it stops resurfacing.)*

## Expected artifacts

- notebook: `notebooks/experiments/editability/othello_arch/othello_arch_discworld.ipynb`
- modules beside it: the model wrapper, the corpus generator, the run registry
  `OTHELLO_ARCH_RUNS.md` (checkpoint, corpus seed range, config hash)
- scratch note: `research/scratch/YYYY-MM-DD-othello-arch-discworld.md`
- figures: `runs/othello_arch/figures/`
- checkpoint: `runs/othello_arch/<code>/best_model.pt` (gitignored)

## Provenance of the decisions above

Settled with Sevan across 2026-08-20/21: noisy objective · no ReLU · dropout 0.1 kept · their
Trainer and schedule · batch 4096 · `block_size` 39 · `d_model` 512 with the registry violation
authorised · seed-range asserts · reuse dataset 4's test/edits · 200k-episode probe split · store
the corpus · run A only, run B only if A shows editability · multiple variables moving at once
accepted deliberately.

**Pinned, explicitly not part of this run:** variable-length / collision-terminated episodes ·
frustum-coordinate (angle, depth) probe targets · off-manifold-distance correlation with edit
failure · GRU replication of the probe-data scaling · restoring genuine partial observability by
varying radius or per-episode reflectivity (2026-08-21 found dataset 4 is nearly fully observable
from a single frame, which weakens the project's "impoverished observation" premise but makes this
run *more* like their setting, not less).
