# Latent-linearity thread — canonical RUN REGISTRY

**The single source of truth for what every run code in `notebooks/experiments/editability/latent_linearity/`
means.** Per `harness/STYLE.md` §5: no notebook may use a run code without copying its row into its own
definitions table, and **figures use the descriptive label, never the raw code**. Adding a run means adding its
row here in the same commit.

Branch `latent_linearity`. Origin: Sevan, 2026-08-19 — extend the ground-truth edit-direction analysis of
`delta_h_analysis` (2026-08-03, GRU + RSSM) to all four architectures, and ask whether the two *learned* edit
pathways write the same latent displacement as the training-free oracles.

> ### ⛔ No models were trained for this thread
> Every checkpoint below is defined by another thread's registry and is used here unchanged. This registry
> records **which** checkpoint, **which state object**, and **why that one** — the second column is the part
> that is specific to this thread and cannot be found in the source registries.

## The unit of analysis is (checkpoint, state object)

An architecture can carry more than one thing that deserves the name "state", and they come apart
(`research/findings/architecture-independence.md`, 2026-08-04). A transformer *carries* a buffer of raw
observations and *recomputes* a residual stream; a latent DiT carries a learned code. So every figure names the
pair, never just the architecture.

## Part 1 — all four architectures (mechanisms 1 and 2, which always apply)

| descriptive label (use this in figures) | run code | source registry | state analysed | `H` | why this state object |
|---|---|---|---|---|---|
| **GRU · hidden state (H=256)** | `H256` | [`../controls/CONTROL_RUNS.md`](../controls/CONTROL_RUNS.md) | the recurrent hidden state `h` | 256 | the only one it has; the thread's baseline capacity |
| **RSSM · det+stoch state (H=320)** | `4_dset4_refined_best` | `research/scratch/2026-06-29-rssm-refinement.md` | `cat(h_det, s_stoch)`, prior mean (`sample=False`) | 320 | the only one it has; matches `delta_h_analysis`'s RSSM |
| **Transformer · residual stream (H=256)** | `W16` | [`../transformers/TRANSFORMER_RUNS.md`](../transformers/TRANSFORMER_RUNS.md) | residual stream at the final point (`probe_layer=4`), current position | 256 | the analogue of `h`. Its *carried* state is a 61-frame buffer of raw observations — not a learned representation, so a Δh measured there would be observation-space geometry, which `orthogonal_edits` already covers |
| **Latent DiT · latent window (H=64)** | `0_latent_dit_z16_w4` | [`../latent_DiT/LATENT_DIT_RUNS.md`](../latent_DiT/LATENT_DIT_RUNS.md) | the carried 4×16 normalised-latent buffer | 64 | the opposite case: here the *carried* state is a learned code. Its activations are contaminated with the denoising iterate at every depth (that registry's own warning), so the latent window is the honest choice |
| **DiT (pixel) · residual stream (H=256)** | `9_dset4_dit_w4_d256` | [`../DiT/DIT_RUNS.md`](../DiT/DIT_RUNS.md) | final-block token features at the current position | 256 | the pixel-space counterpart of the latent DiT, included precisely because the two differ in which object is learned |

All five are trained on `datasets/4_fixed_refl_inview` (2 objects, 40 frames, `obs_res` 128, observation noise
0.2, position noise 0.04, edit frame 20) and evaluated on its `edits` split, N=256, K=15.

**Prediction modes.** RSSM `sample=False` (prior/posterior mean). DiT family `predict_mode="mean"` — the
deterministic conditional-mean readout, the GRU-comparable mode. Not `sample_fresh`: this thread's rollouts are
scored against the clean render, and a generative sample reproduces a noise realisation, which is faithfulness
rather than error but is not what the §4 metrics are calibrated on.

## Part 2 — teleport-trained models (mechanisms 3 and 4)

| descriptive label | run code | source registry | action channel | teleports in training | role |
|---|---|---|---|---|---|
| **GRU · teleport actions given (H=256)** | `XG_A_H256` | [`../action_hidden_size/ACTION_SWEEP_RUNS.md`](../action_hidden_size/ACTION_SWEEP_RUNS.md) | **yes** — `ActionGRUContinuousModel`, continuous teleport-to-absolute-coordinate actions | yes, always cued by an action | the one checkpoint where all four mechanisms exist |
| **GRU · teleports observed, no action channel (H=256)** | `XG_C_H256` | same | no | yes, uncued from the model's point of view | identical recipe and data with the action input removed — isolates action-knowledge from capacity |
| **GRU · never saw a teleport (H=256)** | `H256` | [`../controls/CONTROL_RUNS.md`](../controls/CONTROL_RUNS.md) | no | no | same architecture and capacity with neither; the Part 1 GRU, on the same episodes |

`XG_*` are trained on `datasets/7_cont_teleport` (`p_action = 0.30`, `move_scale = 4.0`, otherwise matched to
dataset 4). Evaluated on **`datasets/15_teleport_eval_single/eval.h5`** — 4,000 sequences at base seed 200000,
generated with `--p-action 0.0` so the world performs **no teleports of its own**; the single edit under test is
synthesised at the edit frame by `scripts/eval_action_sweep.xg_data`, which asserts the set is
intervention-free (`research/GOTCHAS.md`, 2026-08-14).

## Which checkpoints could support Part 2 and do not exist

Checked against every checkpoint under `runs/` on 2026-08-19:

| architecture | teleport actions in its action space | teleports in its training data |
|---|---|---|
| GRU | `XG_A_*`, and the earlier `M_teleport` / `8_action_cond_gru_400ep` | `XG_A_*`, `XG_C_*`, `M_teleport*`, `9_perturbed_passive_gru_400ep` |
| RSSM | **none** — `runs/endogenous_rssm/R*` are the only action-conditioned RSSMs and their actions are **forces**, which cannot express a teleport | none |
| Transformer | none | none |
| DiT / latent DiT | none | none |

So Part 2 is GRU-only. That is a fact about the checkpoint inventory, not about the architectures, and it is what
a follow-up would have to train to make mechanisms 3 and 4 architecture-independent the way 1 and 2 now are.

## Part 3 — the composition random-init control (added 2026-08-21)

`random_baseline.ipynb` asks whether latent object-composition is *learned*. It needs a trained
model and an **untrained** one at the identical config, so the pair is the unit of analysis here.

| descriptive label (use this in figures) | run code | source registry | state analysed | why this one |
|---|---|---|---|---|
| **TRAINED linear enc+dec** | `H256` | [`../controls/CONTROL_RUNS.md`](../controls/CONTROL_RUNS.md) | GRU hidden state `h` (256) | affine encoder and decoder — the family `delta_h_analysis` §7 measured |
| **TRAINED nonlinear enc+dec** | `NL_enc2dec2_s0` | [`../nonlinear_gru/NONLINEAR_GRU_RUNS.md`](../nonlinear_gru/NONLINEAR_GRU_RUNS.md) | GRU hidden state `h` (256) | 2-hidden-layer encoder **and** decoder — the family the 2026-08-05 decoder-artifact correction moved the claim onto |
| **RANDOM s0 / s1 · {linear, nonlinear} enc+dec** | — | *(constructed in `composition_lib.models`)* | GRU hidden state `h` (256) | `GRUModel(ModelConfig(...))` at the identical `enc_hidden_layers` / `dec_hidden_layers`, `torch.manual_seed(s)`, never trained. Two seeds because one cannot distinguish an architectural floor from a lucky draw |

Evaluated on `datasets/4_fixed_refl_inview/edits.h5`, N = 256, edit frame 20, at four displacement
scales {1.0, 0.5, 0.25, 0.125} of the recorded teleport. **No models were trained**; the random
arms are constructed at load time and are reproducible from the seed alone.

## Code in this directory

| file | what it is |
|---|---|
| `edit_directions.py` | the one implementation of the four mechanisms' Δh and the geometry metrics — state plumbing, evidence rendering, cosine / magnitude / consistency / row-space reports |
| `composition_lib.py` | the composition measurement — the real-teleport edit set (with the un-teleport reconstruction), the four counterfactual renders, the observation ceiling, the trained/random model set, and the metrics |
| `figures.py` | figure builders; every one takes `models` and `arms` as ordered lists so an added architecture or mechanism is a data change, not a re-layout |
| `latent_edit_directions.ipynb` | the notebook — orchestration, tables, and the dated results block |
| `random_baseline.ipynb` | the composition random-init control — orchestration, tables, and its `Summary` block |

Metrics come from `scripts/editability_metrics.py` and probes from `pim.extractors.fit_readability_probes`;
neither is re-derived here. Metric definitions: [`../METRICS_AND_EDITORS.md`](../METRICS_AND_EDITORS.md) §4–§5.
