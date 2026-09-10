# REGISTRY — the canonical objects, one table per category

_The single source of truth for what "the probe", "the editor", "the metric" mean.
Every category grows only as needed, and an entry is canonical only if it appears here.
Code paths are the definitions; this file is the index, not a re-derivation. Created
2026-08-31 (housecleaning); supersedes `notebooks/experiments/editability/METRICS_AND_EDITORS.md`._

## Environment instances

An **instance** = one environment class at one fixed generation config, packaged with the
data for every split it defines. Its `instance.json` is a hand-written summary for
humans (2026-09-07: never read by code, never a source of truth); the machine-written
contracts are `train/corpus.json` (`bigcorpus.verify()`) and each split's `config_json`
HDF5 attribute / `dataset.json` (`generate_dataset.py`).
Data-scale comparisons mask one instance's train pool; changed generation params = a new
instance (the dset-17 lesson: it silently used position noise 0.0 and every result on it
was uninterpretable against the canonical eval).

| shorthand | class | definition | splits |
|---|---|---|---|
| `dw-pn04` | discworld | `datasets/discworld/dw-pn04/instance.json` | train 20M (memmap) · probe 120k · eval 10k+10k · edits 10k (EF=20) |
| `dw-noiseless` | discworld | `datasets/discworld/dw-noiseless/instance.json` | identical to `dw-pn04` in every field **except** `obs_noise_std = 0.0` AND `position_noise_std = 0.0` — the noise ablation at mass scale (2026-08-31). Same split sizes; seeds are a FRESH block (base 30e9) |
| `dw-8ray` | discworld | `datasets/discworld/dw-8ray/instance.json` | `dw-noiseless` with the disc radius **doubled (1.0)** and the observation cut to **8 usable rays** — 10 cast with the unchanged caster geometry, the two frustum-wall rays dropped (`SimConfig.drop_edge_rays`, `obs_dim = obs_res − 2`) — the ray-count ablation (2026-09-03). Same split sizes and recipe; seeds a FRESH block (train 60e9, eval 85e9, probe 980e9, probe_large 990e9); edits generated with `--max-edit-attempts 2000` (radius-1 teleports need more draws). At 8 rays ~15 % of edit cases have no differing ray, i.e. no Edit-Index support ⛔ **Filtered edit bench (2026-09-08):** with 8 rays a teleport renders an IDENTICAL frame in 20% of cases and moves a single ray in another 22%, so scoring the first 192 cases used only 163, many marginal. Both 8-ray runs now score `edits_selection.json` — the first 192 cases with ≥2 differing rays, same generator/seeds/split, 192/192 scoreable, mean teleport 2.82 vs 2.16. ONE list shared by the ray-zone and frames-as-tokens benches so the interface ablation stays paired; built by `experiments/interface_ablation/edits_audit/`. |
| `dw-8ray` · **tokens** | discworld, frames as tokens | `datasets/discworld/dw-8ray/tokens/` (`meta.json`, `vocab.npz`; built by `scripts/make_discworld_tokens.py`, `pim/environments/discworld/tokens.py`) | NOT a new instance — a representation of `dw-8ray` (2026-09-05): the same splits with every frame one token of a 422-token vocabulary (421 realisable 8-ray patterns over ray values {0, 0.4, 0.8} + UNK = id 0; ids in ascending pattern code; built over every split, so no eval frame is unseen — `frames_only_outside_train: 0`). `train.i16` (20M, 40) int16, `probe/val/test/edits.npy`. Consumed by `scripts/train.py --repr tokens` and `token_bench`. |
| `oth-uniform` | othello | `datasets/othello/oth-uniform/instance.json` | train [0,20M) · test [90M,+10k) · probe [91M,+20k) · edits = Li's 1001 cases |
| `oth-noflip` | othello | `datasets/othello/oth-noflip/instance.json` | `oth-uniform` with ONE rule changed — a placed disc never recolours the discs it encloses (`OthelloBoardState(flip=False)`); legality, passes, game end, sampling and index law unchanged. ⚠ **Collapses to a colour-free occupancy game** (theorem, 2026-09-07): from the checkerboard opening the enclosure rule keeps colour == parity of r+c forever, every game ends in the SAME full checkerboard, and the legal set equals "mover's parity class + two occupied squares in a row/column direction" (60,000/60,000 positions) — colour never enters the dynamics. Nobody passes; Bayes CE 1.672. Edits = 1001 synthesised cases (`scripts/make_othello_edits.py`). A no-flip world where colour MATTERS needs a non-checkerboard opening (`findings/flip-ablation.md` §4). |
| `oth-adjacent` | othello | `datasets/othello/oth-adjacent/instance.json` | `oth-noflip` with the PLACEMENT rule changed (2026-09-08): a move is legal iff the empty square touches one of the mover's own discs in the 8-neighbourhood (`OthelloBoardState(flip=False, placement='adjacent')`); no recolouring, same passes / game end / uniform sampling / index law / vocabulary. Colour is therefore causally relevant WITHOUT the enclosure geometry — the property the checkerboard theorem took from oth-noflip. Pilot (20k games): 68% of random recolourings change the legal set (mean 2.4 squares), colour == square parity only 45%, 19,999 distinct terminal boards, all games 60 moves, 1.1 passes/game, Bayes CE 2.43. Colour is still trivially decodable from the input (parity of the placing move) — the 'used but input-decodable' cell. Edits = 1001 synthesised cases (`scripts/make_othello_edits.py`). |
| `oth-adjacent-flip` | othello | `datasets/othello/oth-adjacent-flip/instance.json` | `oth-adjacent` with RECOLOURING back on (2026-09-09): a move is legal iff the empty square touches one of the mover's own discs (8-neighbourhood), AND the placed disc recolours the discs it encloses along any line, exactly as in Othello (`OthelloBoardState(flip=True, placement='adjacent')`); same passes / game end / uniform sampling / index law / vocabulary. Colour is used by legality AND rewritten by the dynamics, so it is no longer decodable from the input — the 'used and not input-decodable' cell oth-adjacent could not reach. Pilot gate and corpus pending; edits = 1001 synthesised cases (`scripts/make_othello_edits.py`). `experiments/adjacent_flip_ablation/`. |
| `dw-blink` | discworld | `datasets/discworld/dw-blink/instance.json` | `dw-noiseless` (128 rays, no noise, radius 0.5) plus **blackouts** (2026-09-07): an object leaves the OBSERVATION for min(12, Geometric(1/7)) frames (realised mean ~5.3; prob 0.05/object/frame; never before frame 3; never both at once; ~16% of frames hidden per object) while its physics continues, and the frame before + the last hidden frame carry a **0.5 marker on its edge ray** (ray 0 / ray 127; `obs_id` code −2−j; `pim/environments/discworld/blink.py`). The schedule is a function of the seed, so the edits split hides the same frames as the unedited world. Makes position a NECESSARY carried state (the reappearance frame is unpredictable from the current frame). Eval has **20k edits** so the subsets are populated (~3% reappearance at frame 20, ~18% mid-blackout, ~79% visible); canonical bench = first 192 as everywhere. Seeds train 110e9, eval 135e9, probe 1000e9, probe_large 1010e9. |

⛔ **Seeds are never shared across discworld instances**, even to "pair" worlds:
`always_in_frustum` accepts initial conditions by simulating forward, and the noise
draws are consumed *inside* that acceptance loop, so the same seed with noise off gives
an unrelated world (measured 2026-08-31: 0/5 IC matches, ~5-unit divergence). The
per-instance seed blocks and their disjointness proofs live in
`pim/environments/discworld/bigcorpus.py::INSTANCES`.

## Architectures

Identical across environments up to the input/output projection and the loss — the
invariant `pim/models/` exists to protect. Surface: `pim/models/protocol.py`.

| shorthand | params | definition | task forms |
|---|---|---|---|
| **Transformer-S** | ~3.2M | `pim/models/transformer_s.py` — banded-causal, RoPE, pre-norm | `TransformerS` (MSE) / `TransformerSTokens` (CE) |
| **Transformer-L** | ~25M | `pim/models/transformer_l.py` — Li et al.'s minGPT (vendored), full causal, learned absolute positions. **One class, two interface parameters** (2026-09-09): `input` linear (float observations) \| embedding (ids) and `head` regression (emits a frame; rollout surface) \| categorical (emits a distribution; step-0 only; `output_kind` logits \| raw). The interface is a run parameter (`scripts/train.py --repr frames\|tokens --objective ce\|mse_onehot`), never a property of the environment; the scorer dispatches on what the model emits. Unification gated on all 28 canonical L checkpoints: identical fingerprints, keys and forwards | `TransformerL` (linear + regression, MSE — the discworld preset) / `TransformerLTokens` (embedding + categorical — the Othello preset, also discworld frames-as-tokens) |
| **Recurrent-L** | ~25.4M | `pim/models/recurrent.py` — stacked GRU, 4 × 1024, dropout 0.1 between layers; parameter-matched to Transformer-L (2026-09-02). The **recomputation** test: its hidden state is the only summary of the past, so a write cannot be overwritten by later layers re-deriving the state from earlier positions. Residual point ℓ = layer-ℓ hidden; edits are **carried** in the state (`carry_edits`, the architecture-forced difference; `False` = the transformer's carry-nothing semantics); `state_span` unbounded | `RecurrentL` (MSE); no token head yet |

## Training setup

ONE loop (`pim/training/train.py`), three objectives — `mse_next_obs` (discworld, next
frame), `ce_next_move` (Othello, canonical), and `mse_next_move_onehot` (Othello, MSE of the
61 head outputs against the one-hot next move, 2026-09-04; `scripts/train.py --objective
mse_onehot`, the model then carries `output_kind="raw"` and every scorer reads its head
as-is through `othello.data.move_probs` — GOTCHAS 2026-09-04) — fed by per-environment
`DataSource`s (`pim/training/sources.py`). The canonical recipe (the matched-BIG20M
hyperparameters) IS the `TrainConfig` defaults: AdamW 1e-3 / wd 1e-4 / clip 1.0 /
batch 256 / 2k-step warmup then **constant** LR / seed 0. Entry: `scripts/train.py`.
Every run writes `config.json` + `commit_sha` + `metrics.jsonl` + arch-stamped
checkpoints into `runs/<topic>/<name>/`.

`--limit N` trains on the first N sequences of the pool AND validates on the last 10%
*of that prefix*, so the training-time `val_loss` is a different set at every data-scale
rung and is not comparable across rungs. Every canonical score (probe fits, the
editability bench, the Othello gates) uses the instance's fixed eval/probe/edits
splits, which do not depend on `--limit` and are comparable across rungs.

## Probes

Always held out **by sequence**, never by frame (Othello's frame split is kept only as
Li's replication anchor and always labelled). Cached with the model fingerprint in the
key (`pim/probes/cache.py`). A token model is probed on TOKEN inputs through
`bench.fit_probes(encoder=…, encoder_tag=…)` — the same frames, same span truncation, same
targets; the tag joins the cache key (2026-09-05).

| shorthand | definition | fit |
|---|---|---|
| **LIN** | `pim/probes/linear.py` — one affine map, standardised both ends | closed-form lstsq (regression) / SGD-CE (classification) |
| **MLP-128** | `pim/probes/mlp.py` — Li's §3.2 shape: ONE hidden layer × 128 | SGD, loss in standardised target space |
| NULLSPACE (non-default) | `pim/probes/nullspace.py` — deflation cascade of orthogonal linear probes | float64 min-norm lstsq, orthogonality asserted |

Tripwire: `check_probe_sanity` (MLP ≥ linear on held-out data, or the fit is
memorisation) runs on every paired fit; violations are recorded in `scores.json`.

**The canonical probe grid.** Othello: **2 × L** (LIN, MLP-128 × residual point) at the
locked mine/theirs target, sequence split. Discworld: **2 × 2 × L** (basis × family ×
point) — one probe set per basis, fitted on the **FULL** state.

**Probe targets** (2026-09-09). A probe target is what the probe is asked to read; each
one is its own scored block in `scores.json` and its own ROW of the master tables (never a
column that exists for only some runs). `kind` decides the fit, the skill formula and which
editors apply.

| target | environment | kind | definition | fit recipe | where it exists |
|---|---|---|---|---|---|
| `full` × basis (`cartesian`, `frustum`) | discworld | regression, 8 outputs (u/x, 1/y or y, per object; + velocities) | `bench.bench_arrays`, `arms.fit_probes` | canonical: `probe` split, 30k seq, 200 epochs (LIN closed-form) | every discworld run |
| **`grid-16x8`** | discworld | classification, 128 cells × 3 {empty, obj 0, obj 1} | `pim/environments/discworld/grid_target.py` — cells uniform in the frustum basis (u′, 1/y) over the reachable region; nearer object wins a shared cell; a same-cell teleport is a no-op and is dropped from its bench (`bench.grid_selection`) | `arms.GRID_PROBE_RECIPE`: `probe_250k` split, 200k seq, 50 epochs, streamed (`fit_probe_stream`); LIN is SGD-CE (no closed form) | `noise_ablation/L-dw-noiseless-20m` ONLY (probes fitted 2026-09-08 in `experiments/grid_target_control`, re-keyed into the run's `probes/` 2026-09-09; never fitted by the scorer — `require_cached`). Opt-in per run: `master_eval` SETTINGS `dw_extra_targets` |
| **`appearance`** (+ `-d2`, `-d3`, `-lat`) | discworld | classification, cells = the RUNS of rays a disc lights (the observation-exact partition); dw-8ray: 30 cells × 3 = 90 logits; `-d<k>` splits each run into k depth bands (uniform in 1/y over the run's depth range), `-lat` merges runs by centre (15 cells) | `grid_target.AppearanceTarget` — `covered_rays` is the renderer's own ray–disc test (gated equal to `render_frame`); the realisable runs come from a deterministic dense sweep of the reachable region | `GRID_PROBE_RECIPE` (200k seq, 50 epochs, streamed), fitted by `scripts/fit_probes.py` | `ray_ablation/L-dw-8ray-20m`, `interface_ablation/L-dw-8ray-tok-20m` (overnight 2026-09-09→10, with the grid variants `grid-16x8`, `grid-8x4`, `grid-32x16` as the resolution sweep). Every teleport of the filtered 8-ray bench changes cell; two objects share a cell in 0.31% of frames |
| `mine` (mine/theirs) | othello | classification, 64 tiles × 3 | `othello.data.tokens_and_labels` | 20k probe games, 200 epochs | every Othello run |
| **`mine_signed`** | othello | **regression**, 64 outputs: +1 mine, 0 blank, −1 theirs (`othello.data.signed_mine`) — the same information and frame as `mine`, read by a regression probe: Othello's counterpart of the grid control | `othello.arms.fit_probe_grid(targets=("mine_signed",))`; editors through the regression branches of `linear_arm` (PI: the tile's read-out set to ±1, z-space + y-affine; ND: the tile's row × the sign of the flip, constant magnitude) and `grad_steer_arm` (MSE spec) | 20k probe games, 200 epochs, fitted inline by `master_eval` | every Othello run (block `bases["mine_signed"]`; overnight 2026-09-09→10) + floors on all three instances |

ND is reported on classification targets (Othello and the grid: one categorical change per
case) and never on the regression target. PI's categorical target is the probe's own
read-out with two classes swapped at the edited tile(s) — ONE helper,
`pim.editors.pinv.swap_class_logits`, spelled once for both environments. Probe Skill is
read off any fit through `pim.metrics.probe_skill_from_stats` (R² / 1 − err/majority).
Extra targets are opt-in per run in `master_eval` SETTINGS (`dw_extra_targets`,
`oth_extra_targets`); discworld ones are fitted deliberately by `scripts/fit_probes.py`
(the scorer only ADDS blocks whose probes exist), Othello ones inline.

⛔ **Discworld no longer fits position-only probes** (2026-09-01). Editability instead
sweeps each editor over two *dim sets* through the one full-state probe —
`pim.environments.discworld.bench.DIM_SETS`: `"pos"` drives the position read-outs only
(the dropped dims become hold-the-rest constraints, not free ones), `"all"` drives the
whole state — and the better arm is the reported one, tagged with the dim set that won
it. Nothing is lost: for the **LIN** probe the position rows of a full-state least-squares
fit are BIT-IDENTICAL to a position-only fit, because multi-output least squares
decomposes per output dimension (verified on cached probes: `max|W_full[:4] − W_pos| =
0.0` in both bases), so `"pos"` reproduces the retired probe exactly for PI and ND. The
**MLP** does not decompose — its hidden layer couples the outputs — so for GS the two dim
sets are genuinely different edits, which is why both are swept rather than one assumed.

## Decodability baselines

Two floors, computed **per environment instance** (not per run) and reused by every run on
that instance. They live in `runs/_baselines/<instance>/` — the `_` prefix is the marker
`scan_runs` and `build_full_table` already use to skip a directory, so they sit in `runs/`
without ever being mistaken for a trained run. Rendered as **Table 3**, deliberately its own
table.

| shorthand | definition | what it rules out |
|---|---|---|
| **observation** | `pim/probes/baselines.py::CausalHistory` + `fit_baseline_probe`; entry points `discworld.arms.observation_probes` / `othello.arms.observation_probes` | that the state is simply sitting in the input in probe-readable form. The feature at frame *t* is the zero-left-padded history `obs[0..t]` — exactly what the model has consumed when probed at *t* |
| **random-init** | `pim/probes/baselines.py::random_init_model` + the ORDINARY `fit_probes` path | that the skill comes from random features of the right shape rather than from training. A different MODEL, never a different measurement |

Matched to the model probes in everything else: same families, same `n_seq`, the same
seeded 80/20 split **by sequence** (identical permutation, so literally the same held-out
episodes), same targets and bases. **The binding floor is the higher of the two.**

⛔ **Both floors are keyed per (instance, architecture)** — `baselines.json` is
`archs.<arch>.bases.<basis>.<floor>`, stamped with its own `BASELINE_VERSION` (independent
of `EVAL_VERSION`, so a change to the editor sweep cannot invalidate a floor). Random-init
obviously depends on the architecture; **observation does too**, less obviously, because the
probe is handed the history the model actually consumes and `state_span` is architecture-
dependent (`transformer_l` = `block_size`; `transformer_s` = `n_layers*(window-1)+1`).
Keying on the instance alone would silently have compared a Transformer-S run against a
Transformer-L floor. Both kinds share one `probes/` dir per instance — every cache key
carries the model fingerprint, and the model-free observation key carries the span — so
architectures cannot collide and a fit shared between them is computed once. The scorer
fills in only the architectures a file is missing.

Every fit reports `insample_gap` (in-sample − held-out, on the skill scale) — the overfit
check, Table 3 panel (b). The observation probes read a much wider input (discworld
39×128 = 4,992 features vs the model's 512), giving ~1.5 rows/parameter where the model
probes get ~14, so a large gap there means that floor is an **under**-estimate.

**Probe capacity (Fig 2, `experiments/probe_capacity/`, 2026-09-02).** The floors above are
at MLP-128 on the canonical corpus. Sweeping width LIN → 2048 on 5× the rows shows the
random-init floor *plateaus* — 0.975 on discworld, 0.60 on Othello — below the trained model
in both worlds; the observation floor at the canonical size is an under-estimate (0.70 → 0.88
on discworld with 5× data). See `findings/probe-capacity.md`.

⚠ An Othello board is a **deterministic function of the move sequence**, so its observation
floor is decodable in principle; a low number means "a shallow readout cannot compute the
flip rules", never "the information is absent". Discworld's observation is genuinely lossy
(noisy, and depth is never directly observed — see `research/GOTCHAS.md`).

**Two history layouts for the observation floor (b4, 2026-09-06).** The observation probe
reads the causal input history as a fixed-width vector; `CausalHistory(align="left")` (the
original) puts frame j in block j and zero-fills after the present, so the CURRENT frame sits
in a different block for every row and a LINEAR probe cannot express even a current-frame
lookup (tokenised dw-8ray: left-aligned one-hot LIN 0.726, the lookup alone 0.968).
`align="right"` lays the history out relative to the present (block 0 = now, block k = k
steps back). `baselines.json` carries `observation` / `observation_large` (left) and
`observation_right` / `observation_right_large` (right); Table 3 shows both. Read the
right-aligned LIN cell as the linear floor; the MLP barely differs. Cache keys of the left
fits are unchanged (`align` enters the key only when "right").

## Editors

| shorthand | definition | what it writes |
|---|---|---|
| **PI** | `pim/editors/pinv.py` — pseudoinverse injection, **z-space + y-affine** ("zspace") | the minimum-norm Δh that lands the LIN read-out on the target; α=1 = the exact jump. `"legacy"` reproduces pre-2026-08-31 discworld numbers (the y-affine bug) and is never quoted as PI |
| **ND** | `pim/editors/nanda.py` — Nanda direction addition | α·‖x‖·d̂ along the probe weight rows, standardised. Canonical form is the **target−current contrast** (`subtract_rows`, formerly reported as "ND-sub"): it beat the plain target-row form on every arm (+0.622 vs +0.447, fid 0.23 vs 0.34) and is the more principled direction. ⛔ **Not applicable on discworld** — one fixed direction with a swept scalar is coherent only when the target is CATEGORICAL (flip a tile: same change every case); for a continuous per-case target no single magnitude can serve 192 teleports of differing distance and direction. Discworld ND arms are still computed into `scores.json` but are omitted from the tables (2026-09-01) |
| **GS** | `pim/editors/grad_steer.py` — Li §4.1 MLP gradient steering | descent on the activation through the frozen MLP-128 probe, sequentially from L_s across every later point. ⛔ `target_labels` MUST share a coordinate frame with the probe being steered — a mismatch converges onto a well-formed but WRONG class and looks like a failed editor (the 2026-08-31 GS-mine bug, worth 0.70 Edit Index; `scores.json::probe_sources` records the pairing) |
| nullspace (non-default) | `pim/editors/nullspace.py` | Σₖ Aₖ⁺(tₖ − pₖ(h)) over the whole cascade — the row-space objection's answer |
| oracle: overwrite | `pim/editors/oracle_overwrite.py` · bench wiring `discworld/arms.py::overwrite_oracle_rollout` | the state the model would carry had it SEEN the edited world for the whole window — `counterfactual_history` renders frames 0..EF−1 with the edited object displaced by its teleport vector (noise-matched), then a free-run aligned with every other arm. ⛔ A ONE-frame overwrite (the post-edit frame appended to the pre-edit window) is not accepted by a window model: EI +0.04 on L-dw-20m (2026-09-07) |
| oracle: freeze-interp | `pim/editors/freeze_interpolation.py` · `discworld/arms.py::freeze_oracle_rollout` | N rendered frozen frames (the edited object glides pre → target, time frozen) teacher-forced through the observation channel, then a free-run |

The two oracle editors exist to defend the Edit Index: they score well on discworld, so
a workhorse editor at the unedited floor is a fact about the model, not the measure.
Measured 2026-09-07 on the full 192-case bench (`arms.oracle_arm`): L-dw-noiseless-20m
overwrite **+0.907**, freeze[N=16] **+0.746** (unedited −0.924); L-dw-20m +0.688 / +0.678 at
N=16 (+0.704 at N=32), against the ~+0.82 effective ceiling of clean-render scoring.
`tests/test_oracle_editors.py` pins both ≥ +0.7 on the noiseless run. They are wired
but NOT part of `master_eval`'s default loop.

## Metrics

`pim/metrics/` — arrays in, numbers out, never imports matplotlib. Never re-derive at a
call site.

| metric | definition | notes |
|---|---|---|
| **Probe Skill** | `decodability.py` — 1 − loss/trivial-baseline-loss | THE cross-environment decodability axis; ≡ R² on regression (proven to 1e-12), 1 − err/majority-err on classification. Baseline always from TRAIN |
| error rate (Othello) | probe fit stats (`error_rate`, %) | Li et al.'s native classification quantity, kept only as the anchor to their tables (their §3 numbers are error rates). Not a decodability axis in this project — Probe Skill (= 1 − err/majority-err) is. Discworld has no counterpart row: its Probe Skill *is* R² |
| **Edit Index (ray-zone)** (discworld frame models) | `zone_editability.py::edit_index` — the formula in `edit_index.py` with the two clean renders as references and the differing rays as support; (d_uned − d_edit)/(d_uned + d_edit) on the differing rays | +1 = the edited world, −1 = the unedited; effective range ≈ +0.82…−0.80 because scoring is against the clean render |
| zone RMSEs, scorecard | `zone_editability.py` — target / ghost / collateral / edit-frame | absolute, in intensity units |
| **fidelity ratio** (THE guard) | `zone_editability.py::fidelity_ratio` (discworld) · `set_editability.py::move_fidelity_ratio` (Othello) | ONE definition and polarity in both environments since 2026-09-01: `RMSE(edited prediction, edited-world GT) / RMSE(unsteered prediction, same GT)`, **at the edit step only**. **>1 = the edit degraded the model rather than steering it**; no success claim survives that. It is the ABSOLUTE counterpart to the Edit Index, which is *relative* and so scores a wrecked output mildly positive when it lands marginally nearer the edited world. Discworld: whole frame. Othello: all 64 squares (never the union support — the guard must see collateral damage outside the edit's own zone) |
| **Edit Index (legal-set)** | `set_editability.py::edit_index_legal` — the same formula (`edit_index.py`) with uniform-over-SET reference worlds and the union (headline) or symdiff of the two sets as support | two clients: **Othello** (sets = legal moves before/after the flip; the uniform reference is exact — the generator IS uniform) and **discworld frames-as-tokens models** (sets = the ONE frame each world renders at the edit frame, wired by `environments/discworld/token_bench.py`; +1 = the edited world's frame, −1 = the unedited one; the bridge `zone_edit_index_expected` scores the expected frame on the ray-zone construction and rides along, never as the headline) |
| Li error / legal mass | `set_editability.py` | their §4.2 metric, kept under their name — the anchor to Li et al.'s published numbers (null 2.68 → 0.12), never structural. ⚠ `li_error_vs_pre` is a DIAGNOSTIC, not the guard: it is one half of the pair the Edit Index is already built from, and "higher is better" only holds up to the pre→post separation (2.763 on `L-oth-20m`) — beyond that means drifting away from BOTH worlds |
| gates | `environments/othello/arms.py::gates` | legal mass, top-1, CE with the **exact** Bayes floors (bayes_ce = E[log‖legal‖]) |

⚠ **Do not read the Edit Index without the fidelity ratio.** The index answers *which
world is the output nearer* (relative); the guard answers *did the output get further
from the truth than doing nothing* (absolute). Discworld PI reads EI **+0.22** at
fidelity **1.69** — a destroyed frame, not an edit, and only the guard says so.

Reporting traps (carried from METRICS_AND_EDITORS.md): (a) aggregate probe R² is
variance-weighted — position dominates velocity ~1000:1; quote per-dim; (b) fidelity
ratio cannot see a destructive edit on its own — read it WITH collateral; (c) the EI
scale is ≈ +0.82…−0.80, not ±1 (clean-render reference vs noisy-trained model).

## Canonical runs

The rule: canonical dataset AND canonical architecture AND canonical training setup —
else `runs/archive/`. Runs live at `runs/<topic>/<name>/`; each carries `config.json`,
`commit_sha`, checkpoints, `probes/` (with `INDEX.md`), and its canonical `scores.json`.

⛔ **`runs/` holds trained runs and nothing else** — no driver logs, no chain scripts,
no pilots, no shared caches. Those go to `experiments/<name>/` (`logs/<name>/` for chain
output); one-off experiment artifacts go to `outputs/`.

| run | arch | instance | status |
|---|---|---|---|
| `initial_othello_comparison/L-oth-20m` | Transformer-L (tokens) | oth-uniform | 780k steps, best val 2.02798 (excess over Bayes +0.019) |
| `initial_othello_comparison/L-dw-20m` | Transformer-L (regression) | dw-pn04 | 780k steps, best val 0.022873 (3.16% over the state oracle) |
| `noise_ablation/L-dw-noiseless-20m` | Transformer-L (regression) | dw-noiseless | 780k steps, matched recipe — the noise ablation (2026-08-31). The ONE run carrying the **`grid-16x8` probe target** (2026-09-09): a third `scores.json` block / table row — skill LIN 0.43 / MLP 0.71, unedited −0.93, PI +0.13 / fid 1.58, **ND +0.37 / 0.91**, **GS +0.29 / 0.87** (one-frame, reverting); its `probes/` holds those 18 grid fits beside the regression ones, its `figures/` the grid waterfall; `findings/grid-target-control.md` |
| `ray_ablation/L-dw-8ray-20m` | Transformer-L (regression) | dw-8ray | 780k steps, matched recipe — the ray-count ablation (2026-09-04); the same 8 × 512 stack with `Linear(8, 512)` in/out (25.25M params). Best val MSE 0.00575. Driver `scripts/drivers/dw_8ray.sh`; findings `findings/ray-ablation.md`. Its `probes/` also holds the 18 INLP cascades of `experiments/inlp/8ray` (`findings/inlp-8ray.md`) Rescored 2026-09-09 on the filtered bench: unedited −0.911, **PI +0.282 / fid 0.90** (was +0.297 / 1.11 — now below the guard, i.e. non-destructive), ND −0.148, GS −0.064. Conclusion unchanged. |
| `ray_ablation/R-dw-8ray-20m` | Recurrent-L (regression) | dw-8ray | 780k steps, identical recipe (2026-09-04), the architecture pair of `L-dw-8ray-20m`. Best val 0.00595 at step **40k**, drifting to 0.00625 by the end (no divergence spike this time). Scored: PI +0.191 / fid 1.41, GS −0.61; decodability at the random-init floor. Driver `experiments/recurrent/drivers/recurrent.sh` with `TOPIC=ray_ablation` |
| `architecture_gate/R-dw-20m` | Recurrent-L (regression) | dw-pn04 | 780k steps, matched recipe — the recomputation test (2026-09-02). Best val 0.02302 at step **50k**; val drifted up afterwards and spiked at 525k (constant lr 1e-3 is the transformer's recipe, kept deliberately). Scored: same editability signature as Transformer-L — see `findings/recurrent-l.md` |
| `architecture_gate/R-dw-noiseless-20m` | Recurrent-L (regression) | dw-noiseless | 780k steps, identical recipe (2026-09-02). Best val 0.001112 at step **205k**, unstable afterwards. Scored: PI +0.093 / fid 2.27 — the carried write edits no better than the transformer's; see `findings/recurrent-l.md` |
| `objective_ablation/L-oth-20m-mse` | Transformer-L (tokens, raw head) | oth-uniform | 780k steps, identical to `L-oth-20m` except the objective: `mse_next_move_onehot` (`--objective mse_onehot`, `output_kind="raw"`) — the objective ablation (2026-09-05). Best val Brier 0.013662 at step 770k (19.8 h). The raw head IS a distribution to 4 decimals (mean sum 1.0001, mean negative mass 0.04): legal mass 0.989, top-1 legal 0.998, CE 2.048 vs Bayes 2.011. Scored: skill LIN 0.961 / MLP 0.960 (best point 6; CE model 0.975); PI +0.68 / ND +0.74 / GS +0.73 (CE model +0.61 / +0.62 / +0.65) — as editable as the cross-entropy model. Driver `scripts/drivers/oth_mse.sh`; under a clip-and-renormalise reading every EI is 0.04–0.07 lower (PI +0.63 / ND +0.69 / GS +0.66) and legal mass 0.951 — the negatives cancel illegal-move noise (`experiments/othello_mse_head/`); `findings/othello-mse-head.md`. |
| `interface_ablation/L-dw-8ray-tok-20m` | Transformer-L (tokens) | dw-8ray (frames as tokens) | 780k steps, the interface ablation (2026-09-06): the SAME instance and 20M sequences as `L-dw-8ray-20m`, every frame one token of the 422-token frame vocabulary (`datasets/discworld/dw-8ray/tokens/`), the Othello token model (embedding in, softmax + CE over frames out; `scripts/train.py --repr tokens`). Best val CE 0.4638 at 780k (12.1 h; frame n-gram floors 0.737 → 0.654 for orders 1–6, persistence top-1 0.840 vs the model's 0.860). Scored the Othello way (`token_bench`, frame-set Edit Index †): skill LIN 0.968 (pt 0 = the frame lookup; floor 0.968) / MLP 0.980 (floor 0.968) frustum; unedited −0.755; **PI +0.004 / GS −0.097** — NOT editable: PI lands the probe read-out exactly at α=1 with zero output change, and only 60–175× writes move the output, off the unedited frame and onto nothing (p_post ≤ 0.05). With `L-oth-20m-mse`, neither objective nor interface explains Othello's editability. `findings/interface-ablation.md`; `experiments/dw_tokens/`. Rescored 2026-09-09 on the filtered bench (192/192 scoreable, was 163): unedited −0.779, PI +0.006, GS −0.102 — the degenerate cases were diluting a null, not hiding an effect. |
| `flip_ablation/L-oth-noflip-20m` | Transformer-L (tokens) | oth-noflip | 780k steps, identical to `L-oth-20m` except the instance (2026-09-07). Bayes-optimal (CE excess +0.0015; 96% of the random-init excess gone by step 1k); Probe Skill 1.000 from point 1 = random-init = observation floors (colour is a fixed square pattern on the occupancy set). **NOT editable** (PI −0.001 / fid 3.25, ND +0.086 / 2.57, GS +0.020 / 4.00 vs unedited −0.823) — by construction: legality never depends on colour in this world, so a colour edit has nothing to steer (checkerboard theorem). The cleanest "decodable ≠ used" control; NOT a test of the flip dynamics. `findings/flip-ablation.md`; `experiments/flip_ablation/`. |
| `adjacency_ablation/L-oth-adjacent-20m` | Transformer-L (tokens) | oth-adjacent | 780k steps, identical to `L-oth-noflip-20m` except the placement rule (launched 2026-09-08 13:58, unit `oth_adjacent`, `scripts/drivers/oth_adjacent.sh`). Prediction was that editability returns (colour is used). **It does not**: Bayes-optimal (CE 2.435 vs 2.433, legal mass 1.000), skill 0.988 = the observation floor (colour = placing-move parity), PI −0.05 / fid 2.31, ND +0.12 / 1.05, GS +0.00 / 5.73; extended α (ND to 12, PI to 35) never above +0.13; the read-out LANDS (100% from point 1) and the legal-move distribution does not move. Causal use is not sufficient; the probe reads a lookup copy of colour, not the one legality consumes. `findings/adjacency-ablation.md`. |
| `blink_ablation/L-dw-blink-20m` | Transformer-L (regression) | dw-blink | 780k steps, matched recipe — the blink ablation (launched 2026-09-07 18:46, unit `dw_blink`, `scripts/drivers/dw_blink.sh`). The one discworld instance where position MUST be carried; scored canonically (first 192 edits) and by blink subset in `experiments/blink_ablation/scores/` (reappearance with staleness k, mid-blackout at the reappearance step, visible control; hidden-vs-visible decodability). **Position is carried (MLP 0.97–0.99 / LIN 0.63–0.89 on hidden frames, vs a 0.35 linear observation floor) and still NOT editable**: reappearance PI +0.27 / fid 1.37 (≈ +0.23 at fid ≤ 1.1) vs visible +0.22 / 1.81; GS ≤ 0 everywhere; mid-blackout edits do nothing at reappearance (−0.385 = unedited) unless destructive (fid 44). Canonical: LIN 0.90 / MLP 0.99, PI +0.22 / 1.78, GS −0.09. `findings/blink-ablation.md`. |
| _planned_: S-oth / S-dw | Transformer-S | both | fresh trainings under the new scheme (W16 and the old S rungs failed the rule and are archived) |

## Evaluation

`notebooks/master_eval.ipynb` scores every run under `runs/**` (excluding `archive/` and
`_`-prefixed topics) the identical way — **no metric math in the notebook**; every number
is a call into `pim.*` — and writes `scores.json` into the run dir, stamped with
`EVAL_VERSION` (bump it to force a rescore). `notebooks/build_full_table.ipynb` renders
the one master table from those files. The evaluation settings (probe corpus sizes, α
grids) live in master_eval cell [2] and are recorded into every `scores.json`.
