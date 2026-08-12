# Nonlinear-GRU thread — canonical run registry

**One source of truth for every checkpoint this thread refers to.** Any notebook mentioning a run copies the
rows it uses into its own definitions table, so the notebook still stands alone. Adding a run means adding its
row here **in the same commit**.

Metric and editor definitions come from `../METRICS_AND_EDITORS.md`; conventions from `../../../../CLAUDE.md`.

---

## Why this thread exists

The GRU used throughout the editability thread has an **encoder that is one `nn.Linear` + ReLU** and a
**decoder that is one bare `nn.Linear`** — that is, `decode` is an **affine** function of `h`. Affinity has one
consequence that is easy to miss and that invalidates a specific class of result:

```
decode(h0 + d1 + d2)  ==  decode(h0 + d1) + decode(h0 + d2) - decode(h0)
```

holds **identically, for any vectors `d1, d2`**, with no structure in the latent required. Measured on the
trained H256 baseline in `../delta_h_analysis.ipynb` (§7b, 2026-08-05), the composed-versus-affine gap is
`6.6e-08` — machine precision. So that notebook's "edits superpose across objects" result, *as read off the
decoded observation*, was forced by the decoder rather than earned by the representation. (Its **state-space**
readouts — `cos(composed, direct)` = +0.873 against a +0.059 shuffled floor — were not affected and stand.)

This thread trains GRU variants whose decoder is a genuine MLP, so the identity is broken by construction, and
re-runs the thread's main findings on them.

**What is being asked, precisely:** are the editability findings a property of *implicit recurrent world models*,
or artifacts of a shallow read-in/read-out path? Each finding is re-derived, not assumed.

---

## Architecture knob

`pim/world_models/gru/model.py` gained `enc_hidden_layers` / `dec_hidden_layers` / `mlp_activation`
(defaults `0 / 0 / "relu"` = the original architecture, bit-identical — asserted in
`tests/test_gru_mlp_depth.py`). Depth `k` inserts `k × (Linear(H,H) + activation)` blocks:

- `enc_hidden_layers` — **after** the existing encoder `Linear + ReLU`, before the GRU.
- `dec_hidden_layers` — **before** the existing decoder `Linear`. Any value ≥ 1 makes `decode` nonlinear in `h`.

The blocks live in separate `enc_trunk` / `dec_trunk` submodules that are **absent from `state_dict` at depth 0**,
so every pre-existing checkpoint loads unchanged. `model.has_affine_decoder` reports which regime a model is in.

---

## Runs

All rows share: dataset `datasets/4_fixed_refl_inview` (2 objects, 128-ray 1D scans, 40 frames,
`obs_noise_std=0.2`, `position_noise_std=0.04`, fixed reflectivities, always-in-frustum), trained on
`train.h5` (90k, 10% held out for val) with the **identical recipe**: 400 epochs, batch 256, AdamW
lr 1e-3, weight decay 1e-4, in-memory, `hidden_size=256`, `num_layers=1`, dropout 0, next-frame MSE
teacher forcing, no state supervision. They differ **only** in MLP depth and seed.

| run name | full label used in figures | enc depth | dec depth | `decode` | seed | params | what it is a control for |
|---|---|---|---|---|---|---|---|
| `runs/controls/H256/best_model.pt` | `linear enc+dec · H256 · seed 0` | 0 | 0 | **affine** | 0 | 460,672 | **the baseline** — the exact checkpoint the editability findings and `delta_h_analysis` were established on |
| `runs/nonlinear_gru/NL_enc2dec2_s0` | `nonlinear enc+dec · H256 · seed 0` | 2 | 2 | nonlinear | 0 | 723,840 | **the main variant** — the architecture Sevan asked for |
| `runs/nonlinear_gru/NL_dec2_s0` | `nonlinear dec only · H256 · seed 0` | 0 | 2 | nonlinear | 0 | 592,256 | isolates **which half matters** — the affine-decoder artifact is a decoder property, so this variant alone should be enough to break it |
| `runs/nonlinear_gru/NL_enc2dec2_s1` | `nonlinear enc+dec · H256 · seed 1` | 2 | 2 | nonlinear | 1 | 723,840 | **seed control** for the main variant — single-seed claims are a standing weakness of this repo |

**Suffix key.** `NL` = nonlinear. `enc2dec2` = 2 extra encoder blocks and 2 extra decoder blocks.
`dec2` = decoder-only depth. `s0`/`s1` = training seed.

### Capacity confound — controlled by citation, not by a new run

The nonlinear variants have more parameters than the H256 baseline (724k / 592k vs 461k), so "the finding
changed" could in principle mean "capacity changed". It does not need a new run: the **hidden-size sweep**
(`../controls/hidden_size_sweep.ipynb`, `runs/controls/H{8,32,128,256,512}`) already established that the
editability negative is **capacity-independent across H = 8 … 512**, and `H512` (1,704,064 params) brackets
the nonlinear variants from above. The notebook therefore reports `H512` alongside as the capacity reference.
A parameter-matched *linear* control would sit at H ≈ 326, inside that swept range.

---

## Reading the numbers

- The Edit Index must be read **against each model's own unsteered row** — a model's `−1` end sits at its own
  next-step prediction error, not at a shared constant (`../METRICS_AND_EDITORS.md`, §4).
- Row-space fractions must be read **against `√(d/H)`**, the share a random vector already has. All four runs
  here share `H = 256`, so their chance levels coincide (0.125 for the `d=4` position probe) and the raw
  fractions are directly comparable — this is *not* true against the `H8…H512` sweep, where the enrichment
  ratio `f / chance` is the only comparable form.
