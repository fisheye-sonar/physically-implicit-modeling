# REGISTRY — the canonical objects, one table per category

_The single source of truth for what "the probe", "the editor", "the metric" mean.
Every category grows only as needed, and an entry is canonical only if it appears here.
Code paths are the definitions; this file is the index, not a re-derivation. Created
2026-08-31 (housecleaning); supersedes `notebooks/experiments/editability/METRICS_AND_EDITORS.md`._

## Environment instances

An **instance** = one environment class at one fixed generation config, packaged with the
data for every split it defines (`instance.json` in its directory is the contract).
Data-scale comparisons mask one instance's train pool; changed generation params = a new
instance (the dset-17 lesson: it silently used position noise 0.0 and every result on it
was uninterpretable against the canonical eval).

| shorthand | class | definition | splits |
|---|---|---|---|
| `dw-pn04` | discworld | `datasets/discworld/dw-pn04/instance.json` | train 20M (memmap) · probe 120k · eval 10k+10k · edits 10k (EF=20) |
| `dw-noiseless` | discworld | `datasets/discworld/dw-noiseless/instance.json` | identical to `dw-pn04` in every field **except** `obs_noise_std = 0.0` AND `position_noise_std = 0.0` — the noise ablation at mass scale (2026-08-31). Same split sizes; seeds are a FRESH block (base 30e9) |
| `oth-uniform` | othello | `datasets/othello/oth-uniform/instance.json` | train [0,20M) · test [90M,+10k) · probe [91M,+20k) · edits = Li's 1001 cases |

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
| **Transformer-L** | ~25M | `pim/models/transformer_l.py` — Li et al.'s minGPT (vendored), full causal, learned absolute positions | `TransformerL` (MSE) / `TransformerLTokens` (CE) |

## Training setup

ONE loop (`pim/training/train.py`), two objectives, fed by per-environment
`DataSource`s (`pim/training/sources.py`). The canonical recipe (the matched-BIG20M
hyperparameters) IS the `TrainConfig` defaults: AdamW 1e-3 / wd 1e-4 / clip 1.0 /
batch 256 / 2k-step warmup then **constant** LR / seed 0. Entry: `scripts/train.py`.
Every run writes `config.json` + `commit_sha` + `metrics.jsonl` + arch-stamped
checkpoints into `runs/<topic>/<name>/`.

## Probes

Always held out **by sequence**, never by frame (Othello's frame split is kept only as
Li's replication anchor and always labelled). Cached with the model fingerprint in the
key (`pim/probes/cache.py`).

| shorthand | definition | fit |
|---|---|---|
| **LIN** | `pim/probes/linear.py` — one affine map, standardised both ends | closed-form lstsq (regression) / SGD-CE (classification) |
| **MLP-128** | `pim/probes/mlp.py` — Li's §3.2 shape: ONE hidden layer × 128 | SGD, loss in standardised target space |
| NULLSPACE (non-default) | `pim/probes/nullspace.py` — deflation cascade of orthogonal linear probes | float64 min-norm lstsq, orthogonality asserted |

Tripwire: `check_probe_sanity` (MLP ≥ linear on held-out data, or the fit is
memorisation) runs on every paired fit; violations are recorded in `scores.json`.

## Editors

| shorthand | definition | what it writes |
|---|---|---|
| **PI** | `pim/editors/pinv.py` — pseudoinverse injection, **z-space + y-affine** ("zspace") | the minimum-norm Δh that lands the LIN read-out on the target; α=1 = the exact jump. `"legacy"` reproduces pre-2026-08-31 discworld numbers (the y-affine bug) and is never quoted as PI |
| **ND** | `pim/editors/nanda.py` — Nanda direction addition | α·‖x‖·d̂ along the probe weight row(s), standardised; `add_sub` = target−current |
| **GS** | `pim/editors/grad_steer.py` — Li §4.1 MLP gradient steering | descent on the activation through the frozen MLP-128 probe, sequentially from L_s across every later point. ⛔ `target_labels` MUST share a coordinate frame with the probe being steered — a mismatch converges onto a well-formed but WRONG class and looks like a failed editor (the 2026-08-31 GS-mine bug, worth 0.70 Edit Index; `scores.json::probe_sources` records the pairing) |
| nullspace (non-default) | `pim/editors/nullspace.py` | Σₖ Aₖ⁺(tₖ − pₖ(h)) over the whole cascade — the row-space objection's answer |
| oracle: overwrite | `pim/editors/oracle_overwrite.py` | the state the model would have on the post-edit history (ceiling) |
| oracle: freeze-interp | `pim/editors/freeze_interpolation.py` | N rendered frozen frames teacher-forced through the observation channel |

The two oracle editors exist to defend the Edit Index: they score well on discworld, so
a workhorse editor at the unedited floor is a fact about the model, not the measure.

## Metrics

`pim/metrics/` — arrays in, numbers out, never imports matplotlib. Never re-derive at a
call site.

| metric | definition | notes |
|---|---|---|
| **Probe Skill** | `decodability.py` — 1 − loss/trivial-baseline-loss | THE cross-environment decodability axis; ≡ R² on regression (proven to 1e-12), 1 − err/majority-err on classification. Baseline always from TRAIN |
| R² / error rate | `decodability.py` / probe fit stats | the native per-environment forms, reported alongside |
| **Edit Index** (discworld) | `editability.py::edit_index` — (d_uned − d_edit)/(d_uned + d_edit) on the differing rays | +1 = the edited world, −1 = the unedited; effective range ≈ +0.82…−0.80 because scoring is against the clean render |
| zone RMSEs, scorecard | `editability.py` — target / ghost / collateral / edit-frame | absolute, in intensity units |
| **fidelity ratio** (THE guard) | `editability.py::fidelity_ratio` (discworld) · `othello_moves.py::move_fidelity_ratio` (Othello) | ONE definition and polarity in both environments since 2026-09-01: `RMSE(edited prediction, edited-world GT) / RMSE(unsteered prediction, same GT)`, **at the edit step only**. **>1 = the edit degraded the model rather than steering it**; no success claim survives that. It is the ABSOLUTE counterpart to the Edit Index, which is *relative* and so scores a wrecked output mildly positive when it lands marginally nearer the edited world. Discworld: whole frame. Othello: all 64 squares (never the union support — the guard must see collateral damage outside the edit's own zone) |
| **Edit Index (legal)** (othello) | `othello_moves.py::edit_index_legal` | same formula, uniform-over-legal reference worlds (exact — the generator IS uniform); union support headline, symdiff alongside |
| Li error / legal mass | `othello_moves.py` | their §4.2 metric, kept under their name — the anchor to Li et al.'s published numbers (null 2.68 → 0.12), never structural. ⚠ `li_error_vs_pre` is a DIAGNOSTIC, not the guard: it is one half of the pair the Edit Index is already built from, and "higher is better" only holds up to the pre→post separation (2.763 on `L-oth-20m`) — beyond that means drifting away from BOTH worlds |
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
no pilots, no shared caches. Those go to `logs/` (`logs/drivers/<campaign>/` for chain
output); one-off experiment artifacts go to `outputs/`.

| run | arch | instance | status |
|---|---|---|---|
| `initial_othello_comparison/L-oth-20m` | Transformer-L (tokens) | oth-uniform | 780k steps, best val 2.02798 (excess over Bayes +0.019) |
| `initial_othello_comparison/L-dw-20m` | Transformer-L (regression) | dw-pn04 | 780k steps, best val 0.022873 (3.16% over the state oracle) |
| `noise_ablation/L-dw-noiseless-20m` | Transformer-L (regression) | dw-noiseless | 780k steps, matched recipe — the noise ablation (2026-08-31) |
| _planned_: S-oth / S-dw | Transformer-S | both | fresh trainings under the new scheme (W16 and the old S rungs failed the rule and are archived) |

## Evaluation

`notebooks/master_eval.ipynb` scores every run under `runs/**` (excluding `archive/` and
`_`-prefixed topics) the identical way — **no metric math in the notebook**; every number
is a call into `pim.*` — and writes `scores.json` into the run dir, stamped with
`EVAL_VERSION` (bump it to force a rescore). `notebooks/build_full_table.ipynb` renders
the one master table from those files. The evaluation settings (probe corpus sizes, α
grids) live in master_eval cell [2] and are recorded into every `scores.json`.
