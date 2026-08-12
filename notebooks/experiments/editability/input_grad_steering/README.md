# Input Grad Steering — thread README

Origin: Sevan, 2026-08-11. Question: **can a frozen probe's gradient, backpropagated through the
network to the INPUT OBSERVATION, produce a semantically edited observation** (the object's bump
actually moves) — or does it find an adversarial perturbation that flips the readout without
moving anything the dynamics respect?

Every existing editor in `../METRICS_AND_EDITORS.md` writes to the latent state `h` (or overwrites
history with *externally rendered* frames). This thread fills the empty cell in that matrix: a
**training-free, probe-only write to the observation channel** — for the transformer, that channel
is the *carried* state itself (the only persistent one it has).

## The editor (fold into `../METRICS_AND_EDITORS.md` if it recurs)

| editor | mechanism |
|---|---|
| **Input Grad Steering @Lℓ (n, λ)** | Adam (300 steps, lr 0.02) on a perturbation δ added to the **newest n frames of the observation history**, minimizing `‖probe_ℓ(state(obs+δ)) − target‖² + λ‖δ‖²`, with `obs+δ` clamped to [0,1]. The probe is the **frozen standard linear readability probe** (`pim.extractors.fit_readability_probes`, positions of both objects, 4 dims) at residual point ℓ (transformer) or on `h` (GRU). Target: edited object → teleport target, other object → its true frame-`ef` position. |
| **Render write @1 (oracle)** | Same write surface, oracle content: replace the newest history frame with the **clean render of the edited world** (`ZONES.gt_edited`). The ceiling that separates "the write surface works" from "the gradient found the right direction". |

Distinct from **MLP Grad Steering** (which steers `h` through a frozen 1×128 MLP probe) and from
**Decoder Grad Steering** (which steers `h` through the decoder) — same optimizer, different
variable. λ is the on/off-manifold dial, not a nuisance parameter.

## The direction comparison

**True edit direction** `Δ_true = gt_edited − obs_steered_base`: the clean render of the edited
world at `ef` minus the actual (noisy) observation being modified (`obs[ef−1]`). Per-sample
`cos(δ*, Δ_true)` (reported with the **angle**, per repo convention) says whether the gradient
moves the 128-d observation *toward what the edited world would actually look like*. Two known
impurities, stated wherever the number appears: Δ_true contains the observation-noise realisation
of the base frame, and it is offset by one frame of motion (render at `ef`, base at `ef−1`).
Chance level = empirical shuffled-pair control (`cos(δ*_i, Δ_true_j)`, i≠j), never an analytic
guess.

## Notebooks

- `input_grad_steering_transformer.ipynb` — transformer · window 16 (`runs/transformers/W16`),
  residual-point sweep ℓ ∈ {0..4}, λ sweep, n ∈ {1, all}.
- `input_grad_steering_gru.ipynb` — GRU · H=256 (`runs/controls/H256`), λ sweep, n=1.

## Checkpoints used (rows copied from their registries)

| code | descriptive label | source registry | best val |
|---|---|---|---|
| `W16` | **transformer · window 16** (d256, 4L, RoPE, `state_span` 61 → effective carried state at `ef`=20 is all 20 frames) | `../transformers/TRANSFORMER_RUNS.md` | 0.02359 |
| `H256` | **GRU · H=256 (reference)** | `../controls/CONTROL_RUNS.md` | 0.02362 |

Dataset: `datasets/4_fixed_refl_inview` (edit frame 20, in-frustum teleports, obs noise 0.2).
Metrics: canonical §4 set from `scripts/editability_metrics.py`; probes from
`pim.extractors.fit_readability_probes`. N=64 edits, K=15 rollout, seed 0.
