# The editability scaling sweep — when does an editable world model emerge, and what gates it?

> **⚠ 2026-08-24 — the budget arithmetic below rests on a broken premise.**
> The "saturation at ~1-1.5M steps for 20M games" extrapolation came from the 90k and 900k
> cells' *overfitting turnarounds*, not their convergence. The 20M cell shows a clean
> `step^(-0.53)` power law with no knee at all, so it has no saturation step to extrapolate to;
> a stopping point is a declared tolerance, not a discovered fact. Meanwhile the 900k cell
> drives train loss **below the Bayes floor** (1.9204 < 2.0092) — it is partly memorizing.
> See `research/scratch/2026-08-24-saturation-is-overfitting.md` before trusting any cost
> number here.



**Status:** `planning`, 2026-08-23. The programme this whole line of work has been converging on.
**Tag:** `[reframe]` · **Sub-Q:** 1, 3

## The question

Not "is discworld editable" (we know: no, at the scales tried) but **at what point on the
scale/data/training surface does an editable world model appear, and does the environment move
that point?** That turns a negative result into a scaling law with an environment-dependent
threshold.

## What is already established (2026-08-18 → 08-23)

Editability is **not** explained by: our probe implementation, probe training data, our editor
implementation, data volume alone (222× moved absolute Edit Index +0.059 → +0.098), or
architecture + data + training length jointly at 900k sequences. The **environment** flips it:
their architecture, 900k sequences, matched epochs, reaches **+0.24** on Othello and **−0.11 …
+0.07** on discworld. Full detail: `../scratch/2026-08-21-ours-on-othello.md`,
`../findings/editability.md` (2026-08-22, `replicated`).

Open at the low end: on Othello **only Nanda's direction addition** works at 900k, while on the
*fully trained* published model **PI injection is strongest** (+0.697). So which editor wins is
itself scale-dependent, and the crossover is a thing to measure, not a nuisance.

## The axes

| axis | levels | note |
|---|---|---|
| **environment** | Othello · discworld | the variable that has already been shown to matter |
| **architecture** | Transformer S (3.2M, 4 blocks) · Transformer L (25.3M, 8 blocks) | capacity is confounded with depth/attention/positional encoding — this is *not* a clean capacity axis and must not be described as one |
| **data volume** | 100k · 900k · 20M sequences | nested prefixes of one index-seeded corpus |
| **training length** | free, via checkpoints | **the reason the grid is affordable** |
| **editor** | MLP gradient steering · PI injection (single point, swept) · Nanda direction | report best over (layer, α); keep all three — the winner changes with scale |
| **probe basis** | discworld: Cartesian · frustum · Othello: mine/theirs (**settled**) | separate figure sets per basis, never a fourth line style |

## Decisions already taken

- **x-axis is optimizer steps, never epochs.** An epoch is a different amount of compute at every
  data volume (250 epochs = 88k steps at 100k games, 17.6M at 20M).
- **Edit Index is the absolute post-edit value**, reported beside the model's own null. A
  gain-over-null axis rises with model quality even when editability is flat — measured, 2026-08-22.
- **Othello's basis is mine/theirs.** Absolute colour is catastrophic for the editor and the
  question is settled; no further probes needed there.
- **Velocity stays in the probe target** even though it sits at R² ≈ 0.31 in both bases — it may
  become decodable at scale, and dropping it would foreclose that (Sevan, 2026-08-23).
- **Training-length decline at fixed data ("Peaked") is a secondary phenomenon**, noted alongside,
  not a competing hypothesis.

## Hypotheses

| name | claim | signature on the canonical plot |
|---|---|---|
| **Scale2edit** | discworld does become editable, given enough data × steps × capacity; the gate simply sits further out than Othello's | discworld curves rise, lagging Othello |
| **Insufficiency** | discworld's Bayes-optimal sufficient statistic never coincides with the ground-truth world state, so no editable world model forms at any scale | discworld flat at ≈ 0 forever |
| *(secondary)* **Peaked** | editability is a property of a training regime, not a limit — it rises then declines as memorisation sets in | non-monotonic in steps at fixed data |

⚠ **Insufficiency is only falsifiable once the basis question is settled** — a flat curve is
equally consistent with "no world model" and "we probed the wrong basis". See below.

## Still to decide

1. **The depth coordinate for discworld's frustum basis. STILL OPEN as of 2026-08-25.**
   ⛔ `../scratch/2026-08-23-frustum-basis.md`, cited here and in `pim/simulator/frustum.py`, was
   **never written** — there is no empirical ranking of the five candidates. `inv_y` was used for
   the 20M discworld analysis because it is `frustum.py`'s default and what the `w ∝ 1/depth`
   derivation argues for, **not** because it won a comparison. Linear-only probes make sweeping all
   five ~30 min (`research/scratch/2026-08-25-discworld-at-scale.md` §7). Depth manifests in the observation *only* as apparent
   width, so `y` (what the simulator reports and the model never sees) is likely the wrong choice.
2. **Checkpoint schedule.** Log-spaced in steps, so the training-length axis is evenly sampled on
   the log x-axis: roughly 2^k · 1000 steps.
3. **What to persist** so re-plotting is free: per-checkpoint probe weights, per-arm scorecards,
   and the record list — never a figure that has to be regenerated from a model.
4. Whether Transformer S is worth running at 20M, or whether capacity saturates first.

## Compute, measured on the RTX 5090

Their architecture ~960 steps/min, ours ~5,400. Saturation scales with data volume: 90k games
saturated at ~6k steps, 900k at ~58k, so 20M should saturate near 1–1.5M steps rather than the
17.6M that "250 epochs" would imply. On that basis the whole grid is **~96 h on one 5090**, ~48
H100-hours (≈ $100–150). Before renting: we currently train in fp32, batch 256, no `torch.compile`
— bf16 + a larger batch + compile is plausibly 3–4× on hardware already owned.

⚠ The saturation extrapolation rests on two points. **Run the 20M × Transformer L cell first, with
frequent checkpoints**, so a wrong extrapolation surfaces on day one rather than in week four.

## Artifacts

- figures: `pim/figures/scaling.py` + `notebooks/experiments/editability/scaling/`
  (`collect.py`, `make_figures.py`) — one command regenerates everything
- basis: `pim/simulator/frustum.py`
- run registry and corpora: to be created under `notebooks/experiments/editability/scaling/`
