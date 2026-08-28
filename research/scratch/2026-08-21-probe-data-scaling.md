# 2026-08-21 — Does discworld probe quality improve at Othello-scale probe data?

Script `scratch/probe_scaling.py` (scratchpad copy; ~3 min total on the 5090). Model `W16`
(`runs/transformers/W16`), dataset `4_fixed_refl_inview`, residual point 3 ("block 3 input" —
the thread's best position probe). Probe and fitting loop are the thread's own,
`othello_gpt/othello_probe.fit_probe` at `hidden=512`, 200 epochs, held out **by sequence**
80/20, visible-frames-only. **The sequence count is the only variable.**

## Why

`othello_transfer` (2026-08-20) cleared our *editor implementation*. Two confounds were left
standing for the discworld negative: the data setup and the architecture. On the data side, the
sharpest asymmetry was probe training data — our editability numbers fit probes on **1500
sequences (48k rows)**; Li et al. fit theirs on **~140k games (≈6.7M rows)**, a ~140x gap. If
discworld probe R² climbs with data, "our probes were under-fit" is a live alternative
explanation and every editability number is suspect.

## Result — it does not climb

| source | n_seq | train rows | linear R² | MLP (512) R² | gap |
|---|---|---|---|---|---|
| test | 1,500 | 48,000 | 0.7604 | **0.9349** | +0.1745 |
| train | 1,500 | 48,000 | 0.7546 | 0.9315 | +0.1769 |
| train | 5,000 | 160,000 | 0.7618 | 0.9469 | +0.1852 |
| train | 15,000 | 480,000 | 0.7672 | 0.9538 | +0.1867 |
| train | 45,000 | 1,440,000 | 0.7649 | 0.9593 | +0.1944 |
| train | 90,000 | 2,880,000 | 0.7650 | **0.9604** | +0.1954 |

**60x more probe data buys +0.029 R² (MLP) and +0.010 (linear).** Successive steps give
+0.015 / +0.007 / +0.006 / **+0.001** — saturated by ~1.5M rows, well short of Othello's 6.7M.
Linear extrapolation puts their data scale at ≈0.961.

**Cross-check passes:** the test-split fit at 1500 sequences reproduces the thread's published
**0.9349** exactly, so the curve is anchored to the number actually quoted. Test vs train as the
harvest source moves R² by 0.003 — negligible, so using train episodes for the large scales is
not itself a confound.

## Reading

The 0.96 ceiling is a property of **`W16`'s residual stream**, not of how much data the probe
saw. At 1500 sequences the probe was already within 0.03 of everything the representation
contains at that point.

This compounds with 2026-08-18: the Othello-GPT-style intervention drives the probe read-out from
**3.35 → 0.007–0.018 sim units** and the generation still does not move. Probe *accuracy* was
already known not to be the binding constraint; probe *data* now isn't either. **The
probe-training half of the data confound is eliminated.**

## Scope / caveats

- **`W16` at residual point 3 only.** The GRU carries most of the thread's editability history
  and is a ~2-minute add; not yet run.
- R² 0.96 is not 1.0, and this says more data will not recover the remaining 4%. Whether that
  residual matters for editability is a **separate** question this does not touch.
- Says nothing about the **model**-training data confound (3.6M unique frames against their 1.2B
  unique transitions), which is what the planned run A is for.
