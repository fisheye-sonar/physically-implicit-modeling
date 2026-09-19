# 2026-09-17 — do Othello edits land in the PROBE's output space?

**Why.** The discworld side records the α=1 landing check (`arms.readout_error` /
`readout_landed`, and the paper's "at α = 1 the linear probe reads the target state exactly,
and the next frame does not move"). Othello never recorded it, and the paper's Results asserts
standard Othello "is editable by every editor" without saying that the *exact-landing* write
is not the one that edits. Sevan asked directly.

**Script.** `experiments/probe_readout_landing/scripts/othello_landing.py --run <run>` — the
canonical cached probes, the canonical 1000-case bench, the canonical primitives (`pinv_step`,
`probe_direction`/`addition_delta`, `build_edit_spec`/`_descend`, `inverse_overwrite`). Reads
the probe at the edited residual and compares against the ground-truth post-edit board:
`tile_landed` (the edited square reads its target label) and `board_landed` (all 64 squares).
**Validation: PI's best arm is also decoded and re-scored, and the recomputed symmetric-difference
Edit Index matches `scores.json` to 3 decimals on all four instances** (+0.818, +0.348, +0.175,
−0.312), so the write measured is the canonical write. GPU ~4 min per run. Scores in
`experiments/probe_readout_landing/scores/landing_*.json`.

## PI — lands in probe space on every instance, and landing is uninformative

| instance | tile landed | board landed | Edit Index @ α=1 | @ best α | unedited |
|---|---|---|---|---|---|
| standard (`oth-uniform`) | 99.4% | 85.0% | **−0.700** | +0.818 (α 3) | −0.93 |
| adjacent-flip | 98.4% | 60.2% | **−0.893** | +0.348 (α 5) | −0.96 |
| adjacent-noflip | 99.0% | 86.5% | **−0.957** | +0.175 (α 60) | −0.96 |
| standard-noflip | **100.0%** | **100.0%** | **−0.958** | −0.312 (α 100) | −0.98 |

(Landing is measured at both α; the columns above are α=1, and the best-α readout is within
1 point of it. PI's target differs from the probe's current read-out ONLY at the edited square,
so scaling by α leaves the other 63 squares' logits *exactly* unchanged and only pushes the
edited square further past the swap. The 99.0 → 100.0 drift on adjacent-noflip is the α=1
near-tie at the swapped logits resolving decisively under overshoot.)

Two readings, both sharp:

1. **The exact-landing write does essentially nothing, on every instance including standard
   Othello.** At α=1 the probe reads the requested board and the move distribution sits at the
   unedited floor (−0.70 against −0.93 on standard; −0.96 against −0.98 on standard-noflip).
   Standard Othello's headline +0.82 comes from a **3× overshoot**, adjacent-noflip's +0.18 from
   **60×**, at which point the write is 3.4× the activation norm and the fidelity ratio is 2.1.
2. **Within one model and one editor, identical probe-space landing spans the whole index.**
   Standard Othello reads 99.4% / 85.0% at both α=1 and α=3, with Edit Index −0.700 and +0.818.
   So "the probe reads the target" carries **no** information about whether the output moved.

`board_landed` < 100% is the probe's own decoding error, not the editor's: the write lands the
probe's requested logits exactly, and the comparison here is against the ground-truth board, so
it inherits the ~2.5% per-square miss (mean wrong squares 0.14–0.22). standard-noflip, whose
Probe Skill is 1.000, lands 100.0% of all 64 squares on 100% of cases — **the cleanest cell in
the project: a perfectly read target board with a completely unmoved output.**

## GS — achieves its objective most completely exactly where it is most destructive

| instance | tile landed (MLP probe) | board landed | Edit Index · fidelity |
|---|---|---|---|
| standard | 81.7% | 69.5% | +0.828 · 0.28 |
| adjacent-flip | 89.8% | 0.0% (18.3 wrong squares) | +0.138 · 1.75 |
| adjacent-noflip | **100.0%** | **99.6%** | **−0.157 · 6.68** |
| standard-noflip | **100.0%** | **99.9%** | **−0.523 · 4.25** |

100 Adam steps on the MLP probe's own loss converge perfectly on both no-flip variants — the
hold-the-rest term included, so the other 63 squares are held — and the prediction is destroyed
(fidelity 4–7). GS's *worst* behavioural cells are its *best* read-out cells.

## ND and IM

- **ND never lands the board** except on standard (65.3%): it is a direction addition, so it
  perturbs everything (38–51 of 64 squares misread on the three variants) while still flipping
  the edited tile 64–99% of the time.
- **IM is not built from the forward probe, so this is a consistency check, not a landing check.**
  It tracks editability: standard 98.8% board agreement (+0.806), adjacent-flip 37.9% (+0.558),
  adjacent-noflip 0.9% / 3.6 wrong squares (−0.025), standard-noflip 1.0% tile (−0.554). Where
  `g(s)` produces a residual the forward probe reads back as `s`, IM edits; where it does not,
  IM is destructive.

## What this changes

**The paper's claim should be strengthened and re-pointed.** The current Results says Othello
variants are not editable; the sharper fact is that *the probe-space write lands everywhere and
edits nowhere*, and that even the positive case needs a 3× overshoot past the landing point. This
is the Othello counterpart of the discworld α=1 sentence and it makes the cross-environment story
one claim instead of two: **probe-space landing is not evidence of editability in either
environment.** It also closes the "did you even land the edit?" referee question for the negatives.

Owed: the same check on the discworld categorical arms for symmetry (the continuous ones are
already recorded), and a decision on whether the per-α landing table belongs in the appendix.
