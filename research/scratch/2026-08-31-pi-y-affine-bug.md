# 2026-08-31 — PI injection solved for the wrong units on discworld (the y-affine bug)

**What.** `WorldStateProbe.forward` for regression is a three-stage sandwich:
`z = (h − x_mean)/x_std → net(z) → · y_std + y_mean`. The discworld PI decomposition
(`othello_arch/editability.py::_decompose`) collapsed only the first two stages, so the
editor solved `standardised-y read-out = raw-units target`. On the cartesian probes the
depth row has `y_mean 7.93, y_std 1.91` — asking a dial calibrated in σ-units for the
value 10 is a ~9σ demand, and the pseudoinverse faithfully produced the huge, useless
write. The fingerprint was in the logs the whole time: `readout_err_before = 11.34`
when a teleport is ~3 world units, and best-α pinned at 175, the top of every sweep.

**Why it survived.** Othello was structurally immune — classification probes carry
`y_mean=0, y_std=1` (`fit_probe` sets them so for `n_classes≠None`) — so the side with
a published answer to validate against could never show it. Nine days, every discworld
PI number.

**Measured effect (BIG20M_discworld_L, 30k probes, 192-case bench).** Three arms:
A = legacy (reproduces the published numbers exactly), B = raw-space pinv + affine,
C = z-space pinv + affine (what Othello always did):

| basis/target | A best EI | B best EI | C best EI | C @ α=1 |
|---|---|---|---|---|
| cart/pos | +0.0475 | +0.1618 | +0.1754 | −0.6215 |
| cart/full | +0.0522 | +0.1570 | +0.1615 | −0.6216 |
| frustum/pos | +0.0874 | +0.2332 | +0.1988 | −0.5895 |
| frustum/full | +0.0854 | +0.2381 | +0.1856 | −0.5884 |

Fixing the bug roughly **triples** the best EI — and **sharpens the negative**: at α=1
(the exact jump that provably lands the read-out; `readout_err_after ≈ 0`) the
generation barely moves off the unedited floor (−0.70), and every positive-EI arm needs
writes 4–19× the activation norm with fidelity > 1. The z-vs-raw (whitening) question I
originally suspected is second-order; the affine was the whole story.

**Resolution (canonical since today).**
- `pim/editors/pinv.py`: canonical space = `"zspace"` (y-affine included, min-norm in
  activation-σ units); `"raw"` kept as the variance-blind comparison; `"legacy"` kept
  ONLY to reproduce pre-fix numbers and never quoted as PI.
- The landing check now goes through `probe.forward` itself (`readout_error`), so a
  decomposition can never again agree only with itself —
  `tests/test_editors_canonical.py` asserts zspace/raw land and legacy misses.
- Every stored discworld PI number is the legacy arm; `runs/**/scores.json` under
  EVAL_VERSION ≥ 2026-08-31.1 carries the corrected canon.

**Moral** (GOTCHAS-grade): when an editor's optimum pins at the edge of its sweep, the
sweep isn't too small — the units are wrong. And validate the decomposition against the
probe's own forward pass, not against a re-derivation of it.
