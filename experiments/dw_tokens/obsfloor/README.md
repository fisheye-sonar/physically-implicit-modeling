# dw_tokens_obsfloor — observation-space decodability floors for TOKENISED discworld

**Question.** Table 3 puts the token run (`L-dw-8ray-tok-20m`) against the canonical
observation floor, which reads the causal FLOAT frame history (39 × 8 = 312 features). For
a token model the input is a one-hot frame and its embedding is a linear map of it, so a
linear probe on the one-hot history can implement any per-frame lookup — the fair LINEAR
floor is the one-hot history, and it was missing. (The MLP floor is the same information
either way; it is refitted here on the one-hot for completeness.)

**How.** `scripts/obs_floor.py` — the canonical streamed baseline fit
(`pim.probes.baselines.CausalHistory(kind="one_hot")` + `fit_probe_stream`: same split by
sequence, same targets (full state, cartesian + frustum), same objective and stats as every
Table 3 row) on two feature sets and two corpora:

| features | d_in | LIN params | MLP-128 params | corpora |
|---|---|---|---|---|
| frame one-hot (the model's literal input) | 39 × 422 = 16,458 | 132k | 2.1M | 30k (24k train seqs → 0.4 rows/param) and 250k (200k → 3.7 rows/param, 50 epochs) |
| per-ray one-hot (8 rays × 3 levels) | 936 | 7.5k | 121k | both |

Two things the canonical fit needed for one-hot inputs, both recorded in GOTCHAS
(2026-09-06): one-hot features are NOT standardised (an experiment-side override of the
fit's input affine — rare (position, frame) pairs have std ≈ 0 and blew both fits up), and
the streamed linear closed form now solves the normal equations with a hermitian
pseudo-inverse (CUDA's `gels` assumes full rank; a one-hot design is not — a one-line
canonical fix, identical on full-rank systems).

**Where.** Every fitted probe persists in `probes/` (ProbeCache, kind
`observation_tokens`; `_archive_*` hold the smoke fits made before the two fixes).
`scores/obs_floor_dw-8ray-tokens.json` + `scores/summary.md`, which also lists the float
floors, the token architecture's random-init floor and the trained token model's skills.
Nothing is written into `runs/_baselines/` or the master tables — fold-in is a separate
decision. Driver `drivers/obs_floor.sh`; logs `logs/dw_tokens/obsfloor/`.

**Status.** 2026-09-06 launched after the bridge (unit `dw_tok_exp`). Smoke:
`scores/*_smoke.*`.

## Alignment (found 2026-09-06 while reading the first results)

The canonical `CausalHistory` is left-aligned (block j = frame j), so a linear model over
it cannot express a current-frame lookup; the first LIN floors (0.65–0.73) sat below the
random-init token model's point-0 skill (0.968), which IS that lookup — a contradiction
for a function class that should contain it. `scripts/obs_floor.py::RightAlignedHistory`
lays the history out relative to the present (block 0 = current frame, block k = k steps
back), same information and rows. Both alignments are reported; the right-aligned LIN row
is the fair linear floor for a token model. GOTCHAS 2026-09-06 records the canonical
implication (Table 3's float LIN observation row is understated for the same reason).

## Results (2026-09-06; `scores/summary.md` has every row, both alignments, both corpora)

Frustum basis, 250k corpus (in-sample gaps ≤ 0.004 throughout):

| features | LIN | MLP-128 |
|---|---|---|
| frame one-hot history, right-aligned (the fair floor) | **0.973** | **0.983** |
| frame one-hot history, left-aligned (canonical layout) | 0.726 | 0.978 |
| per-ray one-hot history, right-aligned | 0.871 | 0.980 |
| float frames (canonical Table 3 floor) | 0.279 | 0.925 |
| random-init token model, best point | 0.968 | 0.968 |
| **trained token model, best point** | 0.968 | 0.980 |

Cartesian: right-aligned frame one-hot 0.909 / 0.933 vs the trained token model 0.899 / 0.935.

- **A shallow read of the token input matches the trained token model in both families.**
  The right-aligned one-hot history LIN floor (0.973 / 0.909) sits slightly ABOVE the
  model's LIN skill (0.968 / 0.899), and the one-hot MLP floor (0.983 / 0.933) equals its
  MLP skill (0.980 / 0.935). Training added nothing to decodability over the observation.
- **The current frame is most of it.** A shared lookup of the current one-hot frame alone
  gives 0.968 (frustum) — the random-init model's point 0 IS that lookup — and the full
  39-frame right-aligned history adds only +0.005 linearly.
- **The parameter fear was real on 30k, gone on 250k**: the 2.1M-parameter one-hot MLP
  overfits at 0.4 rows/param (gap 0.02–0.08) and is clean at 3.7 rows/param (gap ≤ 0.004).
- **Alignment matters only for LIN**: left-aligned 0.726 vs right-aligned 0.973 (frame
  one-hot), 0.627 vs 0.871 (per-ray); the MLP learns the alignment either way.
- Not folded into Table 3 (the fold-in decision is open; see the README head and
  GOTCHAS 2026-09-06 on the canonical LIN observation row).
