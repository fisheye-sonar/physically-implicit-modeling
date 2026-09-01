# 2026-08-31 — GS is NOT dead in mine/theirs: the old negative was a target-frame mismatch

**Claim retired.** Every previous record said Li-style gradient steering fails on our
retrained Othello model when driven through mine/theirs probes (best EI **−0.0014**,
2026-08-24), and that only absolute-colour ("state") probes worked. That was a **bug in
the arm, not a fact about the coordinate frame**.

**The bug.** The old arm fed *mine/theirs* probes the benchmark's *absolute-colour*
target labels (`bench.new_class = 2 − ori_color`). The two encodings share the integers
but not the meaning — under mine/theirs `{0,1,2} = {blank, mine, theirs}`, under
absolute `{0,1,2} = {white, blank, black}` — so the descent was driven toward a
well-formed but semantically wrong class index at every case.

**Isolation (this model, pt 0, α 0.05, n_steps 100, β 0.2 — only the frame and the
probe set vary):**

| probes | targets | Edit Index | Li error vs post |
|---|---|---|---|
| NEW canonical mine MLP-128 | **mine** | **+0.6451** | 0.044 |
| NEW canonical mine MLP-128 | absolute | −0.0534 | 1.890 |
| OLD 2026-08-24 mine MLP-128 | **mine** | **+0.6451** | 0.044 |
| OLD 2026-08-24 mine MLP-128 | absolute | −0.0534 | 1.890 |

The old and new probes agree to four decimals in both frames: **probe quality was never
the issue**, and the canonical port changed nothing about the probes. The target frame
moves the result by 0.70 Edit Index. (The old sweep did include α = 0.05 at all nine
points, so this is not a grid artefact either — its α 0.05 row reads −0.053, exactly
reproduced above.)

**Consequence — GS-mine is the strongest editor we have on Othello** (full sweep,
`L-oth-20m`, EVAL_VERSION 2026-08-31.2):

| editor | probes | best EI | Li vs post | Li vs pre | legal mass |
|---|---|---|---|---|---|
| **GS-mine** | mine\|mlp128 | **+0.6459** (pt 4, α 0.05) | **0.036** | 2.272 | 0.9969 |
| ND-sub | mine\|linear | +0.6216 | 0.102 | 2.364 | 0.9941 |
| PI | mine\|linear | +0.6104 | 0.114 | 2.486 | 0.9897 |
| GS (state) | state\|mlp128 | +0.4708 | 0.146 | 1.786 | 0.9884 |

Li vs post **0.036** is well below Li et al.'s published best intervention (0.12), with
guards clean (vs-pre 2.27 ≈ the 2.76 unedited floor; legal mass 0.997). Every workhorse
editor now agrees on Othello, and all four best arms read through **mine/theirs**
probes — Nanda's frame wins for editing as well as for decoding.

**Why it survived.** The mismatch is invisible from either side alone: the probe is
healthy (skill ≈ 0.97), the descent converges, the write lands — on the wrong class.
Only steering the *same* probes in both frames exposes it. `grad_steer_arm` now takes
an explicit `target_labels`, and `PROBE_SOURCES` in every `scores.json` records which
probes each editor read, so the pairing is stated rather than implied.

**Open, for Sevan:** with the frame fixed, `state` probes are no longer needed by any
editor — the grid could drop to mine-only (72 → ~29 fits, Othello probe time ~40 → ~15
min/model). Deferred until he has seen the full table.
