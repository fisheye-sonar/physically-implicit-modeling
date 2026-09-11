# Blink ablation — position as a carried state is still not editable (2026-09-08)

**Question.** In every discworld instance the current frame shows every object, so a model
never has to remember a position; the flip ablation had shown that a decodable variable
the dynamics do not use is not editable. dw-blink removes an object from the observation
for a run of frames (physics unchanged; 0.5 markers on its edge ray the frame before and
the last hidden frame — `pim/environments/discworld/blink.py`), so predicting its
reappearance requires a carried, advanced position. Does editability turn on where the
model must carry the state?

**Answer: no.** `runs/blink_ablation/L-dw-blink-20m` (Transformer-L, 20M sequences, 780k
steps, the matched recipe; val MSE 0.00126) carries position through blackouts — the
probes read it on hidden frames well above every observation floor — and the canonical
editors move it no better on reappearance cases than on visible ones, and not at all
while the object is hidden.

## Canonical scoring (first 192 edits, `scores.json`)

| | dw-blink | dw-noiseless (reference) |
|---|---|---|
| Probe Skill LIN / MLP (frustum) | 0.904 / 0.992 | 0.959 / 0.996 |
| unedited | −0.909 | −0.924 |
| PI | +0.215 / fid 1.78 | +0.233 / 1.95 |
| GS | −0.088 / 1.00 | −0.099 / 0.99 |

Observation floors on dw-blink (right-aligned, large corpus): LIN 0.22 cartesian / 0.35
frustum, MLP 0.81 / 0.91; random-init LIN 0.61 / 0.72. The linear floor collapses because a
current-frame lookup fails on hidden frames; the trained model's linear read (0.90) does not.

## Subset editability (`experiments/blink_ablation/scores/summary.md`, 192 cases each, frustum)

| subset (edited object at frame 20) | unedited | PI best | fid | PI at fid ≤ 1.1 | GS best | fid |
|---|---|---|---|---|---|---|
| reappearance (hidden through 19, visible at 20; k̄ = 5.8) | −0.846 | +0.269 | 1.37 | +0.233 / 1.08 | −0.035 | 0.93 |
| reappearance, k ≥ 3 (k̄ = 7.3) | −0.826 | +0.241 | 1.39 | +0.205 / 1.10 | −0.033 | 0.94 |
| mid-blackout (hidden at 20; scored at reappearance) | −0.385 | −0.148 | 44.6 | −0.385 / 1.10 | −0.382 | 1.84 |
| visible (control) | −0.909 | +0.223 | 1.81 | +0.143 / 1.09 | −0.117 | 0.92 |

Cartesian is the same picture (reappearance PI +0.211 / 1.41, visible +0.181 / 1.75,
mid-blackout −0.196 / 38). The reappearance cases sit in the same weak, destructive PI
regime as the visible control (+0.2 at fidelity 1.4–1.8, ≈ +0.2 at a guarded fidelity),
GS is negative everywhere, and an edit applied while the object is hidden does nothing to
where it reappears unless it destroys the rollout (fidelity 38–45). ND is not reported for
discworld (SETTINGS).

## Decodability by visibility (held-out probe sequences, best point, frustum)

| probe | object | visible frames | hidden frames | 1 frame since seen | 3 | 6 | 10 |
|---|---|---|---|---|---|---|---|
| LIN | 0 / 1 | 0.911 / 0.958 | 0.630 / 0.889 | 0.632 / 0.909 | 0.642 / 0.899 | 0.624 / 0.875 | 0.579 / 0.841 |
| MLP-128 | 0 / 1 | 0.994 / 0.996 | 0.973 / 0.985 | 0.977 / 0.987 | 0.978 / 0.988 | 0.971 / 0.984 | 0.955 / 0.975 |

The position of a hidden object is read from the residual stream at 0.97–0.99 (MLP) and
0.63–0.89 (LIN), decaying only slowly with staleness. The representation carries the
state; the model uses it (it renders the reappearance); it is decodable; and the probe
directions still do not steer it.

## The two floors on the same hidden frames (2026-09-10, `scripts/hidden_frame_floors.py`)

The table above quotes the trained model against whole-split floors. Put on the SAME 4,000
held-out sequences and the same hidden-frame masks (frustum, object 0 / object 1, best point):

| source | visible | hidden | 1 frame since seen | 3 | 6 | 10 |
|---|---|---|---|---|---|---|
| trained · LIN | 0.91 / 0.96 | 0.63 / 0.89 | 0.63 / 0.91 | 0.64 / 0.90 | 0.62 / 0.88 | 0.58 / 0.84 |
| random-init · LIN | 0.66 / 0.88 | 0.33 / 0.52 | 0.45 / 0.65 | 0.37 / 0.57 | 0.28 / 0.45 | 0.15 / 0.26 |
| observation right-aligned · LIN (large) | 0.08 / 0.74 | −0.10 / 0.09 | −0.06 / 0.27 | −0.08 / 0.16 | −0.12 / 0.00 | −0.17 / −0.19 |
| observation left-aligned · LIN (large) | 0.03 / 0.49 | −0.02 / 0.36 | 0.02 / 0.45 | 0.00 / 0.40 | −0.02 / 0.32 | −0.07 / 0.19 |
| **trained · MLP-128** | 0.99 / 1.00 | **0.97 / 0.99** | 0.98 / 0.99 | 0.98 / 0.99 | 0.97 / 0.98 | **0.96 / 0.98** |
| random-init · MLP-128 | 0.95 / 0.98 | 0.77 / 0.88 | 0.86 / 0.92 | 0.83 / 0.91 | 0.74 / 0.86 | 0.58 / 0.74 |
| observation right-aligned · MLP-128 (large) | 0.91 / 0.96 | 0.71 / 0.83 | 0.80 / 0.88 | 0.80 / 0.89 | 0.68 / 0.81 | 0.46 / 0.62 |
| observation left-aligned · MLP-128 (large) | 0.87 / 0.92 | 0.79 / 0.88 | 0.86 / 0.91 | 0.83 / 0.90 | 0.77 / 0.86 | 0.64 / 0.78 |

The observation probes DO extrapolate a hidden object from the frames before the blackout
(MLP 0.7–0.9 one frame in), but their read decays with staleness — to 0.46–0.78 ten frames in
— and so does the random-init model's (0.58–0.74). The trained model's read does not decay
(0.96–0.98 at ten frames). So on hidden frames the trained model holds a position that neither
a shallow read of the history nor untrained features supply: position here is a COMPUTED,
carried variable in the strong sense, with a margin over both floors that grows with staleness
— and it is still not editable through the regression probes. (Object 1, the bright disc, is
easier for every source, as everywhere: GOTCHAS 2026-08-21.) `scores/hidden_frame_floors.json`.

## What this settles

The candidate sufficient condition "the environment forces the model to carry the
variable" is not sufficient. Together with the earlier exclusions (objective, interface,
observation resolution, decodability, weak causal use) every property we can name that
separates Othello from discworld has now been ruled out as the reason on the discworld
side, except the ones that are constitutive of the two worlds: a discrete combinatorial
state read out through a categorical head versus a continuous geometric state read out
through a regression head. The paper's honest framing is therefore "decodability is not
editability" with a matched cross-environment design, not "the conditions for
editability".

Caveats. (i) "Not editable" means not editable through linear / MLP-128 probe directions
with PI and GS; the oracle editors (counterfactual history overwriting) do steer discworld,
so the model responds to observation-level counterfactuals. (ii) The unedited index at the
reappearance step after a mid-blackout is only −0.39, not −0.85: the model's reappearance
prediction is imprecise enough to sit far from BOTH reference worlds, which is a tracking
quality statement worth a waterfall before it is quoted. (iii) 192 cases per subset from
a 20k-edit split; every arm and every case is in `subset_editability.json`.

## Figures

`experiments/blink_ablation/waterfalls/` draws the edit panels SELECTED BY BLINK PHASE
(reappearance k ≥ 3, and the visible control under the same arms), with the two marker rays
widened so they are visible — the canonical panel in the run dir draws the first 192 cases
at mixed phase and hides the markers under the axis spine. Read the 192-case Edit Index in
each column title, not the mean over the six drawn rows. Two things the panels show that the
table does not: reappearance cases carry NO ghost zone (the object was hidden at EF-1, so it
vacates no rays), and the model does not predict the markers in free-run (0.07 mean against a
GT 0.5, within 0.15 of 0.5 on 8-9% of marker frames vs 0.8% of non-marker frames) — correct,
since a blackout start is a coin flip and the MSE-optimal value is the base rate ~0.034.

Assets: `experiments/blink_ablation/` (pilot gates, scripts, scores);
`datasets/discworld/dw-blink/instance.json`; `logs/blink_ablation/dw_blink/`.
