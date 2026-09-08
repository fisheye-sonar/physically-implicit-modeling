# Discworld's decodability is largely NOT a product of training

**Status:** measured 2026-09-01 on the canonical instances, `EVAL_VERSION 2026-09-01.4`.
Source: `runs/_baselines/<instance>/baselines.json`, rendered as Table 3 in
`notebooks/build_full_table.ipynb`. Method and caveats: `research/REGISTRY.md`
§ Decodability baselines.

## The numbers (MLP-128 Probe Skill, best residual point)

| instance · basis | observation | random-init | trained | trained − random-init |
|---|---|---|---|---|
| dw-pn04 · cartesian | +0.460 | +0.870 | +0.973 | **+0.103** |
| dw-pn04 · frustum | +0.696 | +0.960 | +0.996 | **+0.036** |
| dw-noiseless · cartesian | +0.673 | +0.945 | +0.973 | **+0.028** |
| dw-noiseless · frustum | +0.854 | +0.987 | +0.996 | **+0.009** |
| oth-uniform · mine/theirs | +0.724 | +0.583 | +0.976 | **+0.393** |

## What it means

**An untrained transformer already supports 0.87–0.99 MLP decodability of discworld object
state.** On `dw-noiseless` in the frustum basis the trained model beats its own random-init
control by **0.009**. Training buys almost nothing in decodability terms: a random network
is a random feature expansion over the observation history, and object position survives it.

**Othello is the opposite.** Its random-init floor is +0.583 against +0.976 trained — a
+0.393 margin. There, high probe skill really is a fact about what training built.

This sharpens the project's central claim. The discworld result was "decodable but not
editable"; the honest version is stronger and more interesting: **on discworld the
decodability itself is close to vacuous**, so the failure of every editor to write through
that read-out is not a paradox — there was correspondingly little *learned* structure behind
the probe to write into. On Othello, where decodability IS a product of training, editing
works (best Edit Index +0.647, GS). Decodability that training produced is editable;
decodability that falls out of random features is not.

⚠ Do not restate this as "discworld probes are invalid". They read the state, held out by
sequence, with the MLP≥linear tripwire clean. The claim is about **provenance**, not
validity: a number that a random network nearly matches cannot carry the weight of "the
model learned a world model".

## Reading caution

The **observation** floor on discworld is an under-estimate: its MLP overfits (in-sample
minus held-out +0.10 to +0.46, against +0.000–+0.005 for the model probes), because 4,992
input features on 936k rows is ~1.5 rows/parameter. It is not load-bearing here — random-init
is the higher floor everywhere on discworld, and that one's gap is +0.0001 to +0.005. If the
observation floor ever becomes the binding one, shrink its hidden layer first.

Related: `research/findings/state-geometry.md` (the frustum basis), `editability.md`.
