# edit_direction_alignment — is the probe's writable subspace where the true edit direction lives? (2026-09-09)

A reimplementation at scale of the RNN-era latent-geometry analysis: for each edit case an
ORACLE counterfactual history (the simulator run on the edited world) gives the residual the
model produces when the state really is the target, so Δ = h_cf − h is the true edit
direction at every residual point. We measure how much of Δ lies in the linear probe's row
space, in the Haufe et al. (2014) activation-pattern subspace (the forward directions a
min-norm backward probe misses), and in the rows projected onto the top principal
components; against a generic displacement baseline. Discworld adds the causal check
(patch h with Δ, its row-space part, its complement → canonical Edit Index) and the oracle
linear editor's ceiling (Δ regressed on the teleport). Othello's counterfactual histories are
found by search over move substitutions / swaps with the instance's own simulator.

Models: `L-dw-noiseless-20m`, `L-oth-20m`, `L-oth-adjacent-20m`; existing linear probes only;
no canonical code changes. `scripts/{common,discworld_alignment,othello_alignment}.py` →
`scores/*.json`; `summary.md` after the runs.

**Status (2026-09-09): DONE.** Archive result replicated (discworld rows hold 1% of Δ = generic; Haufe/PCA ≤ 4%); standard Othello 4–6× generic at the editable points, adjacent at generic; Haufe does not order the models. New: patching discworld's last-position residual with the full Δ produces the edit (+0.9), the complement alone +0.85, the row part alone nothing — the residual is load-bearing and position's load-bearing code is nonlinear/high-rank. Write-up `research/findings/edit-direction-alignment.md`.

**2026-09-10 — Result 6.** `scripts/fac_probe_alignment.py [run] [target]`: the factorised categorical probes (`appearance-fac`, `L-dw-8ray-20m`) against the same oracle Δ, with the joint-cell and regression probes on the same 66 valid cases; raw rows 2.8× generic / 12× random rank, ND direction cos² 0.056 (28× random), Haufe at the random-pattern floor. `scores/fac_probe_alignment_L-dw-8ray-20m.json`.
