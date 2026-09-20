# history_rewrite — persistent transformer edits by rewriting the history (appendix)

`make_figure.py` → `scores.json` + three waterfalls beside it. Standalone: it imports only `pim.*` and reads
`runs/noise_ablation/L-dw-noiseless-20m` (its `scores.json` for the IM editor's scored point, its cached
inverse map) and the dw-noiseless edit bench. Moved here 2026-09-19 from `experiments/history_rewrite/`.

**Idea (Sevan, 2026-09-16).** A transformer's edit at the last position is forgotten: the next prediction
is recomputed from the unedited observation history. The post-edit target state carries the edited disc's
velocity, so its counterfactual trajectory can be integrated BACKWARDS over the whole teacher-forced
window. At every history step t the residual at the IM point is overwritten with g(s_cf[t]) — the inverse
map of the counterfactual state — and the model's own prediction becomes the rewritten frame t+1. The
rewritten history replaces the window the edit is launched from. Frame 0 is kept (nothing precedes it).

**Arms** (canonical bench, first 32 selected cases, Cartesian block, scored like any arm — Edit Index and
fidelity ratio from `pim`): `unsteered` · `IM` (canonical: original history, g(s_post) at EF−1) ·
`hist` (rewritten history, NO write at EF−1) · `hist+IM` (both). Recorded 2026-09-16: the rewritten history
alone carries the edit (+0.65 / 0.48); with the write on top +0.63 / 0.36 against the canonical +0.61 / 0.36.

**Outputs.** `A_original_history_arms.png` · `B_rewritten_history_arms.png` ·
`C_history_vs_counterfactual_render.png` (the rewritten frames against the simulator's clean render of the
counterfactual history, with the RMSEs) · `scores.json` (every arm's scorecard + the history RMSEs).

Run: `.pim/bin/python paper/figs/history_rewrite/make_figure.py` (GPU, about a minute; n = 32 cases, one seed —
an illustration, not a table number).
