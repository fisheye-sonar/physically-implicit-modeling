# 2026-08-18 — Othello-GPT's probe + intervention, ported exactly: the probing replicates, the editing does not

**Direction:** none open — Sevan-directed, in response to Li et al. (ICLR 2023, arXiv:2210.13382).
**Thread:** `notebooks/experiments/editability/othello_gpt/` (notebook + `othello_probe.py` + `pipeline.py` + README).
**Models:** `runs/transformers/{W2,W4,W16}` (no new world models). **Data:** `datasets/4_fixed_refl_inview`,
edits split, `ef=20`, `K=15`, **N=256** edits; probes on 1500 test sequences held out **by sequence**.
**Findings updated:** `findings/editability.md` (2026-08-18 entry, `replicated`).

## Why

Every editor in this thread is a **probe-derived write**, and every one fails. Othello-GPT is the
strongest published claim that exactly this kind of write *succeeds*. Porting their method unchanged
separates "their method is better than ours" from "their world is different from ours".

## What was copied exactly

Probe families (linear vs **one-hidden-layer** MLP, their §3.1/§3.2); the update rule
`x' ← x − α ∂L(p_θ(x), B')/∂x` on the **activation**; the **sequential multi-layer schedule** (§4.1 /
Fig 2C — write at the last timestep at residual point `L_s` and **every point after it**, alternating
write and compute); the hold-the-rest term with weight `β` (App. G); and their null-intervention
baseline. Two probe targets per Sevan: positions only (4 dims) and the **whole world state** (8 dims,
positions + velocities), identical edit objective.

**Deviations, both deliberate:** (1) regression + R² instead of 3-way classification + error rate —
our state is continuous, probe *shape* unchanged; (2) held out **by sequence**, not by frame — their
by-frame split leaks a trajectory-constant velocity label (`GOTCHAS.md`, +0.34 R² inflation). Also
Adam rather than plain GD for the activation update, which the paper explicitly sanctions (App. G);
plain GD is run alongside and reported, and it does **not** behave the same (below).

## Results

**1. The probing half replicates cleanly.** Best position R²: **linear 0.798 → MLP 0.934** (+0.136),
MLP rising monotonically with depth (0.796 at the encoder port → 0.934 at block 3). Their §3 headline
— a nonlinear world representation is present and a linear probe under-reads it — holds here.

**2. The intervention half does not.** Read-out driven **3.35 → 0.007–0.018** sim units (99.5%
reduction) at every applied layer; Edit Index moves only **−0.684 → −0.538**, a gain of **+0.146** on
a ±1 scale. The waterfall is unambiguous: the object stays on the ghost locator and never reaches the
target, in every sample and every arm.

**3. Ignored, not destroyed, and it reverts within one frame.** Fidelity ratio **0.993–0.999**
everywhere (no arm near the 1.05 guard). Arms collapse onto the unsteered curve **by step 1**; the gap
decays **+0.146 → +0.010** by step 14. Precise word: *reverts*.

**4. Earlier applied layers propagate further.** −0.538 (point 0/1) → −0.565 (2) → −0.606 (3) →
−0.622 (point 4), matching the structural prediction that an edit at point ℓ changes block inputs for
layers > ℓ only.

**5. The full-state probe changes nothing.** −0.539 vs the position probe's −0.538. But this is a
*weak* test of completeness, because velocity is barely readable to begin with (per-dim R² **−0.04 to
0.45**; `obj0 vy` ≈ 0). Reported honestly rather than as "world-state completeness does not help".

**6. The single-frame ceiling on this model is itself low.** Oracle observation (model simply *shown*
the true post-edit frame) reaches **+0.126**, decaying to −0.030. The probe write achieves ~**18%** of
that (+0.146 of +0.810). Any claim about how badly the write fails has to be read against a reference
that is itself far from +1.

**7. The optimiser selects which probe-satisfying write you land on — and it matters.** At a matched
selection rule (lowest read-out error), Adam's write is **1.7–4.9× larger in norm** and moves the
generation; plain GD lands the read-out with a smaller write and moves it essentially not at all
(point 0: read-out 0.192, Edit Index **−0.680** = unsteered). So the set of activations satisfying the
probe is large and the probe constraint does not pin down a member the dynamics honour. This is the
same shape as `2026-08-05-tangent-constrained-injection` (a 252-dim affine solution set) arrived at
from a different direction.

**8. Flat across attention windows.** Gain over each model's own unsteered index: **+0.153** (W2) /
**+0.137** (W4) / **+0.146** (W16), best probe R² 0.932/0.931/0.934.

## Reading (not established)

Does **not** contradict Li et al. — it locates the difference, and this notebook does not separate the
two candidates: (a) **the world** — their board state is discrete and exactly determined by the move
sequence, and the tile they flip is consumed directly by the legal-move computation, whereas ours is
continuous and reaches the output only through a renderer (lines up with `2026-08-05-observation-space-geometry`,
which put `readable ≠ grabbable` in the *world* rather than the model); (b) **the read-out** — their
probe predicts a quantity the next-token computation consumes, ours predicts one merely correlated
with what the decoder consumes. Result 7 sharpens (b).

**What it does settle for the thread:** the probe-derived-write failure is **not** an artefact of this
repo's editor implementations. The strongest published version of that method — its schedule, its
loss, its multi-layer write, its own baseline — fails here too.

## Gates run (all passed, printed in the notebook)

- `state_from_obs.decode` vs one-pass teacher-forced forward: max|diff| **8.3e-07**.
- Identity activation write == plain free-run: max|diff| **exactly 0** (so `Unsteered` is the right null).
- One intervention per episode: max object jump at the edit frame **2.97**, at any other frame **0.246**.

## Open / owed

- Does **not** separate "the world" from "the read-out" (the two candidates above). The cleanest next
  test is a probe trained to predict something the decoder provably consumes.
- Velocity readability is low enough that the full-state arm is a weak completeness test; a world with
  more readable velocity would make it sharper.
- Only the transformer. The same port on the GRU would say whether the two-state structure matters.
- Ceiling caveat: the oracle observation itself only reaches +0.126 on this model.
